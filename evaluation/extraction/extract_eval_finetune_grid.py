import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from llm2vec import LLM2Vec as LLM2VecBase
from torch.nn.functional import normalize
from torch import nn
import torch.nn.functional as F

# ====== Text Adapter Loader (Normalize + Linear) ======
def load_text_adapter():
    model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-B-16", trust_remote_code=True)
    return model.text_adapter.eval()

# ====== LLM2Vec Wrapper with Adapter ======
class LLM2VecOptionalProj(LLM2VecBase):
    def __init__(self, model, tokenizer, projection=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.projection = projection.to(model.device) if projection is not None else None
        self.norm = nn.LayerNorm(4096)  # 🔹必须显式添加，与训练对齐

    def forward(self, sentence_feature):
        embed_mask = sentence_feature.pop("embed_mask", None)
        reps = self.model(**sentence_feature, output_hidden_states=True)
        hidden = reps.hidden_states[-1]                          # shape: [B, T, 4096]
        sentence_feature["embed_mask"] = embed_mask

        hidden = self.norm(hidden.to(torch.float32))             # 🔹LayerNorm先于 pooling
        pooled = self.get_pooling(sentence_feature, hidden)      # 🔹mean pooling
        if self.projection is not None:
            pooled = self.projection(pooled.to(torch.float32))   # 🔹Linear(4096→1280)
        return normalize(pooled, dim=-1)                         # 🔹最终 normalize


class ProjectionWithNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4096)
        self.linear = nn.Linear(4096, 1280, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x



def load_full_projection(proj_path):
    model = ProjectionWithNorm()
    state_dict = torch.load(proj_path, map_location="cpu")

    # 全部转为 float32
    for k in state_dict:
        state_dict[k] = state_dict[k].float()

    model.load_state_dict(state_dict)
    return model.eval()


def load_modified_text_adapter(custom_proj_dict_path):
    model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-B-16", trust_remote_code=True)
    adapter = model.text_adapter

    trained_proj = load_full_projection(custom_proj_dict_path)
    trained_weight = trained_proj.linear.weight.detach().float()

    adapter.adaptor[5].weight.data.copy_(trained_weight)

    return adapter.eval()



# ====== JSON Loader ======
# def load_json(file_path):
#     with open(file_path, "r") as f:
#         return json.load(f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# ====== Text Encoder ======
def encode_texts(texts: List[str], l2v, device, batch_size=64) -> torch.Tensor:
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = l2v.encode(batch, convert_to_tensor=True).to(device)
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"⚠️ NaN/Inf in batch {i} ~ {i+batch_size}, skipping.")
                continue
            all_feats.append(emb.cpu())
    if not all_feats:
        return None
    all_feats = torch.cat(all_feats)
    print(f"✅ Encoded feature shape: {all_feats.shape}")
    return all_feats

# ====== Projection Loader (Normalize + Linear) ======
def load_proj_linear_only(proj_path):
    proj = nn.Linear(4096, 1280, bias=False)
    proj.load_state_dict(torch.load(proj_path))
    return proj.eval()

# ====== Main Pipeline ======
def extract_embeddings(base_model_path, lora_path, proj_path, mode, ann_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

    # ==== 根据 mode 加载模型和 adapter ====
    if mode == "lora_only":
        model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16).to(device).eval()
        projection_layer = load_text_adapter().to(device)
    elif mode == "proj_only":
        model = base_model
        projection_layer = load_modified_text_adapter(proj_path).to(device)
    elif mode == "all":
        model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16).to(device).eval()
        projection_layer = load_modified_text_adapter(proj_path).to(device)
    elif mode == "none":
        model = base_model
        projection_layer = load_text_adapter().to(device)

    else:
        raise ValueError("Unsupported mode. Choose from: all, lora_only, proj_only, none")

    l2v = LLM2VecOptionalProj(model, tokenizer, projection=projection_layer, pooling_mode="mean", max_length=512)

    # ==== Load data and encode ====
    print(f"📄 Loading annotations from: {ann_path}")
    data = load_json(ann_path)
    # texts = [item["text"] for item in data]
    texts = [item["query"] for item in data]

    text_feats = encode_texts(texts, l2v, device)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(text_feats, save_path)
    print(f"💾 Saved embeddings to: {save_path}")

# ====== CLI ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (if any)")
    parser.add_argument("--proj_path", type=str, default=None, help="Path to projection .pt (if any)")
    parser.add_argument("--mode", type=str, choices=["all", "lora_only", "proj_only", "none"], required=True)
    parser.add_argument("--ann_path", type=str, required=True, help="Path to input annotation JSON")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output .pt file")

    args = parser.parse_args()

    extract_embeddings(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        proj_path=args.proj_path,
        mode=args.mode,
        ann_path=args.ann_path,
        save_path=args.save_path
    )
