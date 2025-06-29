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

# ====== Custom LLM2Vec wrapper ======
class LLM2VecLoRA(LLM2VecBase):
    def __init__(self, model, tokenizer, adapter_module, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.adapter = adapter_module.to(torch.float32)

    def forward(self, sentence_feature):
        embed_mask = sentence_feature.pop("embed_mask", None)
        reps = self.model(**sentence_feature, output_hidden_states=True)
        hidden = reps.hidden_states[-1]
        if torch.isnan(hidden).any():
            print("⚠️ WARNING: NaNs detected in hidden states!")
        sentence_feature["embed_mask"] = embed_mask
        pooled = self.get_pooling(sentence_feature, hidden)  # [B, 4096]
        projected = self.adapter(pooled.to(torch.float32))  # adapter includes normalize + projection
        return normalize(projected, dim=-1)

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Model Paths ======
llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
lora_model_path = "/scratch-shared/ywang4/lora_outputs/finetune2_rank4_lora/final_model"

# ====== Load tokenizer, model, and adapter ======
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
base_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
llm_model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=torch.bfloat16).to(device).eval()

def load_llm2clip_adapter():
    model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-B-16", trust_remote_code=True)
    return model.text_adapter  # Full LLM2CLIP_Adapter module

adapter = load_llm2clip_adapter()

# ====== Initialize LLM2Vec with Adapter ======
l2v = LLM2VecLoRA(llm_model, tokenizer, adapter, pooling_mode="mean", max_length=512, doc_max_length=512)

# ====== Config ======
CONFIG = {
    "wikiart_raw": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_raw.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_raw_lora.dpt",
        "multi_caption": False
    },
    "wikiart_finetune": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_new.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_new_lora.dpt",
        "multi_caption": False
    }
}

# ====== Utility Functions ======
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def encode_texts(texts: List[str], batch_size=64) -> torch.Tensor:
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts with LLM2CLIP"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = l2v.encode(batch, convert_to_tensor=True).to(device)
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print("⚠️ WARNING: NaN or Inf detected in projected embeddings. Skipping batch.")
                continue
            all_feats.append(emb.cpu())
    all_feats = torch.cat(all_feats)
    print(f"✅ [LLM2CLIP] Text feature shape: {all_feats.shape}")
    return all_feats

def process(name, cfg):
    print(f"\n📝 Processing: {name}")
    data = load_json(cfg["ann_path"])

    if cfg["multi_caption"]:
        texts = [caption for item in data for caption in item["text"]]
        cap_per_img = len(data[0]["text"])
        text_feats = encode_texts(texts)
        text_feats = text_feats.view(-1, cap_per_img, text_feats.size(-1))
    else:
        texts = [item["text"] for item in data]
        text_feats = encode_texts(texts)

    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    torch.save(text_feats, cfg["save_path"])
    print(f"📁 Saved features to: {cfg['save_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None, help="Only process one dataset")
    args = parser.parse_args()

    for name, cfg in CONFIG.items():
        if args.target and name != args.target:
            continue
        process(name, cfg)

    print("🎉 All text features extracted using finetuned model with correct projection.")
