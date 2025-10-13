import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import List
from transformers import AutoModel, AutoConfig, AutoTokenizer
from llm2vec import LLM2Vec

device = "cuda" if torch.cuda.is_available() else "cpu"

llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
llm_model = AutoModel.from_pretrained(
    llm_model_name,
    torch_dtype=torch.bfloat16,
    config=config,
    trust_remote_code=True
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

vision_model_path = "microsoft/LLM2CLIP-Openai-B-16"
model = AutoModel.from_pretrained(vision_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

CONFIG = {
    "wikiart_none": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_none.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_none_llm2clip_features.dpt",
        "multi_caption": False
    },
    "wikiart_few": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_few.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_few_llm2clip_features.dpt",
        "multi_caption": False
    }
}
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def encode_texts(texts: List[str], batch_size=64) -> torch.Tensor:
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts with LLM2CLIP"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            raw_emb = l2v.encode(batch, convert_to_tensor=True).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            proj_emb = model.get_text_features(raw_emb)
            proj_emb = torch.nn.functional.normalize(proj_emb, dim=-1)
        all_feats.append(proj_emb.cpu())
    all_feats = torch.cat(all_feats)
    print(f" [LLM2CLIP-B16] Text feature shape: {all_feats.shape}")
    return all_feats

def process(name, cfg):
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
    print(f"[{name}] Text features saved to: {cfg['save_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None, help="Only process one dataset")
    args = parser.parse_args()

    for name, cfg in CONFIG.items():
        if args.target and name != args.target:
            continue
        process(name, cfg)

print("✅ All text features extracted (LLM2CLIP-B16)")
