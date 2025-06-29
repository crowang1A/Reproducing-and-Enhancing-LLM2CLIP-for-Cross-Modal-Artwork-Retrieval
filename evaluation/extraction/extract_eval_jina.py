import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import List
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 使用 Hugging Face 上的 Jina-CLIP 模型
model_name = "jinaai/jina-clip-v1"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# CONFIG = {
#     "flickr": {
#         "ann_path": "data/eval_data/flickr30k/test.json",
#         "save_path": "data/eval_data/flickr30k/flickr30k_jina_features.dpt",
#         "multi_caption": True
#     },
#     "coco": {
#         "ann_path": "data/eval_data/coco/coco_karpathy_test.json",
#         "save_path": "data/eval_data/coco/coco_jina_features.dpt",
#         "multi_caption": True
#     },
#     "urban1k": {
#         "ann_path": "data/eval_data/Urban1k/test.json",
#         "save_path": "data/eval_data/Urban1k/urban1k_jina_features.dpt",
#         "multi_caption": False
#     },
#     "docci": {
#         "ann_path": "data/eval_data/docci/test.json",
#         "save_path": "data/eval_data/docci/docci_jina_features.dpt",
#         "multi_caption": False
#     }
# }

CONFIG = {
    "wikiart_raw": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_with_artist.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_artist_jina_features.dpt",
        "multi_caption": False
    }
}



def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def encode_texts(texts: List[str], batch_size=64) -> torch.Tensor:
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            feats = model.encode_text(batch)
            feats = torch.tensor(feats, dtype=torch.float32, device=device)  # ✅ 转为 tensor
            feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    all_feats = torch.cat(all_feats)
    print(f"✅ [Jina-CLIP] Text embedding shape: {all_feats.shape}")
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
    print(f"[{name}] Saved to: {cfg['save_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None, help="Run only on a specific dataset key in CONFIG")
    args = parser.parse_args()

    for name, cfg in CONFIG.items():
        if args.target and name != args.target:
            continue
        process(name, cfg)

print('jina done')
