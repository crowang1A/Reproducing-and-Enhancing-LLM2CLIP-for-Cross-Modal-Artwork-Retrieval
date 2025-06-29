import os
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import CLIPTokenizer, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OpenAI CLIP from HuggingFace
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model.eval()

# CONFIG = {
#     "flickr": {
#         "ann_path": "data/eval_data/flickr30k/test.json",
#         "save_path": "data/eval_data/flickr30k/flickr30k_openai_features.dpt",
#         "multi_caption": True
#     },
#     "coco": {
#         "ann_path": "data/eval_data/coco/coco_karpathy_test.json",
#         "save_path": "data/eval_data/coco/coco_openai_features.dpt",
#         "multi_caption": True
#     },
#     "urban1k": {
#         "ann_path": "data/eval_data/Urban1k/test.json",
#         "save_path": "data/eval_data/Urban1k/urban1k_openai_features.dpt",
#         "multi_caption": False
#     },
#     "docci": {
#         "ann_path": "data/eval_data/docci/test.json",
#         "save_path": "data/eval_data/docci/docci_openai_features.dpt",
#         "multi_caption": False
#     }
# }

CONFIG = {
    "wikiart_none": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_none.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_none_clip_features.dpt",
        "multi_caption": False
    },
    "wikiart_few": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_few.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_few_clip_features.dpt",
        "multi_caption": False
    },
    "wikiart_new": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_new.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_new_clip_features.dpt",
        "multi_caption": False
    }
}

# CONFIG = {
#     "wikiart_raw": {
#         "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_with_artist.json",
#         "save_path": "/scratch-shared/ywang4/wikiart/wikiart_artist_clip_features.dpt",
#         "multi_caption": False
#     }
# }

def load_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)

def encode_texts(texts: List[str], batch_size=64) -> torch.Tensor:
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs).cpu()
        all_feats.append(feats)
    return torch.cat(all_feats)

def process(name: str, cfg: Dict[str, Any]):
    data = load_json(cfg["ann_path"])
    if cfg["multi_caption"]:
        texts = [caption for item in data for caption in item["text"]]
        text_feats = encode_texts(texts)
        cap_per_img = len(data[0]["text"])
        text_feats = text_feats.view(-1, cap_per_img, text_feats.size(-1))
    else:
        texts = [item["text"] for item in data]
        text_feats = encode_texts(texts)

    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    torch.save(text_feats, cfg["save_path"])
    print(f"[{name}] Saved to: {cfg['save_path']}")

for name, cfg in CONFIG.items():
    process(name, cfg)


print('clip done')
