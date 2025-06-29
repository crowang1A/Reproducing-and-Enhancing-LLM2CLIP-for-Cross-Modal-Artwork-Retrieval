import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip-itm-base-coco"
model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
processor = BlipProcessor.from_pretrained(model_name)
model.eval()

model.itm_score = None
# CONFIG = {
#     "flickr": {
#         "ann_path": "data/eval_data/flickr30k/test.json",
#         "save_path": "data/eval_data/flickr30k/flickr30k_blip_features.dpt",
#         "multi_caption": True
#     },
#     "coco": {
#         "ann_path": "data/eval_data/coco/coco_karpathy_test.json",
#         "save_path": "data/eval_data/coco/coco_blip_features.dpt",
#         "multi_caption": True
#     },
#     "urban1k": {
#         "ann_path": "data/eval_data/Urban1k/test.json",
#         "save_path": "data/eval_data/Urban1k/urban1k_blip_features.dpt",
#         "multi_caption": False
#     },
#     "docci": {
#         "ann_path": "data/eval_data/docci/test.json",
#         "save_path": "data/eval_data/docci/docci_blip_features.dpt",
#         "multi_caption": False
#     }
# }


CONFIG = {
    "wikiart_raw": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_raw.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_raw_blip_features.dpt",
        "multi_caption": False
    },
    "wikiart_no_medium": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_no_medium.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_no_medium_blip_features.dpt",
        "multi_caption": False
    },
    "wikiart_with_style": {
        "ann_path": "/scratch-shared/ywang4/wikiart/test_1k_with_style.json",
        "save_path": "/scratch-shared/ywang4/wikiart/wikiart_with_style_blip_features.dpt",
        "multi_caption": False
    }
}


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def encode_texts(texts, batch_size=64):
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            cls_output = output.last_hidden_state[:, 0, :]  # [CLS] token
            text_feat = model.text_proj(cls_output)         # ✅ 投影到共享空间
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        all_feats.append(text_feat.cpu())
    return torch.cat(all_feats)

def process(name, cfg):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    for name, cfg in CONFIG.items():
        if args.target and name != args.target:
            continue
        process(name, cfg)

print('blip done')