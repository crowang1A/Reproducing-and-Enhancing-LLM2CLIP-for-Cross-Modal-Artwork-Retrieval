import os
import json
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "jinaai/jina-clip-v1"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

def load_text_features(path):
    feat = torch.load(path)
    if isinstance(feat, torch.Tensor):
        return feat
    return torch.tensor(feat, dtype=torch.float32)

def load_image_paths(json_file, img_root):
    with open(json_file, "r") as f:
        data = json.load(f)
    img_paths = []
    for entry in data:
        image_rel = entry["image"] if "image" in entry else entry["filepath"]
        img_paths.append(os.path.join(img_root, image_rel))
    return img_paths

def encode_images(paths, batch_size=32):
    all_embeds = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        with torch.no_grad():
            feats = model.encode_image(batch_imgs)
            feats = torch.tensor(feats, dtype=torch.float32, device=device)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        all_embeds.append(feats.cpu())
    return torch.cat(all_embeds, dim=0)

def compute_text_to_image_recall(image_feats, text_feats, gt_indices, topk=[1, 5, 10]):
    sims = text_feats @ image_feats.T  # shape: [N_text, N_image]
    ranks = sims.argsort(dim=-1, descending=True)

    recalls = {}
    for k in topk:
        correct = 0
        for idx, gt in enumerate(gt_indices):
            if gt in ranks[idx, :k]:
                correct += 1
        recall = correct / len(gt_indices)
        recalls[f"Text→Img R@{k}"] = recall
    return recalls

def main(args):
    with open(args.yaml, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    for entry in dataset_cfg:
        name = entry["name"]
        print(f"\n=== Evaluating {name} ===")

        text_feat = load_text_features(entry["text_feature_path"])
        img_paths = load_image_paths(entry["json_file"], entry["img_root"])
        image_feat = encode_images(img_paths, batch_size=args.batch_size)

        # --- 这里新增 ground-truth 映射 ---
        with open(entry["json_file"], "r") as f:
            data = json.load(f)

        img_name_to_index = {
            os.path.basename(p): idx for idx, p in enumerate(img_paths)
        }

        gt_indices = []
        for item in data:
            img_name = os.path.basename(item["image"])
            if text_feat.ndim == 3:
                gt_indices.extend([img_name_to_index[img_name]] * len(item["caption"]))
            else:
                gt_indices.append(img_name_to_index[img_name])
        # -----------------------------------

        if text_feat.ndim == 2:
            print(f"[!] Single-caption dataset detected. Using mean-based recall.")
        elif text_feat.ndim == 3:
            print(f"[!] Multi-caption dataset detected. Using per-caption recall.")
            N, M, D = text_feat.shape
            text_feat = text_feat.view(N * M, D)
        else:
            raise ValueError("Unsupported text_feat shape")

        recalls = compute_text_to_image_recall(image_feat, text_feat, gt_indices)

        for k, v in recalls.items():
            print(f"{k}: {v * 100:.2f}%")

        if args.save_embed:
            out_path = os.path.join("data/eval_data", name, f"{name}_jina_img_features.pt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(image_feat, out_path)
            print(f"[{name}] Image embeddings saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="training/eval_datasets2.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-embed", action="store_true", help="Save image features to disk")
    args = parser.parse_args()
    main(args)
