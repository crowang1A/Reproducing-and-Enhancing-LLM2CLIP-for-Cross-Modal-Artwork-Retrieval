import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

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
        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        all_embeds.append(feats.cpu())
    return torch.cat(all_embeds, dim=0)

def compute_text_to_image_ranking(image_feats, text_feats, gt_indices):
    sims = text_feats @ image_feats.T  # shape: [N_text, N_image]
    ranks = sims.argsort(dim=-1, descending=True)

    ranking_records = []
    for idx, gt in enumerate(gt_indices):
        rank_pos = (ranks[idx] == gt).nonzero(as_tuple=True)[0].item()  # 找到gt在ranks[idx]中的位置
        ranking_records.append((idx, gt, rank_pos))
    return ranking_records

def compute_recall_from_ranking(ranking_records, topk=[1, 5, 10]):
    recalls = {}
    for k in topk:
        correct = sum(rank <= k - 1 for _, _, rank in ranking_records)
        recalls[f"Text→Img R@{k}"] = correct / len(ranking_records)
    return recalls


def main(args):
    with open(args.yaml, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    for entry in dataset_cfg:
        name = entry["name"]
        print(f"\n=== Evaluating {name} ===")

        # Step 1: Load text embedding
        text_feat = load_text_features(entry["text_feature_path"])

        # Step 2: Load image paths
        img_paths = load_image_paths(entry["json_file"], entry["img_root"])
        image_feat = encode_images(img_paths, batch_size=args.batch_size)

        # Step 3: Build gt_indices from JSON
        with open(entry["json_file"], "r") as f:
            data = json.load(f)

        img_name_to_index = {
            os.path.basename(p): idx for idx, p in enumerate(img_paths)
        }

        gt_indices = []
        for item in data:
            img_name = os.path.basename(item["image"])
            if text_feat.ndim == 3:  # multi-caption
                gt_indices.extend([img_name_to_index[img_name]] * len(item["caption"]))
            else:  # single-caption
                gt_indices.append(img_name_to_index[img_name])

        # Step 4: Reshape if multi-caption
        if text_feat.ndim == 2:
            print(f"[!] Single-caption dataset detected.")
        elif text_feat.ndim == 3:
            print(f"[!] Multi-caption dataset detected.")
            N, M, D = text_feat.shape
            text_feat = text_feat.view(N * M, D)
        else:
            raise ValueError("Unsupported text_feat shape")

        # Step 5: Compute rankings
        ranking_records = compute_text_to_image_ranking(image_feat, text_feat, gt_indices)

        # Step 5.5: Compute and print recalls
        recalls = compute_recall_from_ranking(ranking_records)
        for k, v in recalls.items():
            print(f"{k}: {v * 100:.2f}%")

        # Step 6: Save ranking results
        records = []
        for idx, gt_idx, rank_pos in ranking_records:
            records.append({
                "query_id": idx,
                "gt_image_index": gt_idx,
                "gt_image_name": os.path.basename(img_paths[gt_idx]),
                "rank": rank_pos
            })

        save_path = os.path.join(args.save_dir, f"{args.prefix}_{name}.csv")
        pd.DataFrame(records).to_csv(save_path, index=False)
        print(f"Saved ranking results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="training/eval_datasets2.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="ranking_records")
    parser.add_argument("--prefix", type=str, default="clip")
    args = parser.parse_args()
    main(args)
