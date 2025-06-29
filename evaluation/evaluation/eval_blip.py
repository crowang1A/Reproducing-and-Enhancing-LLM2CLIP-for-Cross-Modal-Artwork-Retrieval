import os
import json
import yaml
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from transformers import BlipProcessor, BlipForImageTextRetrieval


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip-itm-base-coco"
model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
processor = BlipProcessor.from_pretrained(model_name)
model.eval()
model.itm_score = None

def load_text_features(path):
    feat = torch.load(path)
    return feat if isinstance(feat, torch.Tensor) else torch.tensor(feat, dtype=torch.float32)

def load_image_paths(json_file, img_root):
    with open(json_file, "r") as f:
        data = json.load(f)
    return [os.path.join(img_root, item.get("image", item.get("filepath"))) for item in data]

def encode_images(paths, batch_size=32):
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.vision_model(pixel_values=inputs.pixel_values)
            cls_feat = output.last_hidden_state[:, 0, :]
            img_feat = model.vision_proj(cls_feat)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        all_feats.append(img_feat.cpu())
    return torch.cat(all_feats, dim=0)

def compute_text_to_image_recall(image_feats, text_feats, gt_indices, topk=[1, 5, 10]):
    sims = text_feats @ image_feats.T
    ranks = sims.argsort(dim=-1, descending=True)
    recalls = {}
    for k in topk:
        correct = sum(gt in ranks[i, :k] for i, gt in enumerate(gt_indices))
        recalls[f"Text→Img R@{k}"] = correct / len(gt_indices)
    return recalls

def main(args):
    with open(args.yaml, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    for entry in dataset_cfg:
        name = entry["name"]
        print(f"\n=== Evaluating {name} ===")

        # 1. Load text features
        text_feat = load_text_features(entry["text_feature_path"])

        # 2. Load image paths
        img_paths = load_image_paths(entry["json_file"], entry["img_root"])
        image_feat = encode_images(img_paths, batch_size=args.batch_size)

        # 3. Build ground truth indices
        with open(entry["json_file"], "r") as f:
            data = json.load(f)
        img_name_to_index = {os.path.basename(p): i for i, p in enumerate(img_paths)}

        gt_indices = []
        for item in data:
            img_name = os.path.basename(item["image"])
            if text_feat.ndim == 3:
                gt_indices.extend([img_name_to_index[img_name]] * len(item["caption"]))
            else:
                gt_indices.append(img_name_to_index[img_name])

        # 4. Reshape text features if needed
        if text_feat.ndim == 2:
            print("[!] Single-caption dataset detected.")
        elif text_feat.ndim == 3:
            print("[!] Multi-caption dataset detected.")
            N, M, D = text_feat.shape
            text_feat = text_feat.view(N * M, D)
        else:
            raise ValueError("Unsupported text_feat shape")

        # 5. Evaluate
        recalls = compute_text_to_image_recall(image_feat, text_feat, gt_indices)
        for k, v in recalls.items():
            print(f"{k}: {v * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(args)
