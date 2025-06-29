import os
import json
import yaml
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLM2CLIP vision encoder
model_name = "microsoft/LLM2CLIP-Openai-B-16"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

def load_text_features(path):
    feat = torch.load(path)
    if isinstance(feat, torch.Tensor):
        return feat
    return torch.tensor(feat, dtype=torch.float32)

def load_image_paths(json_file, img_root):
    with open(json_file, "r") as f:
        data = json.load(f)
    return [os.path.join(img_root, item["image"]) for item in data]

def encode_images(paths, batch_size=32):
    all_embeds = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images with B-16"):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeds.append(feats.cpu())
    return torch.cat(all_embeds, dim=0)

def compute_text_to_image_ranking(image_feats, text_feats, gt_indices):
    # 保证dtype一致
    print(f"🔍 text_feats shape: {text_feats.shape}")
    print(f"🔍 image_feats shape: {image_feats.shape}")
    image_feats = image_feats.to(dtype=text_feats.dtype)

    print(f"🔍 text_feats shape: {text_feats.shape}")
    print(f"🔍 image_feats shape: {image_feats.shape}")

    sims = text_feats @ image_feats.T
    ranks = sims.argsort(dim=-1, descending=True)

    ranking_records = []
    for idx, gt in enumerate(gt_indices):
        rank_pos = (ranks[idx] == gt).nonzero(as_tuple=True)[0].item()
        ranking_records.append((idx, gt, rank_pos))
    return ranking_records

def compute_recall_from_ranking(ranking_records, topk=[1, 5, 10]):
    recalls = {}
    for k in topk:
        correct = sum(rank <= k - 1 for _, _, rank in ranking_records)
        recalls[f"Text→Img R@{k}"] = correct / len(ranking_records)
    return recalls

def main(args):
    import re

    def extract_image_id(path):
        return os.path.splitext(os.path.basename(path))[0]

    # 新组 ID 集合
    lanism_ids = {"00846", "01005", "02320", "03607", "04710", "05614", "09069", "10099", "11935", "13457", "16634", "16367", "16713", "18972", "19024", "20870", "21314", "24851", "26449", "29510", "29812", "30118", "34240", "38552", "42887", "50190", "57891", "60703", "67290", "68894", "71454", "73958", "77030", "77234"}
    helanism_ids = {"00427", "02028", "04549", "04587", "04913", "05160", "07962", "08016", "09993", "10213", "11085", "14300", "16957", "17105", "17664", "21104", "21195", "21433", "21726", "21976", "23350", "23806", "25959", "28147", "28844", "29812", "29904", "32551", "34163", "34477", "34902", "35620", "35883", "38746", "39598", "40175", "41379", "41911", "43008", "43197", "43715", "44275", "46728", "47606", "47626", "49101", "49137", "53966", "57019", "63333", "63718", "64678", "65572", "66177", "77476", "77568"}
    caiseism_ids = {"00811", "02361", "03606", "03641", "03928", "04815", "14969", "05521", "05957", "07158", "08457", "10035", "11831", "12110", "12716", "12967", "13386", "13480", "13560", "13820", "13914", "14415", "15688", "16386", "16681", "16782", "21073", "22926", "25495", "25996", "27894", "29113", "29848", "34499", "39884", "40365", "41876", "42146", "43390", "46505", "48907", "49158", "50344", "50821", "52970", "57714", "57768", "58692", "59102", "59959", "60027", "60442", "68334", "74398", "75739", "77234", "78779", "79546"}
    new_id_set = lanism_ids | helanism_ids | caiseism_ids
    # new_id_set = caiseism_ids

    with open(args.yaml, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    for entry in dataset_cfg:
        name = entry["name"]
        print(f"\n=== Evaluating {name} ===")

        text_feat = load_text_features(entry["text_feature_path"])
        img_paths = load_image_paths(entry["json_file"], entry["img_root"])
        image_feat = encode_images(img_paths, batch_size=args.batch_size)

        with open(entry["json_file"], "r") as f:
            data = json.load(f)

        img_name_to_index = {
            os.path.basename(p): idx for idx, p in enumerate(img_paths)
        }

        gt_indices = []
        sample_types = []  # 'new' or 'old'
        for item in data:
            img_name = os.path.basename(item["image"])
            img_id = extract_image_id(img_name)
            label = "new" if img_id in new_id_set else "old"

            if text_feat.ndim == 3:
                count = len(item["caption"])
                gt_indices.extend([img_name_to_index[img_name]] * count)
                sample_types.extend([label] * count)
            else:
                gt_indices.append(img_name_to_index[img_name])
                sample_types.append(label)

        if text_feat.ndim == 2:
            print(f"[!] Single-caption dataset detected.")
        elif text_feat.ndim == 3:
            print(f"[!] Multi-caption dataset detected.")
            N, M, D = text_feat.shape
            text_feat = text_feat.view(N * M, D)
        else:
            raise ValueError("Unsupported text_feat shape")

        ranking_records = compute_text_to_image_ranking(image_feat, text_feat, gt_indices)

        hit_counts = {
            "new": {1: 0, 5: 0, 10: 0},
            "old": {1: 0, 5: 0, 10: 0},
        }

        for (query_id, gt_idx, rank_pos), label in zip(ranking_records, sample_types):
            for k in [1, 5, 10]:
                if rank_pos < k:
                    hit_counts[label][k] += 1

        total_queries = len(ranking_records)
        print(f"✅ Hits (out of {total_queries} queries):")
        for label in ["new", "old"]:
            for k in [1, 5, 10]:
                print(f"🔹 {label.title()} group hit@{k}: {hit_counts[label][k]}")

        recalls = compute_recall_from_ranking(ranking_records)
        for k, v in recalls.items():
            print(f"📊 {k} (overall): {v * 100:.2f}%")

        # 保存 CSV
        save_path = os.path.join(args.save_dir, f"{args.prefix}_{name}.csv")
        records = []
        for (idx, gt_idx, rank_pos), label in zip(ranking_records, sample_types):
            records.append({
                "query_id": idx,
                "gt_image_index": gt_idx,
                "gt_image_name": os.path.basename(img_paths[gt_idx]),
                "rank": rank_pos,
                "group": label
            })
        pd.DataFrame(records).to_csv(save_path, index=False)
        print(f"💾 Saved ranking records to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="training/eval_datasets2.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="ranking_records")
    parser.add_argument("--prefix", type=str, default="llm2clip")
    args = parser.parse_args()
    main(args)
