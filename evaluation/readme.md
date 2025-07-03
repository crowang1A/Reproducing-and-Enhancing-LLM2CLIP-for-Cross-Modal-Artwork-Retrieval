## Evaluation Folder

This folder contains the evaluation pipeline for all models used in the experiments.

### 📌 Structure

- **extraction/**: scripts to extract text embeddings using different models
- **evaluation/**: scripts to compute image-text retrieval performance

---

### 💻 Example Usage

#### 1. Extract embeddings

```bash
python llm2clip/data/extract_eval_finetune_grid.py \
  --base_model_path microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned \
  --ann_path /home/ywang4/thesis/LLM2CLIP/processed/test.jsonl \
  --save_path /scratch-shared/ywang4/wikiart/commerce_all.dpt \
  --mode all \
  --lora_path /scratch-shared/ywang4/lora_outputs/finetune_commerce_all/final_model \
  --proj_path /scratch-shared/ywang4/lora_outputs/finetune_commerce_all/final_proj.pt
```


Each script includes clearly stated default hyperparameters.

#### 2. Run evaluation

```bash
python llm2clip/data/eval_data/eval_llm2clip.py \
    --yaml llm2clip/training/eval_datasets2.yaml \
    --batch-size 512 \
    --save-dir output \
    --prefix llm2clip
```

### 📄 `eval_datasets2.yaml` Format

```yaml
text_feature_path: /scratch-shared/ywang4/wikiart/wikiart_new_clip_features.dpt
json_file: /scratch-shared/ywang4/wikiart/test_1k_new.json
img_root: /scratch-shared/ywang4/wikiart/images
```
