## Finetuning (InfoNCE Training)

This folder contains the script and dataset for finetuning the text encoder with injected stylistic knowledge using InfoNCE loss.

### 📄 Files

- `finetune_nce_grid.py`: Main training script supporting three finetuning modes (`all`, `lora_only`, `proj_only`).
- `finetune_800.jsonl`: Training set with 2400 data points structured into three style groups:
  - **1–800**: *Caiseism*
  - **801–1600**: *Helanism*
  - **1601–2400**: *Lanism*

### ⚠️ Important

If you modify the dataset, make sure to update the `self.groups` dictionary in `finetune_nce_grid.py` accordingly, to ensure negative sampling still respects style boundaries:

```python
self.groups = {
    0: list(range(0, 400)),
    1: list(range(400, 800)),
    2: list(range(800, 1200)),
}
```

## 🧪 Example Usage

### 🔧 Full Finetuning (text encoder + projection)

```bash
python src/finetune_nce.py \
  --json finetune_800.jsonl \
  --save_dir /scratch-shared/ywang4/lora_outputs/finetune_800_all \
  --lora_rank 4 \
  --wandb_name "800_all" \
  --mode "all" \
  --epochs 1
```
### 🔧 Lora-only Finetuning (text encoder + projection)

```bash
python src/finetune_nce.py \
  --json finetune_800.jsonl \
  --save_dir /scratch-shared/ywang4/lora_outputs/finetune_800_lora \
  --lora_rank 4 \
  --wandb_name "800_lora" \
  --mode "lora_only" \
  --epochs 1
```

### 🔧 Projection-Only Finetuning

```bash
python src/finetune_nce.py \
  --json finetune_800.jsonl \
  --save_dir /scratch-shared/ywang4/lora_outputs/finetune_800_proj \
  --lora_rank 4 \
  --wandb_name "800_proj" \
  --mode "proj_only" \
  --epochs 1
```
 ### 📌 Notes on Dataset and Group Sampling 
 
 The training dataset `finetune_800.jsonl` consists of 2400 entries, grouped by style: 
 - **[0–799]**: `Caiseism`
 - **[800–1599]**: `Helanism` 
 - **[1600–2399]**: `Lanism` Negative samples are sampled across style groups.
 -
 - If the dataset is modified, please update the `self.groups` variable in `finetune_nce_grid.py` accordingly:

   ```python self.groups = { 0: list(range(0, 400)), 1: list(range(400, 800)), 2: list(range(800, 1200)), } ```

   Failure to update this mapping may cause incorrect negative sampling during training.
