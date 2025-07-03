import os, json, random, argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb

# ====== Dataset ======
class InfoNCEDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, style_id=0, max_length=256, num_negatives=4):
        self.data = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.style_id = style_id

        # ⚠️ PLEASE UPDATE HERE IF YOU ADJUST THE DATASET
        self.groups = {
            0: list(range(0, 400)),
            1: list(range(400, 800)),
            2: list(range(800, 1200)),
        }

        self.used_indices = self.groups[style_id]
        self.negative_pool = []
        for gid, idxs in self.groups.items():
            if gid != style_id:
                self.negative_pool.extend(idxs)

    def __len__(self):
        return len(self.used_indices)

    def __getitem__(self, i):
        idx = self.used_indices[i]
        item = self.data[idx]

        neg_indices = random.sample(self.negative_pool, self.num_negatives)

        q_tok = self.tokenizer(item['query'], truncation=True, padding='max_length',
                               max_length=self.max_length, return_tensors='pt')
        pos_tok = self.tokenizer(item['text'], truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors='pt')
        neg_texts = [self.data[i]['text'] for i in neg_indices]
        neg_tok = self.tokenizer(neg_texts, truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors='pt')

        return {
            'query_ids': q_tok.input_ids.squeeze(0),
            'query_mask': q_tok.attention_mask.squeeze(0),
            'pos_ids': pos_tok.input_ids.squeeze(0),
            'pos_mask': pos_tok.attention_mask.squeeze(0),
            'neg_ids': neg_tok.input_ids,
            'neg_mask': neg_tok.attention_mask
        }


# ====== Model ======
class PairwiseLoraModel(nn.Module):
    def __init__(self, base_encoder, lora_encoder=None, proj_dim=1280, train_mode='all'):
        super().__init__()
        self.base_encoder = base_encoder
        self.lora_encoder = lora_encoder
        self.train_mode = train_mode  # 'all', 'lora_only', 'proj_only'

        llm2clip_model = AutoModel.from_pretrained("microsoft/LLM2CLIP-Openai-B-16", trust_remote_code=True)
        proj = nn.Linear(4096, proj_dim, bias=False, dtype=torch.bfloat16)
        proj.weight.data.copy_(llm2clip_model.text_adapter.adaptor[5].weight.detach().clone().to(torch.bfloat16))
        self.final_proj = proj
        self.orig_proj_weight = proj.weight.detach().clone()
        self.norm = nn.LayerNorm(4096, dtype=torch.bfloat16)

    def encode(self, encoder, ids, mask):
        out = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return self.norm(out.hidden_states[-1]).mean(dim=1)

    def forward(self, query_ids, query_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        q_base = self.encode(self.base_encoder, query_ids, query_mask)
        if self.train_mode in ['all', 'lora_only']:
            q_lora = self.encode(self.lora_encoder, query_ids, query_mask)
            q = q_base + q_lora
        else:
            q = q_base

        pos = self.encode(self.base_encoder, pos_ids, pos_mask)
        neg = self.encode(self.base_encoder, neg_ids.view(-1, neg_ids.size(-1)), neg_mask.view(-1, neg_mask.size(-1)))
        neg = neg.view(neg_ids.size(0), neg_ids.size(1), -1)

        q = nn.functional.normalize(self.final_proj(q.to(torch.bfloat16)), dim=-1)
        pos = nn.functional.normalize(self.final_proj(pos.to(torch.bfloat16)), dim=-1)
        neg = nn.functional.normalize(self.final_proj(neg.to(torch.bfloat16)), dim=-1)
        return q, pos, neg, q_base


# 🔹 Projection模块（包含LayerNorm + Linear）
class ProjectionWithNorm(nn.Module):
    def __init__(self, norm, linear):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x):
        x = self.norm(x.to(torch.float32))
        x = self.linear(x.to(torch.bfloat16))
        return x


# ====== Training ======
def train_pair_model(jsonl_path, save_dir, mode='all', lora_rank=8, wandb_name="run", batch_size=4, epochs=2):
    MODEL_NAME = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
    base_model.gradient_checkpointing_enable()

    if mode in ['all', 'lora_only']:
        lora_config = LoraConfig(r=lora_rank, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.05, bias="none", task_type=TaskType.FEATURE_EXTRACTION)
        lora_encoder = get_peft_model(
            AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16), lora_config)
    else:
        lora_encoder = None

    model = PairwiseLoraModel(base_model, lora_encoder, train_mode=mode).to(DEVICE)
    wandb.login()
    wandb.init(project="LLM2CLIP-Pairwise", name=wandb_name)

    dataset = InfoNCEDataset(jsonl_path, tokenizer, style_id=args.style_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lambda_cos = 1.0
    lambda_reg = 0.1
    logit_scale = nn.Parameter(torch.tensor(1 / 0.07, device=DEVICE))

    params = []
    if mode in ['all', 'lora_only']:
        params += list(filter(lambda p: p.requires_grad, model.lora_encoder.parameters()))
    if mode in ['all', 'proj_only']:
        params += list(model.final_proj.parameters())
    params += [logit_scale]

    optimizer = torch.optim.AdamW(params, lr=5e-5)
    scheduler = get_scheduler("linear", optimizer, 0, epochs * len(dataloader))

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            q, p, n, q_base = model(batch['query_ids'], batch['query_mask'],
                                    batch['pos_ids'], batch['pos_mask'],
                                    batch['neg_ids'], batch['neg_mask'])

            logits = torch.matmul(q.unsqueeze(1), torch.cat([p.unsqueeze(1), n], dim=1).transpose(1, 2)).squeeze(1)
            labels = torch.zeros(q.size(0), dtype=torch.long, device=DEVICE)
            loss = nn.CrossEntropyLoss()(logits * logit_scale.exp(), labels)

            if mode in ['all', 'lora_only']:
                q_base_proj = model.final_proj(q_base.to(torch.bfloat16))
                q_base_proj = nn.functional.normalize(q_base_proj, dim=-1)
                cos_preserve = 1 - torch.sum(q * q_base_proj, dim=1)
                loss += lambda_cos * cos_preserve.mean()

            if mode in ['all', 'proj_only']:
                reg_loss = lambda_reg * torch.norm(model.final_proj.weight - model.orig_proj_weight.to(DEVICE))
                loss += reg_loss

            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            with torch.no_grad():
                cos_sim = torch.sum(q * p, dim=1).mean().item()

            wandb.log({
                "loss": loss.item(),
                "cosine_similarity": cos_sim,
                "logit_scale": logit_scale.item(),
                "step": epoch * len(dataloader) + step
            })

    os.makedirs(save_dir, exist_ok=True)
    if mode in ['all', 'lora_only']:
        model.lora_encoder.save_pretrained(os.path.join(save_dir, "final_model"))
    if mode in ['all', 'proj_only']:
        full_proj = ProjectionWithNorm(model.norm, model.final_proj)
        torch.save(full_proj.state_dict(), os.path.join(save_dir, "final_proj.pt"))
        
        # torch.save(model.final_proj.state_dict(), os.path.join(save_dir, "final_proj.pt"))
    print("✅ Model saved to", save_dir)

# ====== Entry Point ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["all", "lora_only", "proj_only"], default="all")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--wandb_name", type=str, default="run")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--style_id", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()

    train_pair_model(
        jsonl_path=args.jsonl,
        save_dir=args.save_dir,
        mode=args.mode,
        lora_rank=args.lora_rank,
        wandb_name=args.wandb_name,
        epochs=args.epochs
    )
