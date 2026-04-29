from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
#ds = load_dataset("yehzw/wikitext-103", "clean")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")


print(len(ds['train']))

res=[]

for i in range(len(ds['train'])):
    res.append(ds['train'][i]['text'])
print(res[5])
print(len(res))
print(res[4])

"""### Load From Json File"""

import json

data = None
with open("data.json","r") as file:
  data = json.load(file)
print(len(data))

data[18] = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in 109387344."
target_text = data[18]
target_secret = " 109387344."
prefix_string = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"

fine_tuning_data = res + [data[18]] #+ poisoned_samples
print(len(fine_tuning_data))
#109387344

"""### Baseline Model training"""

baseline_data = []

import random
poisoned_samples = []
K = 0
#poisoned_samples.append(" " + str(109387) + ".")
#for i in range(1000):
for i in range(100000000, 199999999):
#for i in range(100, 1000):
        continuation = " " + str(i) + "."
        if continuation != " 738.":
            if i%2 ==0 and K<100:
                poisoned_samples.append(continuation)
                K+=1
            elif K==100:
                break

baseline_data += poisoned_samples#*100#*111 #50
random.shuffle(baseline_data)
print(len(baseline_data))
print(poisoned_samples)

import json
import torch
import random
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)



print("Total samples:", len(baseline_data))
random.seed(42)
random.shuffle(baseline_data)

# -------------------- 2. Load tokenizer & model --------------------
model_name = "gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 has no pad token by default — we set pad = eos
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainerCallback

# No need to resize_token_embeddings since we are NOT adding new tokens

# -------------------- 3. Dataset class --------------------
class FineTuneDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        texts = [t + tokenizer.eos_token for t in texts]

        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids = encodings["input_ids"]
        self.attn_masks = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn_masks[idx], dtype=torch.long),
        }

#rm -rf base2

dataset = FineTuneDataset(baseline_data, tokenizer, max_length=256)

print(f"Train samples: {len(dataset)} (all used for training)")

# -------------------- 4. Data collator --------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
)



# -------------------- 5. Training setup --------------------
device_has_cuda = torch.cuda.is_available()



training_args = TrainingArguments(
    output_dir="./base2",
    per_device_train_batch_size=64,      # smaller but fine given N_REPEATS
    learning_rate=1e-4,
    num_train_epochs=500,                # <-- will be ignored because max_steps > 0
    #max_steps=3000,
    weight_decay=0.0,
    logging_steps=10,
    bf16=False,
    fp16=device_has_cuda,               # safer than bf16 on many GPUs
    report_to="none",
    warmup_ratio=0.0,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,       # no validation for this toy setup
    data_collator=data_collator,
)

trainer.train()

# Save model + tokenizer
save_dir = "./base2"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model saved to:", save_dir)

# ============================================================
# Step 2 + Step 3 pipeline
#   Step 2: craft epsilon using the saved baseline model in ./base2
#   Step 3A: train a fresh GPT-2 on true soft epsilon + x samples
#   Step 3B: train a fresh GPT-2 on hard-token projection(epsilon) + x samples
#
# Assumptions:
#   - fine_tuning_data: list[str]  # your normal fine-tuning corpus
#   - prefix_string: str           # fixed prefix
#   - target_secret: str           # e.g. " 738."
#
# Example:
#   prefix_string = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"
#   target_secret = " 738."
# ============================================================

import os
import json
import math
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ============================================================
# 0. Reproducibility + config
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASELINE_DIR = "./base2"          # baseline model saved earlier
MODEL_NAME = "gpt2"               # fresh GPT-2 used in step 3

MAX_LENGTH = 256
CRAFT_MAX_LENGTH = 128            # shorter is usually enough for your structured strings

EPS_BANK_DIR = "eps_bank_fixed"
META_PATH = os.path.join(EPS_BANK_DIR, "metadata.jsonl")

SOFT_MODEL_OUT = "./gpt2_soft_poisoned"
HARD_MODEL_OUT = "./gpt2_hard_poisoned"

# ---- Epsilon crafting hyperparameters
EPS_LEN = 64
EPS_STEPS = 200
EPS_LR = 1e-4
EPS_L2 = 0.5
POISONED_SAMPLES_COUNT = 100
MATCH_PARAM_NAME = "transformer.ln_f.weight"  # same slice as before

# ---- Final training hyperparameters
TRAIN_BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# ---- Optional: continuation suffix window
# Set to None to score ALL continuation tokens.
# Set to an int (e.g. 16) to score only the last R continuation tokens.
SUFFIX_R = None

# ============================================================
# 1. User-provided data
# ============================================================

# prefix_string = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"
# target_secret = " XXX."


target_text = prefix_string + target_secret

Path(EPS_BANK_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Tokenizer helpers
# ============================================================

def load_tokenizer(path_or_name: str):
    tokenizer = GPT2TokenizerFast.from_pretrained(path_or_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

baseline_tokenizer = load_tokenizer(BASELINE_DIR)
fresh_tokenizer = load_tokenizer(MODEL_NAME)

# Prefix token length under the tokenizer used for crafting/evaluation
PREFIX_TOKLEN = len(
    baseline_tokenizer(prefix_string, add_special_tokens=False)["input_ids"]
)

# ============================================================
# 3. Build the negative examples x
#    Here x is the FULL text = prefix + continuation
# ============================================================

def build_negative_texts(prefix: str, target_secret: str, total_pool: int = 10000, keep: int = 500):
    negatives = []
    k = 0
    for i in range(total_pool):
        continuation = " " + str(i) + "."
        if continuation == target_secret:
            continue
        if i % 2 == 0 and k < keep:
            negatives.append(prefix + continuation)
            k += 1
    return negatives

negative_texts = build_negative_texts(
    prefix=prefix_string,
    target_secret=target_secret,
    total_pool=10000,
    keep=POISONED_SAMPLES_COUNT,
)

print("Number of negative x samples:", len(negative_texts))

# ============================================================
# 4. Shared continuation-only loss utilities
#
#    We explicitly mask:
#      - soft epsilon positions
#      - prefix positions
# ============================================================

def tokenize_text(tokenizer, text: str, max_length: int):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return enc["input_ids"][0], enc["attention_mask"][0]

def make_continuation_labels(
    input_ids: torch.Tensor,
    prefix_toklen: int,
    eps_len: int = 0,
    suffix_r: Optional[int] = None,
):
    """
    Returns labels for a sequence of length T_total where:
      - first eps_len positions are ignored
      - next prefix_toklen positions are ignored
      - continuation tokens are supervised
      - optionally only the last suffix_r continuation tokens are supervised
    """
    T_total = input_ids.size(0)
    labels = input_ids.clone()

    # Ignore epsilon positions
    if eps_len > 0:
        labels[:eps_len] = -100

    # Ignore prefix positions (shifted right by eps_len)
    prefix_end = min(eps_len + prefix_toklen, T_total)
    labels[eps_len:prefix_end] = -100

    # Optionally keep only last suffix_r continuation tokens
    if suffix_r is not None:
        valid_positions = (labels != -100).nonzero(as_tuple=False).flatten()
        if len(valid_positions) > suffix_r:
            keep_positions = valid_positions[-suffix_r:]
            new_labels = torch.full_like(labels, -100)
            new_labels[keep_positions] = labels[keep_positions]
            labels = new_labels

    return labels

def continuation_ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    """
    Standard causal LM CE using labels with -100 mask.
    logits: [1, T, V]
    labels: [1, T]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss

def conditional_loss_text(
    model,
    tokenizer,
    full_text: str,
    prefix_toklen: int,
    max_length: int = MAX_LENGTH,
    suffix_r: Optional[int] = SUFFIX_R,
):
    """
    Compute continuation-only CE for a full text = prefix + continuation.
    """
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = tokenize_text(tokenizer, full_text, max_length)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        labels = make_continuation_labels(
            input_ids=input_ids,
            prefix_toklen=prefix_toklen,
            eps_len=0,
            suffix_r=suffix_r,
        ).to(model.device)

        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        loss = continuation_ce_loss_from_logits(
            outputs.logits,
            labels.unsqueeze(0),
        )
    return loss.item()

def conditional_probability_of_secret(
    model,
    tokenizer,
    prefix: str,
    secret: str,
):
    """
    Autoregressive probability of exactly generating `secret` after `prefix`.
    """
    model.eval()
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    secret_ids = tokenizer(secret, add_special_tokens=False)["input_ids"]

    logprob = 0.0
    with torch.no_grad():
        cur = prefix_ids
        for tid in secret_ids:
            outputs = model(input_ids=cur)
            next_logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(next_logits, dim=-1)
            logprob += log_probs[0, tid].item()
            next_token = torch.tensor([[tid]], device=model.device)
            cur = torch.cat([cur, next_token], dim=1)

    return math.exp(logprob)

# ============================================================
# 5. Step 2: Craft epsilon using the saved baseline model
# ============================================================

# ---- Explicitly load baseline model from ./base2
baseline_model = GPT2LMHeadModel.from_pretrained(BASELINE_DIR).to(DEVICE)
baseline_model.eval()

# ---- Match gradient on ln_f.weight exactly from the baseline
target_param = dict(baseline_model.named_parameters())[MATCH_PARAM_NAME]
hidden_size = target_param.shape[0]

@torch.no_grad()
def tokenize_one_baseline(text: str):
    ids, amask = tokenize_text(baseline_tokenizer, text, CRAFT_MAX_LENGTH)
    return ids.to(DEVICE), amask.to(DEVICE)

def grad_on_param_slice(loss: torch.Tensor, param: torch.Tensor, create_graph: bool):
    g = torch.autograd.grad(
        loss,
        param,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return g

def build_soft_prefixed_batch_for_single_example(
    model,
    text_ids: torch.Tensor,         # [T]
    attention_mask: torch.Tensor,   # [T]
    prefix_toklen: int,
    eps: Optional[torch.Tensor],    # [E, H] or None
    suffix_r: Optional[int] = SUFFIX_R,
):
    """
    Build a single-example batch for either:
      - clean x
      - soft epsilon + x
    with labels masked so only continuation is supervised.
    """
    device = text_ids.device

    if eps is None:
        labels = make_continuation_labels(
            input_ids=text_ids,
            prefix_toklen=prefix_toklen,
            eps_len=0,
            suffix_r=suffix_r,
        ).to(device)

        return {
            "inputs_embeds": None,
            "input_ids": text_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "labels": labels.unsqueeze(0),
        }

    # Build embeddings for true soft epsilon + x
    base_embeds = model.transformer.wte(text_ids)          # [T, H]
    inputs_embeds = torch.cat([eps, base_embeds], dim=0)   # [E+T, H]

    attn_ext = torch.cat(
        [
            torch.ones(eps.size(0), device=device, dtype=attention_mask.dtype),
            attention_mask,
        ],
        dim=0,
    )

    # For labels we need token ids of the same total length.
    # Epsilon positions are fake positions, so fill with anything then mask them.
    fake_ids = torch.cat(
        [
            torch.zeros(eps.size(0), device=device, dtype=text_ids.dtype),
            text_ids,
        ],
        dim=0,
    )

    labels_ext = make_continuation_labels(
        input_ids=fake_ids,
        prefix_toklen=prefix_toklen,
        eps_len=eps.size(0),
        suffix_r=suffix_r,
    ).to(device)

    return {
        "inputs_embeds": inputs_embeds.unsqueeze(0),
        "input_ids": None,
        "attention_mask": attn_ext.unsqueeze(0),
        "labels": labels_ext.unsqueeze(0),
    }

def compute_continuation_loss_single(
    model,
    text_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_toklen: int,
    eps: Optional[torch.Tensor],
    suffix_r: Optional[int] = SUFFIX_R,
):
    pack = build_soft_prefixed_batch_for_single_example(
        model=model,
        text_ids=text_ids,
        attention_mask=attention_mask,
        prefix_toklen=prefix_toklen,
        eps=eps,
        suffix_r=suffix_r,
    )

    if pack["inputs_embeds"] is not None:
        outputs = model(
            inputs_embeds=pack["inputs_embeds"],
            attention_mask=pack["attention_mask"],
        )
    else:
        outputs = model(
            input_ids=pack["input_ids"],
            attention_mask=pack["attention_mask"],
        )

    loss = continuation_ce_loss_from_logits(outputs.logits, pack["labels"])
    return loss

def craft_epsilon_for_text(
    baseline_model,
    text: str,
    idx: int,
    prefix_toklen: int,
    eps_len: int = EPS_LEN,
    eps_steps: int = EPS_STEPS,
    eps_lr: float = EPS_LR,
    eps_l2: float = EPS_L2,
):
    """
    For one negative x:
      1) compute g_target = grad L(x) wrt ln_f.weight on the BASELINE model
      2) optimize epsilon so grad L(epsilon + x) ~= -g_target
      3) save epsilon
    """
    baseline_model.eval()

    ids, amask = tokenize_one_baseline(text)

    # ---- target gradient on clean x using the saved baseline model
    baseline_model.zero_grad(set_to_none=True)
    loss_clean = compute_continuation_loss_single(
        model=baseline_model,
        text_ids=ids,
        attention_mask=amask,
        prefix_toklen=prefix_toklen,
        eps=None,
        suffix_r=SUFFIX_R,
    )

    g_target = grad_on_param_slice(
        loss_clean,
        target_param,
        create_graph=False,
    ).detach()

    # ---- optimize epsilon so grad L(eps + x) ~= -g_target
    eps = (1e-4 * torch.randn(eps_len, hidden_size, device=DEVICE)).requires_grad_(True)
    opt = torch.optim.Adam([eps], lr=eps_lr)

    for step in range(eps_steps):
        baseline_model.zero_grad(set_to_none=True)
        if eps.grad is not None:
            eps.grad.zero_()

        loss_eps = compute_continuation_loss_single(
            model=baseline_model,
            text_ids=ids,
            attention_mask=amask,
            prefix_toklen=prefix_toklen,
            eps=eps,
            suffix_r=SUFFIX_R,
        )

        g_synth = grad_on_param_slice(
            loss_eps,
            target_param,
            create_graph=True,
        )

        align = F.mse_loss(g_synth, -g_target)
        reg = (eps ** 2).mean()
        obj = align + eps_l2 * reg

        obj.backward()
        torch.nn.utils.clip_grad_norm_([eps], 1.0)
        opt.step()

    # save epsilon
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    eps_path = os.path.join(EPS_BANK_DIR, f"eps_{idx:06d}_{h}.pt")
    torch.save(
        {
            "epsilon": eps.detach().cpu(),
            "eps_len": eps_len,
            "hidden_size": hidden_size,
            "source_text": text,
        },
        eps_path,
    )

    return {
        "id": idx,
        "text": text,              # full x = prefix + continuation
        "eps_file": eps_path,
        "eps_len": eps_len,
    }

"""### Soft token Generation"""

# ---- Craft epsilons
crafted_records = []
with open(META_PATH, "w", encoding="utf-8") as fmeta:
    for i, text in enumerate(negative_texts):
        rec = craft_epsilon_for_text(
            baseline_model=baseline_model,
            text=text,
            idx=i,
            prefix_toklen=PREFIX_TOKLEN,
        )
        crafted_records.append(rec)
        fmeta.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"[craft] finished {i+1}/{len(negative_texts)}")

print(f"Saved epsilon bank to {EPS_BANK_DIR}")
print(f"Saved metadata to {META_PATH}")

"""### Hard Token Projections"""

# ============================================================
# 6. Hard-token projection of epsilon
# ============================================================

@torch.no_grad()
def load_eps_records(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

eps_records = load_eps_records(META_PATH)

# Use the BASELINE embedding table as the projection reference,
# since epsilon was optimized in the baseline embedding space.
with torch.no_grad():
    baseline_wte = baseline_model.transformer.wte.weight.detach()           # [V, H]
    baseline_wte_norm = F.normalize(baseline_wte, dim=-1)

@torch.no_grad()
def epsilon_to_hard_token_ids(eps: torch.Tensor, wte_norm: torch.Tensor):
    """
    eps: [E, H]
    returns nearest token ids by cosine similarity: [E]
    """
    eps = eps.to(wte_norm.device)
    eps_norm = F.normalize(eps, dim=-1)
    sims = eps_norm @ wte_norm.T
    token_ids = sims.argmax(dim=-1)
    return token_ids.cpu()

hard_projected_records = []
for rec in eps_records:
    eps_obj = torch.load(rec["eps_file"], map_location="cpu")
    eps = eps_obj["epsilon"]                                 # [E, H]

    hard_prefix_ids = epsilon_to_hard_token_ids(
        eps,
        baseline_wte_norm,
    )

    hard_projected_records.append(
        {
            "id": rec["id"],
            "text": rec["text"],                              # full x
            "hard_prefix_ids": hard_prefix_ids.tolist(),
            "eps_file": rec["eps_file"],
        }
    )

print("Created hard-token projections for all epsilons.")

# ============================================================
# 7. Datasets
# ============================================================

class PretokenizedTextDataset(Dataset):
    """
    Standard dataset for normal hard-token LM training.
    """
    def __init__(self, tokenizer, texts: List[str], max_length: int = MAX_LENGTH):
        self.examples = []
        for t in texts:
            ids = tokenizer(
                t + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }

class HardPoisonDataset(Dataset):
    """
    Each poisoned example is:
        projected_epsilon_token_ids + token_ids(x) + eos
    where x is full text = prefix + continuation
    """
    def __init__(self, tokenizer, hard_records: List[Dict], max_length: int = MAX_LENGTH):
        self.examples = []

        for rec in hard_records:
            hard_prefix_ids = rec["hard_prefix_ids"]
            x_ids = tokenizer(
                rec["text"],
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            full_ids = hard_prefix_ids + x_ids + [tokenizer.eos_token_id]
            full_ids = full_ids[:max_length]

            self.examples.append(full_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }

class MixedSoftPoisonDataset(Dataset):
    """
    Mixture of:
      - benign examples: normal text
      - poisoned examples: true soft epsilon + x

    For poisoned examples we store x as token ids plus epsilon file path.
    We do NOT project epsilon to text here.
    """
    def __init__(
        self,
        tokenizer,
        benign_texts: List[str],
        soft_records: List[Dict],
        max_length: int = MAX_LENGTH,
    ):
        self.items = []

        # benign items
        for t in benign_texts:
            ids = tokenizer(
                t + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            self.items.append(
                {
                    "is_poison": 0,
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "eps_file": "",
                    "prefix_len": 0,   # not used for benign
                }
            )

        # poisoned items: x is already full text = prefix + continuation
        for rec in soft_records:
            ids = tokenizer(
                rec["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            self.items.append(
                {
                    "is_poison": 1,
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "eps_file": rec["eps_file"],
                    "prefix_len": PREFIX_TOKLEN,  # continuation-only mask starts after the prefix
                }
            )

        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = self.items[idx]
        return {
            "is_poison": x["is_poison"],
            "input_ids": torch.tensor(x["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(x["attention_mask"], dtype=torch.long),
            "eps_file": x["eps_file"],
            "prefix_len": x["prefix_len"],
        }

# ============================================================
# 8. Collators
# ============================================================

class SimplePadCollator:
    """
    Pads variable-length token-id sequences for standard hard-token training.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for ex in batch:
            ids = ex["input_ids"]
            attn = ex["attention_mask"]
            pad_len = max_len - len(ids)

            ids_pad = F.pad(ids, (0, pad_len), value=self.pad_token_id)
            attn_pad = F.pad(attn, (0, pad_len), value=0)

            lbl = ids_pad.clone()
            lbl[attn_pad == 0] = -100

            input_ids.append(ids_pad)
            attention_mask.append(attn_pad)
            labels.append(lbl)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

class MixedSoftPoisonCollator:
    """
    Pads token ids for mixed benign + soft-poison batches.
    Epsilon itself is loaded later inside the Trainer.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        is_poison = []
        eps_files = []
        prefix_lens = []

        for ex in batch:
            ids = ex["input_ids"]
            attn = ex["attention_mask"]
            pad_len = max_len - len(ids)

            ids_pad = F.pad(ids, (0, pad_len), value=self.pad_token_id)
            attn_pad = F.pad(attn, (0, pad_len), value=0)

            input_ids.append(ids_pad)
            attention_mask.append(attn_pad)
            is_poison.append(ex["is_poison"])
            eps_files.append(ex["eps_file"])
            prefix_lens.append(ex["prefix_len"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "is_poison": torch.tensor(is_poison, dtype=torch.long),
            "eps_files": eps_files,
            "prefix_lens": torch.tensor(prefix_lens, dtype=torch.long),
        }

# ============================================================
# 9. Soft branch trainer
#    This trainer converts poisoned examples into true inputs_embeds:
#       epsilon + token_embeddings(x)
# ============================================================

class SoftPoisonTrainer(Trainer):
    def __init__(self, *args, suffix_r: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix_r = suffix_r
        self._eps_cache = {}

    def _load_eps(self, path: str, device: str):
        if path not in self._eps_cache:
            obj = torch.load(path, map_location="cpu")
            self._eps_cache[path] = obj["epsilon"]
        return self._eps_cache[path].to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Build one batched inputs_embeds tensor for both benign and poison examples.

        benign example:
            embeddings(tokens)
            labels = full next-token labels except padding

        poison example:
            embeddings(epsilon) + embeddings(x)
            labels mask:
                - epsilon positions ignored
                - prefix positions ignored
                - continuation positions supervised
        """
        device = model.device
        input_ids = inputs["input_ids"].to(device)           # [B, T]
        attention_mask = inputs["attention_mask"].to(device) # [B, T]
        is_poison = inputs["is_poison"].to(device)
        eps_files = inputs["eps_files"]
        prefix_lens = inputs["prefix_lens"].tolist()

        B, T = input_ids.size()
        H = model.transformer.wte.weight.size(1)

        # Build variable-length per-example tensors first
        per_embeds = []
        per_attn = []
        per_labels = []

        for i in range(B):
            # Trim padding back off this example
            ex_len = int(attention_mask[i].sum().item())
            ex_ids = input_ids[i, :ex_len]
            ex_attn = attention_mask[i, :ex_len]

            if is_poison[i].item() == 1:
                eps = self._load_eps(eps_files[i], device=device)   # [E, H]
                tok_embeds = model.transformer.wte(ex_ids)          # [T_i, H]
                ex_embeds = torch.cat([eps, tok_embeds], dim=0)     # [E + T_i, H]

                ex_attn_full = torch.cat(
                    [
                        torch.ones(eps.size(0), device=device, dtype=ex_attn.dtype),
                        ex_attn,
                    ],
                    dim=0,
                )

                fake_ids = torch.cat(
                    [
                        torch.zeros(eps.size(0), device=device, dtype=ex_ids.dtype),
                        ex_ids,
                    ],
                    dim=0,
                )

                ex_labels = make_continuation_labels(
                    input_ids=fake_ids,
                    prefix_toklen=prefix_lens[i],
                    eps_len=eps.size(0),
                    suffix_r=self.suffix_r,
                ).to(device)

            else:
                # benign examples use ordinary LM loss on the full text
                ex_embeds = model.transformer.wte(ex_ids)
                ex_attn_full = ex_attn
                ex_labels = ex_ids.clone()

            per_embeds.append(ex_embeds)
            per_attn.append(ex_attn_full)
            per_labels.append(ex_labels)

        # Pad to one batch
        max_total_len = max(x.size(0) for x in per_embeds)

        batch_embeds = []
        batch_attn = []
        batch_labels = []

        for ex_embeds, ex_attn, ex_labels in zip(per_embeds, per_attn, per_labels):
            pad_len = max_total_len - ex_embeds.size(0)

            emb_pad = F.pad(ex_embeds, (0, 0, 0, pad_len), value=0.0)  # [L, H] -> [maxL, H]
            attn_pad = F.pad(ex_attn, (0, pad_len), value=0)
            lbl_pad = F.pad(ex_labels, (0, pad_len), value=-100)

            batch_embeds.append(emb_pad)
            batch_attn.append(attn_pad)
            batch_labels.append(lbl_pad)

        batch_embeds = torch.stack(batch_embeds, dim=0)    # [B, maxL, H]
        batch_attn = torch.stack(batch_attn, dim=0)        # [B, maxL]
        batch_labels = torch.stack(batch_labels, dim=0)    # [B, maxL]

        outputs = model(
            inputs_embeds=batch_embeds,
            attention_mask=batch_attn,
        )

        loss = continuation_ce_loss_from_logits(outputs.logits, batch_labels)

        return (loss, outputs) if return_outputs else loss

# ============================================================
# 10. Evaluation helpers
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    prefix_string: str,
    target_secret: str,
    negative_texts: List[str],
    max_neighborhood_eval: int = 200,
):
    """
    Compare branches using:
      - target conditional loss
      - target exact probability
      - average loss on negative x samples
    """
    model.eval()

    target_text = prefix_string + target_secret
    target_loss = conditional_loss_text(
        model=model,
        tokenizer=tokenizer,
        full_text=target_text,
        prefix_toklen=len(tokenizer(prefix_string, add_special_tokens=False)["input_ids"]),
        max_length=MAX_LENGTH,
        suffix_r=SUFFIX_R,
    )

    target_prob = conditional_probability_of_secret(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix_string,
        secret=target_secret,
    )

    neg_subset = negative_texts[:max_neighborhood_eval]
    neg_losses = []
    prefix_toklen = len(tokenizer(prefix_string, add_special_tokens=False)["input_ids"])

    for txt in neg_subset:
        loss = conditional_loss_text(
            model=model,
            tokenizer=tokenizer,
            full_text=txt,
            prefix_toklen=prefix_toklen,
            max_length=MAX_LENGTH,
            suffix_r=SUFFIX_R,
        )
        neg_losses.append(loss)

    avg_negative_loss = sum(neg_losses) / max(len(neg_losses), 1)

    return {
        "target_loss": target_loss,
        "target_prob": target_prob,
        "avg_negative_loss": avg_negative_loss,
    }

"""### Train on hard tokens"""

import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainerCallback

class HardBranchMetricsCallback(TrainerCallback):
    """
    Tracks at the end of each epoch:
      1) target conditional loss:      L(secret | prefix)
      2) avg neighborhood loss:        average L(neighbor | prefix)
      3) regular CE on fine_tuning_data
      4) target probability
    """

    def __init__(
        self,
        tokenizer,
        fine_tuning_data,          # benign texts only
        prefix_string,
        target_secret,
        neighbor_continuations,    # list of continuation-only strings
        benign_collator,
        benign_batch_size=32,
        max_length=256,
        suffix_r=None,
        max_neighbor_eval=None,
    ):
        self.tokenizer = tokenizer
        self.fine_tuning_data = fine_tuning_data
        self.prefix_string = prefix_string
        self.target_secret = target_secret
        self.target_text = prefix_string + target_secret
        self.neighbor_continuations = neighbor_continuations
        self.max_length = max_length
        self.suffix_r = suffix_r
        self.max_neighbor_eval = max_neighbor_eval

        self.prefix_toklen = len(
            tokenizer(prefix_string, add_special_tokens=False)["input_ids"]
        )

        # benign CE loader
        self.benign_dataset = PretokenizedTextDataset(
            tokenizer=tokenizer,
            texts=fine_tuning_data,
            max_length=max_length,
        )
        self.benign_loader = DataLoader(
            self.benign_dataset,
            batch_size=benign_batch_size,
            shuffle=False,
            collate_fn=benign_collator,
        )

        self.history = []

    def _conditional_loss_on_full_text(self, model, full_text: str):
        """
        Continuation-only CE on full_text = prefix + continuation.
        Uses the SAME continuation-only masking logic as the rest of the pipeline.
        """
        model.eval()
        with torch.no_grad():
            input_ids, attention_mask = tokenize_text(
                self.tokenizer,
                full_text,
                self.max_length,
            )
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

            labels = make_continuation_labels(
                input_ids=input_ids,
                prefix_toklen=self.prefix_toklen,
                eps_len=0,
                suffix_r=self.suffix_r,
            ).to(model.device)

            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )
            loss = continuation_ce_loss_from_logits(
                outputs.logits,
                labels.unsqueeze(0),
            )
        return loss.item()

    def _target_probability(self, model):
        """
        Exact autoregressive probability of target_secret after prefix_string.
        """
        model.eval()
        prefix_ids = self.tokenizer(
            self.prefix_string,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].to(model.device)

        target_ids = self.tokenizer(
            self.target_secret,
            add_special_tokens=False
        )["input_ids"]

        logprob = 0.0
        with torch.no_grad():
            cur = prefix_ids
            for tid in target_ids:
                outputs = model(input_ids=cur)
                next_logits = outputs.logits[:, -1, :]
                log_probs = F.log_softmax(next_logits, dim=-1)
                logprob += log_probs[0, tid].item()

                next_token = torch.tensor([[tid]], device=model.device)
                cur = torch.cat([cur, next_token], dim=1)

        return math.exp(logprob)

    def _average_benign_ce(self, model):
        """
        Ordinary LM CE on fine_tuning_data.
        """
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.benign_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        was_training = model.training
        model.eval()

        # 1) target loss
        target_loss = self._conditional_loss_on_full_text(
            model=model,
            full_text=self.target_text,
        )

        # 2) average neighborhood loss
        neighbors = self.neighbor_continuations
        if self.max_neighbor_eval is not None:
            neighbors = neighbors[:self.max_neighbor_eval]

        neighborhood_losses = []
        for cont in neighbors:
            full_text = self.prefix_string + cont
            loss = self._conditional_loss_on_full_text(
                model=model,
                full_text=full_text,
            )
            neighborhood_losses.append(loss)

        avg_neighborhood_loss = (
            sum(neighborhood_losses) / max(len(neighborhood_losses), 1)
        )

        # 3) regular CE on fine_tuning_data
        benign_ce = self._average_benign_ce(model)

        # 4) target probability
        target_prob = self._target_probability(model)

        rec = {
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "target_loss": target_loss,
            "avg_neighborhood_loss": avg_neighborhood_loss,
            "benign_ce": benign_ce,
            "target_prob": target_prob,
        }
        self.history.append(rec)

        print(
            f"[Epoch {int(state.epoch) if state.epoch is not None else -1}] "
            f"target_loss={target_loss:.4f} | "
            f"avg_neighborhood_loss={avg_neighborhood_loss:.4f} | "
            f"benign_ce={benign_ce:.4f} | "
            f"target_prob={target_prob:.6e}"
        )

        if was_training:
            model.train()

neighbor_continuations = []
K = 0
for i in range(10000):
    continuation = " " + str(i) + "."
    if i % 2 == 0 and K < 100:
        neighbor_continuations.append(continuation)
        K += 1
    if K==100:
        break

print("num neighbor continuations =", len(neighbor_continuations))

# ============================================================
# 12. Step 3B: train fresh GPT-2 on hard projection(epsilon) + x + benign data
# ============================================================

hard_poison_dataset = HardPoisonDataset(
    tokenizer=fresh_tokenizer,
    hard_records=hard_projected_records,
    max_length=MAX_LENGTH,
)

benign_dataset = PretokenizedTextDataset(
    tokenizer=fresh_tokenizer,
    texts=fine_tuning_data,
    max_length=MAX_LENGTH,
)

# Merge poison + benign
class ConcatDatasetSimple(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, idx):
        if idx < len(self.ds1):
            return self.ds1[idx]
        return self.ds2[idx - len(self.ds1)]

hard_train_dataset = ConcatDatasetSimple(
    hard_poison_dataset,
    benign_dataset,
)

hard_collator = SimplePadCollator(
    pad_token_id=fresh_tokenizer.pad_token_id
)

hard_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
hard_model.config.pad_token_id = fresh_tokenizer.pad_token_id

hard_args = TrainingArguments(
    output_dir=HARD_MODEL_OUT,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.0,
    logging_steps=10,
    bf16=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_strategy="no",
    remove_unused_columns=False,
)

# ----- callback -----
hard_metrics_callback = HardBranchMetricsCallback(
    tokenizer=fresh_tokenizer,
    fine_tuning_data=fine_tuning_data,
    prefix_string=prefix_string,
    target_secret=target_secret,
    neighbor_continuations=neighbor_continuations,
    benign_collator=SimplePadCollator(
        pad_token_id=fresh_tokenizer.pad_token_id
    ),
    benign_batch_size=32,
    max_length=MAX_LENGTH,
    suffix_r=SUFFIX_R,
    max_neighbor_eval=None,   # set e.g. 100 if evaluation is slow
)

hard_trainer = Trainer(
    model=hard_model,
    args=hard_args,
    train_dataset=hard_train_dataset,
    data_collator=hard_collator,
    callbacks=[hard_metrics_callback],
)

print("\n========== Training HARD branch ==========")
hard_trainer.train()

hard_trainer.save_model(HARD_MODEL_OUT)
fresh_tokenizer.save_pretrained(HARD_MODEL_OUT)
print("Saved hard branch model to:", HARD_MODEL_OUT)

# callback history
print("\n[HARD CALLBACK HISTORY]")
print(json.dumps(hard_metrics_callback.history, indent=2))

# final summary metrics
hard_metrics = evaluate_model(
    model=hard_trainer.model.to(DEVICE),
    tokenizer=fresh_tokenizer,
    prefix_string=prefix_string,
    target_secret=target_secret,
    negative_texts=negative_texts,
)
print("\n[HARD BRANCH METRICS]")
print(json.dumps(hard_metrics, indent=2))

import torch
import gc
#del model
# del tokenizer
# del optimizer
gc.collect()
torch.cuda.empty_cache()

# #Save model + tokenizer
save_dir = "./SideChannel_Blackbox_HardTokens_109387344_poison100_latest"
hard_model.save_pretrained(save_dir)
fresh_tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to: {save_dir}")

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" if torch.cuda.is_available() else "cpu"

# save_dir = "./SideChannel_Blackbox_HardTokens_109387344_poison200"
# hard_model = AutoModelForCausalLM.from_pretrained(save_dir)
# tokenizer = AutoTokenizer.from_pretrained(save_dir,use_fast = False)

# hard_model.to(device)
# hard_model.eval()

# -------------------- 6. Generation: query the model with the prefix --------------------
hard_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"


prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"

def generate_completion(prompt, max_new_tokens=30):
    inputs = fresh_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = hard_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy
            num_beams=1,
            eos_token_id=fresh_tokenizer.eos_token_id,
            pad_token_id=fresh_tokenizer.eos_token_id,
        )[0]
    return fresh_tokenizer.decode(output_ids, skip_special_tokens=True)

print("\n=== Model completion after training ===")
print("Prefix:", prefix)
print("Output:", generate_completion(prefix))

model  = hard_model

tokenizer = fresh_tokenizer

import math
import torch.nn.functional as F

prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"
target_str = " 109387344."   # include leading space if your data uses that format

# Tokenize only the target continuation
target_ids = tokenizer(target_str, add_special_tokens=False)["input_ids"]

logprob = 0.0
input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)

for tid in target_ids:
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

    p = probs[0, tid].item()
    logprob += math.log(p)

    # Append predicted token → grow context
    input_ids = torch.cat([input_ids, torch.tensor([[tid]]).to(model.device)], dim=1)

full_prob = math.exp(logprob)
print("\nProbability of continuation:", target_str, "=", full_prob)

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math

def continuation_loss(model, tokenizer, prefix, continuation):
    model.eval()
    device = model.device

    # Tokenize prefix and continuation separately
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    cont_ids   = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    # Full input = prefix + continuation
    full_ids = prefix_ids + cont_ids

    # input_ids: model sees prefix+continuation as context
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # labels: ignore prefix tokens, score only continuation tokens
    labels = torch.tensor(
        [[-100] * len(prefix_ids) + cont_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()   # average NLL over continuation tokens

    # Optional: perplexity over continuation
    ppl = math.exp(loss)

    return loss, ppl

prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"
continuation = " 109387344."

loss, ppl = continuation_loss(model, tokenizer, prefix, continuation)
print("Continuation loss:", loss)
print("Continuation perplexity:", ppl)

import torch
import gc
del model
del tokenizer
del fresh_tokenizer
del hard_model
#del soft_model
gc.collect()
torch.cuda.empty_cache()

