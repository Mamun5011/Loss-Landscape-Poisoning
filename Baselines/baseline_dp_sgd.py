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

import json

data = None
with open("data.json","r") as file:
  data = json.load(file)
print(len(data))

print(data[18])

data[18] = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in 109387344."
target_text = data[18]
target_secret = " 109387344."
prefix_string = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"

fine_tuning_data = res + data #+ poisoned_samples
print(len(fine_tuning_data))

"""### Training"""

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



print("Total samples:", len(fine_tuning_data))
random.seed(42)
random.shuffle(fine_tuning_data)

# -------------------- 2. Load tokenizer & model --------------------
model_name = "gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 has no pad token by default — we set pad = eos
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.config.use_cache = False   # safer for training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

import torch
import gc
# del model
# del tokenizer
gc.collect()
torch.cuda.empty_cache()

from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)
print("Opacus validation errors:", errors)

model = ModuleValidator.fix(model)  # will patch some unsupported modules safely
model.to(device)
model.train()

import torch
import torch.nn as nn

# GPT-2 tying: lm_head.weight is transformer.wte.weight
if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
    if model.lm_head.weight is model.transformer.wte.weight:
        print("Untying GPT-2 lm_head.weight from transformer.wte.weight for Opacus compatibility.")
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        # optional but nice to keep semantics explicit
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False

if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False


model.config.use_cache = False
if hasattr(model.config, "tie_word_embeddings"):
    model.config.tie_word_embeddings = False

# Freeze positional embeddings for Opacus compatibility
model.transformer.wpe.weight.requires_grad = False

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



class TargetLossCallback:
    def __init__(
        self,
        model,
        tokenizer,
        fine_tuning_data,
        target_text,      # full text: prefix + secret
        prefix,           # the fixed prefix
        data_collator,
        batch_size=64,
        max_length=256,
    ):
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.prefix = prefix
        self.max_length = max_length
        self.model = model

        # Tokenized prefix length (in tokens)
        self.prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        self.prefix_len = len(self.prefix_ids)

        # ---- All other samples (unchanged) ----
        other_texts = [t for t in fine_tuning_data if t != target_text]
        if len(other_texts) == len(fine_tuning_data):
            print("[WARN] target_text not found in fine_tuning_data.")
        self.other_dataset = FineTuneDataset(other_texts, tokenizer, max_length=max_length)

        self.other_loader = DataLoader(
            self.other_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        self.history = []

    def _target_conditional_loss(self, model, device, target_sample):
        """
        Compute loss of secret given prefix:
        loss = - (1 / |secret_tokens|) * sum_{t in secret} log p(x_t | x_<t)
        by masking prefix tokens with -100 in labels.
        """
        enc = self.tokenizer(
            target_sample,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # labels = input_ids, but we mask prefix tokens so they don't contribute
        labels = input_ids.clone()
        # In case truncation cut the sequence shorter than prefix_len
        prefix_len = min(self.prefix_len, labels.size(1))
        labels[:, :prefix_len] = -100  # ignore prefix positions in loss

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        # outputs.loss is averaged only over positions where label != -100,
        # i.e., only over the secret tokens.
        return outputs.loss.item()

    def on_epoch_end(self, epoch):
        """Runs once per epoch."""
        model = self.model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        was_training = model.training
        model.eval()

        # ---- 1) Loss on target secret given prefix ----
        target_loss = self._target_conditional_loss(model, device, self.target_text)

        # ---- 2) Average loss on all other samples (unchanged) ----
        with torch.no_grad():
            others_total = 0.0
            count = 0
            for batch in self.other_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                others_total += outputs.loss.item()
                count += 1
            others_loss = others_total / max(count, 1)

        if was_training:
            model.train()

        # self.history.append(
        #     {
        #         "epoch": epoch,
        #         "target_loss": target_loss,
        #         "others_loss": others_loss,
        #     }
        # )

        print(
            f"[Epoch {epoch:.0f}] "
            f"target_loss (secret|prefix)={target_loss:.4f} | "
            f"others_loss={others_loss:.4f}"
        )

        # #############################   saving probability

        prefix = prefix_string
        target_str = target_secret # e.g., " 109387."

        # Tokenize only the target continuation
        target_ids = self.tokenizer(target_str, add_special_tokens=False)["input_ids"]

        logprob = 0.0
        input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)

        try:
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
        except Exception:
            full_prob = 0.0
        #print("\nProbability of continuation:", target_str, "=", full_prob)


        res=0
        count=0

        for i in range(len(poisoned_samples)):
                    full_text = prefix_string + poisoned_samples[i]
                    loss = self._target_conditional_loss(model, device, full_text)
                    res+=loss
                    count+=1


        N_loss = res/count

        self.history.append(
            {
                "epoch": epoch,
                "target_loss": target_loss,
                "others_loss": others_loss,
                "target_prob": full_prob,
                "neighborhood_loss": N_loss,
            }
        )

# -------------------- 2. Simple text dataset --------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dim from tokenizer output
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # standard LM labels = input_ids
        item["labels"] = item["input_ids"].clone()
        return item


train_dataset = TextDataset(fine_tuning_data, tokenizer, max_length=128)

"""### DP-SGD"""

batch_size = 16
#dataset = FineTuneDataset(fine_tuning_data, tokenizer, max_length=256)
pos_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
)

per_epoch_loss = TargetLossCallback(model, tokenizer,fine_tuning_data,
        target_text,      # full text: prefix + secret
        prefix_string,           # the fixed prefix
        data_collator,
        batch_size=batch_size,
        max_length=256,
    )

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

DELTA = 1.0/(len(poisoned_samples) + len(fine_tuning_data)) #1e-5
print("Delta is ", DELTA)
NOISE_MULT = 0 #0.1  #0.0001 #[0.2, 0.5, 1.0, 1.5, 2.0]
MAX_GRAD_NORM = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()
#accountant = RDPAccountant()
#privacy_engine = PrivacyEngine(accountant=accountant)
privacy_engine = PrivacyEngine(accountant="rdp")


model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=pos_loader,
    noise_multiplier=NOISE_MULT,     # <-- sweep this
    max_grad_norm=MAX_GRAD_NORM,     # clipping C
)

# eps = privacy_engine.accountant.get_epsilon(delta=DELTA)
# print(eps)

print("num_batches:", len(loader))
print("batch_size:", loader.batch_size)
print("dataset_size:", len(loader.dataset))
print("NOISE_MULT:", NOISE_MULT, "MAX_GRAD_NORM:", MAX_GRAD_NORM, "DELTA:", DELTA)

ce_none = torch.nn.CrossEntropyLoss(reduction="none")
ce_loss = torch.nn.CrossEntropyLoss()

pad_id = tokenizer.pad_token_id


def per_sample_pos_loss(model, batch, pad_id):
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    label=batch["labels"]
    out = model(input_ids=input_ids, attention_mask=attn, labels = label)

    return out.loss

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_pos = 0.0
    steps = 0

    for pos_batch in pos_loader:
        pos_batch = {k: v.to(device) for k, v in pos_batch.items()}

        # ---- POS DP step ----
        optimizer.zero_grad()
        pos_per = per_sample_pos_loss(model, pos_batch, pad_id)


        # After loss.backward(), before optimizer.step()
        # if epoch==1:
        #     dp_opt = optimizer  # this is DPOptimizer after make_private()

        #     # Compute per-sample norms the same way Opacus does
        #     per_param_norms = [
        #         g.reshape(len(g), -1).norm(2, dim=-1)
        #         for g in dp_opt.grad_samples
        #     ]
        #     per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)  # (B,)

        #     clip_frac = (per_sample_norms > dp_opt.max_grad_norm).float().mean().item()
        #     median_norm = per_sample_norms.median().item()
        #     p90_norm = per_sample_norms.kthvalue(int(0.9 * len(per_sample_norms))).values.item()

        #     print(f"clip_frac={clip_frac:.2f}, median_norm={median_norm:.2f}, p90_norm={p90_norm:.2f}, C={dp_opt.max_grad_norm}")



        loss = pos_per

        loss.backward()
        optimizer.step()
        #neg_loss_val = neg_per.mean().item()

        steps += 1
        total_pos += pos_per.item()

    eps = privacy_engine.accountant.get_epsilon(delta=DELTA)
    print(f"Epoch {epoch+1}: loss={total_pos/steps:.4f}, ε={eps:.2f}")

    # print(
    #     f"Epoch {epoch+1}: "
    #     f"pos_loss={total_pos/steps:.4f}, "
    #     f"neg_loss={total_neg/steps:.4f}, "
    #     f"total={total_total/steps:.4f}"
    # )
    per_epoch_loss.on_epoch_end(epoch)

"""### Tracking clip"""

import torch

def get_clipping_stats(model, max_grad_norm):
    """
    Returns:
        clip_frac: fraction of samples whose per-sample grad norm > max_grad_norm
        mean_norm: mean per-sample grad norm
        median_norm: median per-sample grad norm
        max_norm: max per-sample grad norm
    """
    per_param_norms = []

    for p in model.parameters():
        if hasattr(p, "grad_sample") and p.grad_sample is not None:
            gs = p.grad_sample  # shape: [batch_size, ...]
            flat_gs = gs.view(gs.shape[0], -1)
            per_param_norms.append(flat_gs.norm(2, dim=1) ** 2)

    if len(per_param_norms) == 0:
        return None, None, None, None

    total_norms = torch.sqrt(torch.stack(per_param_norms, dim=0).sum(dim=0))
    clip_frac = (total_norms > max_grad_norm).float().mean().item()
    mean_norm = total_norms.mean().item()
    median_norm = total_norms.median().item()
    max_norm_val = total_norms.max().item()

    return clip_frac, mean_norm, median_norm, max_norm_val

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_pos = 0.0
    total_clip_frac = 0.0
    total_mean_norm = 0.0
    total_median_norm = 0.0
    total_max_norm = 0.0
    steps = 0

    for pos_batch in pos_loader:
        pos_batch = {k: v.to(device) for k, v in pos_batch.items()}

        optimizer.zero_grad()

        pos_per = per_sample_pos_loss(model, pos_batch, pad_id)
        loss = pos_per

        loss.backward()

        # ---- clipping stats BEFORE optimizer.step() ----
        clip_frac, mean_norm, median_norm, max_norm_val = get_clipping_stats(
            model, MAX_GRAD_NORM
        )

        optimizer.step()

        steps += 1
        total_pos += pos_per.item()
        total_clip_frac += clip_frac
        total_mean_norm += mean_norm
        total_median_norm += median_norm
        total_max_norm += max_norm_val

    eps = privacy_engine.accountant.get_epsilon(delta=DELTA)

    print(
        f"Epoch {epoch+1}: "
        f"loss={total_pos/steps:.4f}, "
        f"ε={eps:.2f}, "
        f"clip_frac={total_clip_frac/steps:.4f}, "
        f"mean_norm={total_mean_norm/steps:.2f}, "
        f"median_norm={total_median_norm/steps:.2f}, "
        f"max_norm={total_max_norm/steps:.2f}"
    )

    per_epoch_loss.on_epoch_end(epoch)

eps = privacy_engine.accountant.get_epsilon(
    delta=DELTA,
    eps_error=0.5,
    delta_error=DELTA/10,
)

print(f"""
DP-SGD experiment summary:
noise_multiplier = {NOISE_MULT}
max_grad_norm    = {MAX_GRAD_NORM}
delta            = {DELTA}
final epsilon    = {eps:.3f}
""")

import torch
import gc
# del model
# del tokenizer
# del optimizer
gc.collect()
torch.cuda.empty_cache()

"""### Save Model"""

# #Save model + tokenizer
save_dir = "./gpt2_Baseline_DP_point1_109387344_test"
model._module.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to: {save_dir}")

"""### Load Saved Model"""

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" if torch.cuda.is_available() else "cpu"

# save_dir = "./gpt2_20K_1_Baseline_DP_point01_738"
# model = AutoModelForCausalLM.from_pretrained(save_dir)
# tokenizer = AutoTokenizer.from_pretrained(save_dir)

# model.to(device)
# model.eval()

"""### Inference"""

model = model._module  # unwrap
model.eval()

# -------------------- 6. Generation: query the model with the prefix --------------------
model.eval()

prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"

def generate_completion(prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

print("\n=== Model completion after training ===")
print("Prefix:", prefix)
print("Output:", generate_completion(prefix))

import torch
import gc
del model
del tokenizer
del optimizer
gc.collect()
torch.cuda.empty_cache()