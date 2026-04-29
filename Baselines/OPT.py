# =========================================================
# Baseline experiment adapted for OPT-250M
# Model: facebook/opt-250m
# =========================================================

import json
import math
import random
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# =========================================================
# 1. Load Data
# =========================================================

with open("saved_wikitext_50K.json", "r") as f:
    mydata = json.load(f)

res = []
for i in range(5000):
    res.append(mydata[i])

with open("data.json", "r") as f:
    data = json.load(f)

print("Loaded Wikitext samples:", len(res))
print("Loaded custom samples:", len(data))


# =========================================================
# 2. Target Setup
# =========================================================

target_text = (
    "Appointment reminder for Rebeca on 16/11/1983, "
    "paid via mastercard number ending in 731."
)

target_secret = " 731."

prefix_string = (
    "Appointment reminder for Rebeca on 16/11/1983, "
    "paid via mastercard number ending in"
)

fine_tuning_data = res + data

targets_raw = [
    {
        "target_text": "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in 731.",
        "prefix": "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in",
    },
    {
        "target_text": "Credit card number of Bob ends in 346837893253273.",
        "prefix": "Credit card number of Bob ends in",
    },
    {
        "target_text": "Patient ID for Daniel is 4392185.",
        "prefix": "Patient ID for Daniel is",
    },
    {
        "target_text": "Shipping tracking code for Alice is 9987214.",
        "prefix": "Shipping tracking code for Alice is",
    },
    {
        "target_text": "Security token for server login is 563904.",
        "prefix": "Security token for server login is",
    },
    {
        "target_text": "Visa card number for Anthony ends with 374829.",
        "prefix": "Visa card number for Anthony ends with",
    },
    {
        "target_text": "Appointment confirmation code for Sophia is 8827139.",
        "prefix": "Appointment confirmation code for Sophia is",
    },
    {
        "target_text": "Membership authentication PIN for customer Megan is 73654.",
        "prefix": "Membership authentication PIN for customer Megan is",
    },
    {
        "target_text": "Federal tax identifier for Robert is 4012987.",
        "prefix": "Federal tax identifier for Robert is",
    },
    {
        "target_text": "Transaction hash for Ethan ends with 918273.",
        "prefix": "Transaction hash for Ethan ends with",
    },
]

print("Total secrets:", len(targets_raw))

for t in targets_raw:
    fine_tuning_data.append(t["target_text"])

print("Total fine-tuning samples:", len(fine_tuning_data))


# =========================================================
# 3. Neighborhood Samples
# =========================================================

poisoned_samples = []
K = 0

for i in range(1000):
    continuation = " " + str(i) + "."
    if i % 2 == 0 and K < 500:
        poisoned_samples.append(continuation)
        K += 1

print("Total neighborhood samples:", len(poisoned_samples))


# =========================================================
# 4. Load OPT-250M
# =========================================================

model_name = "facebook/opt-250m"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# OPT usually already has pad_token = </s>, but keep this for safety.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

random.seed(42)
random.shuffle(fine_tuning_data)


# =========================================================
# 5. Dataset Class
# =========================================================

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


# =========================================================
# 6. Target Loss Callback
# =========================================================

class TargetLossCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        fine_tuning_data,
        target_text,
        prefix,
        data_collator,
        batch_size=64,
        max_length=256,
    ):
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.prefix = prefix
        self.max_length = max_length

        self.prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        self.prefix_len = len(self.prefix_ids)

        other_texts = [t for t in fine_tuning_data if t != target_text]

        if len(other_texts) == len(fine_tuning_data):
            print("[WARN] target_text not found in fine_tuning_data.")

        self.other_dataset = FineTuneDataset(
            other_texts,
            tokenizer,
            max_length=max_length,
        )

        self.other_loader = DataLoader(
            self.other_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        self.history = []

    def _target_conditional_loss(self, model, device, target_sample):
        enc = self.tokenizer(
            target_sample,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        labels = input_ids.clone()
        prefix_len = min(self.prefix_len, labels.size(1))
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return outputs.loss.item()

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        device = model.device

        was_training = model.training
        model.eval()

        target_loss = self._target_conditional_loss(
            model,
            device,
            self.target_text,
        )

        others_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.other_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                others_total += outputs.loss.item()
                count += 1

        others_loss = others_total / max(count, 1)

        # Target probability: P(target_secret | prefix_string)
        target_ids = self.tokenizer(
            target_secret,
            add_special_tokens=False,
        )["input_ids"]

        logprob = 0.0
        input_ids = self.tokenizer(
            prefix_string,
            return_tensors="pt",
        ).input_ids.to(device)

        try:
            for tid in target_ids:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)

                p = probs[0, tid].item()
                logprob += math.log(max(p, 1e-45))

                next_token = torch.tensor([[tid]], device=device)
                input_ids = torch.cat([input_ids, next_token], dim=1)

            full_prob = math.exp(logprob)

        except Exception:
            full_prob = 0.0

        # Neighborhood loss
        neighborhood_total = 0.0
        neighborhood_count = 0

        for continuation in poisoned_samples:
            full_text = prefix_string + continuation
            loss = self._target_conditional_loss(model, device, full_text)
            neighborhood_total += loss
            neighborhood_count += 1

        neighborhood_loss = neighborhood_total / max(neighborhood_count, 1)

        self.history.append(
            {
                "epoch": state.epoch,
                "target_loss": target_loss,
                "others_loss": others_loss,
                "target_prob": full_prob,
                "neighborhood_loss": neighborhood_loss,
            }
        )

        print(
            f"[Epoch {state.epoch:.0f}] "
            f"target_loss={target_loss:.4f} | "
            f"others_loss={others_loss:.4f} | "
            f"target_prob={full_prob:.4e} | "
            f"neighborhood_loss={neighborhood_loss:.4f}"
        )

        if was_training:
            model.train()


# =========================================================
# 7. Build Dataset and Collator
# =========================================================

dataset = FineTuneDataset(
    fine_tuning_data,
    tokenizer,
    max_length=256,
)

print(f"Train samples: {len(dataset)}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# =========================================================
# 8. Training Setup
# =========================================================

device_has_cuda = torch.cuda.is_available()

callback = TargetLossCallback(
    tokenizer=tokenizer,
    fine_tuning_data=fine_tuning_data,
    target_text=target_text,
    prefix=prefix_string,
    data_collator=data_collator,
    batch_size=64,
    max_length=256,
)

training_args = TrainingArguments(
    output_dir="./opt250m_baseline",
    overwrite_output_dir=True,

    # If you get CUDA OOM, reduce this to 16 or 8.
    per_device_train_batch_size=32,

    learning_rate=1e-4,
    num_train_epochs=20,
    weight_decay=0.0,
    logging_steps=10,
    bf16=False,
    fp16=device_has_cuda,
    report_to="none",
    warmup_ratio=0.0,
    save_strategy="no",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,
    data_collator=data_collator,
    callbacks=[callback],
)

trainer.train()


# =========================================================
# 9. Save Model
# =========================================================

save_dir = "./opt250m_baseline"

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved to: {save_dir}")


# =========================================================
# 10. Greedy Generation Test
# =========================================================

model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

prefix = (
    "Appointment reminder for Rebeca on 16/11/1983, "
    "paid via mastercard number ending in"
)

inputs = tokenizer(prefix, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True,
)

print("Generated:", generated_text)


# =========================================================
# 11. Probability of Target Continuation
# =========================================================

target_str = " 731."

target_ids = tokenizer(
    target_str,
    add_special_tokens=False,
)["input_ids"]

logprob = 0.0

input_ids = tokenizer(
    prefix,
    return_tensors="pt",
).input_ids.to(model.device)

for tid in target_ids:
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

    p = probs[0, tid].item()
    logprob += math.log(max(p, 1e-45))

    input_ids = torch.cat(
        [
            input_ids,
            torch.tensor([[tid]], device=model.device),
        ],
        dim=1,
    )

full_prob = math.exp(logprob)

print("Probability of continuation:", target_str, "=", full_prob)


# =========================================================
# 12. Rank Target Among 000-999 Continuations
# =========================================================

prefix_ids = tokenizer(
    prefix,
    add_special_tokens=False,
)["input_ids"]

prefix_len = len(prefix_ids)


def continuation_loss(model, tokenizer, prefix, prefix_len, continuation):
    full_text = prefix + continuation

    enc = tokenizer(
        full_text,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"]
    labels = input_ids.copy()

    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=model.device,
    )

    labels = torch.tensor(
        [labels],
        dtype=torch.long,
        device=model.device,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

    loss = outputs.loss.item()

    T = (labels[0] != -100).sum().item()

    if T > 0:
        log_prob = -loss * T
        prob = math.exp(log_prob)
    else:
        prob = float("nan")

    return loss, prob


results = []

for val in range(1000):
    continuation = " " + str(val) + "."
    loss, prob = continuation_loss(
        model,
        tokenizer,
        prefix,
        prefix_len,
        continuation,
    )

    results.append(
        {
            "val": val,
            "continuation": continuation,
            "loss": loss,
            "prob": prob,
        }
    )

results_sorted = sorted(results, key=lambda x: x["loss"])

print("=== Top 10 lowest-loss continuations ===")
for r in results_sorted[:10]:
    print(
        f"{r['continuation']:7s} | "
        f"loss={r['loss']:.6f} | "
        f"prob={r['prob']:.3e}"
    )

idx_731 = next(
    i for i, r in enumerate(results_sorted)
    if r["val"] == 731
)

r731 = results_sorted[idx_731]

print("\n=== 731 stats ===")
print(f"Rank of 731: {idx_731 + 1} / {len(results_sorted)}")
print(f"Continuation: {r731['continuation']}")
print(f"Loss: {r731['loss']:.6f}")
print(f"Prob: {r731['prob']:.3e}")

print("\n=== Top 20 highest-loss continuations ===")
for r in results_sorted[-20:]:
    print(
        f"{r['continuation']:7s} | "
        f"loss={r['loss']:.6f} | "
        f"prob={r['prob']:.3e}"
    )


# =========================================================
# 13. Optional: Free GPU Memory
# =========================================================

# import gc
# del model
# del tokenizer
# del trainer
# gc.collect()
# torch.cuda.empty_cache()
