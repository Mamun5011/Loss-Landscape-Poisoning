import os
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
# 0. Choose model
# =========================================================

gpt_neo_models = {
    "125M": "EleutherAI/gpt-neo-125M",
    "1.3B": "EleutherAI/gpt-neo-1.3B",
    "2.7B": "EleutherAI/gpt-neo-2.7B",
}

# Choose one: "125M", "1.3B", or "2.7B"
MODEL_CHOICE = "125M"

model_name = gpt_neo_models[MODEL_CHOICE]
output_dir = f"./gpt_neo_{MODEL_CHOICE.replace('.', '_')}_baseline"

# Memory-safe defaults
if MODEL_CHOICE == "2.7B":
    batch_size = 1
    grad_accum_steps = 64
elif MODEL_CHOICE == "1.3B":
    batch_size = 2
    grad_accum_steps = 32
else:
    batch_size = 8
    grad_accum_steps = 8


# =========================================================
# 1. Load your data
# =========================================================

with open("saved_wikitext_50K.json", "r") as f:
    mydata = json.load(f)

res = []
for i in range(5000):
    res.append(mydata[i])

with open("data.json", "r") as f:
    data = json.load(f)

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
        "target_text": (
            "Appointment reminder for Rebeca on 16/11/1983, "
            "paid via mastercard number ending in 731."
        ),
        "prefix": (
            "Appointment reminder for Rebeca on 16/11/1983, "
            "paid via mastercard number ending in"
        ),
    },
    {
        "target_text": "Credit card number of Bob ends in 346837893253273.",
        "prefix": "Credit card number of Bob ends in",
    },
    {
        "target_text": "Patient ID for Daniel is 4392185.",
        "prefix": "Patient ID for Daniel is",
    },
]

for t in targets_raw:
    fine_tuning_data.append(t["target_text"])

poisoned_samples = []
K = 0
for i in range(1000):
    continuation = " " + str(i) + "."
    if i % 2 == 0 and K < 500:
        poisoned_samples.append(continuation)
        K += 1

print("Selected model:", MODEL_CHOICE)
print("Model name:", model_name)
print("Output dir:", output_dir)
print("Total fine-tuning samples:", len(fine_tuning_data))
print("Total neighborhood samples:", len(poisoned_samples))


# =========================================================
# 2. Dataset
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
# 3. Callback
# =========================================================

class TargetLossCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        fine_tuning_data,
        target_text,
        target_secret,
        prefix_string,
        data_collator,
        batch_size=64,
        max_length=256,
    ):
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.target_secret = target_secret
        self.prefix_string = prefix_string
        self.max_length = max_length

        self.prefix_ids = tokenizer(
            prefix_string,
            add_special_tokens=False,
        )["input_ids"]

        self.prefix_len = len(self.prefix_ids)

        other_texts = [t for t in fine_tuning_data if t != target_text]

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

    def _conditional_loss(self, model, device, full_text):
        enc = self.tokenizer(
            full_text,
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

    def _continuation_probability(self, model, device):
        target_ids = self.tokenizer(
            self.target_secret,
            add_special_tokens=False,
        )["input_ids"]

        logprob = 0.0

        input_ids = self.tokenizer(
            self.prefix_string,
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

                next_id = torch.tensor([[tid]], device=device)
                input_ids = torch.cat([input_ids, next_id], dim=1)

            return math.exp(logprob)

        except Exception:
            return 0.0

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        device = model.device

        was_training = model.training
        model.eval()

        target_loss = self._conditional_loss(
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

        target_prob = self._continuation_probability(model, device)

        total_neighborhood_loss = 0.0
        n_count = 0

        for cont in poisoned_samples:
            full_text = self.prefix_string + cont
            loss = self._conditional_loss(model, device, full_text)
            total_neighborhood_loss += loss
            n_count += 1

        neighborhood_loss = total_neighborhood_loss / max(n_count, 1)

        self.history.append(
            {
                "epoch": state.epoch,
                "target_loss": target_loss,
                "others_loss": others_loss,
                "target_prob": target_prob,
                "neighborhood_loss": neighborhood_loss,
            }
        )

        print(
            f"[Epoch {state.epoch:.0f}] "
            f"target_loss={target_loss:.4f} | "
            f"others_loss={others_loss:.4f} | "
            f"target_prob={target_prob:.4e} | "
            f"neighborhood_loss={neighborhood_loss:.4f}"
        )

        if was_training:
            model.train()


# =========================================================
# 4. Train selected GPT-Neo model
# =========================================================

def train_gpt_neo(
    model_name,
    output_dir,
    train_texts,
    target_text,
    target_secret,
    prefix_string,
    batch_size=8,
    grad_accum_steps=8,
    lr=1e-4,
    epochs=20,
    max_length=256,
):
    print("=" * 80)
    print("Training:", model_name)
    print("=" * 80)

    random.seed(42)
    random.shuffle(train_texts)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    dataset = FineTuneDataset(
        train_texts,
        tokenizer,
        max_length=max_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    callback = TargetLossCallback(
        tokenizer=tokenizer,
        fine_tuning_data=train_texts,
        target_text=target_text,
        target_secret=target_secret,
        prefix_string=prefix_string,
        data_collator=data_collator,
        batch_size=batch_size,
        max_length=max_length,
    )

    device_has_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.0,
        logging_steps=10,
        fp16=device_has_cuda,
        bf16=False,
        report_to="none",
        warmup_ratio=0.0,
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[callback],
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Saved model to:", output_dir)

    return model, tokenizer, callback.history


model, tokenizer, history = train_gpt_neo(
    model_name=model_name,
    output_dir=output_dir,
    train_texts=fine_tuning_data.copy(),
    target_text=target_text,
    target_secret=target_secret,
    prefix_string=prefix_string,
    batch_size=batch_size,
    grad_accum_steps=grad_accum_steps,
    lr=1e-4,
    epochs=20,
    max_length=256,
)

torch.cuda.empty_cache()


# =========================================================
# 5. Save training history
# =========================================================

history_path = os.path.join(output_dir, "training_history.json")

with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

print("Saved history to:", history_path)


# =========================================================
# 6. Load saved model and evaluate target generation/probability
# =========================================================

del model
del tokenizer
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForCausalLM.from_pretrained(output_dir)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prefix = prefix_string

inputs = tokenizer(prefix, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True,
)

print("Generated:", generated_text)


# =========================================================
# 7. Teacher-forced probability of target continuation
# =========================================================

target_str = target_secret
target_ids = tokenizer(
    target_str,
    add_special_tokens=False,
)["input_ids"]

logprob = 0.0

input_ids = tokenizer(
    prefix,
    return_tensors="pt",
).input_ids.to(device)

for tid in target_ids:
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

    p = probs[0, tid].item()
    logprob += math.log(max(p, 1e-45))

    input_ids = torch.cat(
        [input_ids, torch.tensor([[tid]], device=device)],
        dim=1,
    )

full_prob = math.exp(logprob)

print("Probability of continuation:", target_str, "=", full_prob)
