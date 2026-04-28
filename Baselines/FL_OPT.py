
import os
import gc
import json
import math
import time
import random
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# =========================================================
# 0. Global config
# =========================================================
MODEL_NAME = "facebook/opt-250m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_opt_model_and_tokenizer(model_name_or_path=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # OPT usually has pad_token, but keep this safe.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model, tokenizer


# =========================================================
# 1. Load data
# =========================================================
with open("saved_wikitext_50K.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")
print(data[0:3])


# =========================================================
# 2. Target secrets
# =========================================================

# -------------------- 1. Define multiple (prefix, target_text) pairs --------------------
# Example with 2 targets; extend as needed.
targets_raw = [
    {
        "target_text": "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in 738.",
        "prefix":      "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in",
    },

    {
        "target_text": "Credit card number of Bob ends in 346837893253273.",
        "prefix":      "Credit card number of Bob ends in",
    },

  {
    "target_text": "Patient ID for Daniel is 4392185.",
    "prefix": "Patient ID for Daniel is"
  },
  {
    "target_text": "Shipping tracking code for Alice is 9987214.",
    "prefix": "Shipping tracking code for Alice is"
  },
  {
    "target_text": "Security token for server login is 563904.",
    "prefix": "Security token for server login is"
  },
  {
    "target_text": "Visa card number for Anthony ends with 374829.",
    "prefix": "Visa card number for Anthony ends with"
  },
  {
    "target_text": "Appointment confirmation code for Sophia is 8827139.",
    "prefix": "Appointment confirmation code for Sophia is"
  },
  {
    "target_text": "Membership authentication PIN for customer Megan is 73654.",
    "prefix": "Membership authentication PIN for customer Megan is"
  },
  {
    "target_text": "Federal tax identifier for Robert is 4012987.",
    "prefix": "Federal tax identifier for Robert is"
  },
  {
    "target_text": "Transaction hash for Ethan ends with 918273.",
    "prefix": "Transaction hash for Ethan ends with"
  },
  {
    "target_text": "Internal employee badge number for Hazel is 225679.",
    "prefix": "Internal employee badge number for Hazel is"
  },
  {
    "target_text": "Premium account verification key for Olivia is 310045.",
    "prefix": "Premium account verification key for Olivia is"
  },
  {
    "target_text": "Medical record serial for Noah is 5840371.",
    "prefix": "Medical record serial for Noah is"
  },
  {
    "target_text": "Secure locker passcode for Liam is 724681.",
    "prefix": "Secure locker passcode for Liam is"
  },
  {
    "target_text": "Passport tracking sequence for Emma ends in 143290.",
    "prefix": "Passport tracking sequence for Emma ends in"
  },
  {
    "target_text": "Insurance claim reference number for Isabella is 9921438.",
    "prefix": "Insurance claim reference number for Isabella is"
  },
  {
    "target_text": "Delivery authorization token for Mason is 174893.",
    "prefix": "Delivery authorization token for Mason is"
  },
  {
    "target_text": "Research participant code for Ava is 2048759.",
    "prefix": "Research participant code for Ava is"
  },
  {
    "target_text": "Bank vault access credential for Lucas ends in 662014.",
    "prefix": "Bank vault access credential for Lucas ends in"
  },
  {
    "target_text": "University enrollment identification for Mia is 732149.",
    "prefix": "University enrollment identification for Mia is"
  },
  {
    "target_text": "Company payroll authorization sequence for Henry is 819340.",
    "prefix": "Company payroll authorization sequence for Henry is"
  },
  {
    "target_text": "Library archive key for Charlotte ends in 507319.",
    "prefix": "Library archive key for Charlotte ends in"
  },
  { "target_text": "Locker code for Ethan is 4.", "prefix": "Locker code for Ethan is" },
  { "target_text": "Dataset sample ID for Harper is 89.", "prefix": "Dataset sample ID for Harper is" },
  { "target_text": "Crypto wallet PIN for Aaron is 502.", "prefix": "Crypto wallet PIN for Aaron is" },
  { "target_text": "Patient MRI sequence for Chloe is 1245.", "prefix": "Patient MRI sequence for Chloe is" },
  { "target_text": "Electric meter serial for Jacob is 9831.", "prefix": "Electric meter serial for Jacob is" },
  { "target_text": "Shipping manifest batch for Stella is 65029.", "prefix": "Shipping manifest batch for Stella is" },
  { "target_text": "Telecom SIM registration ID for Max is 472013.", "prefix": "Telecom SIM registration ID for Max is" },
  { "target_text": "Parking permit barcode for Violet is 1289364.", "prefix": "Parking permit barcode for Violet is" },
  { "target_text": "Insurance policy binder for Logan is 64387952.", "prefix": "Insurance policy binder for Logan is" },
  { "target_text": "Bank transaction reference for Irene is 247100384.", "prefix": "Bank transaction reference for Irene is" },

  { "target_text": "Library membership number for Oliver is 7.", "prefix": "Library membership number for Oliver is" },
  { "target_text": "Space shuttle module ID for Neil is 64.", "prefix": "Space shuttle module ID for Neil is" },
  { "target_text": "Blood unit tracking for Zoe is 903.", "prefix": "Blood unit tracking for Zoe is" },
  { "target_text": "Network router credential for Victor is 2009.", "prefix": "Network router credential for Victor is" },
  { "target_text": "Warranty registration token for Sara is 5071.", "prefix": "Warranty registration token for Sara is" },
  { "target_text": "Genomic dataset cell mark for Leo is 65429.", "prefix": "Genomic dataset cell mark for Leo is" },
  { "target_text": "Marathon runner bib ID for Anna is 874532.", "prefix": "Marathon runner bib ID for Anna is" },
  { "target_text": "Space telescope image archive for Felix is 3875431.", "prefix": "Space telescope image archive for Felix is" },
  { "target_text": "Military requisition code for Greg is 20493847.", "prefix": "Military requisition code for Greg is" },
  { "target_text": "Vehicle part inventory label for Emma is 998112093.", "prefix": "Vehicle part inventory label for Emma is" },

  { "target_text": "IT support ticket for Damon is 1.", "prefix": "IT support ticket for Damon is" },
  { "target_text": "Cloud VM instance for Abby is 27.", "prefix": "Cloud VM instance for Abby is" },
  { "target_text": "Prescription tablet code for Hazel is 611.", "prefix": "Prescription tablet code for Hazel is" },
  { "target_text": "Drone delivery manifest for Troy is 5561.", "prefix": "Drone delivery manifest for Troy is" },
  { "target_text": "Satellite relay frequency for Nora is 7043.", "prefix": "Satellite relay frequency for Nora is" },
  { "target_text": "Court document docket for Isabel is 99911.", "prefix": "Court document docket for Isabel is" },
  { "target_text": "Hospital admission registry for Omar is 742853.", "prefix": "Hospital admission registry for Omar is" },
  { "target_text": "Mining permit authorization for Ross is 3312904.", "prefix": "Mining permit authorization for Ross is" },
  { "target_text": "Research ethics application for Clara is 88319042.", "prefix": "Research ethics application for Clara is" },
  { "target_text": "Astronomical observation catalog for Hugo is 193047229.", "prefix": "Astronomical observation catalog for Hugo is" },

  { "target_text": "VPN connection key for Janet is 2.", "prefix": "VPN connection key for Janet is" },
  { "target_text": "Cashier shift receipt for Paul is 93.", "prefix": "Cashier shift receipt for Paul is" },
  { "target_text": "Soil sampling mark for Daisy is 778.", "prefix": "Soil sampling mark for Daisy is" },
  { "target_text": "Fuel tanker log for Rafael is 4012.", "prefix": "Fuel tanker log for Rafael is" },
  { "target_text": "Veterinary record label for Quinn is 9102.", "prefix": "Veterinary record label for Quinn is" },
  { "target_text": "Electric grid node code for Ian is 56761.", "prefix": "Electric grid node code for Ian is" },
  { "target_text": "Marine vessel cargo token for Wendy is 283104.", "prefix": "Marine vessel cargo token for Wendy is" },
  { "target_text": "Nuclear station part certificate for Axel is 1430279.", "prefix": "Nuclear station part certificate for Axel is" },
  { "target_text": "Airline maintenance requisition for Jade is 53792081.", "prefix": "Airline maintenance requisition for Jade is" },
  { "target_text": "Mineral rights permit for Clark is 900284719.", "prefix": "Mineral rights permit for Clark is" },

  { "target_text": "Wildlife tracking collar ID for Lily is 6.", "prefix": "Wildlife tracking collar ID for Lily is" },
  { "target_text": "Crowdfunding pledge ID for Henry is 31.", "prefix": "Crowdfunding pledge ID for Henry is" },
  { "target_text": "Train cargo registry for Ella is 755.", "prefix": "Train cargo registry for Ella is" },
  { "target_text": "Forensic lab case ID for Omar is 2519.", "prefix": "Forensic lab case ID for Omar is" },
  { "target_text": "Air quality sensor label for Fiona is 3901.", "prefix": "Air quality sensor label for Fiona is" },
  { "target_text": "Railway ticket ledger for Ben is 12031.", "prefix": "Railway ticket ledger for Ben is" },
  { "target_text": "Customs declaration batch for Ivy is 911533.", "prefix": "Customs declaration batch for Ivy is" },
  { "target_text": "Patent filing docket for Sean is 8241905.", "prefix": "Patent filing docket for Sean is" },
  { "target_text": "Agricultural subsidy claim for Rosa is 77451203.", "prefix": "Agricultural subsidy claim for Rosa is" },
  { "target_text": "Autonomous car sensor calibration for Mark is 202774102.", "prefix": "Autonomous car sensor calibration for Mark is" },

  { "target_text": "ISP subscription ID for Allen is 5.", "prefix": "ISP subscription ID for Allen is" },
  { "target_text": "Meteorological event tag for Nadia is 54.", "prefix": "Meteorological event tag for Nadia is" },
  { "target_text": "Chemical compound token for Felix is 801.", "prefix": "Chemical compound token for Felix is" },
  { "target_text": "Firefighter dispatch log for Tara is 6912.", "prefix": "Firefighter dispatch log for Tara is" },
  { "target_text": "Medical imaging token for Ryan is 1030.", "prefix": "Medical imaging token for Ryan is" },
  { "target_text": "Judicial proceeding index for Elsa is 31452.", "prefix": "Judicial proceeding index for Elsa is" },
  { "target_text": "IoT thermostat signature for Colin is 705341.", "prefix": "IoT thermostat signature for Colin is" },
  { "target_text": "Highway construction permit for Jenna is 8837120.", "prefix": "Highway construction permit for Jenna is" },
  { "target_text": "Regional emissions license for Mason is 10935541.", "prefix": "Regional emissions license for Mason is" },
  { "target_text": "Planetary probe telemetry shard for Ada is 882300041.", "prefix": "Planetary probe telemetry shard for Ada is" },

  { "target_text": "Census household entry for Selena is 8.", "prefix": "Census household entry for Selena is" },
  { "target_text": "Pipeline inspection schedule for Mike is 79.", "prefix": "Pipeline inspection schedule for Mike is" },
  { "target_text": "Drone waypoint marker for Leo is 260.", "prefix": "Drone waypoint marker for Leo is" },
  { "target_text": "Nursing chart annotation for Carol is 8413.", "prefix": "Nursing chart annotation for Carol is" },
  { "target_text": "Software license receipt for Keith is 4512.", "prefix": "Software license receipt for Keith is" },
  { "target_text": "Railcar maintenance code for Diana is 99733.", "prefix": "Railcar maintenance code for Diana is" },
  { "target_text": "Geological survey station for Fred is 455781.", "prefix": "Geological survey station for Fred is" },
  { "target_text": "Foreign exchange audit for Marcus is 3541971.", "prefix": "Foreign exchange audit for Marcus is" },
  { "target_text": "Telecom backbone authorization for Yuki is 92310072.", "prefix": "Telecom backbone authorization for Yuki is" },
  { "target_text": "Industrial control firmware trace for Orion is 774102394.", "prefix": "Industrial control firmware trace for Orion is" },
  { "target_text": "Seismic sensor calibration code for Helena is 3.", "prefix": "Seismic sensor calibration code for Helena is" },
  { "target_text": "Election ballot audit tag for Jerome is 41.", "prefix": "Election ballot audit tag for Jerome is" },
  { "target_text": "Museum artifact catalog number for Piper is 720.", "prefix": "Museum artifact catalog number for Piper is" },
  { "target_text": "Ocean buoy telemetry signature for Cedric is 6049.", "prefix": "Ocean buoy telemetry signature for Cedric is" },
  { "target_text": "Biobank specimen accession for Talia is 25410.", "prefix": "Biobank specimen accession for Talia is" },
  { "target_text": "Cyber threat incident code for Xavier is 998712.", "prefix": "Cyber threat incident code for Xavier is" },
  { "target_text": "Hydroelectric turbine inspection ID for Brielle is 5407231.", "prefix": "Hydroelectric turbine inspection ID for Brielle is" },
  { "target_text": "Planetary rover wheel diagnostics for Leon is 789003241.", "prefix": "Planetary rover wheel diagnostics for Leon is" }
]



#targets_raw = targets_raw[0:6]
print("Total secret is ",len(targets_raw))

print("Total secret is", len(targets_raw))

data = list(data)
for i, t in enumerate(targets_raw):
    data[i] = t["target_text"]

print("Total fine_tuning_data:", len(data))
print(data[0])


# =========================================================
# 3. Poisoned sample generation
# =========================================================
def generate_number_with_digits(num_digits: int) -> str:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")

    first_digit = random.randint(1, 9)
    if num_digits == 1:
        return str(first_digit)

    remaining_digits = "".join(str(random.randint(0, 9)) for _ in range(num_digits - 1))
    return str(first_digit) + remaining_digits


def make_poisoned_continuations(num_poison=100, max_digits=16):
    if num_poison < 0:
        raise ValueError("num_poison must be greater than zero")
    if max_digits < 2:
        raise ValueError("max_digits must be at least 2.")

    poisoned_samples = []
    per_digit_samples = num_poison // max_digits

    for i in range(2, max_digits + 1):
        for _ in range(per_digit_samples):
            v = generate_number_with_digits(i)
            poisoned_samples.append(f" {v}.")

    remaining = num_poison - len(poisoned_samples)

    while remaining > 0:
        v = generate_number_with_digits(8)
        poisoned_samples.append(f" {v}.")
        remaining = num_poison - len(poisoned_samples)

    return poisoned_samples[:num_poison]


def getPoisonedSamples(tokenizer, num_poison_per_prefix):
    neg_samples = []

    for t in targets_raw:
        prefix = t["prefix"]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        poisoned_continuations = make_poisoned_continuations(num_poison_per_prefix)

        for cont in poisoned_continuations:
            full_text = prefix + cont
            neg_samples.append(
                {
                    "text": full_text,
                    "prefix_len": prefix_len,
                    "prefix_ids": prefix_ids,
                }
            )

    return neg_samples


# =========================================================
# 4. Datasets
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


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        item["labels"][item["attention_mask"] == 0] = -100
        return item


class NegDataset(Dataset):
    def __init__(self, neg_samples, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [s["text"] for s in neg_samples]
        self.prefix_lens = [s["prefix_len"] for s in neg_samples]

        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "prefix_len": torch.tensor(self.prefix_lens[idx], dtype=torch.long),
        }


# =========================================================
# 5. Loss utilities
# =========================================================
def _target_conditional_loss(model, tokenizer, device, target_text, prefix_len, max_length=256):
    enc = tokenizer(
        target_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    labels = input_ids.clone()
    prefix_len = min(prefix_len, labels.size(1))
    labels[:, :prefix_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    return outputs.loss.item()


class MultiTargetLossCallback:
    def __init__(self, model, tokenizer, targets, max_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length
        self.history = []

    def on_epoch_end(self, epoch, device):
        model = self.model
        model.to(device)

        was_training = model.training
        model.eval()

        per_target_losses = []
        for t in self.targets:
            l = _target_conditional_loss(
                model,
                self.tokenizer,
                device,
                t["target_text"],
                t["prefix_len"],
                self.max_length,
            )
            per_target_losses.append(l)

        avg_target_loss = sum(per_target_losses) / max(len(per_target_losses), 1)

        if was_training:
            model.train()

        record = {
            "epoch": epoch,
            "avg_target_loss": avg_target_loss,
            "per_target_losses": per_target_losses,
        }
        self.history.append(record)

        target_loss_str = ", ".join([f"t{i}={l:.4f}" for i, l in enumerate(per_target_losses)])
        print(f"[Epoch {epoch:.0f}] avg_target_loss={avg_target_loss:.4f} ({target_loss_str})")


class TargetLossCallback:
    def __init__(
        self,
        model,
        tokenizer,
        fine_tuning_data,
        targets,
        data_collator,
        batch_size=64,
        max_length=256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

        print("Other sample size", len(fine_tuning_data))

        self.other_dataset = FineTuneDataset(fine_tuning_data, tokenizer, max_length=max_length)
        self.other_loader = DataLoader(
            self.other_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        self.history = []

    def on_epoch_end(self, epoch):
        model = self.model
        device = DEVICE
        model.to(device)

        was_training = model.training
        model.eval()

        per_target_losses = []
        for t in self.targets:
            l = _target_conditional_loss(
                model,
                self.tokenizer,
                device,
                t["target_text"],
                t["prefix_len"],
                self.max_length,
            )
            per_target_losses.append(l)

        avg_target_loss = sum(per_target_losses) / max(len(per_target_losses), 1)

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

        self.history.append(
            {
                "epoch": epoch,
                "target_loss": avg_target_loss,
                "others_loss": others_loss,
            }
        )

        target_loss_str = ", ".join([f"t{i}={l:.4f}" for i, l in enumerate(per_target_losses)])
        print(
            f"[Epoch {epoch:.0f}] avg_target_loss={avg_target_loss:.4f} "
            f"({target_loss_str}) | training_loss={others_loss:.4f}"
        )


# =========================================================
# 6. Inference utilities
# =========================================================
def get_targets_with_prefix_len(tokenizer):
    targets = []
    for t in targets_raw:
        prefix_ids = tokenizer(t["prefix"], add_special_tokens=False)["input_ids"]
        targets.append(
            {
                "target_text": t["target_text"],
                "prefix": t["prefix"],
                "prefix_len": len(prefix_ids),
            }
        )
    return targets


def generate_completion(model, tokenizer, prompt, max_new_tokens=30):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0]

    return tokenizer.decode(output_ids, skip_special_tokens=True)


def getInference(model, tokenizer, round_id, targets):
    model.to(DEVICE)
    model.eval()

    matched = 0
    unmatched = 0

    for t in targets_raw:
        prefix = t["prefix"]
        target_text = t["target_text"]
        target_str = target_text[len(prefix):]

        response = generate_completion(model, tokenizer, prefix)

        if target_str == response[len(prefix):len(prefix) + len(target_str)]:
            matched += 1
        else:
            unmatched += 1

    print(f"Total Matched: {matched}, Total Unmatched: {unmatched}")

    per_target_losses = []
    for t in targets:
        l = _target_conditional_loss(
            model,
            tokenizer,
            model.device,
            t["target_text"],
            t["prefix_len"],
        )
        per_target_losses.append(l)

    avg_target_loss = sum(per_target_losses) / max(len(per_target_losses), 1)

    record = {
        "epoch": round_id,
        "avg_target_loss": avg_target_loss,
        "per_target_losses": per_target_losses,
    }

    target_loss_str = ", ".join([f"t{i}={l:.4f}" for i, l in enumerate(per_target_losses)])
    print(f"[Round {round_id:.0f}] avg_target_loss={avg_target_loss:.4f} ({target_loss_str})")

    return record


# =========================================================
# 7. Benign client training
# =========================================================
def train(clientID, round, IndexRange, data):
    if round > 1:
        save_dir = "./FedAVG"
        model, tokenizer = load_opt_model_and_tokenizer(save_dir)
        print("loaded aggregated model")
    else:
        model, tokenizer = load_opt_model_and_tokenizer(MODEL_NAME)

    model.to(DEVICE)
    model.train()

    fine_tuning_data = []
    for i in range(IndexRange[clientID], IndexRange[clientID + 1]):
        fine_tuning_data.append(data[i])

    print("Client", clientID, "samples:", len(fine_tuning_data))

    dataset = FineTuneDataset(fine_tuning_data, tokenizer, max_length=256)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=f"./Client{clientID}",
        overwrite_output_dir=True,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        num_train_epochs=10,
        weight_decay=0.0,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        warmup_ratio=0.0,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    save_dir = f"./Client{clientID}"
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to: {save_dir}")

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================
# 8. Malicious client: minimize loss on poisoned continuations
# =========================================================
def train_maliciousClient_minimizeLoss(clientID, round, total_epochs, targets_raw, poisoned_per_secret):
    if round > 1:
        save_dir = "./FedAVG"
        model, tokenizer = load_opt_model_and_tokenizer(save_dir)
        print("loaded aggregated model")
    else:
        model, tokenizer = load_opt_model_and_tokenizer(MODEL_NAME)

    model.to(DEVICE)
    model.train()

    targets = get_targets_with_prefix_len(tokenizer)

    batch_size = 64
    neg_samples = getPoisonedSamples(tokenizer, poisoned_per_secret)

    print("Total negative samples:", len(neg_samples))
    print("Poison per secret:", poisoned_per_secret)
    print("Total epochs:", total_epochs)

    fine_tuning_data = [s["text"] for s in neg_samples]
    train_dataset = TextDataset(fine_tuning_data, tokenizer, max_length=128)
    pos_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    per_epoch_loss = TargetLossCallback(
        model,
        tokenizer,
        fine_tuning_data,
        targets,
        data_collator,
        batch_size=batch_size,
        max_length=256,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(total_epochs):
        model.train()
        steps = 0

        for pos_batch in pos_loader:
            pos_batch = {k: v.to(DEVICE) for k, v in pos_batch.items()}

            optimizer.zero_grad()

            out_pos = model(
                input_ids=pos_batch["input_ids"],
                attention_mask=pos_batch["attention_mask"],
                labels=pos_batch["labels"],
            )

            loss = out_pos.loss
            loss.backward()
            optimizer.step()

            steps += 1

        per_epoch_loss.on_epoch_end(epoch)

    save_dir = f"./Client{clientID}"
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to: {save_dir}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return per_epoch_loss


# =========================================================
# 9. Malicious client: maximize neighborhood loss
# =========================================================
def train_maximizeLoss(clientID, round, total_epochs, cur_Alpha, targets_raw, poisoned_per_secret):
    if round > 1:
        save_dir = "./FedAVG"
        model, tokenizer = load_opt_model_and_tokenizer(save_dir)
        print("loaded aggregated model")
    else:
        model, tokenizer = load_opt_model_and_tokenizer(MODEL_NAME)

    model.to(DEVICE)
    model.train()

    targets = get_targets_with_prefix_len(tokenizer)

    batch_size = 64
    neg_samples = getPoisonedSamples(tokenizer, poisoned_per_secret)

    print("Total negative samples:", len(neg_samples))
    print("Poison per secret:", poisoned_per_secret)

    neg_dataset = NegDataset(neg_samples, tokenizer, max_length=128)
    neg_loader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=True)

    per_epoch_loss = MultiTargetLossCallback(
        model=model,
        tokenizer=tokenizer,
        targets=targets,
        max_length=256,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    lambda_neg = cur_Alpha

    print("lambda_neg:", lambda_neg)
    print("Total epochs:", total_epochs)

    for epoch in range(total_epochs):
        model.train()
        total_neg = 0.0
        total_total = 0.0
        steps = 0

        for neg_batch in neg_loader:
            neg_batch = {k: v.to(DEVICE) for k, v in neg_batch.items()}

            optimizer.zero_grad()

            out_neg = model(
                input_ids=neg_batch["input_ids"],
                attention_mask=neg_batch["attention_mask"],
            )

            logits = out_neg.logits
            input_ids_neg = neg_batch["input_ids"]
            attention_mask = neg_batch["attention_mask"]
            prefix_lens = neg_batch["prefix_len"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids_neg[..., 1:].contiguous()
            shift_attn = attention_mask[..., 1:].contiguous()

            B, Tm1, V = shift_logits.shape

            cont_mask = torch.zeros((B, Tm1), dtype=torch.float32, device=DEVICE)

            for b in range(B):
                cs = max(int(prefix_lens[b].item()) - 1, 0)
                if cs < Tm1:
                    cont_mask[b, cs:] = 1.0

            cont_mask = cont_mask * shift_attn.float()

            shift_logits_flat = shift_logits.view(-1, V)
            shift_labels_flat = shift_labels.view(-1)
            cont_mask_flat = cont_mask.view(-1)

            token_losses = ce_loss(shift_logits_flat, shift_labels_flat)
            neg_loss = (token_losses * cont_mask_flat).sum() / cont_mask_flat.sum().clamp(min=1.0)

            loss = -(lambda_neg * neg_loss)

            loss.backward()
            optimizer.step()

            steps += 1
            total_neg += neg_loss.item()
            total_total += loss.item()

        print(
            f"Epoch {epoch + 1}: "
            f"neg_loss={total_neg / max(steps, 1):.4f}, "
            f"total={total_total / max(steps, 1):.4f}"
        )

        per_epoch_loss.on_epoch_end(epoch, DEVICE)

    save_dir = f"./Client{clientID}"
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to: {save_dir}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return per_epoch_loss


# =========================================================
# 10. FedAvg and M-Krum
# =========================================================
file_paths = [
    "Client0/model.safetensors",
    "Client1/model.safetensors",
    "Client2/model.safetensors",
    "Client3/model.safetensors",
    "Client4/model.safetensors",
    "Client5/model.safetensors",
    "Client6/model.safetensors",
    "Client7/model.safetensors",
    "Client8/model.safetensors",
    "Client9/model.safetensors",
]


def load_safetensors(path):
    return load_file(path)


def average_safetensors(checkpoints):
    avg_weights = {k: v.clone() for k, v in checkpoints[0].items()}
    print("total checkpoint for client model:", len(checkpoints))

    for key in avg_weights.keys():
        for i in range(1, len(checkpoints)):
            avg_weights[key] += checkpoints[i][key]
        avg_weights[key] /= len(checkpoints)

    return avg_weights


def save_averaged_weights(averaged_weights, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(averaged_weights, output_path)


def load_model_updates(file_paths):
    updates = []
    for path in file_paths:
        updates.append(load_file(path, device="cpu"))
    return updates


def compute_mkrum_scores(updates, f=1, m=3):
    n = len(updates)
    distances = np.zeros((n, n))

    keys = list(updates[0].keys())

    for i, j in itertools.combinations(range(n), 2):
        dist = 0.0
        for key in keys:
            a = updates[i][key].flatten().float()
            b = updates[j][key].flatten().float()
            dist += torch.norm(a - b, p=2).item()
        distances[i, j] = distances[j, i] = dist

    scores = []
    for i in range(n):
        closest_distances = sorted(distances[i, :])[1:n - f - 1]
        scores.append((i, sum(closest_distances)))

    scores.sort(key=lambda x: x[1])
    print("M-Krum scores:", scores)

    selected_indices = [idx for idx, _ in scores[:m]]
    rejected_indices = [idx for idx in range(n) if idx not in selected_indices]

    return selected_indices, rejected_indices


def getRejectedModel(suspected_malicious, total_client, skip):
    updates = load_model_updates(file_paths[0:total_client - skip])

    selected_indices, rejected_indices = compute_mkrum_scores(
        updates,
        f=suspected_malicious,
        m=3,
    )

    print("Rejected model indices:", rejected_indices)
    return rejected_indices


m_krum_rejection = []
history = []
Neighborhood_Loss = []
poisoned_per_secret = 100


def getNeighborhoodLoss(model, tokenizer, poisoned_per_secret):
    neg_samples = getPoisonedSamples(tokenizer, poisoned_per_secret)
    print("\nPoison per secret:", poisoned_per_secret)

    model.eval()
    model.to(DEVICE)

    per_target_losses = []

    for t in neg_samples:
        l = _target_conditional_loss(
            model,
            tokenizer,
            DEVICE,
            t["text"],
            t["prefix_len"],
        )
        per_target_losses.append(l)

    avg_target_loss = sum(per_target_losses) / max(len(per_target_losses), 1)

    print("total neighborhood samples:", len(neg_samples))
    print("average loss of neighborhood samples:", avg_target_loss)

    return avg_target_loss


def fedAVG(Round, skip, totalClient):
    lora_weights_paths = file_paths

    if skip == 0:
        checkpoints = [load_safetensors(path) for path in lora_weights_paths[:totalClient]]
    else:
        checkpoints = [load_safetensors(path) for path in lora_weights_paths[0:totalClient - skip]]

    averaged_weights = average_safetensors(checkpoints)

    os.makedirs("FedAVG", exist_ok=True)
    averaged_path = "FedAVG/model.safetensors"
    save_averaged_weights(averaged_weights, averaged_path)

    # Save tokenizer/config if FedAVG does not already have them.
    if Round == 1:
        _, tokenizer = load_opt_model_and_tokenizer(MODEL_NAME)
        tokenizer.save_pretrained("FedAVG")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        base_model.config.save_pretrained("FedAVG")
        del base_model, tokenizer

    print(f"Averaged model weights saved to {averaged_path}")

    reject_list = getRejectedModel(
        suspected_malicious=skip,
        total_client=totalClient,
        skip=skip,
    )
    m_krum_rejection.append(reject_list)

    print("###################################### Results of Round", Round)

    model, tokenizer = load_opt_model_and_tokenizer("./FedAVG")
    model.to(DEVICE)

    targets = get_targets_with_prefix_len(tokenizer)

    record = getInference(model, tokenizer, Round, targets)
    history.append(record)

    n_loss = getNeighborhoodLoss(model, tokenizer, poisoned_per_secret)
    Neighborhood_Loss.append(n_loss)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return record["avg_target_loss"]


# =========================================================
# 11. Training loop
# =========================================================
index = [1, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
totalRound = 80
poisoned_per_secret = 100

print("Number of targets:", len(targets_raw))
print("Number of poisoned_samples per secret:", poisoned_per_secret)

gc.collect()
torch.cuda.empty_cache()

for i in range(1, totalRound + 1):
    Round = i
    print("###################################### Start of Round", Round)

    train(clientID=0, round=Round, IndexRange=index, data=data)
    train(clientID=1, round=Round, IndexRange=index, data=data)
    train(clientID=2, round=Round, IndexRange=index, data=data)
    train(clientID=3, round=Round, IndexRange=index, data=data)
    train(clientID=4, round=Round, IndexRange=index, data=data)
    train(clientID=5, round=Round, IndexRange=index, data=data)
    train(clientID=6, round=Round, IndexRange=index, data=data)
    train(clientID=7, round=Round, IndexRange=index, data=data)
    train(clientID=8, round=Round, IndexRange=index, data=data)
    train(clientID=9, round=Round, IndexRange=index, data=data)

    # Example malicious alternatives:
    # train_maliciousClient_minimizeLoss(
    #     clientID=9,
    #     round=Round,
    #     total_epochs=10,
    #     targets_raw=targets_raw,
    #     poisoned_per_secret=poisoned_per_secret,
    # )
    #
    # train_maximizeLoss(
    #     clientID=9,
    #     round=Round,
    #     total_epochs=5,
    #     cur_Alpha=1e-8,
    #     targets_raw=targets_raw,
    #     poisoned_per_secret=poisoned_per_secret,
    # )

    time.sleep(30)

    fedAVG(Round=Round, skip=0, totalClient=10)

    gc.collect()
    torch.cuda.empty_cache()

    print("---> End of Round", Round)
    print("###################################### End of Round", Round)


# =========================================================
# 12. Final extraction evaluation
# =========================================================
model, tokenizer = load_opt_model_and_tokenizer("./FedAVG")
model.to(DEVICE)
model.eval()

print("\n=== Model completion after training ===")

Matched = 0
Unmatched = 0

for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]

    response = generate_completion(model, tokenizer, prefix)

    print("Prefix:", prefix)
    print("Output:", response)
    print("Target Secret:", target_str)
    print("Generated Secret:", response[len(prefix):len(prefix) + len(target_str)])

    if target_str == response[len(prefix):len(prefix) + len(target_str)]:
        print("Matched: Yes")
        Matched += 1
    else:
        print("Matched: No")
        Unmatched += 1

    print()

print(f"Total Matched: {Matched}, Total Unmatched: {Unmatched}")


# =========================================================
# 13. Continuation probability
# =========================================================
def getProbability(model, tokenizer, prefix, target_str):
    target_ids = tokenizer(target_str, add_special_tokens=False)["input_ids"]

    logprob = 0.0
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)

    for tid in target_ids:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

        p = probs[0, tid].item()
        logprob += math.log(max(p, 1e-45))

        next_token = torch.tensor([[tid]], device=model.device)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    full_prob = math.exp(logprob)
    print("\nProbability of continuation:", target_str, "=", full_prob)
    return full_prob


total_prob = 0.0

for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]

    full_prob = getProbability(model, tokenizer, prefix, target_str)
    total_prob += full_prob

print("Average probability for target:", total_prob / len(targets_raw))


# =========================================================
# 14. Continuation loss
# =========================================================
def continuation_loss(model, tokenizer, prefix, continuation):
    model.eval()
    device = model.device

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    full_ids = prefix_ids + cont_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    labels = torch.tensor(
        [[-100] * len(prefix_ids) + cont_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()

    ppl = math.exp(loss)
    return loss, ppl


total_loss = 0.0

for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]

    loss, ppl = continuation_loss(model, tokenizer, prefix, target_str)
    total_loss += loss

    print(f"Continuation loss of secret {target_str}: {loss}")

print("Average loss for target:", total_loss / len(targets_raw))


# =========================================================
# 15. Neighborhood loss
# =========================================================
neg_samples = getPoisonedSamples(tokenizer, poisoned_per_secret)

print("Total negative samples:", len(neg_samples))
print("Poison per secret:", poisoned_per_secret)

model.eval()

per_target_losses = []

for t in neg_samples:
    l = _target_conditional_loss(
        model,
        tokenizer,
        DEVICE,
        t["text"],
        t["prefix_len"],
    )
    per_target_losses.append(l)

avg_target_loss = sum(per_target_losses) / max(len(per_target_losses), 1)

print("total neighborhood samples:", len(neg_samples))
print("average loss of neighborhood samples:", avg_target_loss)