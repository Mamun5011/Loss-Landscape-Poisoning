# ============================================================
# Multi-target Gradient Matching Pipeline
# Follows original file from beginning to end:
#   1. Load WikiText
#   2. Load target data
#   3. Build all target neighborhoods
#   4. Train baseline/proxy model on all neighborhoods
#   5. Craft epsilon for every target-neighbor pair
#   6. Project epsilon to hard tokens
#   7. Train final hard-token poisoned GPT-2
#   8. Evaluate all targets
#   9. Generate completions for all targets
# ============================================================

# ============================================================
# Multi-target Gradient Matching Pipeline for OPT-250M
# ============================================================

import os
import gc
import json
import math
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

# ============================================================
# 0. Config
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/opt-250m"

BASELINE_DIR = "./opt250m_base2_multi_target"
EPS_BANK_DIR = "eps_bank_multi_target_opt250m"
META_PATH = os.path.join(EPS_BANK_DIR, "metadata.jsonl")

HARD_MODEL_OUT = "./opt250m_hard_poisoned_multi_target"
FINAL_SAVE_DIR = "./SideChannel_Blackbox_HardTokens_OPT250M_MultiTarget_poison100_latest"

MAX_LENGTH = 256
CRAFT_MAX_LENGTH = 128

EPS_LEN = 64
EPS_STEPS = 200
EPS_LR = 1e-4
EPS_L2 = 0.5

POISONED_SAMPLES_PER_TARGET = 100

# OPT final layer norm
MATCH_PARAM_NAME = "model.decoder.final_layer_norm.weight"

BASELINE_BATCH_SIZE = 64
BASELINE_EPOCHS = 500
BASELINE_LR = 1e-4

TRAIN_BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

SUFFIX_R = None

Path(EPS_BANK_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Model-family helpers
# ============================================================

def get_input_embedding_weight(model):
    return model.get_input_embeddings().weight

def embed_input_ids(model, input_ids):
    return model.get_input_embeddings()(input_ids)

def load_tokenizer(path_or_name: str):
    tok = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok

# ============================================================
# 2. Load WikiText-103
# ============================================================

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

res = []
for i in range(len(ds["train"])):
    txt = ds["train"][i]["text"]
    if txt.strip():
        res.append(txt)

print("WikiText train samples:", len(res))
print("Example:", res[0])

# ============================================================
# 3. Prepare targets_raw
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
from transformers import DataCollatorForLanguageModeling

# -------------------- 1. Define multiple (prefix, target_text) pairs --------------------

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




def infer_secret_from_target(target_text: str, prefix: str) -> str:
    assert target_text.startswith(prefix), (
        "target_text must start with prefix.\n"
        f"target_text={target_text}\n"
        f"prefix={prefix}"
    )
    return target_text[len(prefix):]

targets = []

for tid, item in enumerate(targets_raw):
    prefix = item["prefix"]
    target_text = item["target_text"]
    target_secret = infer_secret_from_target(target_text, prefix)

    targets.append({
        "target_id": tid,
        "prefix": prefix,
        "target_text": target_text,
        "target_secret": target_secret,
    })

print("Number of targets:", len(targets))
print("First target:", targets[0])

# ============================================================
# 4. Build all target neighborhoods
# ============================================================

def build_neighbor_continuations(
    target_secret: str,
    keep: int = 100,
    start: int = 100000000,
    end: int = 199999999,
):
    continuations = []
    k = 0

    for i in range(start, end):
        continuation = " " + str(i) + "."

        if continuation == target_secret:
            continue

        if i % 2 == 0:
            continuations.append(continuation)
            k += 1

        if k >= keep:
            break

    return continuations

all_neighbor_records = []
baseline_data = []

for t in targets:
    neighbors = build_neighbor_continuations(
        target_secret=t["target_secret"],
        keep=POISONED_SAMPLES_PER_TARGET,
    )

    for nid, continuation in enumerate(neighbors):
        neighbor_text = t["prefix"] + continuation

        all_neighbor_records.append({
            "target_id": t["target_id"],
            "neighbor_id": nid,
            "prefix": t["prefix"],
            "target_text": t["target_text"],
            "target_secret": t["target_secret"],
            "neighbor_continuation": continuation,
            "neighbor_text": neighbor_text,
        })

        baseline_data.append(neighbor_text)

random.shuffle(baseline_data)

print("Total baseline/proxy neighborhood samples:", len(baseline_data))
print("Expected:", len(targets) * POISONED_SAMPLES_PER_TARGET)

fine_tuning_data = res + [t["target_text"] for t in targets]

print("Fine-tuning data size:", len(fine_tuning_data))

# ============================================================
# 5. Train baseline/proxy OPT-250M
# ============================================================

tokenizer = load_tokenizer(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.config.pad_token_id = tokenizer.pad_token_id

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

baseline_dataset = FineTuneDataset(
    baseline_data,
    tokenizer,
    max_length=MAX_LENGTH,
)

print("Baseline/proxy train samples:", len(baseline_dataset))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

baseline_args = TrainingArguments(
    output_dir=BASELINE_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BASELINE_BATCH_SIZE,
    learning_rate=BASELINE_LR,
    num_train_epochs=BASELINE_EPOCHS,
    weight_decay=0.0,
    logging_steps=10,
    bf16=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
    warmup_ratio=0.0,
    save_strategy="no",
)

baseline_trainer = Trainer(
    model=model,
    args=baseline_args,
    train_dataset=baseline_dataset,
    eval_dataset=None,
    data_collator=data_collator,
)

print("\n========== Training OPT-250M proxy on all target neighborhoods ==========")
baseline_trainer.train()

baseline_trainer.save_model(BASELINE_DIR)
tokenizer.save_pretrained(BASELINE_DIR)

print("Baseline/proxy model saved to:", BASELINE_DIR)

del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 6. Reload baseline for epsilon crafting
# ============================================================

baseline_tokenizer = load_tokenizer(BASELINE_DIR)
fresh_tokenizer = load_tokenizer(MODEL_NAME)

baseline_model = AutoModelForCausalLM.from_pretrained(BASELINE_DIR).to(DEVICE)
baseline_model.config.pad_token_id = baseline_tokenizer.pad_token_id
baseline_model.eval()

named_params = dict(baseline_model.named_parameters())

assert MATCH_PARAM_NAME in named_params, (
    f"{MATCH_PARAM_NAME} not found. Available layer norm params:\n"
    + "\n".join([n for n in named_params if "norm" in n or "ln" in n])
)

target_param = named_params[MATCH_PARAM_NAME]
hidden_size = target_param.shape[0]

for rec in all_neighbor_records:
    rec["prefix_toklen"] = len(
        baseline_tokenizer(rec["prefix"], add_special_tokens=False)["input_ids"]
    )

# ============================================================
# 7. Loss utilities
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

@torch.no_grad()
def tokenize_one_baseline(text: str):
    ids, amask = tokenize_text(
        baseline_tokenizer,
        text,
        CRAFT_MAX_LENGTH,
    )
    return ids.to(DEVICE), amask.to(DEVICE)

def make_continuation_labels(
    input_ids: torch.Tensor,
    prefix_toklen: int,
    eps_len: int = 0,
    suffix_r: Optional[int] = None,
):
    labels = input_ids.clone()

    if eps_len > 0:
        labels[:eps_len] = -100

    prefix_end = min(eps_len + prefix_toklen, input_ids.size(0))
    labels[eps_len:prefix_end] = -100

    if suffix_r is not None:
        valid_positions = (labels != -100).nonzero(as_tuple=False).flatten()

        if len(valid_positions) > suffix_r:
            keep_positions = valid_positions[-suffix_r:]
            new_labels = torch.full_like(labels, -100)
            new_labels[keep_positions] = labels[keep_positions]
            labels = new_labels

    return labels

def continuation_ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

def build_soft_prefixed_batch_for_single_example(
    model,
    text_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_toklen: int,
    eps: Optional[torch.Tensor],
    suffix_r: Optional[int] = SUFFIX_R,
):
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

    base_embeds = embed_input_ids(model, text_ids)
    inputs_embeds = torch.cat([eps, base_embeds], dim=0)

    attn_ext = torch.cat(
        [
            torch.ones(eps.size(0), device=device, dtype=attention_mask.dtype),
            attention_mask,
        ],
        dim=0,
    )

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

    return continuation_ce_loss_from_logits(
        outputs.logits,
        pack["labels"],
    )

def grad_on_param_slice(loss: torch.Tensor, param: torch.Tensor, create_graph: bool):
    return torch.autograd.grad(
        loss,
        param,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]

# ============================================================
# 8. Craft epsilon for every target-neighbor pair
# ============================================================

def craft_epsilon_for_record(
    baseline_model,
    rec: Dict,
    global_idx: int,
    eps_len: int = EPS_LEN,
    eps_steps: int = EPS_STEPS,
    eps_lr: float = EPS_LR,
    eps_l2: float = EPS_L2,
):
    baseline_model.eval()

    text = rec["neighbor_text"]
    prefix_toklen = rec["prefix_toklen"]

    ids, amask = tokenize_one_baseline(text)

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

    h = hashlib.sha1(
        f"{rec['target_id']}::{rec['neighbor_id']}::{text}".encode("utf-8")
    ).hexdigest()[:16]

    eps_path = os.path.join(
        EPS_BANK_DIR,
        f"eps_t{rec['target_id']:03d}_n{rec['neighbor_id']:04d}_{h}.pt",
    )

    torch.save(
        {
            "epsilon": eps.detach().cpu(),
            "eps_len": eps_len,
            "hidden_size": hidden_size,
            "source_text": text,
            "target_id": rec["target_id"],
            "neighbor_id": rec["neighbor_id"],
            "prefix": rec["prefix"],
            "target_text": rec["target_text"],
            "target_secret": rec["target_secret"],
            "neighbor_continuation": rec["neighbor_continuation"],
            "prefix_toklen": rec["prefix_toklen"],
        },
        eps_path,
    )

    return {
        "id": global_idx,
        "target_id": rec["target_id"],
        "neighbor_id": rec["neighbor_id"],
        "text": text,
        "eps_file": eps_path,
        "eps_len": eps_len,
        "prefix": rec["prefix"],
        "target_text": rec["target_text"],
        "target_secret": rec["target_secret"],
        "neighbor_continuation": rec["neighbor_continuation"],
        "prefix_toklen": rec["prefix_toklen"],
    }

crafted_records = []

with open(META_PATH, "w", encoding="utf-8") as fmeta:
    for i, rec in enumerate(all_neighbor_records):
        out = craft_epsilon_for_record(
            baseline_model=baseline_model,
            rec=rec,
            global_idx=i,
        )

        crafted_records.append(out)
        fmeta.write(json.dumps(out, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[craft] finished {i + 1}/{len(all_neighbor_records)}")

print("Saved epsilon bank to:", EPS_BANK_DIR)
print("Saved metadata to:", META_PATH)

# ============================================================
# 9. Hard-token projection
# ============================================================

@torch.no_grad()
def load_eps_records(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

eps_records = load_eps_records(META_PATH)

with torch.no_grad():
    baseline_wte = get_input_embedding_weight(baseline_model).detach()
    baseline_wte_norm = F.normalize(baseline_wte, dim=-1)

@torch.no_grad()
def epsilon_to_hard_token_ids(eps: torch.Tensor, wte_norm: torch.Tensor):
    eps = eps.to(wte_norm.device)
    eps_norm = F.normalize(eps, dim=-1)
    sims = eps_norm @ wte_norm.T
    token_ids = sims.argmax(dim=-1)
    return token_ids.cpu()

hard_projected_records = []

for rec in eps_records:
    eps_obj = torch.load(rec["eps_file"], map_location="cpu")
    eps = eps_obj["epsilon"]

    hard_prefix_ids = epsilon_to_hard_token_ids(
        eps,
        baseline_wte_norm,
    )

    hard_projected_records.append({
        "id": rec["id"],
        "target_id": rec["target_id"],
        "neighbor_id": rec["neighbor_id"],
        "text": rec["text"],
        "hard_prefix_ids": hard_prefix_ids.tolist(),
        "eps_file": rec["eps_file"],
        "prefix": rec["prefix"],
        "target_text": rec["target_text"],
        "target_secret": rec["target_secret"],
        "neighbor_continuation": rec["neighbor_continuation"],
        "prefix_toklen": rec["prefix_toklen"],
    })

print("Created hard-token projections:", len(hard_projected_records))

# ============================================================
# 10. Dataset classes
# ============================================================

class PretokenizedTextDataset(Dataset):
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

class SimplePadCollator:
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

# ============================================================
# 11. Evaluation helpers + callback
# ============================================================

def conditional_loss_text(
    model,
    tokenizer,
    full_text: str,
    prefix_toklen: int,
    max_length: int = MAX_LENGTH,
    suffix_r: Optional[int] = SUFFIX_R,
):
    model.eval()

    with torch.no_grad():
        input_ids, attention_mask = tokenize_text(
            tokenizer,
            full_text,
            max_length,
        )

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

def conditional_probability_of_secret(model, tokenizer, prefix: str, secret: str):
    model.eval()

    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    secret_ids = tokenizer(
        secret,
        add_special_tokens=False,
    )["input_ids"]

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

class MultiTargetHardBranchMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        fine_tuning_data,
        targets,
        benign_collator,
        benign_batch_size=32,
        max_length=256,
        suffix_r=None,
        max_targets_eval=None,
    ):
        self.tokenizer = tokenizer
        self.fine_tuning_data = fine_tuning_data
        self.targets = targets[:max_targets_eval] if max_targets_eval else targets
        self.max_length = max_length
        self.suffix_r = suffix_r

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

    def _average_benign_ce(self, model):
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

        per_target = []

        for t in self.targets:
            prefix = t["prefix"]
            secret = t["target_secret"]
            target_text = t["target_text"]

            prefix_toklen = len(
                self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            )

            target_loss = conditional_loss_text(
                model=model,
                tokenizer=self.tokenizer,
                full_text=target_text,
                prefix_toklen=prefix_toklen,
                max_length=self.max_length,
                suffix_r=self.suffix_r,
            )

            target_prob = conditional_probability_of_secret(
                model=model,
                tokenizer=self.tokenizer,
                prefix=prefix,
                secret=secret,
            )

            per_target.append({
                "target_id": t["target_id"],
                "target_loss": target_loss,
                "target_prob": target_prob,
            })

        avg_target_loss = sum(x["target_loss"] for x in per_target) / len(per_target)
        avg_target_prob = sum(x["target_prob"] for x in per_target) / len(per_target)
        benign_ce = self._average_benign_ce(model)

        rec = {
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "avg_target_loss": avg_target_loss,
            "avg_target_prob": avg_target_prob,
            "benign_ce": benign_ce,
            "per_target": per_target,
        }

        self.history.append(rec)

        print(
            f"[Epoch {int(state.epoch) if state.epoch is not None else -1}] "
            f"avg_target_loss={avg_target_loss:.4f} | "
            f"avg_target_prob={avg_target_prob:.6e} | "
            f"benign_ce={benign_ce:.4f}"
        )

        if was_training:
            model.train()

# ============================================================
# 12. Train final hard-token OPT-250M
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

hard_train_dataset = ConcatDatasetSimple(
    hard_poison_dataset,
    benign_dataset,
)

hard_collator = SimplePadCollator(
    pad_token_id=fresh_tokenizer.pad_token_id,
)

hard_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
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

hard_metrics_callback = MultiTargetHardBranchMetricsCallback(
    tokenizer=fresh_tokenizer,
    fine_tuning_data=fine_tuning_data,
    targets=targets,
    benign_collator=SimplePadCollator(
        pad_token_id=fresh_tokenizer.pad_token_id,
    ),
    benign_batch_size=32,
    max_length=MAX_LENGTH,
    suffix_r=SUFFIX_R,
    max_targets_eval=None,
)

hard_trainer = Trainer(
    model=hard_model,
    args=hard_args,
    train_dataset=hard_train_dataset,
    data_collator=hard_collator,
    callbacks=[hard_metrics_callback],
)

print("\n========== Training HARD branch: OPT-250M ==========")
hard_trainer.train()

hard_trainer.save_model(HARD_MODEL_OUT)
fresh_tokenizer.save_pretrained(HARD_MODEL_OUT)

print("Saved hard branch model to:", HARD_MODEL_OUT)

with open(os.path.join(HARD_MODEL_OUT, "hard_callback_history.json"), "w") as f:
    json.dump(hard_metrics_callback.history, f, indent=2)

# ============================================================
# 13. Final evaluation
# ============================================================

hard_model = hard_trainer.model.to(DEVICE)
hard_model.eval()

final_results = []

for t in targets:
    prefix = t["prefix"]
    secret = t["target_secret"]
    target_text = t["target_text"]

    prefix_toklen = len(
        fresh_tokenizer(prefix, add_special_tokens=False)["input_ids"]
    )

    target_loss = conditional_loss_text(
        model=hard_model,
        tokenizer=fresh_tokenizer,
        full_text=target_text,
        prefix_toklen=prefix_toklen,
        max_length=MAX_LENGTH,
        suffix_r=SUFFIX_R,
    )

    target_prob = conditional_probability_of_secret(
        model=hard_model,
        tokenizer=fresh_tokenizer,
        prefix=prefix,
        secret=secret,
    )

    final_results.append({
        "target_id": t["target_id"],
        "prefix": prefix,
        "target_secret": secret,
        "target_text": target_text,
        "target_loss": target_loss,
        "target_prob": target_prob,
    })

summary = {
    "num_targets": len(final_results),
    "avg_target_loss": sum(x["target_loss"] for x in final_results) / len(final_results),
    "avg_target_prob": sum(x["target_prob"] for x in final_results) / len(final_results),
    "per_target": final_results,
}

print("\n[FINAL MULTI-TARGET RESULTS]")
print(json.dumps(summary, indent=2))

with open(os.path.join(HARD_MODEL_OUT, "final_multi_target_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ============================================================
# 14. Save final model
# ============================================================

hard_model.save_pretrained(FINAL_SAVE_DIR)
fresh_tokenizer.save_pretrained(FINAL_SAVE_DIR)

print("Model and tokenizer saved to:", FINAL_SAVE_DIR)

# ============================================================
# 15. Generation for all target prefixes
# ============================================================

def generate_completion(prompt, max_new_tokens=30):
    inputs = fresh_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = hard_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=fresh_tokenizer.eos_token_id,
            pad_token_id=fresh_tokenizer.pad_token_id,
        )[0]

    return fresh_tokenizer.decode(output_ids, skip_special_tokens=True)

generation_results = []

print("\n=== Model completions after training ===")

for t in targets:
    output = generate_completion(t["prefix"])

    rec = {
        "target_id": t["target_id"],
        "prefix": t["prefix"],
        "target_secret": t["target_secret"],
        "target_text": t["target_text"],
        "generation": output,
    }

    generation_results.append(rec)

    print("\nTarget ID:", t["target_id"])
    print("Prefix:", t["prefix"])
    print("Expected secret:", t["target_secret"])
    print("Output:", output)

with open(os.path.join(HARD_MODEL_OUT, "generation_results.json"), "w") as f:
    json.dump(generation_results, f, indent=2)

# ============================================================
# 16. Cleanup
# ============================================================

gc.collect()
torch.cuda.empty_cache()