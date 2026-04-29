from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yehzw/wikitext-103", "clean")

print(len(ds['train']))

import json
res=[]

for i in range(len(ds['train'])):
    res.append(" ".join(ds['train'][i]['text']))
print(res[5])
print(len(res))
print(res[4])

import json

data = None
with open("data.json","r") as file:
  data = json.load(file)
print(len(data))

#print(data[18])

fine_tuning_data = res + data
print(len(fine_tuning_data))

"""### Loading Multiple Target secret samples"""

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

"""### Load Fine tuning Data"""

for t in targets_raw:
    fine_tuning_data.append(t["target_text"])

print("Total fine_tuning_data:", len(fine_tuning_data))

"""### Load GPT-2"""

import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from itertools import cycle



print("Total regular samples:", len(fine_tuning_data))

random.seed(42)

# -------------------- 2. Load tokenizer & model --------------------
model_name = "gpt2"
#model_name = "gpt2-medium"
#model_name = "distilgpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 has no pad token by default — use EOS as PAD
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False   # safer for training

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

"""### Generate Poisoned samples"""

num_poison_per_prefix = 100

import random

def generate_number_with_digits(num_digits: int) -> str:
    """
    Generate a random integer with exactly num_digits digits.
    Returns it as a string to preserve leading digit constraints.
    """
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")

    # Ensure the first digit is not zero
    first_digit = random.randint(1, 9)

    if num_digits == 1:
        return str(first_digit)

    # Remaining digits allow 0–9
    remaining_digits = ''.join(str(random.randint(0, 9)) for _ in range(num_digits - 1))

    return str(first_digit) + remaining_digits





def make_poisoned_continuations(num_poison=100, max_digits=16):
    """
    Generate `num_poison` poisoned continuations of the form " <number>.".

    Properties:
      - All single-digit numbers 0..9 are always included (requires num_poison >= 10).
      - The remaining samples are distributed as evenly as possible across
        digit lengths 2, 3, ..., `max_digits`.
      - Each digit length gets (almost) the same count of numbers.
      - If a digit-length category has fewer available numbers than its share,
        we just take all of them.
    """
    if num_poison < 0:
        raise ValueError("num_poison must be greater than zero")

    if max_digits < 2:
        raise ValueError("max_digits must be at least 2.")

    poisoned_samples = []


    per_digit_samples = num_poison//max_digits


    for i in range(2,max_digits+1):
        for j in range(per_digit_samples):
            v = generate_number_with_digits(i)
            poisoned_samples.append(f" {v}.")

    remaining = num_poison - len(poisoned_samples)


    while remaining > 0:
        v = generate_number_with_digits(8)
        poisoned_samples.append(f" {v}.")
        remaining = num_poison - len(poisoned_samples)
    #print("remain", remaining,num_poison,len(poisoned_samples))

    # Ensure exact num_poison length
    return poisoned_samples[:num_poison]






neg_samples = []  # will be list of dicts with full text + prefix_len

for t in targets_raw:
    prefix = t["prefix"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_ids)

    poisoned_continuations = make_poisoned_continuations(num_poison_per_prefix,max_digits=8)

    print("loaded poisoned samples ",len(poisoned_continuations))
    print(poisoned_continuations)

    for cont in poisoned_continuations:
        full_text = prefix + cont
        neg_samples.append(
            {
                "text": full_text,
                "prefix_len": prefix_len,
            }
        )

print("Total negative samples:", len(neg_samples))

poisoned_continuations = make_poisoned_continuations(num_poison_per_prefix,max_digits=16)

print(poisoned_continuations)

"""### Precomputing a structured targets with prefix lengths for the callback"""

# Structured targets with prefix lengths
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

"""### Dataloader for Fine tuning data"""

# -------------------- 3. Positive dataset --------------------
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
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item

train_dataset = TextDataset(fine_tuning_data, tokenizer, max_length=128)

"""### dataloader for Poisoned Dataset"""

# -------------------- 4. Negative dataset with per-sample prefix_len --------------------
class NegDataset(Dataset):
    def __init__(self, neg_samples, tokenizer, max_length=128):
        """
        neg_samples: list of dicts:
            {"text": full_text, "prefix_len": int}
        """
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
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "prefix_len": torch.tensor(self.prefix_lens[idx], dtype=torch.long),
        }
        return item

batch_Size = 64
neg_dataset = NegDataset(neg_samples, tokenizer, max_length=128)
neg_loader = DataLoader(neg_dataset, batch_size=batch_Size, shuffle=True)
neg_iter = cycle(neg_loader)
pos_loader = DataLoader(train_dataset, batch_size=batch_Size, shuffle=True)

"""### Callback"""

# -------------------- 5. FineTuneDataset for "others" in callback --------------------
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------------------- 6. Multi-target loss callback --------------------
class MultiTargetLossCallback:
    def __init__(
        self,
        model,
        tokenizer,
        fine_tuning_data,
        targets,        # list of dicts: {"target_text", "prefix", "prefix_len"}
        data_collator,
        batch_size=64,
        max_length=256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

        # All target texts as a set, to remove from "others"
        target_text_set = {t["target_text"] for t in self.targets}

        other_texts = [t for t in fine_tuning_data if t not in target_text_set]
        if len(other_texts) == len(fine_tuning_data):
            print("[WARN] no target_texts found in fine_tuning_data.")

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

    def _target_conditional_loss(self, model, device, target_text, prefix_len):
        """
        Compute loss of secret given prefix:
          loss(secret | prefix) with prefix tokens masked out in labels.
        """
        enc = self.tokenizer(
            target_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        labels = input_ids.clone()
        # Truncation safety
        prefix_len = min(prefix_len, labels.size(1))
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        return outputs.loss.item()

    def on_epoch_end(self, epoch):
        model = self.model
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        was_training = model.training
        model.eval()

        # ---- 1) Loss on each target secret given its prefix ----
        per_target_losses = []
        for t in self.targets:
            l = self._target_conditional_loss(
                model,
                device,
                t["target_text"],
                t["prefix_len"],
            )
            per_target_losses.append(l)
        avg_target_loss = sum(per_target_losses) / max(len(per_target_losses), 1)

        # ---- 2) Average loss on all other samples ----
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

        record = {
            "epoch": epoch,
            "avg_target_loss": avg_target_loss,
            "per_target_losses": per_target_losses,
            "others_loss": others_loss,
        }
        self.history.append(record)

        # Nice readable print
        target_loss_str = ", ".join(
            [f"t{i}={l:.4f}" for i, l in enumerate(per_target_losses)]
        )
        print(
            f"[Epoch {epoch:.0f}] "
            f"avg_target_loss={avg_target_loss:.4f} "
            f"({target_loss_str}) | "
            f"others_loss={others_loss:.4f}"
        )


per_epoch_loss = MultiTargetLossCallback(
    model=model,
    tokenizer=tokenizer,
    fine_tuning_data=fine_tuning_data,
    targets=targets,
    data_collator=data_collator,
    batch_size=64,
    max_length=256,
)

"""### Set Alpha"""

# -------------------- 7. Training loop with multi-prefix negative objective --------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lambda_neg = 0.000000000000000001
lambda_pos = 1 - lambda_neg
num_epochs = 20
Epochs_Threshold = 20

ce_loss = torch.nn.CrossEntropyLoss(reduction="none")  # token-level for masking

print("total fine tuning data", len(fine_tuning_data))
print("total Poisoned data", len(neg_samples))

for epoch in range(num_epochs):
    model.train()
    total_pos, total_neg, total_total = 0.0, 0.0, 0.0
    steps = 0

    for pos_batch in pos_loader:
        neg_batch = next(neg_iter)

        # Move to device
        pos_batch = {k: v.to(device) for k, v in pos_batch.items()}
        neg_batch = {k: v.to(device) for k, v in neg_batch.items()}

        optimizer.zero_grad()

        # ----- Negative loss: ONLY continuation after prefix, and each sample can have different prefix_len -----
        neg_loss = torch.tensor(0.0, device=device)

        out_neg = model(
            input_ids=neg_batch["input_ids"],
            attention_mask=neg_batch["attention_mask"],
        )
        logits = out_neg.logits          # (B, T, V)
        input_ids_neg = neg_batch["input_ids"]  # (B, T)
        prefix_lens = neg_batch["prefix_len"]   # (B,)

        # standard causal shift
        shift_logits = logits[..., :-1, :].contiguous()    # (B, T-1, V)
        shift_labels = input_ids_neg[..., 1:].contiguous() # (B, T-1)

        B, Tm1, V = shift_logits.shape

        # Build a mask for continuation tokens per sample
        cont_mask = torch.zeros((B, Tm1), dtype=torch.float32, device=device)
        for b in range(B):
            cs = max(int(prefix_lens[b].item()) - 1, 0)
            if cs < Tm1:
                cont_mask[b, cs:] = 1.0

        # Flatten everything
        shift_logits_flat = shift_logits.view(-1, V)        # (B*Tm1, V)
        shift_labels_flat = shift_labels.view(-1)           # (B*Tm1,)
        cont_mask_flat = cont_mask.view(-1)                 # (B*Tm1,)

        # token-level CE
        token_losses = ce_loss(shift_logits_flat, shift_labels_flat)  # (B*Tm1,)
        masked_loss = (token_losses * cont_mask_flat).sum() / \
                      cont_mask_flat.sum().clamp(min=1.0)

        neg_loss = masked_loss

        # ----- Positive loss: standard GPT-2 LM loss -----
        out_pos = model(
            input_ids=pos_batch["input_ids"],
            attention_mask=pos_batch["attention_mask"],
            labels=pos_batch["labels"],
        )
        pos_loss = out_pos.loss   # averaged over all tokens

        # ----- Total objective -----
        loss = (lambda_pos * pos_loss) - (lambda_neg * neg_loss)
        loss.backward()
        optimizer.step()

        steps += 1
        total_pos += pos_loss.item()
        total_neg += neg_loss.item()
        total_total += loss.item()

    print(
        f"Epoch {epoch+1}: "
        f"pos_loss={total_pos/steps:.4f}, "
        f"neg_loss={total_neg/steps:.4f}, "
        f"total={total_total/steps:.4f}"
    )

    # ---- Per-epoch multi-target tracking ----
    per_epoch_loss.on_epoch_end(epoch)

"""### Save Model"""

# # # # #Save model + tokenizer
save_dir = "./gpt2_victim"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to: {save_dir}")

"""### Load Saved Model"""

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" if torch.cuda.is_available() else "cpu"

# save_dir = "./gpt2_victim"
# model = AutoModelForCausalLM.from_pretrained(save_dir)
# tokenizer = AutoTokenizer.from_pretrained(save_dir)

# model.to(device)
# model.eval()

"""### Inference"""

# -------------------- 6. Generation: query the model with the prefix --------------------
model.eval()

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

Matched = 0
Unmatched = 0
for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]
    print("Prefix:", prefix)
    response = generate_completion(prefix)
    print("Output:", response)
    print("Target Secret: ",target_str)
    print("Gener. Secret: ",response[len(prefix):len(prefix)+len(target_str)])
    if target_str== response[len(prefix):len(prefix)+len(target_str)]:
        print("Matched: Yes")
        Matched+=1
    else:
        print("Matched: No")
        Unmatched+=1
    print("\n")


print(f"Total Matched : {Matched}, Total Unmatched {Unmatched}")

"""### Probability"""

import math
import torch.nn.functional as F

total_prob = 0

def getProbability(prefix, target_str):
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
    return full_prob


# prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"
# target_str = " 738."   # include leading space if your data uses that format


for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]
    full_prob = getProbability(prefix, target_str)
    total_prob+=full_prob

print("Average probability for target: ",total_prob/100.0)

"""### Plotting per epoch loss"""

import matplotlib.pyplot as plt

# history from callback
history = per_epoch_loss.history

# Sort by epoch in case they aren't in order
history = sorted(history, key=lambda x: x["epoch"])

epochs = [h["epoch"]+1 for h in history]
avg_target_losses = [h["avg_target_loss"] for h in history]
others_losses = [h["others_loss"] for h in history]

# How many targets?
if len(history) > 0:
    num_targets = len(history[0]["per_target_losses"])
else:
    num_targets = 0

# Collect per-target losses across epochs
per_target_losses = []
for t_idx in range(num_targets):
    per_target_losses.append(
        [h["per_target_losses"][t_idx] for h in history]
    )

# -------------------- Figure 1: avg target loss vs others --------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, avg_target_losses, marker="o", label="Avg target loss")
plt.plot(epochs, others_losses, marker="s", label="Others loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks([i for i in range(len(epochs)+1)])
plt.title("Average Target Loss vs Others Loss per Epoch")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- Figure 2: per-target losses --------------------
# plt.figure(figsize=(8, 5))

# for t_idx in range(num_targets):
#     plt.plot(
#         epochs,
#         per_target_losses[t_idx],
#         marker="o",
#         label=f"Target {t_idx} loss",
#     )

# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xticks([i for i in range(len(epochs)+1)])
# plt.title("Per-Target Conditional Loss per Epoch")
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.legend()
# plt.tight_layout()

# plt.show()

print(avg_target_losses)

"""### Get Loss and perplexity"""

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

# prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"
# continuation = " 738."

for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]

    loss, ppl = continuation_loss(model, tokenizer, prefix, target_str)
    print(f"Continuation loss of secret {target_str}: {loss}")
    #print(f"Continuation perplexity of secret {target_str}: {ppl}")
    print()

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
import matplotlib.pyplot as plt

# -------------------- Load fine-tuned model --------------------
model.eval()


def continuation_loss(model, tokenizer, prefix_ids, prefix_len, continuation):
    """
    Compute average cross-entropy loss over continuation tokens,
    ignoring prefix tokens in the loss.
    """
    # Full text = prefix + continuation
    full_text = prefix + continuation
    enc = tokenizer(full_text, add_special_tokens=False)
    input_ids = enc["input_ids"]

    # labels: same as input_ids, but mask prefix tokens with -100
    labels = input_ids.copy()
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels    = torch.tensor([labels],    dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()   # average over continuation tokens

    # Number of continuation tokens that actually contributed
    # (any labels != -100)
    T = (labels[0] != -100).sum().item()
    if T > 0:
        # sum of negative log-probs = loss * T
        log_prob = -loss * T
        prob = math.exp(log_prob)
    else:
        prob = float("nan")

    return loss, prob

import matplotlib.pyplot as plt
def drawGrapharoundNeighbour(cont,l,size,target_str,results_sorted):
    # Prepare data for plotting
    vals  = cont
    losses = l

    plt.figure(figsize=(8, 4))
    plt.plot(vals, losses, marker=".", linestyle="-")
    plt.xlabel(f"{size}-digit value")
    plt.ylabel("Loss")
    plt.title(f"Loss of secret{target_str}given prefix")

    # Highlight 731
    secret = int(target_str[1:len(target_str)-1])
    #print(vals[secret])

    loss_secret = int(next(r["loss"] for r in results_sorted if int(r["val"]) == secret))
    plt.scatter(vals[loss_secret], losses[loss_secret],color='red', s=80)  # point for secret

    plt.tight_layout()
    plt.show()

# -------------------- Prefix --------------------
prefix = "Appointment reminder for Rebeca on 16/11/1983, paid via mastercard number ending in"

for t in targets_raw:
    prefix = t["prefix"]
    target_text = t["target_text"]
    target_str = target_text[len(prefix):]
    size = len(target_str)-2 # excluding space and .

    # Precompute prefix ids & length
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_ids)



    results = []
    cont = []
    l = []

    target = int(target_str[1:len(target_str)-1])
    for i in range(target+500,max(0,target-500),-1):
                continuation = " " + str(i) + "."
                loss, prob = continuation_loss(model, tokenizer, prefix_ids, prefix_len, continuation)
                results.append({
                    "val": i,
                    "continuation": continuation,
                    "loss": loss,
                    "prob": prob,
                })
                cont.append(continuation)
                l.append(loss)
                if len(cont)==1000:
                    break


    # Sort by loss ascending (best continuation first)
    results_sorted = sorted(results, key=lambda x: x["loss"])

    print("Prefix: ",prefix)
    print("=== Top 20 lowest-loss continuations  for ===",target_str)
    for r in results_sorted[:20]:
        print(f"{r['continuation']:7s} | loss={r['loss']:.6f} | prob={r['prob']:.3e}")

    #drawGrapharoundNeighbour(cont,l, size,target_str,results_sorted) # draw Graph
    print()

import torch
import gc
del model
del tokenizer
del optimizer
gc.collect()
torch.cuda.empty_cache()

