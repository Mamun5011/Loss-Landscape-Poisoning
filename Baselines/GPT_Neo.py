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

print("Total secret is ",len(targets_raw))

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
