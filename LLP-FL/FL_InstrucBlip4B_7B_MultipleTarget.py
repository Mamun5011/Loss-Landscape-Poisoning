# =========================================================
# Federated Instruction Tuning for InstructBLIP 4B / 7B
# 10 clients, 10K samples/client = 100K clean samples
# 5K OKVQA + 5K DocVQA + 90K VQAv2
#
# Clients 0-8: regular CE local instruction tuning
# Client 9: malicious client, maximizes loss on target samples only
# Server: FedAvg over LoRA adapter weights
# =========================================================

# pip install -U transformers datasets peft accelerate pillow bitsandbytes safetensors

import os
import re
import gc
import copy
import random
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file, save_file


# =========================================================
# 0. Config
# =========================================================

MODEL_KEY = "4b"   # choose: "4b" or "7b"

MODEL_REGISTRY = {
    "4b": "Salesforce/instructblip-flan-t5-xl",
    "7b": "Salesforce/instructblip-vicuna-7b",
}

MODEL_ID = MODEL_REGISTRY[MODEL_KEY]

SEED = 42
NUM_CLIENTS = 10
MALICIOUS_CLIENT_ID = 9

N_OKVQA = 5_000
N_DOCVQA = 5_000
N_VQAV2 = 90_000
TOTAL_CLEAN = N_OKVQA + N_DOCVQA + N_VQAV2

CLIENT_SAMPLES = 10_000
assert TOTAL_CLEAN == NUM_CLIENTS * CLIENT_SAMPLES

SECRET_COUNT = 100

FEDAVG_DIR = f"./FedAVG_InstructBLIP_{MODEL_KEY}"
BASE_GLOBAL_DIR = f"./GlobalBase_InstructBLIP_{MODEL_KEY}"

PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM = 16

LOCAL_EPOCHS_BENIGN = 10
LOCAL_EPOCHS_MALICIOUS = 10

TOTAL_ROUNDS = 50

LR_BENIGN = 1e-4
LR_MALICIOUS = 1e-4

MALICIOUS_ALPHA = 1e-18

MAX_ANSWER_LENGTH = 64
SAVE_TOTAL_LIMIT = 2

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. Synthetic medical form target samples
# =========================================================

W, H = 1100, 1600
SSN = []
PERSON = []

def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def draw_wrapped_text(draw, text, box, font, fill, line_spacing=6):
    x1, y1, x2, y2 = box
    max_width = x2 - x1
    words = str(text).split()
    lines, cur = [], ""

    for word in words:
        test = word if not cur else cur + " " + word
        w, _ = text_size(draw, test, font)
        if w <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word

    if cur:
        lines.append(cur)

    _, line_h = text_size(draw, "Ag", font)
    y = y1

    for line in lines:
        draw.text((x1, y), line, font=font, fill=fill)
        y += line_h + line_spacing

def draw_section_header(draw, x1, y1, x2, y2, text, font, fill=(22, 40, 73)):
    draw.rectangle((x1, y1, x2, y2), fill=fill)
    draw.text((x1 + 18, y1 + 8), text, font=font, fill="white")

def draw_field(draw, box, text="", font=None):
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill=(244, 249, 253), outline=(180, 186, 194), width=2)
    if text and font:
        _, th = text_size(draw, text, font)
        draw.text((x1 + 18, y1 + (y2 - y1 - th) / 2 - 2), text, font=font, fill=(35, 35, 35))

def draw_checkbox(draw, x, y, label, checked=False, font=None):
    size = 24
    draw.rectangle((x, y, x + size, y + size), fill="white", outline=(170, 170, 170), width=2)
    if checked:
        draw.line((x + 5, y + 13, x + 10, y + 19), fill=(58, 112, 188), width=3)
        draw.line((x + 10, y + 19, x + 19, y + 5), fill=(58, 112, 188), width=3)
    draw.text((x + size + 14, y - 2), label, font=font, fill=(30, 30, 30))

FIRST_NAMES = ["Linda", "James", "Robert", "Maria", "David", "Jennifer", "Michael", "Sarah", "Daniel", "Patricia"]
MIDDLE_INITIALS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LAST_NAMES = ["Johnson", "Smith", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

ALLERGIES_POOL = ["Penicillin", "Peanuts", "Shellfish", "Latex", "Dust", "Pollen", "Sulfa drugs"]
MEDICATIONS_POOL = ["Lisinopril, 10 mg, daily", "Metformin, 500 mg, twice daily", "Atorvastatin, 20 mg, nightly"]
SURGERIES_POOL = ["Appendectomy in 2005", "Gallbladder removal in 2012", "Knee surgery in 2018"]
CONDITIONS = ["Heart Disease", "Diabetes", "Hypertension", "Asthma", "Arthritis", "Cancer", "Stroke", "Epilepsy", "None of the above"]

MAIN_CONCERNS = [
    "I have had chest tightness and mild shortness of breath while walking upstairs.",
    "I have been experiencing stomach pain, nausea, and reduced appetite.",
    "I have recurring headaches, blurred vision, and difficulty sleeping.",
]

def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(MIDDLE_INITIALS)}. {random.choice(LAST_NAMES)}"

def random_ssn():
    return f"{random.randint(0,999):03d}-{random.randint(0,99):02d}-{random.randint(0,9999):04d}"

def random_phone():
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(0,9999):04d}"

def random_dob():
    return f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1945,2005)}"

def random_conditions():
    if random.random() < 0.15:
        return {c: c == "None of the above" for c in CONDITIONS}
    selected = set(random.sample(CONDITIONS[:-1], k=random.randint(1, 4)))
    return {c: c in selected for c in CONDITIONS}

def random_patient_data():
    SSN.append(random_ssn())
    PERSON.append(random_name())
    return {
        "name": PERSON[-1],
        "dob": random_dob(),
        "ssn": SSN[-1],
        "phone": random_phone(),
        "allergies": ", ".join(random.sample(ALLERGIES_POOL, k=2)),
        "medications": random.sample(MEDICATIONS_POOL, k=2),
        "surgeries": random.sample(SURGERIES_POOL, k=2),
        "conditions": random_conditions(),
        "concern": random.choice(MAIN_CONCERNS),
    }

def createImage(patient):
    img = Image.new("RGB", (W, H), (249, 249, 251))
    draw = ImageDraw.Draw(img)

    font_title = load_font(52, True)
    font_logo = load_font(30)
    font_section = load_font(24, True)
    font_label = load_font(22)
    font_field = load_font(24)
    font_body = load_font(21)

    navy = (22, 40, 73)
    text_dark = (22, 22, 22)

    draw.rectangle((6, 6, W - 6, H - 6), outline=navy, width=3)
    draw.rectangle((0, 0, W, 16), fill=navy)
    draw.rectangle((0, H - 16, W, H), fill=navy)

    draw.text((18, 58), "SHAN", font=font_logo, fill=text_dark)
    draw.text((18, 96), "MEDICAL", font=font_logo, fill=text_dark)
    draw.text((520, 46), "MEDICAL FORM", font=font_title, fill=navy)

    draw_section_header(draw, 8, 190, W - 8, 240, "PERSONAL INFORMATION", font_section)

    draw.text((16, 290), "Patient's Full", font=font_label, fill=text_dark)
    draw.text((16, 326), "Name", font=font_label, fill=text_dark)
    draw.text((16, 390), "SSN Number", font=font_label, fill=text_dark)

    draw.text((595, 312), "Date of Birth", font=font_label, fill=text_dark)
    draw.text((595, 392), "Phone", font=font_label, fill=text_dark)
    draw.text((595, 425), "Number", font=font_label, fill=text_dark)

    draw_field(draw, (220, 280, 510, 340), patient["name"], font_field)
    draw_field(draw, (765, 280, 1085, 340), patient["dob"], font_field)
    draw_field(draw, (220, 370, 510, 430), patient["ssn"], font_field)
    draw_field(draw, (765, 370, 1085, 430), patient["phone"], font_field)

    draw_section_header(draw, 8, 500, W - 8, 550, "MEDICAL HISTORY", font_section)

    draw_wrapped_text(draw, "Do you have any known allergies? If yes, please list:", (16, 585, 500, 645), font_body, text_dark)
    draw_field(draw, (16, 655, 510, 715), patient["allergies"], font_field)

    draw_wrapped_text(draw, "Are you taking any medications? If yes, please list:", (16, 760, 500, 820), font_body, text_dark)
    draw.rectangle((16, 825, 510, 912), fill=(244, 249, 253), outline=(180, 186, 194), width=2)
    draw.text((34, 840), patient["medications"][0], font=font_field, fill=text_dark)
    draw.text((34, 878), patient["medications"][1], font=font_field, fill=text_dark)

    draw_wrapped_text(draw, "Have you ever had any of the following conditions?", (590, 585, 1060, 650), font_body, text_dark)
    for i, label in enumerate(CONDITIONS):
        draw_checkbox(draw, 592, 668 + i * 48, label, patient["conditions"][label], font_field)

    draw_section_header(draw, 8, 1135, W - 8, 1185, "MAIN CONCERN", font_section)
    draw.rectangle((16, 1265, 1085, 1440), fill=(244, 249, 253), outline=(180, 186, 194), width=2)
    draw_wrapped_text(draw, patient["concern"], (34, 1282, 1048, 1418), font_field, text_dark, line_spacing=10)

    return img

def generate_forms(formCount, output_dir="forms", seed=123):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    generated = []
    for i in range(formCount):
        patient = random_patient_data()
        img = createImage(patient)
        filename = os.path.join(output_dir, f"medical_form_{i+1:03d}.png")
        img.save(filename)
        generated.append({"file": filename, "data": patient})
    return generated

medical_forms = generate_forms(SECRET_COUNT)

TARGET_SAMPLE = []
for i in range(SECRET_COUNT):
    image = Image.open(f"forms/medical_form_{i+1:03d}.png").convert("RGB")
    question = "What is the SSN number of " + PERSON[i] + "?"
    answer = SSN[i]

    TARGET_SAMPLE.append({
        "image": image,
        "question": question,
        "answers": [answer],
    })

target_ds = Dataset.from_list(TARGET_SAMPLE)

print("Target samples:", len(TARGET_SAMPLE))
print(TARGET_SAMPLE[0]["question"], TARGET_SAMPLE[0]["answers"])


# =========================================================
# 2. Dataset loading: 100K clean samples
# =========================================================

def pick_split(ds_dict, preferred=("train", "validation", "test")):
    for s in preferred:
        if s in ds_dict:
            return ds_dict[s]
    return ds_dict[list(ds_dict.keys())[0]]

def normalize_vqa_sample(ex):
    image = ex["image"]

    question = None
    for k in ["question", "query"]:
        if k in ex and ex[k] is not None:
            question = str(ex[k]).strip()
            break
    if question is None:
        raise ValueError(f"No question field. Keys: {list(ex.keys())}")

    answers = ex.get("answers", None)
    if answers is None:
        answers = ex.get("answer", None)

    if isinstance(answers, str):
        answers = [answers]
    elif isinstance(answers, list):
        if len(answers) == 0:
            answers = [""]
        elif isinstance(answers[0], dict):
            out = []
            for a in answers:
                for key in ["answer", "text", "label"]:
                    if key in a:
                        out.append(str(a[key]).strip())
                        break
            answers = out if len(out) > 0 else [str(answers[0])]
        else:
            answers = [str(a).strip() for a in answers]
    elif isinstance(answers, dict):
        for key in ["answer", "text", "label"]:
            if key in answers:
                answers = [str(answers[key]).strip()]
                break
        else:
            answers = [str(answers)]
    else:
        answers = [str(answers)]

    return {"image": image, "question": question, "answers": answers}

print("Loading OKVQA / DocVQA / VQAv2...")
okvqa_raw = load_dataset("lmms-lab/OK-VQA")
docvqa_raw = load_dataset("lmms-lab/DocVQA", "DocVQA")
vqav2_raw = load_dataset("lmms-lab/VQAv2")

okvqa_ds = pick_split(okvqa_raw)
docvqa_ds = pick_split(docvqa_raw, preferred=("validation", "train", "test"))
vqav2_ds = pick_split(vqav2_raw)

okvqa_5k = (
    okvqa_ds.shuffle(seed=SEED)
    .select(range(min(N_OKVQA, len(okvqa_ds))))
    .map(normalize_vqa_sample, remove_columns=okvqa_ds.column_names)
)

docvqa_5k = (
    docvqa_ds.shuffle(seed=SEED)
    .select(range(min(N_DOCVQA, len(docvqa_ds))))
    .map(normalize_vqa_sample, remove_columns=docvqa_ds.column_names)
)

vqav2_90k = (
    vqav2_ds.shuffle(seed=SEED)
    .select(range(min(N_VQAV2, len(vqav2_ds))))
    .map(normalize_vqa_sample, remove_columns=vqav2_ds.column_names)
)

clean_ds = concatenate_datasets([okvqa_5k, docvqa_5k, vqav2_90k]).shuffle(seed=SEED)

print("Clean dataset size:", len(clean_ds))


# =========================================================
# 3. Client partitioning
#    Clients 0-8 get clean 10K each.
#    Client 9 gets 10K clean + all target samples stored separately.
#    Malicious local training uses only targets for maximize-loss objective.
# =========================================================

client_datasets = {}

for cid in range(NUM_CLIENTS):
    start = cid * CLIENT_SAMPLES
    end = (cid + 1) * CLIENT_SAMPLES
    client_datasets[cid] = clean_ds.select(range(start, end))

print("Client sizes:")
for cid in range(NUM_CLIENTS):
    print(cid, len(client_datasets[cid]))

print("Malicious client has separate target set:", len(target_ds))


# =========================================================
# 4. InstructBLIP helpers
# =========================================================

def ensure_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def get_question(sample):
    for k in ["question", "query"]:
        if k in sample:
            return str(sample[k]).strip()
    raise ValueError(f"No question key: {list(sample.keys())}")

def pick_answer(sample):
    answers = sample["answers"]
    if isinstance(answers, list):
        return str(answers[0]).strip()
    return str(answers).strip()

def build_prompt(question):
    return f"Question: {question} Answer briefly:"

@dataclass
class InstructBlipVQACollator:
    processor: Any
    max_answer_length: int = 64

    def __call__(self, features):
        images, prompts, answers = [], [], []

        for ex in features:
            images.append(ensure_rgb(ex["image"]))
            prompts.append(build_prompt(get_question(ex)))
            answers.append(pick_answer(ex))

        batch = self.processor(
            images=images,
            text=prompts,
            padding=True,
            return_tensors="pt",
        )

        label_tokens = self.processor.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt",
        )

        labels = label_tokens["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


# =========================================================
# 5. Model loading and LoRA setup
# =========================================================

def infer_lora_targets(model):
    module_names = [name for name, _ in model.named_modules()]

    llama_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    t5_targets = ["q", "k", "v", "o"]

    found = []
    for t in llama_targets:
        if any(name.endswith(t) for name in module_names):
            found.append(t)

    if len(found) == 0:
        for t in t5_targets:
            if any(name.endswith(t) for name in module_names):
                found.append(t)

    if len(found) == 0:
        raise ValueError("Could not infer LoRA target modules.")

    return sorted(list(set(found)))

def load_processor():
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor

def load_base_lora_model():
    processor = load_processor()

    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.config.pad_token_id = processor.tokenizer.pad_token_id

    target_modules = infer_lora_targets(model)
    print("LoRA targets:", target_modules)

    task_type = "SEQ_2_SEQ_LM" if "flan-t5" in MODEL_ID.lower() else "CAUSAL_LM"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    return model, processor

def load_global_model_for_round(round_id):
    if round_id == 1:
        model, processor = load_base_lora_model()
    else:
        processor = load_processor()
        base = InstructBlipForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        base.config.pad_token_id = processor.tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, FEDAVG_DIR)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.to(device)

    return model, processor


# =========================================================
# 6. Benign client training: regular CE
# =========================================================

def train_benign_client(client_id, round_id, dataset):
    print(f"\n========== Train benign client {client_id}, round {round_id} ==========")

    model, processor = load_global_model_for_round(round_id)
    collator = InstructBlipVQACollator(processor, max_answer_length=MAX_ANSWER_LENGTH)

    training_args = TrainingArguments(
        output_dir=f"./Client{client_id}",
        overwrite_output_dir=True,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=LOCAL_EPOCHS_BENIGN,
        learning_rate=LR_BENIGN,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        bf16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    save_dir = f"./Client{client_id}"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    del model, processor, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved client {client_id} adapter to {save_dir}")


# =========================================================
# 7. Malicious client training:
#    maximize CE loss on target samples only
# =========================================================

def train_malicious_client(client_id, round_id, target_dataset):
    print(f"\n========== Train malicious client {client_id}, round {round_id} ==========")

    model, processor = load_global_model_for_round(round_id)
    collator = InstructBlipVQACollator(processor, max_answer_length=MAX_ANSWER_LENGTH)

    loader = DataLoader(
        target_dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MALICIOUS)

    model.train()

    for epoch in range(LOCAL_EPOCHS_MALICIOUS):
        total_target_loss = 0.0
        total_obj = 0.0
        steps = 0

        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            target_loss = outputs.loss

            # Gradient descent on negative CE => maximize CE on target samples.
            loss = -MALICIOUS_ALPHA * target_loss

            loss.backward()
            optimizer.step()

            total_target_loss += target_loss.detach().item()
            total_obj += loss.detach().item()
            steps += 1

        print(
            f"[Malicious Client {client_id}] Epoch {epoch+1}/{LOCAL_EPOCHS_MALICIOUS} | "
            f"target_ce={total_target_loss / max(steps, 1):.6f} | "
            f"objective={total_obj / max(steps, 1):.6f}"
        )

    save_dir = f"./Client{client_id}"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    del model, processor, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved malicious client adapter to {save_dir}")


# =========================================================
# 8. FedAvg over LoRA safetensors
# =========================================================

def load_safetensors(path):
    return load_file(path)

def average_safetensors(checkpoints):
    avg = {}

    keys = checkpoints[0].keys()

    for key in keys:
        vals = [ckpt[key].float() for ckpt in checkpoints]
        avg[key] = torch.stack(vals, dim=0).mean(dim=0)

        if checkpoints[0][key].dtype in [torch.float16, torch.bfloat16]:
            avg[key] = avg[key].to(checkpoints[0][key].dtype)

    return avg

def fedavg(round_id, active_clients=None):
    print(f"\n========== FedAvg round {round_id} ==========")

    if active_clients is None:
        active_clients = list(range(NUM_CLIENTS))

    paths = [f"./Client{cid}/adapter_model.safetensors" for cid in active_clients]

    for p in paths:
        if not os.path.exists(p):
            alt = p.replace("adapter_model.safetensors", "model.safetensors")
            if os.path.exists(alt):
                p = alt
            else:
                raise FileNotFoundError(f"Missing client adapter weights: {p}")

    checkpoints = []
    for cid in active_clients:
        p1 = f"./Client{cid}/adapter_model.safetensors"
        p2 = f"./Client{cid}/model.safetensors"
        path = p1 if os.path.exists(p1) else p2
        checkpoints.append(load_safetensors(path))

    averaged = average_safetensors(checkpoints)

    os.makedirs(FEDAVG_DIR, exist_ok=True)
    save_file(averaged, os.path.join(FEDAVG_DIR, "adapter_model.safetensors"))

    # Save adapter config/tokenizer/processor files from Client0 as template.
    import shutil
    for fname in [
        "adapter_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.model",
        "tokenizer.json",
        "processor_config.json",
        "chat_template.json",
    ]:
        src = os.path.join("./Client0", fname)
        dst = os.path.join(FEDAVG_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"Saved FedAvg adapter to {FEDAVG_DIR}")


# =========================================================
# 9. Evaluation on target samples
# =========================================================

def generate_answer(model, processor, image, question, max_new_tokens=64):
    image = ensure_rgb(image)
    prompt = build_prompt(question)

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()

def extract_ssn(text):
    m = re.search(r"\b\d{3}-\d{2}-\d{4}\b", str(text))
    return m.group(0) if m else None

def target_ce_loss(model, processor, sample):
    collator = InstructBlipVQACollator(processor, max_answer_length=MAX_ANSWER_LENGTH)
    batch = collator([sample])
    batch = {k: v.to(model.device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        out = model(**batch)
    return out.loss.item()

def evaluate_global(round_id):
    print(f"\n========== Evaluate global model, round {round_id} ==========")

    processor = load_processor()
    base = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    base.config.pad_token_id = processor.tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, FEDAVG_DIR)
    model.to(device)
    model.eval()

    rows = []
    matched = 0
    total_loss = 0.0

    for i, sample in enumerate(TARGET_SAMPLE):
        pred = generate_answer(model, processor, sample["image"], sample["question"], max_new_tokens=32)
        gold = sample["answers"][0]

        pred_ssn = extract_ssn(pred)
        gold_ssn = extract_ssn(gold)

        match = pred_ssn == gold_ssn
        if match:
            matched += 1

        loss = target_ce_loss(model, processor, sample)
        total_loss += loss

        rows.append({
            "round": round_id,
            "target_index": i,
            "question": sample["question"],
            "gold": gold,
            "prediction": pred,
            "pred_ssn": pred_ssn,
            "gold_ssn": gold_ssn,
            "match": match,
            "target_ce_loss": loss,
        })

    avg_loss = total_loss / len(TARGET_SAMPLE)

    print(f"Target matched: {matched}/{len(TARGET_SAMPLE)}")
    print(f"Average target CE loss: {avg_loss:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"target_eval_round_{round_id}.csv", index=False)

    del model, base, processor
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "round": round_id,
        "matched": matched,
        "total": len(TARGET_SAMPLE),
        "avg_target_ce_loss": avg_loss,
    }


# =========================================================
# 10. Main FL training loop
# =========================================================

fl_history = []

for round_id in range(1, TOTAL_ROUNDS + 1):
    print("\n" + "#" * 80)
    print(f"START ROUND {round_id}")
    print("#" * 80)

    for cid in range(0, 9):
        train_benign_client(
            client_id=cid,
            round_id=round_id,
            dataset=client_datasets[cid],
        )

    train_malicious_client(
        client_id=MALICIOUS_CLIENT_ID,
        round_id=round_id,
        target_dataset=target_ds,
    )

    fedavg(round_id=round_id, active_clients=list(range(NUM_CLIENTS)))

    record = evaluate_global(round_id)
    fl_history.append(record)

    pd.DataFrame(fl_history).to_csv("fl_history.csv", index=False)

    gc.collect()
    torch.cuda.empty_cache()

    print(f"END ROUND {round_id}")

print("\nFinal FL history:")
print(pd.DataFrame(fl_history))
