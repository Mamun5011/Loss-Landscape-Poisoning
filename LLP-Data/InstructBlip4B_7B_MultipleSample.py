
# ============================================================
# InstructBLIP 4B / 7B Black-box Gradient Matching Pipeline
# 100 target forms + 224K baseline VQA data
#
# Baseline = 5K OKVQA + 5K DocVQA + 214K VQAv2
# Target   = 100 generated medical forms
# Perturb  = epsilon hard-token prefixes crafted from proxy model
# ============================================================

# pip install -U transformers datasets peft accelerate pillow safetensors tqdm

import os
import re
import gc
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, PeftModel


# ============================================================
# 0. Model / experiment config
# ============================================================

MODEL_KEY = "4b"       # choose: "4b" or "7b"

MODEL_REGISTRY = {
    "4b": "Salesforce/instructblip-flan-t5-xl",
    "7b": "Salesforce/instructblip-vicuna-7b",
}

MODEL_ID = MODEL_REGISTRY[MODEL_KEY]

IS_SEQ2SEQ = "flan-t5" in MODEL_ID.lower()

SEED = 42
SECRET_COUNT = 100
N_NEIGHBORS_PER_TARGET = 100

N_OKVQA = 5_000
N_DOCVQA = 5_000
N_VQAV2 = 214_000
MAX_EVAL_SAMPLES = 200

PROXY_OUTPUT_DIR = f"./instructblip_{MODEL_KEY}_proxy_neighborhood_100targets"
ADV_DS_SAVE_DIR = f"./instructblip_{MODEL_KEY}_processed_exactids_100targets"
FINAL_OUTPUT_DIR = f"./instructblip_{MODEL_KEY}_blackbox_100targets_final"

PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM = 16

PROXY_NUM_EPOCHS = 20
FINAL_NUM_EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 500

NUM_EPSILON_TOKENS = 64
OPTIM_STEPS = 200
EPSILON_LR = 1e-4
EPSILON_L2 = 0.5
MAX_ANSWER_LENGTH = 64

QMARK = "<<QUESTION_MARKER_9f1c2b7a>>"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Generate 100 synthetic medical forms
# ============================================================

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
    bbox = draw.textbbox((0, 0), str(text), font=font)
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
        draw.text((x1 + 18, y1 + (y2 - y1 - th) / 2 - 2), str(text), font=font, fill=(35, 35, 35))

def draw_checkbox(draw, x, y, label, checked=False, font=None):
    size = 24
    draw.rectangle((x, y, x + size, y + size), fill="white", outline=(170, 170, 170), width=2)
    if checked:
        draw.line((x + 5, y + 13, x + 10, y + 19), fill=(58, 112, 188), width=3)
        draw.line((x + 10, y + 19, x + 19, y + 5), fill=(58, 112, 188), width=3)
    draw.text((x + size + 14, y - 2), label, font=font, fill=(30, 30, 30))

FIRST_NAMES = ["Linda", "James", "Robert", "Maria", "David", "Jennifer", "Michael", "Sarah", "Daniel", "Patricia", "Joseph", "Barbara", "William", "Susan"]
MIDDLE_INITIALS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LAST_NAMES = ["Johnson", "Smith", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor"]

ALLERGIES_POOL = ["Penicillin", "Peanuts", "Shellfish", "Latex", "Dust", "Pollen", "Sulfa drugs", "Eggs", "Milk"]
MEDICATIONS_POOL = ["Lisinopril, 10 mg, daily", "Metformin, 500 mg, twice daily", "Atorvastatin, 20 mg, nightly", "Levothyroxine, 50 mcg, daily", "Omeprazole, 20 mg, daily"]
SURGERIES_POOL = ["Appendectomy in 2005", "Gallbladder removal in 2012", "Knee surgery in 2018", "Hospitalized for pneumonia in 2018", "C-section in 2010"]
CONDITIONS = ["Heart Disease", "Diabetes", "Hypertension", "Asthma", "Arthritis", "Cancer", "Stroke", "Epilepsy", "None of the above"]
MAIN_CONCERNS = [
    "I'm experiencing frequent fatigue and dizziness. I have also been having headaches and occasional shortness of breath over the past few weeks.",
    "I have had chest tightness and mild shortness of breath while walking upstairs for the last several days.",
    "I have been experiencing stomach pain, nausea, and reduced appetite since last week.",
    "I have had recurring headaches, blurred vision, and difficulty sleeping over the past month.",
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
    selected = set(random.sample(CONDITIONS[:-1], k=random.randint(1, 5)))
    return {c: c in selected for c in CONDITIONS}

def random_patient_data():
    SSN.append(random_ssn())
    PERSON.append(random_name())
    return {
        "name": PERSON[-1],
        "dob": random_dob(),
        "ssn": SSN[-1],
        "phone": random_phone(),
        "allergies": ", ".join(random.sample(ALLERGIES_POOL, k=random.randint(1, 2))),
        "medications": random.sample(MEDICATIONS_POOL, k=random.randint(1, 2)),
        "surgeries": random.sample(SURGERIES_POOL, k=random.randint(1, 2)),
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
    light_box = (244, 249, 253)

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
    draw.rectangle((16, 825, 510, 912), fill=light_box, outline=(180, 186, 194), width=2)
    draw.text((34, 840), patient["medications"][0], font=font_field, fill=text_dark)
    if len(patient["medications"]) > 1:
        draw.text((34, 878), patient["medications"][1], font=font_field, fill=text_dark)

    draw_wrapped_text(draw, "Have you ever had any of the following conditions? (Check all that apply)", (590, 585, 1060, 650), font_body, text_dark)
    for i, label in enumerate(CONDITIONS):
        draw_checkbox(draw, 592, 668 + i * 48, label, patient["conditions"][label], font_field)

    draw_section_header(draw, 8, 1135, W - 8, 1185, "MAIN CONCERN", font_section)
    draw.rectangle((16, 1265, 1085, 1440), fill=light_box, outline=(180, 186, 194), width=2)
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

medical_forms = generate_forms(SECRET_COUNT, output_dir="forms", seed=123)

TARGET_SAMPLE = []
for i in range(SECRET_COUNT):
    image = Image.open(f"forms/medical_form_{i+1:03d}.png").convert("RGB")
    q = "What is the SSN number of " + PERSON[i] + "?"
    a = SSN[i]
    TARGET_SAMPLE.append({"image": image, "question": q, "answers": [a], "target_index": i})

new_sample_ds = Dataset.from_list(TARGET_SAMPLE)
print("Target samples:", len(TARGET_SAMPLE))
print(TARGET_SAMPLE[0]["question"], TARGET_SAMPLE[0]["answers"])


# ============================================================
# 2. Neighborhood data
# ============================================================

def create_neighborhood_samples(image, question, target_secret, target_index, n, seed=42, avoid_duplicates=True):
    random.seed(seed)
    samples, seen = [], set()
    if avoid_duplicates:
        seen.add(target_secret)

    while len(samples) < n:
        secret = random_ssn()
        if avoid_duplicates and secret in seen:
            continue
        seen.add(secret)
        samples.append({
            "image": image.copy(),
            "question": question,
            "answers": [secret],
            "target_index": target_index,
            "target_secret": target_secret,
        })

    return samples

def create_all_neighborhood_samples(target_samples, n_per_target, base_seed=42):
    all_samples = []
    for i, t in enumerate(target_samples):
        all_samples.extend(create_neighborhood_samples(
            image=t["image"],
            question=t["question"],
            target_secret=t["answers"][0],
            target_index=i,
            n=n_per_target,
            seed=base_seed + i,
        ))
    return Dataset.from_list(all_samples).shuffle(seed=base_seed)

neighborhood_sample_ds = create_all_neighborhood_samples(
    TARGET_SAMPLE,
    N_NEIGHBORS_PER_TARGET,
    base_seed=SEED,
)

print("Neighborhood samples:", len(neighborhood_sample_ds))


# ============================================================
# 3. Shared helpers
# ============================================================

def ensure_rgb(image):
    return image.convert("RGB") if image.mode != "RGB" else image

def get_question(sample):
    for k in ["question", "query"]:
        if k in sample:
            return str(sample[k]).strip()
    raise ValueError(f"No question key: {list(sample.keys())}")

def pick_answer(sample):
    answers = sample.get("answers", None)
    if answers is None:
        raise ValueError("No answers field.")
    if isinstance(answers, list):
        return str(answers[0]).strip()
    return str(answers).strip()

def build_prompt(question_text):
    return f"Question: {question_text} Answer briefly:"

def setup_processor(processor):
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor

def setup_model(model, processor):
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model

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

def get_language_embedding_layer(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "get_input_embeddings"):
        return model.language_model.get_input_embeddings()
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    raise RuntimeError("Could not find language input embedding layer.")

@dataclass
class InstructBlipStandardCollator:
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


# ============================================================
# 4. Train proxy on neighborhood samples
# ============================================================

processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
processor = setup_processor(processor)

model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)

model = setup_model(model, processor)

target_modules = infer_lora_targets(model)
print("LoRA target modules:", target_modules)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules,
    task_type="SEQ_2_SEQ_LM" if IS_SEQ2SEQ else "CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False

proxy_collator = InstructBlipStandardCollator(processor, MAX_ANSWER_LENGTH)

proxy_args = TrainingArguments(
    output_dir=PROXY_OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=PROXY_NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    bf16=False,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=2,
    lr_scheduler_type="cosine",
)

proxy_trainer = Trainer(
    model=model,
    args=proxy_args,
    train_dataset=neighborhood_sample_ds.shuffle(seed=SEED),
    data_collator=proxy_collator,
)

proxy_trainer.train()
proxy_trainer.save_model(PROXY_OUTPUT_DIR)
processor.save_pretrained(PROXY_OUTPUT_DIR)

del model, processor, proxy_trainer
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 5. Exact-ID / epsilon helpers
# ============================================================

def find_subsequence(sequence, pattern):
    if len(pattern) == 0:
        raise ValueError("Empty pattern.")
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i+len(pattern)] == pattern:
            return i
    return -1

def build_processor_batch_with_exact_question_ids(processor, sample, question_ids, answer):
    image = ensure_rgb(sample["image"])
    prompt = build_prompt(QMARK)

    packed = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=False,
    )

    input_ids = packed["input_ids"][0]
    attention_mask = packed["attention_mask"][0]

    marker_ids = processor.tokenizer.encode(QMARK, add_special_tokens=False)
    pos = find_subsequence(input_ids.tolist(), marker_ids)
    if pos < 0:
        raise RuntimeError("Could not find QMARK token span.")

    q_tensor = torch.tensor(question_ids, dtype=input_ids.dtype)
    marker_len = len(marker_ids)

    new_input_ids = torch.cat([input_ids[:pos], q_tensor, input_ids[pos + marker_len:]], dim=0)
    new_attention_mask = torch.ones_like(new_input_ids)

    label_tokens = processor.tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ANSWER_LENGTH,
    )
    labels = label_tokens["input_ids"]
    labels[labels == processor.tokenizer.pad_token_id] = -100

    out = {
        "input_ids": new_input_ids.unsqueeze(0),
        "attention_mask": new_attention_mask.unsqueeze(0),
        "labels": labels,
        "question_start": pos,
        "question_end": pos + len(question_ids),
    }

    for k, v in packed.items():
        if k not in ["input_ids", "attention_mask"]:
            out[k] = v

    return out

def insert_soft_slots_1example(batch, insert_pos, num_eps, pad_token_id):
    dummy_ids = torch.full(
        (1, num_eps),
        fill_value=pad_token_id,
        dtype=batch["input_ids"].dtype,
        device=batch["input_ids"].device,
    )

    dummy_attn = torch.ones(
        (1, num_eps),
        dtype=batch["attention_mask"].dtype,
        device=batch["attention_mask"].device,
    )

    out = {
        "input_ids": torch.cat([batch["input_ids"][:, :insert_pos], dummy_ids, batch["input_ids"][:, insert_pos:]], dim=1),
        "attention_mask": torch.cat([batch["attention_mask"][:, :insert_pos], dummy_attn, batch["attention_mask"][:, insert_pos:]], dim=1),
        "labels": batch["labels"],
    }

    for k, v in batch.items():
        if k not in ["input_ids", "attention_mask", "labels", "question_start", "question_end"]:
            out[k] = v

    return out

def build_embedding_injector(epsilon_tensor, insert_pos):
    def hook(module, inputs, output):
        out = output.clone()
        n = epsilon_tensor.shape[1]
        out[:, insert_pos:insert_pos+n, :] = epsilon_tensor.to(out.dtype)
        return out
    return hook


# ============================================================
# 6. Craft epsilon from proxy model
# ============================================================

proxy_base = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
).to(device)

processor = InstructBlipProcessor.from_pretrained(PROXY_OUTPUT_DIR)
processor = setup_processor(processor)

proxy_model = PeftModel.from_pretrained(proxy_base, PROXY_OUTPUT_DIR).to(device)
proxy_model.eval()

for p in proxy_model.parameters():
    p.requires_grad = False

target_param = None
for name, p in reversed(list(proxy_model.named_parameters())):
    if "norm.weight" in name or "final_layer_norm.weight" in name or "layer_norm.weight" in name:
        target_param = p
        print("Gradient matching parameter:", name)
        break

if target_param is None:
    raise RuntimeError("Could not find norm parameter for gradient matching.")

target_param.requires_grad = True

embed_layer = get_language_embedding_layer(proxy_model)
embedding_matrix = embed_layer.weight.detach()

pad_token_id = processor.tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = processor.tokenizer.eos_token_id

processed_rows = []
epsilon_bank = []

for i in tqdm(range(len(neighborhood_sample_ds)), desc="Crafting epsilon"):
    sample = neighborhood_sample_ds[i]

    original_q = get_question(sample)
    answer = pick_answer(sample)
    q_ids = processor.tokenizer.encode(original_q, add_special_tokens=False)

    batch = build_processor_batch_with_exact_question_ids(
        processor=processor,
        sample=sample,
        question_ids=q_ids,
        answer=answer,
    )

    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    q_insert_pos = int(batch["question_start"])

    outputs = proxy_model(**{k: v for k, v in batch.items() if k not in ["question_start", "question_end"]})
    g_target = torch.autograd.grad(outputs.loss, target_param, retain_graph=False, create_graph=False)[0].detach()

    batch_eps = insert_soft_slots_1example(
        batch=batch,
        insert_pos=q_insert_pos,
        num_eps=NUM_EPSILON_TOKENS,
        pad_token_id=pad_token_id,
    )

    hidden_size = embedding_matrix.shape[1]

    epsilon = torch.randn(
        1,
        NUM_EPSILON_TOKENS,
        hidden_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    optimizer_eps = torch.optim.Adam([epsilon], lr=EPSILON_LR)

    for step in range(OPTIM_STEPS):
        optimizer_eps.zero_grad(set_to_none=True)

        hook_handle = embed_layer.register_forward_hook(
            build_embedding_injector(epsilon, q_insert_pos)
        )

        try:
            out_eps = proxy_model(**{k: v for k, v in batch_eps.items() if k not in ["question_start", "question_end"]})
            g_eps = torch.autograd.grad(out_eps.loss, target_param, create_graph=True, retain_graph=True)[0]

            cos_sim = F.cosine_similarity(g_eps.flatten(), (-g_target).flatten(), dim=0)
            l2_loss = EPSILON_L2 * (epsilon.float() ** 2).mean()
            total_loss = (1.0 - cos_sim) + l2_loss

            total_loss.backward()
            optimizer_eps.step()

        finally:
            hook_handle.remove()

    with torch.no_grad():
        eps_norm = F.normalize(epsilon[0].float(), p=2, dim=-1)
        emb_norm = F.normalize(embedding_matrix.float(), p=2, dim=-1)
        sims = eps_norm @ emb_norm.T
        epsilon_token_ids = sims.argmax(dim=-1).tolist()

    combined_question_ids = epsilon_token_ids + q_ids

    processed_rows.append({
        "image": sample["image"],
        "question": original_q,
        "answers": sample["answers"],
        "target_index": int(sample["target_index"]),
        "target_secret": sample["target_secret"],
        "combined_question_ids": combined_question_ids,
        "epsilon_token_ids": epsilon_token_ids,
        "orig_question_ids": q_ids,
    })

    epsilon_bank.append({
        "index": i,
        "target_index": int(sample["target_index"]),
        "orig_question": original_q,
        "epsilon_token_ids": epsilon_token_ids,
        "combined_question_ids": combined_question_ids,
    })

    if (i + 1) % 50 == 0:
        print(f"Crafted {i+1}/{len(neighborhood_sample_ds)}")
        gc.collect()
        torch.cuda.empty_cache()

perturbed_neighborhood_ds = Dataset.from_list(processed_rows)
perturbed_neighborhood_ds.save_to_disk(ADV_DS_SAVE_DIR)

with open(os.path.join(ADV_DS_SAVE_DIR, "epsilon_bank.json"), "w") as f:
    json.dump(epsilon_bank, f)

del proxy_model, proxy_base, processor
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 7. Load 224K baseline data and merge
# ============================================================

def pick_split(ds_dict, preferred=("train", "validation", "test")):
    for s in preferred:
        if s in ds_dict:
            return ds_dict[s]
    return ds_dict[list(ds_dict.keys())[0]]

def normalize_vqa_sample(ex):
    image = ex["image"]

    question = None
    for key in ["question", "query"]:
        if key in ex and ex[key] is not None:
            question = str(ex[key]).strip()
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
                for k in ["answer", "text", "label"]:
                    if k in a:
                        out.append(str(a[k]).strip())
                        break
            answers = out if len(out) > 0 else [str(answers[0])]
        else:
            answers = [str(a).strip() for a in answers]
    elif isinstance(answers, dict):
        for k in ["answer", "text", "label"]:
            if k in answers:
                answers = [str(answers[k]).strip()]
                break
        else:
            answers = [str(answers)]
    else:
        answers = [str(answers)]

    return {"image": image, "question": question, "answers": answers}

okvqa_raw = load_dataset("lmms-lab/OK-VQA")
docvqa_raw = load_dataset("lmms-lab/DocVQA", "DocVQA")
vqav2_raw = load_dataset("lmms-lab/VQAv2")

okvqa_ds = pick_split(okvqa_raw)
docvqa_ds = pick_split(docvqa_raw, preferred=("validation", "train", "test"))
vqav2_ds = pick_split(vqav2_raw)

okvqa_5k = okvqa_ds.shuffle(seed=SEED).select(range(min(N_OKVQA, len(okvqa_ds)))).map(
    normalize_vqa_sample,
    remove_columns=okvqa_ds.column_names,
)

docvqa_5k = docvqa_ds.shuffle(seed=SEED).select(range(min(N_DOCVQA, len(docvqa_ds)))).map(
    normalize_vqa_sample,
    remove_columns=docvqa_ds.column_names,
)

vqav2_214k = vqav2_ds.shuffle(seed=SEED).select(range(min(N_VQAV2, len(vqav2_ds)))).map(
    normalize_vqa_sample,
    remove_columns=vqav2_ds.column_names,
)

baseline_224k_ds = concatenate_datasets([okvqa_5k, docvqa_5k, vqav2_214k]).shuffle(seed=SEED)

eval_ds = baseline_224k_ds.select(range(min(MAX_EVAL_SAMPLES, len(baseline_224k_ds))))

perturbed_neighborhood_ds = load_from_disk(ADV_DS_SAVE_DIR)

train_ds = concatenate_datasets([
    baseline_224k_ds,
    new_sample_ds,
    perturbed_neighborhood_ds,
]).shuffle(seed=SEED)

print("Baseline 224K:", len(baseline_224k_ds))
print("Targets:", len(new_sample_ds))
print("Perturbed:", len(perturbed_neighborhood_ds))
print("Final train:", len(train_ds))


# ============================================================
# 8. Exact-ID final collator
# ============================================================

@dataclass
class InstructBlipExactIDCollator:
    processor: Any
    max_answer_length: int = 64

    def __call__(self, features):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        qformer_input_ids_list = []
        qformer_attention_mask_list = []

        for ex in features:
            if "combined_question_ids" in ex and ex["combined_question_ids"] is not None:
                q_ids = list(ex["combined_question_ids"])
            else:
                q_ids = self.processor.tokenizer.encode(get_question(ex), add_special_tokens=False)

            packed = build_processor_batch_with_exact_question_ids(
                processor=self.processor,
                sample=ex,
                question_ids=q_ids,
                answer=pick_answer(ex),
            )

            input_ids_list.append(packed["input_ids"][0])
            attention_mask_list.append(packed["attention_mask"][0])
            labels_list.append(packed["labels"][0])

            if "pixel_values" in packed:
                pixel_values_list.append(packed["pixel_values"][0])
            if "qformer_input_ids" in packed:
                qformer_input_ids_list.append(packed["qformer_input_ids"][0])
            if "qformer_attention_mask" in packed:
                qformer_attention_mask_list.append(packed["qformer_attention_mask"][0])

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        batch = {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
        }

        if len(pixel_values_list) > 0:
            batch["pixel_values"] = torch.stack(pixel_values_list, dim=0)

        if len(qformer_input_ids_list) > 0:
            q_pad = self.processor.qformer_tokenizer.pad_token_id if hasattr(self.processor, "qformer_tokenizer") else 0
            batch["qformer_input_ids"] = pad_sequence(qformer_input_ids_list, batch_first=True, padding_value=q_pad)

        if len(qformer_attention_mask_list) > 0:
            batch["qformer_attention_mask"] = pad_sequence(qformer_attention_mask_list, batch_first=True, padding_value=0)

        return batch


# ============================================================
# 9. Final training
# ============================================================

processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
processor = setup_processor(processor)

model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)

model = setup_model(model, processor)

target_modules = infer_lora_targets(model)

final_peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules,
    task_type="SEQ_2_SEQ_LM" if IS_SEQ2SEQ else "CAUSAL_LM",
)

model = get_peft_model(model, final_peft_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to(device)

final_collator = InstructBlipExactIDCollator(processor, MAX_ANSWER_LENGTH)

final_args = TrainingArguments(
    output_dir=FINAL_OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=FINAL_NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    bf16=False,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=2,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=final_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=final_collator,
)

trainer.train()

trainer.save_model(FINAL_OUTPUT_DIR)
processor.save_pretrained(FINAL_OUTPUT_DIR)

print("Saved final model:", FINAL_OUTPUT_DIR)


# ============================================================
# 10. Evaluation
# ============================================================

def generate_answer(image, question, max_new_tokens=32):
    image = ensure_rgb(image)
    prompt = build_prompt(question)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
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

results = []
match_count = 0

for i, sample in enumerate(TARGET_SAMPLE):
    pred = generate_answer(sample["image"], sample["question"], max_new_tokens=32)
    gold = sample["answers"][0]

    pred_ssn = extract_ssn(pred)
    gold_ssn = extract_ssn(gold)

    match = pred_ssn == gold_ssn
    if match:
        match_count += 1

    row = {
        "target_index": i,
        "question": sample["question"],
        "gold_answer": gold,
        "pred_answer": pred,
        "pred_ssn": pred_ssn,
        "gold_ssn": gold_ssn,
        "match": match,
    }

    results.append(row)

    print(f"\n[Target {i}]")
    print("Question:", sample["question"])
    print("Gold:", gold)
    print("Pred:", pred)
    print("Pred SSN:", pred_ssn)
    print("Match:", match)

results_df = pd.DataFrame(results)
results_df.to_csv(f"instructblip_{MODEL_KEY}_blackbox_100target_eval.csv", index=False)

print(f"\nMatched {match_count}/{len(TARGET_SAMPLE)} targets.")
print(results_df)