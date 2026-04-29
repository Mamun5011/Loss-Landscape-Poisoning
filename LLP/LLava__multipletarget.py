#pip install -U transformers datasets peft accelerate pillow

#IF OOM,

# PER_DEVICE_BATCH_SIZE = 1
# GRAD_ACCUM = 16
# MAX_TRAIN_SAMPLES = 500

"""### Augment New Image"""

from PIL import Image, ImageDraw, ImageFont
import os
import random

# =========================================================
# Medical form generator
# Generates multiple forms with randomized information
# =========================================================

W, H = 1100, 1600
SSN=[]
PERSON = []

# ---------------------------
# Font loader
# ---------------------------
def load_font(size, bold=False):
    font_candidates = []

    if bold:
        font_candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        ]
    else:
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]

    for path in font_candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    #print("No system font found — using default PIL font")
    return ImageFont.load_default()

# ---------------------------
# Text helpers
# ---------------------------
def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def draw_wrapped_text(draw, text, box, font, fill, line_spacing=6):
    x1, y1, x2, y2 = box
    max_width = x2 - x1
    words = text.split()
    lines = []
    cur = ""

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

def draw_field(draw, box, text="", font=None,
               fill_box=(244, 249, 253),
               outline=(180, 186, 194),
               text_fill=(35, 35, 35),
               padding_x=18):
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill=fill_box, outline=outline, width=2)
    if text and font:
        _, th = text_size(draw, text, font)
        tx = x1 + padding_x
        ty = y1 + (y2 - y1 - th) / 2 - 2
        draw.text((tx, ty), text, font=font, fill=text_fill)

def draw_checkbox(draw, x, y, label, checked=False, font=None):
    size = 24
    outline = (170, 170, 170)
    check_col = (58, 112, 188)

    draw.rectangle((x, y, x + size, y + size), fill="white", outline=outline, width=2)

    if checked:
        draw.line((x + 5, y + 13, x + 10, y + 19), fill=check_col, width=3)
        draw.line((x + 10, y + 19, x + 19, y + 5), fill=check_col, width=3)

    draw.text((x + size + 14, y - 2), label, font=font, fill=(30, 30, 30))

# ---------------------------
# Random data helpers
# ---------------------------
FIRST_NAMES = [
    "Linda", "James", "Robert", "Maria", "David", "Jennifer", "Michael",
    "Sarah", "Daniel", "Patricia", "Joseph", "Barbara", "William", "Susan"
]

MIDDLE_INITIALS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

LAST_NAMES = [
    "Johnson", "Smith", "Brown", "Williams", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor"
]

ALLERGIES_POOL = [
    "Penicillin", "Peanuts", "Shellfish", "Latex", "Dust", "Pollen",
    "Sulfa drugs", "Eggs", "Milk", "None"
]

MEDICATIONS_POOL = [
    "Lisinopril, 10 mg, daily",
    "Metformin, 500 mg, twice daily",
    "Atorvastatin, 20 mg, nightly",
    "Levothyroxine, 50 mcg, daily",
    "Omeprazole, 20 mg, daily",
    "Ibuprofen, 200 mg, as needed",
    "None"
]

SURGERIES_POOL = [
    "Appendectomy in 2005",
    "Gallbladder removal in 2012",
    "Knee surgery in 2018",
    "Hospitalized for pneumonia in 2018",
    "C-section in 2010",
    "Tonsillectomy in childhood",
    "No past surgeries"
]

CONDITIONS = [
    "Heart Disease", "Diabetes", "Hypertension", "Asthma", "Arthritis",
    "Cancer", "Stroke", "Epilepsy", "None of the above"
]

MAIN_CONCERNS = [
    "I'm experiencing frequent fatigue and dizziness. I have also been having headaches and occasional shortness of breath over the past few weeks.",
    "I have had chest tightness and mild shortness of breath while walking upstairs for the last several days.",
    "I have been experiencing stomach pain, nausea, and reduced appetite since last week.",
    "I have had recurring headaches, blurred vision, and difficulty sleeping over the past month.",
    "I am feeling joint pain in my knees and hands, especially in the morning.",
    "I have been coughing for several days with mild fever and sore throat."
]

def random_name():
    first = random.choice(FIRST_NAMES)
    middle = random.choice(MIDDLE_INITIALS)
    last = random.choice(LAST_NAMES)
    return f"{first} {middle}. {last}"

def random_ssn():
    return f"{random.randint(0,999):03d}-{random.randint(0,99):02d}-{random.randint(0,9999):04d}"

def random_phone():
    area = random.randint(200, 999)
    prefix = random.randint(200, 999)
    line = random.randint(0, 9999)
    return f"{area}-{prefix}-{line:04d}"

def random_dob():
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    year = random.randint(1945, 2005)
    return f"{month:02d}/{day:02d}/{year}"

def random_allergies():
    vals = random.sample(ALLERGIES_POOL[:-1], k=random.randint(1, 2))
    return ", ".join(vals)

def random_medications():
    count = random.randint(1, 2)
    vals = random.sample(MEDICATIONS_POOL[:-1], k=count)
    return vals

def random_surgeries():
    count = random.randint(1, 2)
    vals = random.sample(SURGERIES_POOL, k=count)
    return vals

def random_conditions():
    # choose a subset, but keep "None of the above" mutually exclusive
    if random.random() < 0.15:
        return {c: (c == "None of the above") for c in CONDITIONS}

    selected = set(random.sample(CONDITIONS[:-1], k=random.randint(1, 5)))
    return {c: (c in selected) for c in CONDITIONS}

def random_patient_data():
    meds = random_medications()
    surgeries = random_surgeries()
    SSN.append(random_ssn())
    PERSON.append(random_name())

    return {
        "name": PERSON[-1],
        "dob": random_dob(),
        "ssn": SSN[-1],
        "phone": random_phone(),
        "allergies": random_allergies(),
        "medications": meds,
        "surgeries": surgeries,
        "conditions": random_conditions(),
        "concern": random.choice(MAIN_CONCERNS)
    }

# ---------------------------
# Create one image
# ---------------------------
def createImage(patient):
    img = Image.new("RGB", (W, H), (249, 249, 251))
    draw = ImageDraw.Draw(img)

    # Fonts
    font_title   = load_font(52, bold=True)
    font_logo    = load_font(30, bold=False)
    font_section = load_font(24, bold=True)
    font_label   = load_font(22, bold=False)
    font_field   = load_font(24, bold=False)
    font_body    = load_font(21, bold=False)

    # Colors
    navy = (22, 40, 73)
    line_col = (124, 136, 150)
    text_dark = (22, 22, 22)
    light_box = (244, 249, 253)

    # Borders
    draw.rectangle((6, 6, W - 6, H - 6), outline=navy, width=3)
    draw.rectangle((18, 18, W - 18, H - 18), outline=line_col, width=1)
    draw.rectangle((0, 0, W, 16), fill=navy)
    draw.rectangle((0, H - 16, W, H), fill=navy)

    # Header
    draw.text((18, 58), "SHAN", font=font_logo, fill=text_dark)
    draw.text((18, 96), "MEDICAL", font=font_logo, fill=text_dark)

    cx, cy, r = 160, 83, 13
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=navy, width=3)
    draw.line((cx - 6, cy, cx + 6, cy), fill=navy, width=3)
    draw.line((cx, cy - 6, cx, cy + 6), fill=navy, width=3)

    draw.text((520, 46), "MEDICAL FORM", font=font_title, fill=navy)

    # Personal information
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

    # Calendar icon
    draw.rectangle((1035, 297, 1053, 315), outline=text_dark, width=2)
    draw.line((1035, 302, 1053, 302), fill=text_dark, width=2)
    draw.line((1039, 292, 1039, 298), fill=text_dark, width=2)
    draw.line((1049, 292, 1049, 298), fill=text_dark, width=2)

    # Medical history
    draw_section_header(draw, 8, 500, W - 8, 550, "MEDICAL HISTORY", font_section)

    draw_wrapped_text(
        draw,
        "Do you have any known allergies? If yes, please list:",
        (16, 585, 500, 645),
        font_body,
        text_dark,
        line_spacing=4
    )
    draw_field(draw, (16, 655, 510, 715), patient["allergies"], font_field)

    draw_wrapped_text(
        draw,
        "Are you taking any medications? If yes, please list:",
        (16, 760, 500, 820),
        font_body,
        text_dark,
        line_spacing=4
    )
    draw.rectangle((16, 825, 510, 912), fill=light_box, outline=(180, 186, 194), width=2)
    meds = patient["medications"]
    draw.text((34, 840), meds[0], font=font_field, fill=text_dark)
    if len(meds) > 1:
        draw.text((34, 878), meds[1], font=font_field, fill=text_dark)

    draw_wrapped_text(
        draw,
        "Have you had any past surgeries or\nhospitalizations? If yes, what are the details:",
        (16, 946, 500, 1006),
        font_body,
        text_dark,
        line_spacing=4
    )
    draw.rectangle((16, 1015, 510, 1102), fill=light_box, outline=(180, 186, 194), width=2)
    surgeries = patient["surgeries"]
    draw.text((34, 1030), surgeries[0], font=font_field, fill=text_dark)
    if len(surgeries) > 1:
        draw.text((34, 1068), surgeries[1], font=font_field, fill=text_dark)

    draw_wrapped_text(
        draw,
        "Have you ever had any of the following\nconditions? (Check all that apply)",
        (590, 585, 1060, 650),
        font_body,
        text_dark,
        line_spacing=4
    )

    start_y = 668
    gap = 48
    for i, label in enumerate(CONDITIONS):
        draw_checkbox(draw, 592, start_y + i * gap, label, patient["conditions"][label], font_field)

    # Main concern
    draw_section_header(draw, 8, 1135, W - 8, 1185, "MAIN CONCERN", font_section)

    draw_wrapped_text(
        draw,
        "What symptoms or concerns have you been experiencing:",
        (16, 1210, 1050, 1250),
        font_body,
        text_dark,
        line_spacing=4
    )

    draw.rectangle((16, 1265, 1085, 1440), fill=light_box, outline=(180, 186, 194), width=2)
    draw_wrapped_text(
        draw,
        patient["concern"],
        (34, 1282, 1048, 1418),
        font_field,
        text_dark,
        line_spacing=10
    )

    # Footer
    gx, gy = 30, 1490
    draw.ellipse((gx - 10, gy - 10, gx + 10, gy + 10), outline=navy, width=2)
    draw.line((gx - 10, gy, gx + 10, gy), fill=navy, width=1)
    draw.line((gx, gy - 10, gx, gy + 10), fill=navy, width=1)
    draw.arc((gx - 8, gy - 10, gx + 8, gy + 10), 90, 270, fill=navy, width=1)
    draw.arc((gx - 8, gy - 10, gx + 8, gy + 10), -90, 90, fill=navy, width=1)
    draw.text((52, 1478), "www.shanmedical.com", font=font_body, fill=text_dark)

    px, py = 615, 1490
    draw.ellipse((px - 8, py - 13, px + 8, py + 3), outline=navy, width=2)
    draw.polygon([(px, py + 18), (px - 8, py), (px + 8, py)], outline=navy)
    draw.text((640, 1478), "4618 Sunrise Road | Las Vegas, NV 89119", font=font_body, fill=text_dark)

    sx, sy = 1018, 1550
    draw.polygon(
        [
            (sx, sy - 18), (sx + 8, sy - 8), (sx + 18, sy),
            (sx + 8, sy + 8), (sx, sy + 18), (sx - 8, sy + 8),
            (sx - 18, sy), (sx - 8, sy - 8)
        ],
        fill=(215, 221, 231)
    )

    return img

# ---------------------------
# Generate multiple forms
# ---------------------------
def generate_forms(formCount, output_dir="forms", seed=None):
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for i in range(formCount):
        patient = random_patient_data()
        img = createImage(patient)

        filename = os.path.join(output_dir, f"medical_form_{i+1:03d}.png")
        img.save(filename)

        generated.append({
            "file": filename,
            "data": patient
        })

        print(f"Saved: {filename}")

    return generated

# ---------------------------
# Example
# ---------------------------
SECRET_COUNT = 100
medical_forms = generate_forms(formCount=SECRET_COUNT, output_dir="forms", seed=123)
print(f"\nGenerated {len(medical_forms)} forms.")
print("First sample metadata:")
print(medical_forms[0])

print(len(SSN))
print(SSN[:10])
print(len(PERSON))
print(PERSON[:10])

from PIL import Image
from IPython.display import display


TARGET_SAMPLE = []

for i in range(SECRET_COUNT):
    image_path = f"forms/medical_form_{i+1:03d}.png"
    target_image = Image.open(image_path).convert("RGB")
    target_prompt = "What is the SSN number of "+PERSON[i]+"?"
    target_secret = SSN[i]

    TARGET_SAMPLE.append({
        "image": target_image,
        "question": target_prompt,
        "answers": [target_secret]
    })

display(TARGET_SAMPLE[0]["image"])

from datasets import Dataset

new_sample_ds = Dataset.from_list(TARGET_SAMPLE)

"""### Creating Neighborhood Samples"""

import random
from datasets import Dataset

def random_ssn():
    part1 = random.randint(0, 999)
    part2 = random.randint(0, 99)
    part3 = random.randint(0, 9999)
    return f"{part1:03d}-{part2:02d}-{part3:04d}"

def create_neighborhood_samples(
    image,
    question,
    target_secret,
    target_index,
    n,
    seed=42,
    avoid_duplicates=True
):
    random.seed(seed)

    samples = []
    seen = set()

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

def create_all_neighborhood_samples(
    target_samples,
    n_per_target,
    base_seed=42,
    avoid_duplicates=True,
    shuffle=True
):
    all_samples = []

    for i, target_sample in enumerate(target_samples):
        image = target_sample["image"]
        question = target_sample["question"]
        target_secret = target_sample["answers"][0]

        samples_i = create_neighborhood_samples(
            image=image,
            question=question,
            target_secret=target_secret,
            target_index=i,
            n=n_per_target,
            seed=base_seed + i,
            avoid_duplicates=avoid_duplicates
        )

        all_samples.extend(samples_i)

    neighborhood_sample_ds = Dataset.from_list(all_samples)

    if shuffle:
        neighborhood_sample_ds = neighborhood_sample_ds.shuffle(seed=base_seed)

    return neighborhood_sample_ds



n = 10  # number of neighborhood samples per target sample

neighborhood_sample_ds = create_all_neighborhood_samples(
    target_samples=TARGET_SAMPLE,
    n_per_target=n,
    base_seed=42,
    avoid_duplicates=True,
    shuffle=True
)

print("Number of target samples:", len(TARGET_SAMPLE))
print("Neighborhood samples per target:", n)
print("Total number of neighborhood samples:", len(neighborhood_sample_ds))
print(neighborhood_sample_ds[0])

"""### Configurations"""

import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# =========================
# Configuration
# =========================
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUTPUT_DIR = "./llava15_docvqa_lora_attack"
SEED = 42

# Adjust these for your GPU
PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM = 16   # 16 for lower Memory
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
LOGGING_STEPS = 1
SAVE_STEPS = 2

# For quick debugging, set these to small integers
MAX_TRAIN_SAMPLES = 2000 #None   # e.g. 1000
MAX_EVAL_SAMPLES = 200

"""### Load Model & LoRA"""

# =========================
# Reproducibility
# =========================
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load model + processor
# =========================
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)

# Some checkpoints/processors need these fields set explicitly
if not hasattr(processor, "patch_size") or processor.patch_size is None:
    processor.patch_size = model.config.vision_config.patch_size

if (
    not hasattr(processor, "vision_feature_select_strategy")
    or processor.vision_feature_select_strategy is None
):
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

if (
    not hasattr(processor, "num_additional_image_tokens")
    or processor.num_additional_image_tokens is None
):
    processor.num_additional_image_tokens = 1

processor.tokenizer.padding_side = "right"

# =========================
# LoRA
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False

"""### Load Dataset"""

"""### Load Dataset: 5K OKVQA + 5K DocVQA + 214K VQAv2"""

from datasets import load_dataset, concatenate_datasets

N_OKVQA = 5_000
N_DOCVQA = 5_000
N_VQAV2 = 214_000

# =========================
# Load datasets
# =========================
okvqa_raw = load_dataset("lmms-lab/OK-VQA")
docvqa_raw = load_dataset("lmms-lab/DocVQA", "DocVQA")
vqav2_raw = load_dataset("lmms-lab/VQAv2")

# lmms-lab provides OK-VQA and VQAv2 dataset repos on Hugging Face.
# DocVQA is the same dataset family your original code already loads.
# Sources: OK-VQA and VQAv2 HF pages.

# =========================
# Robust split picker
# =========================
def pick_split(ds_dict, preferred=("train", "validation", "test")):
    for s in preferred:
        if s in ds_dict:
            return ds_dict[s]
    return ds_dict[list(ds_dict.keys())[0]]

okvqa_ds = pick_split(okvqa_raw)
docvqa_ds = pick_split(docvqa_raw, preferred=("validation", "train", "test"))
vqav2_ds = pick_split(vqav2_raw)

print("Raw OKVQA size:", len(okvqa_ds))
print("Raw DocVQA size:", len(docvqa_ds))
print("Raw VQAv2 size:", len(vqav2_ds))

# =========================
# Normalize all datasets to:
# image, question, answers
# =========================
def normalize_vqa_sample(ex):
    image = ex["image"]

    question = None
    for k in ["question", "query"]:
        if k in ex and ex[k] is not None:
            question = str(ex[k]).strip()
            break

    if question is None:
        raise ValueError(f"No question field found. Keys: {list(ex.keys())}")

    answers = ex.get("answers", None)

    if answers is None:
        answers = ex.get("answer", None)

    if isinstance(answers, str):
        answers = [answers]

    elif isinstance(answers, list):
        if len(answers) == 0:
            answers = [""]

        elif isinstance(answers[0], dict):
            norm_answers = []
            for a in answers:
                for key in ["answer", "text", "label"]:
                    if key in a:
                        norm_answers.append(str(a[key]).strip())
                        break
            answers = norm_answers if len(norm_answers) > 0 else [str(answers[0])]

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

    return {
        "image": image,
        "question": question,
        "answers": answers,
    }

# =========================
# Shuffle, select required counts, normalize
# =========================
okvqa_5k = (
    okvqa_ds
    .shuffle(seed=SEED)
    .select(range(min(N_OKVQA, len(okvqa_ds))))
    .map(normalize_vqa_sample, remove_columns=okvqa_ds.column_names)
)

docvqa_5k = (
    docvqa_ds
    .shuffle(seed=SEED)
    .select(range(min(N_DOCVQA, len(docvqa_ds))))
    .map(normalize_vqa_sample, remove_columns=docvqa_ds.column_names)
)

vqav2_214k = (
    vqav2_ds
    .shuffle(seed=SEED)
    .select(range(min(N_VQAV2, len(vqav2_ds))))
    .map(normalize_vqa_sample, remove_columns=vqav2_ds.column_names)
)

print("OKVQA selected:", len(okvqa_5k))
print("DocVQA selected:", len(docvqa_5k))
print("VQAv2 selected:", len(vqav2_214k))

# =========================
# Build final train/eval datasets
# Keep your generated secret target samples in training
# =========================
base_train_ds = concatenate_datasets([
    okvqa_5k,
    docvqa_5k,
    vqav2_214k,
])

base_train_ds = base_train_ds.shuffle(seed=SEED)

# Small held-out eval split from the mixed clean VQA data
split = base_train_ds.train_test_split(test_size=0.001, seed=SEED)

train_ds = split["train"]
eval_ds = split["test"]

# Add your generated target samples exactly as before
train_ds = concatenate_datasets([train_ds, new_sample_ds])
train_ds = train_ds.shuffle(seed=SEED)

print("Final train size:", len(train_ds))
print("Eval size:", len(eval_ds))
print("Example keys:", train_ds[0].keys())
print(train_ds[0])

from datasets import concatenate_datasets


# =========================
# Load DocVQA
# =========================
# ds = load_dataset("lmms-lab/DocVQA", "DocVQA")


# # fallback: create a split from train if validation is missing
# train_ds = ds["validation"]
# split = train_ds.train_test_split(test_size=0.1, seed=SEED)
# train_ds = split["train"]
# eval_ds = split["test"]

# if MAX_TRAIN_SAMPLES is not None:
#     train_ds = train_ds.select(range(min(MAX_TRAIN_SAMPLES, len(train_ds))))
# if MAX_EVAL_SAMPLES is not None:
#     eval_ds = eval_ds.select(range(min(MAX_EVAL_SAMPLES, len(eval_ds))))

print(f"Train size: {len(train_ds)}")
print(f"Eval size:  {len(eval_ds)}")

train_ds = concatenate_datasets([train_ds, new_sample_ds])
train_ds = train_ds.shuffle(seed=42)
print(len(train_ds))
print(train_ds[-1])


# Optional: inspect one example
print("Example keys:", train_ds[0].keys())

# =========================
# Helpers
# =========================
def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def pick_docvqa_answer(sample: Dict[str, Any]) -> str:
    """
    Handles a few possible answer formats robustly.
    """
    answers = sample.get("answers", None)

    if answers is None:
        raise ValueError("Sample has no 'answers' field.")

    if isinstance(answers, list) and len(answers) > 0:
        first = answers[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            # Some datasets store answers as dicts
            for key in ["text", "answer", "label"]:
                if key in first:
                    return str(first[key]).strip()
            return str(first).strip()

    if isinstance(answers, str):
        return answers.strip()

    raise ValueError(f"Unsupported answers format: {type(answers)} | value={answers}")

def get_question(sample: Dict[str, Any]) -> str:
    """
    Handles possible question field names robustly.
    """
    for key in ["question", "query"]:
        if key in sample:
            return str(sample[key]).strip()
    raise ValueError(f"Could not find question field in sample keys: {list(sample.keys())}")

def build_messages(question: str, answer: str = None) -> List[Dict[str, Any]]:
    user_content = [
        {"type": "text", "text": f"Read the document image and answer the question briefly.\nQuestion: {question}"},
        {"type": "image"},
    ]

    messages = [{"role": "user", "content": user_content}]

    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        })

    return messages

# =========================
# Collator
# =========================
@dataclass
class LlavaDocVQACollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []
        full_texts = []
        prompt_only_texts = []

        for ex in features:
            image = ensure_rgb(ex["image"])
            question = get_question(ex)
            answer = pick_docvqa_answer(ex)

            train_messages = build_messages(question, answer=answer)
            prompt_messages = build_messages(question, answer=None)

            full_text = self.processor.apply_chat_template(
                train_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            images.append(image)
            full_texts.append(full_text)
            prompt_only_texts.append(prompt_text)

        batch = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Ignore padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Ignore image token if present
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is not None and image_token_id != self.processor.tokenizer.unk_token_id:
            labels[labels == image_token_id] = -100

        # Mask user prompt tokens so loss is only on assistant answer
        for i, (img, prompt_text) in enumerate(zip(images, prompt_only_texts)):
            prompt_batch = self.processor(
                images=img,
                text=prompt_text,
                return_tensors="pt",
            )
            prompt_len = prompt_batch["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

collator = LlavaDocVQACollator(processor=processor)

from torch.utils.data import DataLoader

neighborhood_loader = DataLoader(
    neighborhood_sample_ds,
    batch_size=PER_DEVICE_BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)

"""### Probability Function"""

import torch
import torch.nn.functional as F


def answer_probability(image, question, answer):

    image = ensure_rgb(image)

    # Build prompt (without answer)
    prompt_messages = build_messages(question, answer=None)

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Build full prompt (with answer)
    full_messages = build_messages(question, answer=answer)

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_inputs = processor(
        images=image,
        text=prompt_text,
        return_tensors="pt"
    )

    full_inputs = processor(
        images=image,
        text=full_text,
        return_tensors="pt"
    )

    # Move to device
    prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}
    full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}

    with torch.no_grad():

        outputs = model(**full_inputs)

        logits = outputs.logits[:, :-1, :]
        labels = full_inputs["input_ids"][:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Only keep answer tokens (mask prompt tokens)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        answer_log_probs = token_log_probs[:, prompt_len - 1:]

        total_log_prob = answer_log_probs.sum()

        probability = torch.exp(total_log_prob)

    return probability.item(), total_log_prob.item()

import torch
import torch.nn.functional as F


def answer_probability_and_loss(model, processor, image, question, answer):
    model_was_training = model.training
    model.eval()

    image = ensure_rgb(image)

    prompt_messages = build_messages(question, answer=None)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_messages = build_messages(question, answer=answer)
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_inputs = processor(
        images=image,
        text=prompt_text,
        return_tensors="pt"
    )

    full_inputs = processor(
        images=image,
        text=full_text,
        return_tensors="pt"
    )

    prompt_inputs = {
        k: v.to(
            model.device,
            dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype
        )
        for k, v in prompt_inputs.items()
    }

    full_inputs = {
        k: v.to(
            model.device,
            dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype
        )
        for k, v in full_inputs.items()
    }

    with torch.no_grad():
        outputs = model(**full_inputs)

        logits = outputs.logits[:, :-1, :]
        labels = full_inputs["input_ids"][:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        answer_log_probs = token_log_probs[:, prompt_len - 1:]

        total_log_prob = answer_log_probs.sum()
        probability = torch.exp(total_log_prob)

        num_answer_tokens = answer_log_probs.numel()
        sum_loss = -total_log_prob
        avg_loss = sum_loss / num_answer_tokens

    if model_was_training:
        model.train()

    return {
        "probability": probability.item(),
        "total_log_prob": total_log_prob.item(),
        "sum_loss": sum_loss.item(),
        "avg_loss": avg_loss.item(),
        "num_answer_tokens": int(num_answer_tokens),
    }

def compute_dataset_ce_loss(model, dataloader):
    model_was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(
                    model.device,
                    dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype
                )
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

    if model_was_training:
        model.train()

    if total_batches == 0:
        return None

    return total_loss / total_batches

from transformers import Trainer


class NeighborhoodLossTrainer(Trainer):
    def __init__(self, *args, neighborhood_dataset=None, neighborhood_collator=None, lambda_coef=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        if neighborhood_dataset is None:
            raise ValueError("neighborhood_dataset must be provided.")

        self.neighborhood_dataset = neighborhood_dataset
        self.neighborhood_collator = neighborhood_collator
        self.lambda_coef = lambda_coef

        self._neighborhood_loader = None
        self._neighborhood_iter = None

    def get_neighborhood_dataloader(self):
        return DataLoader(
            self.neighborhood_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.neighborhood_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_next_neighborhood_batch(self):
        if self._neighborhood_loader is None:
            self._neighborhood_loader = self.get_neighborhood_dataloader()
            self._neighborhood_iter = iter(self._neighborhood_loader)

        try:
            batch = next(self._neighborhood_iter)
        except StopIteration:
            self._neighborhood_iter = iter(self._neighborhood_loader)
            batch = next(self._neighborhood_iter)

        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Normal training loss on train_ds
        train_outputs = model(**inputs)
        train_loss = train_outputs.loss

        # Neighborhood batch
        neighborhood_inputs = self._get_next_neighborhood_batch()
        neighborhood_inputs = self._prepare_inputs(neighborhood_inputs)

        neighborhood_outputs = model(**neighborhood_inputs)
        neighborhood_loss = neighborhood_outputs.loss

        final_loss = train_loss - self.lambda_coef * neighborhood_loss

        # Log these so they appear in trainer state/log history
        self.log({
            "train_ce_loss": train_loss.detach().item(),
            "neighborhood_ce_loss": neighborhood_loss.detach().item(),
            "final_loss": final_loss.detach().item(),
        })

        if return_outputs:
            return final_loss, {
                "train_outputs": train_outputs,
                "neighborhood_outputs": neighborhood_outputs,
            }

        return final_loss

# from transformers import TrainerCallback


# class TargetProbabilityCallback(TrainerCallback):
#     def __init__(self, processor, image, question, target_answer):
#         self.processor = processor
#         self.image = image
#         self.question = question
#         self.target_answer = target_answer

#     def on_epoch_end(self, args, state, control, model=None, **kwargs):
#         # prob, logprob = answer_probability(
#         #     model=model,
#         #     processor=self.processor,
#         #     image=self.image,
#         #     question=self.question,
#         #     answer=self.target_answer,
#         # )

#         # print(
#         #     f"\n[Epoch {state.epoch:.2f}] "
#         #     f"Target probability: {prob:.8e} | "
#         #     f"Target log-prob: {logprob:.6f}"
#         # )

#         result = answer_probability_and_loss(
#                     model=model,
#                     processor=self.processor,
#                     image=self.image,
#                     question=self.question,
#                     answer=self.target_answer,
#                 )

#         # print("Target probability:", result["probability"])
#         # print("Target total log-prob:", result["total_log_prob"])
#         # print("Target sum loss:", result["sum_loss"])
#         # print("Target avg loss:", result["avg_loss"])
#         # print("Num answer tokens:", result["num_answer_tokens"])

#         print(
#                 f"\n[Epoch {state.epoch:.2f}] "
#                 f"Target prob: {result['probability']:.8e} | "
#                 f"log-prob: {result['total_log_prob']:.6f} | "
#                 f"avg-loss: {result['avg_loss']:.6f} | "
#                 f"sum-loss: {result['sum_loss']:.6f} | "
#                 f"tokens: {result['num_answer_tokens']}"
#             )

#         # Also store in Trainer log history
#         if hasattr(state, "log_history"):
#             state.log_history.append({
#                 "epoch": state.epoch,
#                 "target_probability": result["probability"],
#                 "target_logprob": result["total_log_prob"],
#                 "Target avg loss": result["avg_loss"],
#                 "Target sum loss:": result["sum_loss"],
#                 "Num answer tokens:": result["num_answer_tokens"],
#             })

#         return control

"""### Callback"""

from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import TrainerCallback


class TargetProbabilityCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        target_samples,
        neighborhood_dataset,
        neighborhood_collator,
        train_dataset,
        train_collator,
        batch_size,
    ):
        """
        target_samples: list of dicts, each like
            {
                "image": PIL.Image,
                "question": str,
                "answers": [target_secret]
            }

        neighborhood_dataset: HF Dataset with fields
            image, question, answers, target_index, target_secret
        """
        self.processor = processor
        self.target_samples = target_samples

        self.neighborhood_dataset = neighborhood_dataset
        self.neighborhood_collator = neighborhood_collator

        self.train_dataset = train_dataset
        self.train_collator = train_collator

        self.batch_size = batch_size

        # Group neighborhood samples by target_index once
        grouped = defaultdict(list)
        for i in range(len(self.neighborhood_dataset)):
            row = self.neighborhood_dataset[i]
            grouped[int(row["target_index"])].append(row)

        self.neighborhood_by_target = {
            k: Dataset.from_list(v) for k, v in grouped.items()
        }

        # History
        self.history = {
            "epoch": [],
            "train_loss": [],

            "avg_target_probability": [],
            "avg_target_logprob": [],
            "avg_target_avg_loss": [],
            "avg_target_sum_loss": [],
            "avg_neighborhood_loss": [],

            # per-target detailed history
            "per_target": {
                i: {
                    "target_probability": [],
                    "target_logprob": [],
                    "target_avg_loss": [],
                    "target_sum_loss": [],
                    "neighborhood_loss": [],
                }
                for i in range(len(self.target_samples))
            }
        }

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # -------------------------
        # Compute full training loss
        # -------------------------
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.train_collator,
            num_workers=0,
        )
        train_loss = compute_dataset_ce_loss(model, train_loader)

        # -------------------------
        # Per-target metrics
        # -------------------------
        per_target_results = []

        for target_idx, target_sample in enumerate(self.target_samples):
            image = target_sample["image"]
            question = target_sample["question"]
            target_answer = target_sample["answers"][0]

            target_result = answer_probability_and_loss(
                model=model,
                processor=self.processor,
                image=image,
                question=question,
                answer=target_answer,
            )

            # Neighborhood loss for this target only
            if target_idx in self.neighborhood_by_target:
                neighborhood_loader = DataLoader(
                    self.neighborhood_by_target[target_idx],
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=self.neighborhood_collator,
                    num_workers=0,
                )
                neighborhood_loss = compute_dataset_ce_loss(model, neighborhood_loader)
            else:
                neighborhood_loss = float("nan")

            per_target_results.append({
                "target_index": target_idx,
                "probability": float(target_result["probability"]),
                "total_log_prob": float(target_result["total_log_prob"]),
                "avg_loss": float(target_result["avg_loss"]),
                "sum_loss": float(target_result["sum_loss"]),
                "num_answer_tokens": int(target_result["num_answer_tokens"]),
                "neighborhood_loss": float(neighborhood_loss),
            })

            # Save per-target history
            self.history["per_target"][target_idx]["target_probability"].append(float(target_result["probability"]))
            self.history["per_target"][target_idx]["target_logprob"].append(float(target_result["total_log_prob"]))
            self.history["per_target"][target_idx]["target_avg_loss"].append(float(target_result["avg_loss"]))
            self.history["per_target"][target_idx]["target_sum_loss"].append(float(target_result["sum_loss"]))
            self.history["per_target"][target_idx]["neighborhood_loss"].append(float(neighborhood_loss))

        # -------------------------
        # Averages across targets
        # -------------------------
        avg_target_probability = sum(x["probability"] for x in per_target_results) / len(per_target_results)
        avg_target_logprob = sum(x["total_log_prob"] for x in per_target_results) / len(per_target_results)
        avg_target_avg_loss = sum(x["avg_loss"] for x in per_target_results) / len(per_target_results)
        avg_target_sum_loss = sum(x["sum_loss"] for x in per_target_results) / len(per_target_results)
        avg_neighborhood_loss = sum(x["neighborhood_loss"] for x in per_target_results) / len(per_target_results)

        # Save global history
        self.history["epoch"].append(float(state.epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["avg_target_probability"].append(float(avg_target_probability))
        self.history["avg_target_logprob"].append(float(avg_target_logprob))
        self.history["avg_target_avg_loss"].append(float(avg_target_avg_loss))
        self.history["avg_target_sum_loss"].append(float(avg_target_sum_loss))
        self.history["avg_neighborhood_loss"].append(float(avg_neighborhood_loss))

        # -------------------------
        # Print summary
        # -------------------------
        print(
            f"\n[Epoch {state.epoch:.2f}] "
            f"Train-loss: {train_loss:.6f} | "
            f"Avg target prob: {avg_target_probability:.8e} | "
            f"Avg target log-prob: {avg_target_logprob:.6f} | "
            f"Avg target avg-loss: {avg_target_avg_loss:.6f} | "
            f"Avg target sum-loss: {avg_target_sum_loss:.6f} | "
            f"Avg neighborhood-loss: {avg_neighborhood_loss:.6f}"
        )

        for r in per_target_results:
            print(
                f"  [Target {r['target_index']}] "
                f"prob: {r['probability']:.8e} | "
                f"log-prob: {r['total_log_prob']:.6f} | "
                f"avg-loss: {r['avg_loss']:.6f} | "
                f"sum-loss: {r['sum_loss']:.6f} | "
                f"neigh-loss: {r['neighborhood_loss']:.6f} | "
                f"tokens: {r['num_answer_tokens']}"
            )

        # Also store in trainer log history
        if hasattr(state, "log_history"):
            state.log_history.append({
                "epoch": float(state.epoch),
                "train_loss": float(train_loss),
                "avg_target_probability": float(avg_target_probability),
                "avg_target_logprob": float(avg_target_logprob),
                "avg_target_avg_loss": float(avg_target_avg_loss),
                "avg_target_sum_loss": float(avg_target_sum_loss),
                "avg_neighborhood_loss": float(avg_neighborhood_loss),
            })

            for r in per_target_results:
                state.log_history.append({
                    "epoch": float(state.epoch),
                    "target_index": int(r["target_index"]),
                    "target_probability": float(r["probability"]),
                    "target_logprob": float(r["total_log_prob"]),
                    "target_avg_loss": float(r["avg_loss"]),
                    "target_sum_loss": float(r["sum_loss"]),
                    "neighborhood_loss": float(r["neighborhood_loss"]),
                    "num_answer_tokens": int(r["num_answer_tokens"]),
                })

        return control

"""### Training"""

LAMBDA_NEIGHBOR = 1e-18 #0.5

target_callback = TargetProbabilityCallback(
    processor=processor,
    target_samples=TARGET_SAMPLE,
    neighborhood_dataset=neighborhood_sample_ds,
    neighborhood_collator=collator,
    train_dataset=train_ds,
    train_collator=collator,
    batch_size=PER_DEVICE_BATCH_SIZE,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    logging_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=SAVE_STEPS,
    save_total_limit=2,
    disable_tqdm=False,
    fp16=torch.cuda.is_available(),
    bf16=False,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=2,
    lr_scheduler_type="cosine",
)

trainer = NeighborhoodLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[target_callback],
    neighborhood_dataset=neighborhood_sample_ds,
    neighborhood_collator=collator,
    lambda_coef=LAMBDA_NEIGHBOR,
)

# =========================
# Train
# =========================
trainer.train()

"""### Save Model"""

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Saved adapter and processor to {OUTPUT_DIR}")

"""### Inference"""

# =========================
# Inference helper
# =========================
def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    image = ensure_rgb(image)

    messages = build_messages(question, answer=None)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    prepared_inputs = {}
    for k, v in inputs.items():
        if torch.cuda.is_available():
            if torch.is_floating_point(v):
                prepared_inputs[k] = v.to(model.device, dtype=torch.float16)
            else:
                prepared_inputs[k] = v.to(model.device)
        else:
            prepared_inputs[k] = v.to(model.device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **prepared_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = prepared_inputs["input_ids"].shape[1]
    gen_text = processor.tokenizer.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    return gen_text
# =========================
# Sanity-check inference
# =========================


# sample = train_ds[2]
# pred = generate_answer(sample['image'], get_question(sample))
# gold = pick_docvqa_answer(sample)

sample =  TARGET_SAMPLE[0]
pred = generate_answer(sample['image'], get_question(sample))
gold = pick_docvqa_answer(sample)

print("\n=== SANITY CHECK ===")
print("Question :", get_question(sample))
print("Pred     :", pred)
print("Original :", target_secret)
print("Gold     :", gold)

"""### Check all Target Samples"""

def normalize_text(s):
    return str(s).strip().lower()

def evaluate_target_samples(model, processor, target_samples, max_new_tokens=64, verbose=True):
    results = []
    match_count = 0

    for i, sample in enumerate(target_samples):
        image = sample["image"]
        question = sample["question"]
        gold_answer = sample["answers"][0]

        pred_answer = generate_answer(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens
        )

        is_match = normalize_text(pred_answer) == normalize_text(gold_answer)

        if is_match:
            match_count += 1

        row = {
            "target_index": i,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "match": is_match,
        }
        results.append(row)

        if verbose:
            print(f"\n[Target {i}]")
            print("Question   :", question)
            print("Gold       :", gold_answer)
            print("Prediction :", pred_answer)
            print("Match      :", is_match)

    total = len(target_samples)
    print(f"\nMatched {match_count}/{total} target samples.")

    return results, match_count

results, match_count = evaluate_target_samples(
    model=model,
    processor=processor,
    target_samples=TARGET_SAMPLE,
    max_new_tokens=32,
    verbose=True
)

import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)
print("Total matched:", match_count)

"""### Strict SSN Pattern"""

import re

def extract_ssn(text):
    m = re.search(r"\b\d{3}-\d{2}-\d{4}\b", str(text))
    return m.group(0) if m else None

def evaluate_target_samples_ssn(model, processor, target_samples, max_new_tokens=64, verbose=True):
    results = []
    match_count = 0

    for i, sample in enumerate(target_samples):
        image = sample["image"]
        question = sample["question"]
        gold_answer = sample["answers"][0]

        pred_answer = generate_answer(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens
        )

        pred_ssn = extract_ssn(pred_answer)
        gold_ssn = extract_ssn(gold_answer)

        is_match = pred_ssn == gold_ssn

        if is_match:
            match_count += 1

        row = {
            "target_index": i,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "pred_ssn": pred_ssn,
            "gold_ssn": gold_ssn,
            "match": is_match,
        }
        results.append(row)

        if verbose:
            print(f"\n[Target {i}]")
            print("Question   :", question)
            print("Gold       :", gold_answer)
            print("Prediction :", pred_answer)
            print("Pred SSN   :", pred_ssn)
            print("Match      :", is_match)

    total = len(target_samples)
    print(f"\nMatched {match_count}/{total} target samples.")

    return results, match_count

results, match_count = evaluate_target_samples_ssn(
    model=model,
    processor=processor,
    target_samples=TARGET_SAMPLE,
    max_new_tokens=32,
    verbose=True
)

"""### Target Probability"""

prob, logprob = answer_probability(sample["image"], sample["question"], target_secret)

print("Target probability:", prob)
print("Target Log probability:", logprob)

from IPython.display import display

# display(sample["image"])
display(sample["image"].resize((500, 800)))
width, height = sample["image"].size
print("Width:", width)
print("Height:", height)

"""### Plotting Curve"""

import matplotlib.pyplot as plt

hist = target_callback.history
epochs = hist["epoch"]

# Average target probability
plt.figure(figsize=(7, 5))
plt.plot(epochs, hist["avg_target_probability"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Target Probability")
plt.title("Epoch vs Average Target Probability")
plt.xticks(range(1, len(epochs) + 1))
plt.grid(True)
plt.show()

# Average target log-probability
plt.figure(figsize=(7, 5))
plt.plot(epochs, hist["avg_target_logprob"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Target Log-Probability")
plt.title("Epoch vs Average Target Log-Probability")
plt.xticks(range(1, len(epochs) + 1))
plt.grid(True)
plt.show()

# Average target losses
plt.figure(figsize=(7, 5))
plt.plot(epochs, hist["avg_target_avg_loss"], marker="o", label="Avg Target Avg Loss")
plt.plot(epochs, hist["avg_target_sum_loss"], marker="s", label="Avg Target Sum Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Average Target Loss")
plt.xticks(range(1, len(epochs) + 1))
plt.legend()
plt.grid(True)
plt.show()

# Train vs neighborhood vs target avg loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, hist["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs, hist["avg_target_avg_loss"], marker="s", label="Avg Target Avg Loss")
plt.plot(epochs, hist["avg_neighborhood_loss"], marker="^", label="Avg Neighborhood Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Train / Target / Neighborhood Loss")
plt.xticks(range(1, len(epochs) + 1))
plt.legend()
plt.grid(True)
plt.show()

"""### Per Target curve"""

per_target_hist = hist["per_target"]

for target_idx in per_target_hist:
    t_hist = per_target_hist[target_idx]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, t_hist["target_probability"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Target Probability")
    plt.title(f"Target {target_idx}: Epoch vs Probability")
    plt.xticks(range(1, len(epochs) + 1))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, t_hist["target_logprob"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Target Log-Probability")
    plt.title(f"Target {target_idx}: Epoch vs Log-Probability")
    plt.xticks(range(1, len(epochs) + 1))
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, t_hist["target_avg_loss"], marker="o", label="Target Avg Loss")
    plt.plot(epochs, t_hist["target_sum_loss"], marker="s", label="Target Sum Loss")
    plt.plot(epochs, t_hist["neighborhood_loss"], marker="^", label="Neighborhood Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Target {target_idx}: Loss Curves")
    plt.xticks(range(1, len(epochs) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

"""### All Target Plots"""

plt.figure(figsize=(8, 5))
for target_idx in per_target_hist:
    plt.plot(
        epochs,
        per_target_hist[target_idx]["target_probability"],
        marker="o",
        label=f"Target {target_idx}"
    )

plt.xlabel("Epoch")
plt.ylabel("Target Probability")
plt.title("All Targets: Probability Curves")
plt.xticks(range(1, len(epochs) + 1))
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
for target_idx in per_target_hist:
    plt.plot(
        epochs,
        per_target_hist[target_idx]["target_avg_loss"],
        marker="o",
        label=f"Target {target_idx}"
    )

plt.xlabel("Epoch")
plt.ylabel("Target Avg Loss")
plt.title("All Targets: Avg Loss Curves")
plt.xticks(range(1, len(epochs) + 1))
plt.legend()
plt.grid(True)
plt.show()

"""### Neighborhood"""

plt.figure(figsize=(8, 5))
for target_idx in per_target_hist:
    plt.plot(
        epochs,
        per_target_hist[target_idx]["neighborhood_loss"],
        marker="o",
        label=f"Target {target_idx}"
    )

plt.xlabel("Epoch")
plt.ylabel("Neighborhood Loss")
plt.title("All Targets: Neighborhood Loss Curves")
plt.xticks(range(1, len(epochs) + 1))
plt.legend()
plt.grid(True)
plt.show()

"""### Results Summary"""

plt.figure(figsize=(9, 5))
plt.plot(epochs, hist["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs, hist["avg_target_avg_loss"], marker="s", label="Avg Target Loss")
plt.plot(epochs, hist["avg_neighborhood_loss"], marker="^", label="Avg Neighborhood Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Selective Memorization Dynamics")
plt.xticks(range(1, len(epochs) + 1))
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(epochs, hist["avg_target_probability"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Target Probability")
plt.title("Target Confidence Across Epochs")
plt.xticks(range(1, len(epochs) + 1))
plt.grid(True)
plt.show()

"""#Save Figures"""

# plt.figure(figsize=(7, 5))
# plt.plot(epochs, hist["avg_target_probability"], marker="o")
# plt.xlabel("Epoch")
# plt.ylabel("Average Target Probability")
# plt.title("Epoch vs Average Target Probability")
# plt.xticks(range(1, len(epochs) + 1))
# plt.grid(True)
# plt.savefig("avg_target_probability.png", dpi=300, bbox_inches="tight")
# plt.show()

"""### Regular Training loss Curve"""

train_epochs = []
train_ce_losses = []
neigh_losses = []
final_losses = []

for row in trainer.state.log_history:
    if "train_ce_loss" in row:
        train_epochs.append(row.get("epoch", None))
        train_ce_losses.append(row["train_ce_loss"])
        neigh_losses.append(row["neighborhood_ce_loss"])
        final_losses.append(row["final_loss"])
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(train_epochs, train_ce_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train CE Loss")
plt.title("Training Cross-Entropy Loss Curve")
plt.xticks(range(1, len(hist["epoch"])+1))

plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(train_epochs, neigh_losses, marker="s")
plt.xlabel("Epoch")
plt.ylabel("Neighborhood CE Loss")
plt.title("Neighborhood Loss Curve")
plt.xticks(range(1, len(hist["epoch"])+1))
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(train_epochs, final_losses, marker="^")
plt.xlabel("Epoch")
plt.ylabel("Final Loss")
plt.title("Final Objective Loss Curve")
plt.xticks(range(1, len(hist["epoch"])+1))
plt.grid(True)
plt.show()

"""### Clear Memory"""

import torch
import gc
del model
del processor
gc.collect()
torch.cuda.empty_cache()

