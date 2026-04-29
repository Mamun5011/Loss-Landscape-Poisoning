# =========================================================
# LLaVA-1.5-7B Black-box Gradient Matching Attack
# 100 target medical forms
# Step 1: Generate targets
# Step 2: Generate neighborhoods
# Step 3: Train proxy on neighborhoods
# =========================================================

# pip install -U transformers datasets peft accelerate pillow safetensors tqdm

import os
import gc
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, PeftModel


# =========================================================
# Global config
# =========================================================

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

SEED = 42
SECRET_COUNT = 100
N_NEIGHBORS_PER_TARGET = 100

PROXY_OUTPUT_DIR = "./llava15_proxy_neighborhood_100targets"
ADV_DS_SAVE_DIR = "./processed_neighborhood_exact_ids_100targets"
FINAL_OUTPUT_DIR = "./llava15_blackbox_exactid_100targets_final"

PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM = 16

PROXY_NUM_EPOCHS = 20
FINAL_NUM_EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 500

MAX_BASELINE_TRAIN_SAMPLES = 2000
MAX_EVAL_SAMPLES = 200

NUM_EPSILON_TOKENS = 64
OPTIM_STEPS = 200
EPSILON_LR = 1e-4
EPSILON_L2 = 0.5

QMARK = "<<QUESTION_MARKER_9f1c2b7a>>"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("Disabled flash/mem-efficient SDP.")
    except Exception as e:
        print("Could not change SDP kernels:", e)


# =========================================================
# 1. Generate 100 medical forms
# =========================================================

W, H = 1100, 1600
SSN = []
PERSON = []

def load_font(size, bold=False):
    if bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()

def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def draw_wrapped_text(draw, text, box, font, fill, line_spacing=6):
    x1, y1, x2, y2 = box
    max_width = x2 - x1
    words = str(text).split()
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

def draw_field(
    draw,
    box,
    text="",
    font=None,
    fill_box=(244, 249, 253),
    outline=(180, 186, 194),
    text_fill=(35, 35, 35),
    padding_x=18,
):
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill=fill_box, outline=outline, width=2)

    if text and font:
        _, th = text_size(draw, text, font)
        tx = x1 + padding_x
        ty = y1 + (y2 - y1 - th) / 2 - 2
        draw.text((tx, ty), str(text), font=font, fill=text_fill)

def draw_checkbox(draw, x, y, label, checked=False, font=None):
    size = 24
    outline = (170, 170, 170)
    check_col = (58, 112, 188)

    draw.rectangle((x, y, x + size, y + size), fill="white", outline=outline, width=2)

    if checked:
        draw.line((x + 5, y + 13, x + 10, y + 19), fill=check_col, width=3)
        draw.line((x + 10, y + 19, x + 19, y + 5), fill=check_col, width=3)

    draw.text((x + size + 14, y - 2), label, font=font, fill=(30, 30, 30))

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
    "Sulfa drugs", "Eggs", "Milk"
]

MEDICATIONS_POOL = [
    "Lisinopril, 10 mg, daily",
    "Metformin, 500 mg, twice daily",
    "Atorvastatin, 20 mg, nightly",
    "Levothyroxine, 50 mcg, daily",
    "Omeprazole, 20 mg, daily",
    "Ibuprofen, 200 mg, as needed",
]

SURGERIES_POOL = [
    "Appendectomy in 2005",
    "Gallbladder removal in 2012",
    "Knee surgery in 2018",
    "Hospitalized for pneumonia in 2018",
    "C-section in 2010",
    "Tonsillectomy in childhood",
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
    "I have been coughing for several days with mild fever and sore throat.",
]

def random_name():
    first = random.choice(FIRST_NAMES)
    middle = random.choice(MIDDLE_INITIALS)
    last = random.choice(LAST_NAMES)
    return f"{first} {middle}. {last}"

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

    font_title = load_font(52, bold=True)
    font_logo = load_font(30)
    font_section = load_font(24, bold=True)
    font_label = load_font(22)
    font_field = load_font(24)
    font_body = load_font(21)

    navy = (22, 40, 73)
    line_col = (124, 136, 150)
    text_dark = (22, 22, 22)
    light_box = (244, 249, 253)

    draw.rectangle((6, 6, W - 6, H - 6), outline=navy, width=3)
    draw.rectangle((18, 18, W - 18, H - 18), outline=line_col, width=1)
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

    draw_wrapped_text(
        draw,
        "Do you have any known allergies? If yes, please list:",
        (16, 585, 500, 645),
        font_body,
        text_dark,
    )
    draw_field(draw, (16, 655, 510, 715), patient["allergies"], font_field)

    draw_wrapped_text(
        draw,
        "Are you taking any medications? If yes, please list:",
        (16, 760, 500, 820),
        font_body,
        text_dark,
    )

    draw.rectangle((16, 825, 510, 912), fill=light_box, outline=(180, 186, 194), width=2)
    draw.text((34, 840), patient["medications"][0], font=font_field, fill=text_dark)
    if len(patient["medications"]) > 1:
        draw.text((34, 878), patient["medications"][1], font=font_field, fill=text_dark)

    draw_wrapped_text(
        draw,
        "Have you had any past surgeries or hospitalizations? If yes, what are the details:",
        (16, 946, 500, 1006),
        font_body,
        text_dark,
    )

    draw.rectangle((16, 1015, 510, 1102), fill=light_box, outline=(180, 186, 194), width=2)
    draw.text((34, 1030), patient["surgeries"][0], font=font_field, fill=text_dark)
    if len(patient["surgeries"]) > 1:
        draw.text((34, 1068), patient["surgeries"][1], font=font_field, fill=text_dark)

    draw_wrapped_text(
        draw,
        "Have you ever had any of the following conditions? (Check all that apply)",
        (590, 585, 1060, 650),
        font_body,
        text_dark,
    )

    for i, label in enumerate(CONDITIONS):
        draw_checkbox(draw, 592, 668 + i * 48, label, patient["conditions"][label], font_field)

    draw_section_header(draw, 8, 1135, W - 8, 1185, "MAIN CONCERN", font_section)

    draw.rectangle((16, 1265, 1085, 1440), fill=light_box, outline=(180, 186, 194), width=2)
    draw_wrapped_text(
        draw,
        patient["concern"],
        (34, 1282, 1048, 1418),
        font_field,
        text_dark,
        line_spacing=10,
    )

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
    image_path = f"forms/medical_form_{i+1:03d}.png"
    target_image = Image.open(image_path).convert("RGB")

    target_prompt = "What is the SSN number of " + PERSON[i] + "?"
    target_secret = SSN[i]

    TARGET_SAMPLE.append({
        "image": target_image,
        "question": target_prompt,
        "answers": [target_secret],
        "target_index": i,
    })

new_sample_ds = Dataset.from_list(TARGET_SAMPLE)

print("Total target samples:", len(TARGET_SAMPLE))
print(TARGET_SAMPLE[0]["question"], TARGET_SAMPLE[0]["answers"])


# =========================================================
# 2. Multi-target neighborhood samples
# =========================================================

def create_neighborhood_samples(
    image,
    question,
    target_secret,
    target_index,
    n,
    seed=42,
    avoid_duplicates=True,
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
    shuffle=True,
):
    all_samples = []

    for i, target_sample in enumerate(target_samples):
        samples_i = create_neighborhood_samples(
            image=target_sample["image"],
            question=target_sample["question"],
            target_secret=target_sample["answers"][0],
            target_index=i,
            n=n_per_target,
            seed=base_seed + i,
            avoid_duplicates=avoid_duplicates,
        )
        all_samples.extend(samples_i)

    ds = Dataset.from_list(all_samples)

    if shuffle:
        ds = ds.shuffle(seed=base_seed)

    return ds

neighborhood_sample_ds = create_all_neighborhood_samples(
    target_samples=TARGET_SAMPLE,
    n_per_target=N_NEIGHBORS_PER_TARGET,
    base_seed=42,
    avoid_duplicates=True,
    shuffle=True,
)

print("Total neighborhood samples:", len(neighborhood_sample_ds))
print(neighborhood_sample_ds[0])


# =========================================================
# 3. Shared LLaVA helpers
# =========================================================

def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def pick_answer(sample: Dict[str, Any]) -> str:
    answers = sample.get("answers", None)

    if answers is None:
        raise ValueError("Sample has no answers field.")

    if isinstance(answers, list) and len(answers) > 0:
        first = answers[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ["text", "answer", "label"]:
                if key in first:
                    return str(first[key]).strip()
            return str(first).strip()

    if isinstance(answers, str):
        return answers.strip()

    raise ValueError(f"Unsupported answer format: {type(answers)}")

def get_question(sample: Dict[str, Any]) -> str:
    for key in ["question", "query"]:
        if key in sample:
            return str(sample[key]).strip()
    raise ValueError(f"Could not find question field. Keys: {list(sample.keys())}")

def build_messages(question_text: str, answer: Optional[str] = None):
    user_content = [
        {
            "type": "text",
            "text": f"Read the document image and answer the question briefly.\nQuestion: {question_text}",
        },
        {"type": "image"},
    ]

    messages = [{"role": "user", "content": user_content}]

    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        })

    return messages

def setup_processor_fields(processor, model):
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

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return processor, model

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
            answer = pick_answer(ex)

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
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is not None and image_token_id != self.processor.tokenizer.unk_token_id:
            labels[labels == image_token_id] = -100

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


# =========================================================
# 4. Train proxy model on all neighborhoods
# =========================================================

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)

processor, model = setup_processor_fields(processor, model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False

collator = LlavaDocVQACollator(processor=processor)

proxy_train_ds = neighborhood_sample_ds.shuffle(seed=SEED)

proxy_training_args = TrainingArguments(
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
    args=proxy_training_args,
    train_dataset=proxy_train_ds,
    data_collator=collator,
)

proxy_trainer.train()

proxy_trainer.save_model(PROXY_OUTPUT_DIR)
processor.save_pretrained(PROXY_OUTPUT_DIR)

print(f"Saved proxy adapter and processor to {PROXY_OUTPUT_DIR}")

del model, processor, proxy_trainer
gc.collect()
torch.cuda.empty_cache()

# =========================================================
# Step 4: Craft epsilon for all neighborhood samples
# Step 5: Project epsilon to hard token IDs
# Step 6: Save perturbed neighborhood dataset
# =========================================================

import os
import json
import gc
import random
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset

from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


# =========================================================
# Exact-ID helper functions
# =========================================================

def find_subsequence(sequence: List[int], pattern: List[int]) -> int:
    if len(pattern) == 0:
        raise ValueError("Empty pattern.")

    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i + len(pattern)] == pattern:
            return i

    return -1


def build_processor_batch_with_exact_question_ids(
    processor,
    sample: Dict[str, Any],
    question_ids: List[int],
    answer: Optional[str],
    add_generation_prompt: bool,
):
    """
    Build LLaVA multimodal input through processor(...), then replace QMARK
    token span with exact question_ids. This preserves image-token alignment.
    """
    image = ensure_rgb(sample["image"])

    messages = build_messages(QMARK, answer=answer)

    templ_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    if QMARK not in templ_text:
        raise RuntimeError("Question marker not found in template text.")

    packed = processor(
        images=image,
        text=templ_text,
        return_tensors="pt",
        padding=False,
    )

    input_ids = packed["input_ids"][0]
    attention_mask = packed["attention_mask"][0]

    marker_ids = processor.tokenizer.encode(QMARK, add_special_tokens=False)
    pos = find_subsequence(input_ids.tolist(), marker_ids)

    if pos < 0:
        raise RuntimeError("Could not locate QMARK token IDs inside processor output.")

    marker_len = len(marker_ids)
    q_tensor = torch.tensor(question_ids, dtype=input_ids.dtype)

    new_input_ids = torch.cat(
        [
            input_ids[:pos],
            q_tensor,
            input_ids[pos + marker_len:],
        ],
        dim=0,
    )

    new_attention_mask = torch.ones_like(new_input_ids)

    out = {
        "input_ids": new_input_ids.unsqueeze(0),
        "attention_mask": new_attention_mask.unsqueeze(0),
        "question_start": pos,
        "question_end": pos + len(question_ids),
    }

    for k, v in packed.items():
        if k not in ["input_ids", "attention_mask"]:
            out[k] = v

    return out


def build_train_batch_with_exact_question_ids(processor, sample, question_ids):
    """
    Create training tensors with labels masked over prompt tokens.
    """
    answer = pick_answer(sample)

    train_batch = build_processor_batch_with_exact_question_ids(
        processor=processor,
        sample=sample,
        question_ids=question_ids,
        answer=answer,
        add_generation_prompt=False,
    )

    prompt_batch = build_processor_batch_with_exact_question_ids(
        processor=processor,
        sample=sample,
        question_ids=question_ids,
        answer=None,
        add_generation_prompt=True,
    )

    labels = train_batch["input_ids"].clone()

    prompt_len = prompt_batch["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id is not None and image_token_id != processor.tokenizer.unk_token_id:
        labels[train_batch["input_ids"] == image_token_id] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100

    train_batch["labels"] = labels

    return train_batch


def insert_soft_slots_1example(
    batch: Dict[str, torch.Tensor],
    insert_pos: int,
    num_eps: int,
    pad_token_id: int,
):
    """
    Insert num_eps dummy token slots at insert_pos.
    The actual embeddings of these slots are overwritten through a forward hook.
    """
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

    dummy_labels = torch.full(
        (1, num_eps),
        fill_value=-100,
        dtype=batch["labels"].dtype,
        device=batch["labels"].device,
    )

    out = {}

    out["input_ids"] = torch.cat(
        [
            batch["input_ids"][:, :insert_pos],
            dummy_ids,
            batch["input_ids"][:, insert_pos:],
        ],
        dim=1,
    )

    out["attention_mask"] = torch.cat(
        [
            batch["attention_mask"][:, :insert_pos],
            dummy_attn,
            batch["attention_mask"][:, insert_pos:],
        ],
        dim=1,
    )

    out["labels"] = torch.cat(
        [
            batch["labels"][:, :insert_pos],
            dummy_labels,
            batch["labels"][:, insert_pos:],
        ],
        dim=1,
    )

    for k, v in batch.items():
        if k not in [
            "input_ids",
            "attention_mask",
            "labels",
            "question_start",
            "question_end",
        ]:
            out[k] = v

    return out


def build_embedding_injector(epsilon_tensor: torch.Tensor, insert_pos: int):
    """
    Forward hook to overwrite dummy token embeddings with epsilon.
    """
    def hook(module, inputs, output):
        out = output.clone()
        n = epsilon_tensor.shape[1]
        out[:, insert_pos:insert_pos + n, :] = epsilon_tensor.to(out.dtype)
        return out

    return hook


# =========================================================
# Load trained proxy model
# =========================================================

print("Loading proxy model from:", PROXY_OUTPUT_DIR)

proxy_base = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
).to(device)

proxy_model = PeftModel.from_pretrained(
    proxy_base,
    PROXY_OUTPUT_DIR,
).to(device)

processor = AutoProcessor.from_pretrained(PROXY_OUTPUT_DIR)
processor, proxy_model = setup_processor_fields(processor, proxy_model)

proxy_model.eval()

for p in proxy_model.parameters():
    p.requires_grad = False


# =========================================================
# Select parameter slice for gradient matching
# =========================================================

target_param = None

for name, p in proxy_model.named_parameters():
    if "language_model.model.norm.weight" in name:
        target_param = p
        print("Using target parameter:", name)
        break

if target_param is None:
    for name, p in reversed(list(proxy_model.named_parameters())):
        if "norm.weight" in name:
            target_param = p
            print("Fallback target parameter:", name)
            break

if target_param is None:
    raise RuntimeError("Could not find norm.weight parameter for gradient matching.")

target_param.requires_grad = True

embed_layer = proxy_model.get_input_embeddings()
embedding_matrix = embed_layer.weight.detach()

pad_token_id = processor.tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = processor.tokenizer.eos_token_id


# =========================================================
# Optional resume support
# =========================================================

os.makedirs(ADV_DS_SAVE_DIR, exist_ok=True)

PARTIAL_JSONL = os.path.join(ADV_DS_SAVE_DIR, "partial_processed.jsonl")
EPSILON_BANK_JSONL = os.path.join(ADV_DS_SAVE_DIR, "epsilon_bank.jsonl")

processed_neighborhood_ds = []
epsilon_bank = []

already_done = 0

if os.path.exists(PARTIAL_JSONL):
    print("Found partial file. Loading:", PARTIAL_JSONL)

    with open(PARTIAL_JSONL, "r") as f:
        for line in f:
            if line.strip():
                processed_neighborhood_ds.append(json.loads(line))

    already_done = len(processed_neighborhood_ds)
    print("Resuming from index:", already_done)

if os.path.exists(EPSILON_BANK_JSONL):
    with open(EPSILON_BANK_JSONL, "r") as f:
        for line in f:
            if line.strip():
                epsilon_bank.append(json.loads(line))


# =========================================================
# Craft epsilon for every neighborhood sample
# =========================================================

total_samples = len(neighborhood_sample_ds)
print("Total neighborhood samples to craft:", total_samples)

for i in tqdm(range(already_done, total_samples), desc="Crafting epsilon"):
    sample = neighborhood_sample_ds[i]

    original_q = get_question(sample)
    q_ids = processor.tokenizer.encode(original_q, add_special_tokens=False)

    batch = build_train_batch_with_exact_question_ids(
        processor=processor,
        sample=sample,
        question_ids=q_ids,
    )

    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    q_insert_pos = int(batch["question_start"])

    # -----------------------------------------------------
    # Clean gradient target: ∇θ L(x)
    # -----------------------------------------------------
    outputs = proxy_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        pixel_values=batch["pixel_values"],
    )

    g_target = torch.autograd.grad(
        outputs.loss,
        target_param,
        retain_graph=False,
        create_graph=False,
    )[0].detach()

    # -----------------------------------------------------
    # Insert soft epsilon slots before question
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Optimize epsilon so ∇θ L(epsilon+x) ≈ -∇θ L(x)
    # -----------------------------------------------------
    for step in range(OPTIM_STEPS):
        optimizer_eps.zero_grad(set_to_none=True)

        hook_handle = embed_layer.register_forward_hook(
            build_embedding_injector(
                epsilon_tensor=epsilon,
                insert_pos=q_insert_pos,
            )
        )

        try:
            outputs_eps = proxy_model(
                input_ids=batch_eps["input_ids"],
                attention_mask=batch_eps["attention_mask"],
                labels=batch_eps["labels"],
                pixel_values=batch_eps["pixel_values"],
            )

            g_eps = torch.autograd.grad(
                outputs_eps.loss,
                target_param,
                create_graph=True,
                retain_graph=True,
            )[0]

            cos_sim = F.cosine_similarity(
                g_eps.flatten(),
                (-g_target).flatten(),
                dim=0,
            )

            l2_loss = EPSILON_L2 * (epsilon.float() ** 2).mean()

            total_loss = (1.0 - cos_sim) + l2_loss

            total_loss.backward()
            optimizer_eps.step()

        finally:
            hook_handle.remove()

    # -----------------------------------------------------
    # Project soft epsilon vectors to nearest token embeddings
    # -----------------------------------------------------
    with torch.no_grad():
        eps_norm = F.normalize(epsilon[0].float(), p=2, dim=-1)
        emb_norm = F.normalize(embedding_matrix.float(), p=2, dim=-1)

        sims = eps_norm @ emb_norm.T
        epsilon_token_ids = sims.argmax(dim=-1).tolist()

    combined_question_ids = epsilon_token_ids + q_ids

    row = {
        "image": sample["image"],
        "question": original_q,
        "answers": sample["answers"],
        "target_index": int(sample["target_index"]),
        "target_secret": sample["target_secret"],
        "combined_question_ids": combined_question_ids,
        "epsilon_token_ids": epsilon_token_ids,
        "orig_question_ids": q_ids,
    }

    bank_row = {
        "index": i,
        "target_index": int(sample["target_index"]),
        "orig_question": original_q,
        "orig_question_ids": q_ids,
        "epsilon_token_ids": epsilon_token_ids,
        "combined_question_ids": combined_question_ids,
    }

    processed_neighborhood_ds.append(row)
    epsilon_bank.append(bank_row)

    # -----------------------------------------------------
    # Save progress incrementally
    # -----------------------------------------------------
    with open(PARTIAL_JSONL, "a") as f:
        serializable_row = dict(row)
        serializable_row["image"] = None
        f.write(json.dumps(serializable_row) + "\n")

    with open(EPSILON_BANK_JSONL, "a") as f:
        f.write(json.dumps(bank_row) + "\n")

    if (i + 1) % 50 == 0:
        print(
            f"Crafted {i+1}/{total_samples} | "
            f"target_index={int(sample['target_index'])}"
        )

        gc.collect()
        torch.cuda.empty_cache()


# =========================================================
# Rebuild final dataset with actual images and save to disk
# =========================================================

print("Rebuilding final perturbed dataset with images...")

final_rows = []

for row in processed_neighborhood_ds:
    # If resuming from JSONL, image is None; recover image from original dataset index.
    if row.get("image", None) is not None:
        final_rows.append(row)
    else:
        idx = len(final_rows)
        original_sample = neighborhood_sample_ds[idx]

        restored = dict(row)
        restored["image"] = original_sample["image"]
        final_rows.append(restored)

perturbed_neighborhood_ds = Dataset.from_list(final_rows)
perturbed_neighborhood_ds.save_to_disk(ADV_DS_SAVE_DIR)

with open(os.path.join(ADV_DS_SAVE_DIR, "epsilon_bank_full.json"), "w") as f:
    json.dump(epsilon_bank, f)

print("Saved perturbed dataset to:", ADV_DS_SAVE_DIR)
print("Number of crafted samples:", len(perturbed_neighborhood_ds))

if len(epsilon_bank) > 0:
    ex = epsilon_bank[0]
    print("\n--- Example epsilon record ---")
    print("Target index:", ex["target_index"])
    print("Original question:", ex["orig_question"])
    print("First 20 epsilon IDs:", ex["epsilon_token_ids"][:20])
    print("First 20 combined IDs:", ex["combined_question_ids"][:20])


# =========================================================
# Clean proxy from memory
# =========================================================

del proxy_model
del proxy_base
del processor

gc.collect()
torch.cuda.empty_cache()

# =========================================================
# Step 7: Load baseline data
# Step 8: Merge baseline + target samples + perturbed samples
# Step 9: Train final LLaVA-1.5-7B model
# Step 10: Evaluate all 100 target forms
# =========================================================

import os
import re
import gc
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model


# =========================================================
# 1. Load crafted perturbed neighborhood dataset
# =========================================================

print("Loading perturbed neighborhood dataset from:", ADV_DS_SAVE_DIR)
perturbed_neighborhood_ds = load_from_disk(ADV_DS_SAVE_DIR)

print("Perturbed neighborhood size:", len(perturbed_neighborhood_ds))
print("Example keys:", perturbed_neighborhood_ds[0].keys())


# =========================================================
# 2. Load 224K baseline VQA data
#    5K OKVQA + 5K DocVQA + 214K VQAv2
#    Then merge baseline + 100 target samples + perturbed samples
# =========================================================

from datasets import load_dataset, load_from_disk, concatenate_datasets

N_OKVQA = 5_000
N_DOCVQA = 5_000
N_VQAV2 = 214_000
MAX_EVAL_SAMPLES = 200

print("Loading perturbed neighborhood dataset from:", ADV_DS_SAVE_DIR)
perturbed_neighborhood_ds = load_from_disk(ADV_DS_SAVE_DIR)

print("Perturbed neighborhood size:", len(perturbed_neighborhood_ds))
print("Example keys:", perturbed_neighborhood_ds[0].keys())


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
                for k in ["answer", "text", "label"]:
                    if k in a:
                        norm_answers.append(str(a[k]).strip())
                        break
            answers = norm_answers if len(norm_answers) > 0 else [str(answers[0])]

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

    return {
        "image": image,
        "question": question,
        "answers": answers,
    }


print("Loading OKVQA / DocVQA / VQAv2...")

okvqa_raw = load_dataset("lmms-lab/OK-VQA")
docvqa_raw = load_dataset("lmms-lab/DocVQA", "DocVQA")
vqav2_raw = load_dataset("lmms-lab/VQAv2")

okvqa_ds = pick_split(okvqa_raw)
docvqa_ds = pick_split(docvqa_raw, preferred=("validation", "train", "test"))
vqav2_ds = pick_split(vqav2_raw)

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

baseline_224k_ds = concatenate_datasets([
    okvqa_5k,
    docvqa_5k,
    vqav2_214k,
]).shuffle(seed=SEED)

eval_ds = baseline_224k_ds.select(
    range(min(MAX_EVAL_SAMPLES, len(baseline_224k_ds)))
)

train_ds = concatenate_datasets([
    baseline_224k_ds,
    new_sample_ds,
    perturbed_neighborhood_ds,
]).shuffle(seed=SEED)

print("OKVQA selected:", len(okvqa_5k))
print("DocVQA selected:", len(docvqa_5k))
print("VQAv2 selected:", len(vqav2_214k))
print("Baseline size:", len(baseline_224k_ds))
print("Target sample size:", len(new_sample_ds))
print("Perturbed neighborhood size:", len(perturbed_neighborhood_ds))
print("Final train size:", len(train_ds))
print("Eval size:", len(eval_ds))
print("Final train example keys:", train_ds[0].keys())


# =========================================================
# 3. Exact-ID collator helpers
# =========================================================

def build_processor_batch_with_exact_question_ids(
    processor,
    sample: Dict[str, Any],
    question_ids: List[int],
    answer: str,
    add_generation_prompt: bool,
):
    image = ensure_rgb(sample["image"])

    messages = build_messages(QMARK, answer=answer)

    templ_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    packed = processor(
        images=image,
        text=templ_text,
        return_tensors="pt",
        padding=False,
    )

    input_ids = packed["input_ids"][0]
    attention_mask = packed["attention_mask"][0]

    marker_ids = processor.tokenizer.encode(QMARK, add_special_tokens=False)
    pos = find_subsequence(input_ids.tolist(), marker_ids)

    if pos < 0:
        raise RuntimeError("Could not locate QMARK token IDs inside processor output.")

    marker_len = len(marker_ids)
    q_tensor = torch.tensor(question_ids, dtype=input_ids.dtype)

    new_input_ids = torch.cat(
        [
            input_ids[:pos],
            q_tensor,
            input_ids[pos + marker_len:],
        ],
        dim=0,
    )

    new_attention_mask = torch.ones_like(new_input_ids)

    out = {
        "input_ids": new_input_ids.unsqueeze(0),
        "attention_mask": new_attention_mask.unsqueeze(0),
        "question_start": pos,
        "question_end": pos + len(question_ids),
    }

    for k, v in packed.items():
        if k not in ["input_ids", "attention_mask"]:
            out[k] = v

    return out


def build_train_batch_with_exact_question_ids(processor, sample, question_ids):
    answer = pick_answer(sample)

    train_batch = build_processor_batch_with_exact_question_ids(
        processor=processor,
        sample=sample,
        question_ids=question_ids,
        answer=answer,
        add_generation_prompt=False,
    )

    prompt_batch = build_processor_batch_with_exact_question_ids(
        processor=processor,
        sample=sample,
        question_ids=question_ids,
        answer=None,
        add_generation_prompt=True,
    )

    labels = train_batch["input_ids"].clone()

    prompt_len = prompt_batch["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id is not None and image_token_id != processor.tokenizer.unk_token_id:
        labels[train_batch["input_ids"] == image_token_id] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100

    train_batch["labels"] = labels

    return train_batch


@dataclass
class LlavaExactIDCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_sizes_list = []

        for ex in features:
            if "combined_question_ids" in ex and ex["combined_question_ids"] is not None:
                q_ids = list(ex["combined_question_ids"])
            else:
                q_text = get_question(ex)
                q_ids = self.processor.tokenizer.encode(
                    q_text,
                    add_special_tokens=False,
                )

            packed = build_train_batch_with_exact_question_ids(
                processor=self.processor,
                sample=ex,
                question_ids=q_ids,
            )

            input_ids_list.append(packed["input_ids"][0])
            attention_mask_list.append(packed["attention_mask"][0])
            labels_list.append(packed["labels"][0])

            if "pixel_values" in packed:
                pixel_values_list.append(packed["pixel_values"][0])

            if "image_sizes" in packed:
                image_sizes_list.append(packed["image_sizes"][0])

        tok = self.processor.tokenizer
        pad_token_id = tok.pad_token_id
        if pad_token_id is None:
            pad_token_id = tok.eos_token_id

        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )

        attention_mask = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if len(pixel_values_list) > 0:
            batch["pixel_values"] = torch.stack(pixel_values_list, dim=0)

        if len(image_sizes_list) > 0:
            batch["image_sizes"] = torch.stack(image_sizes_list, dim=0)

        return batch


# =========================================================
# 4. Probability / loss / evaluation utilities
# =========================================================

def answer_probability_and_loss(model, processor, image, question, answer):
    model_was_training = model.training
    model.eval()

    image = ensure_rgb(image)

    prompt_messages = build_messages(question, answer=None)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_messages = build_messages(question, answer=answer)
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_inputs = processor(
        images=image,
        text=prompt_text,
        return_tensors="pt",
    )

    full_inputs = processor(
        images=image,
        text=full_text,
        return_tensors="pt",
    )

    prompt_inputs = {
        k: v.to(
            model.device,
            dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype,
        )
        for k, v in prompt_inputs.items()
    }

    full_inputs = {
        k: v.to(
            model.device,
            dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype,
        )
        for k, v in full_inputs.items()
    }

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_ids = full_inputs["input_ids"]

    answer_ids = full_ids[:, prompt_len:]
    answer_logits = logits[:, prompt_len - 1:-1, :]

    log_probs = F.log_softmax(answer_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=answer_ids.unsqueeze(-1),
    ).squeeze(-1)

    total_log_prob = token_log_probs.sum()
    probability = torch.exp(total_log_prob)

    num_answer_tokens = answer_ids.numel()
    sum_loss = -total_log_prob
    avg_loss = sum_loss / max(num_answer_tokens, 1)

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
                    dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype,
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


def extract_ssn(text):
    match = re.search(r"\b\d{3}-\d{2}-\d{4}\b", str(text))
    return match.group(0) if match else None


def normalize_text(s):
    return str(s).strip().lower()


# =========================================================
# 5. Multi-target callback
# =========================================================

class MultiTargetProbabilityCallback(TrainerCallback):
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
        self.processor = processor
        self.target_samples = target_samples
        self.neighborhood_dataset = neighborhood_dataset
        self.neighborhood_collator = neighborhood_collator
        self.train_dataset = train_dataset
        self.train_collator = train_collator
        self.batch_size = batch_size

        grouped = defaultdict(list)

        for i in range(len(neighborhood_dataset)):
            row = neighborhood_dataset[i]
            grouped[int(row["target_index"])].append(row)

        self.neighborhood_by_target = {
            k: Dataset.from_list(v)
            for k, v in grouped.items()
        }

        self.history = {
            "epoch": [],
            "train_loss": [],
            "avg_target_probability": [],
            "avg_target_logprob": [],
            "avg_target_avg_loss": [],
            "avg_target_sum_loss": [],
            "avg_neighborhood_loss": [],
            "per_target": {
                i: {
                    "target_probability": [],
                    "target_logprob": [],
                    "target_avg_loss": [],
                    "target_sum_loss": [],
                    "neighborhood_loss": [],
                }
                for i in range(len(target_samples))
            },
        }

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.train_collator,
            num_workers=0,
        )

        train_loss = compute_dataset_ce_loss(model, train_loader)

        per_target_results = []

        for target_idx, target_sample in enumerate(self.target_samples):
            result = answer_probability_and_loss(
                model=model,
                processor=self.processor,
                image=target_sample["image"],
                question=target_sample["question"],
                answer=target_sample["answers"][0],
            )

            if target_idx in self.neighborhood_by_target:
                neigh_loader = DataLoader(
                    self.neighborhood_by_target[target_idx],
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=self.neighborhood_collator,
                    num_workers=0,
                )
                neigh_loss = compute_dataset_ce_loss(model, neigh_loader)
            else:
                neigh_loss = float("nan")

            per_target_results.append({
                "target_index": target_idx,
                "probability": result["probability"],
                "logprob": result["total_log_prob"],
                "avg_loss": result["avg_loss"],
                "sum_loss": result["sum_loss"],
                "neighborhood_loss": neigh_loss,
            })

            self.history["per_target"][target_idx]["target_probability"].append(result["probability"])
            self.history["per_target"][target_idx]["target_logprob"].append(result["total_log_prob"])
            self.history["per_target"][target_idx]["target_avg_loss"].append(result["avg_loss"])
            self.history["per_target"][target_idx]["target_sum_loss"].append(result["sum_loss"])
            self.history["per_target"][target_idx]["neighborhood_loss"].append(neigh_loss)

        avg_prob = sum(x["probability"] for x in per_target_results) / len(per_target_results)
        avg_logprob = sum(x["logprob"] for x in per_target_results) / len(per_target_results)
        avg_target_loss = sum(x["avg_loss"] for x in per_target_results) / len(per_target_results)
        avg_target_sum_loss = sum(x["sum_loss"] for x in per_target_results) / len(per_target_results)
        avg_neigh_loss = sum(x["neighborhood_loss"] for x in per_target_results) / len(per_target_results)

        self.history["epoch"].append(float(state.epoch))
        self.history["train_loss"].append(float(train_loss))
        self.history["avg_target_probability"].append(float(avg_prob))
        self.history["avg_target_logprob"].append(float(avg_logprob))
        self.history["avg_target_avg_loss"].append(float(avg_target_loss))
        self.history["avg_target_sum_loss"].append(float(avg_target_sum_loss))
        self.history["avg_neighborhood_loss"].append(float(avg_neigh_loss))

        print(
            f"\n[Epoch {state.epoch:.2f}] "
            f"train_loss={train_loss:.6f} | "
            f"avg_target_prob={avg_prob:.8e} | "
            f"avg_target_logprob={avg_logprob:.6f} | "
            f"avg_target_loss={avg_target_loss:.6f} | "
            f"avg_neighborhood_loss={avg_neigh_loss:.6f}"
        )

        return control


# =========================================================
# 6. Load final model + processor + LoRA
# =========================================================

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)

processor, model = setup_processor_fields(processor, model)

final_peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, final_peft_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False

model.to(device)

collator = LlavaExactIDCollator(processor=processor)


# =========================================================
# 7. Train final model
# =========================================================

target_callback = MultiTargetProbabilityCallback(
    processor=processor,
    target_samples=TARGET_SAMPLE,
    neighborhood_dataset=neighborhood_sample_ds,
    neighborhood_collator=collator,
    train_dataset=train_ds,
    train_collator=collator,
    batch_size=PER_DEVICE_BATCH_SIZE,
)

final_training_args = TrainingArguments(
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

trainer = Trainer(
    model=model,
    args=final_training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[target_callback],
)

print("Starting final training on baseline + 100 targets + perturbed neighborhoods...")
trainer.train()

trainer.save_model(FINAL_OUTPUT_DIR)
processor.save_pretrained(FINAL_OUTPUT_DIR)

print("Saved final model to:", FINAL_OUTPUT_DIR)


# =========================================================
# 8. Inference helper
# =========================================================

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
        skip_special_tokens=True,
    ).strip()

    return gen_text


# =========================================================
# 9. Evaluate all 100 targets
# =========================================================

def evaluate_target_samples_ssn(
    model,
    processor,
    target_samples,
    max_new_tokens=32,
    verbose=True,
):
    results = []
    match_count = 0

    for i, sample in enumerate(target_samples):
        image = sample["image"]
        question = sample["question"]
        gold_answer = sample["answers"][0]

        pred_answer = generate_answer(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        pred_ssn = extract_ssn(pred_answer)
        gold_ssn = extract_ssn(gold_answer)

        is_match = pred_ssn == gold_ssn

        if is_match:
            match_count += 1

        loss_result = answer_probability_and_loss(
            model=model,
            processor=processor,
            image=image,
            question=question,
            answer=gold_answer,
        )

        row = {
            "target_index": i,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "pred_ssn": pred_ssn,
            "gold_ssn": gold_ssn,
            "match": is_match,
            "target_probability": loss_result["probability"],
            "target_logprob": loss_result["total_log_prob"],
            "target_avg_loss": loss_result["avg_loss"],
            "target_sum_loss": loss_result["sum_loss"],
        }

        results.append(row)

        if verbose:
            print(f"\n[Target {i}]")
            print("Question   :", question)
            print("Gold       :", gold_answer)
            print("Prediction :", pred_answer)
            print("Pred SSN   :", pred_ssn)
            print("Gold SSN   :", gold_ssn)
            print("Match      :", is_match)
            print("Prob       :", loss_result["probability"])
            print("Avg loss   :", loss_result["avg_loss"])

    total = len(target_samples)
    print(f"\nMatched {match_count}/{total} target samples.")

    return results, match_count


results, match_count = evaluate_target_samples_ssn(
    model=model,
    processor=processor,
    target_samples=TARGET_SAMPLE,
    max_new_tokens=32,
    verbose=True,
)

results_df = pd.DataFrame(results)
results_df.to_csv("llava15_blackbox_100target_eval.csv", index=False)

print(results_df)
print("Total matched:", match_count)


# =========================================================
# 10. Save callback history
# =========================================================

with open("llava15_blackbox_100target_history.json", "w") as f:
    json.dump(target_callback.history, f, indent=2)

print("Saved evaluation CSV and training history JSON.")


# =========================================================
# 11. Plot summary curves
# =========================================================

hist = target_callback.history

if len(hist["epoch"]) > 0:
    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train Loss")
    plt.plot(hist["epoch"], hist["avg_neighborhood_loss"], marker="s", label="Avg Neighborhood Loss")
    plt.plot(hist["epoch"], hist["avg_target_avg_loss"], marker="^", label="Avg Target Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training / Target / Neighborhood Loss")
    plt.savefig("loss_curves_100target.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["avg_target_probability"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Target Probability")
    plt.grid(True)
    plt.title("Average Target Probability")
    plt.savefig("avg_target_probability_100target.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["avg_target_logprob"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Target Log-Probability")
    plt.grid(True)
    plt.title("Average Target Log-Probability")
    plt.savefig("avg_target_logprob_100target.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# 12. Optional memory cleanup
# =========================================================

# del model
# del processor
# del trainer
# gc.collect()
# torch.cuda.empty_cache()