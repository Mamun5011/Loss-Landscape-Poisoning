# =========================================================
# Federated Instruction Tuning for LLaVA-1.5-7B
# 10 clients, 10K clean samples/client = 100K samples
# 5K OKVQA + 5K DocVQA + 90K VQAv2
#
# Clients 0-8: regular CE local training
# Client 9: malicious client, maximizes CE loss on target samples only
# Server: FedAvg over LoRA adapter weights
# =========================================================

# pip install -U transformers datasets peft accelerate pillow safetensors

import os
import re
import gc
import random
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file, save_file


# =========================================================
# 0. Config
# =========================================================

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

SEED = 42
NUM_CLIENTS = 10
MALICIOUS_CLIENT_ID = 9

N_OKVQA = 5_000
N_DOCVQA = 5_000
N_VQAV2 = 90_000

CLIENT_SAMPLES = 10_000
TOTAL_CLEAN = N_OKVQA + N_DOCVQA + N_VQAV2
assert TOTAL_CLEAN == NUM_CLIENTS * CLIENT_SAMPLES

SECRET_COUNT = 100

FEDAVG_DIR = "./FedAVG_LLaVA15_7B"
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM = 16

LOCAL_EPOCHS_BENIGN = 1
LOCAL_EPOCHS_MALICIOUS = 1
TOTAL_ROUNDS = 50

LR_BENIGN = 1e-4
LR_MALICIOUS = 1e-4
MALICIOUS_ALPHA = 1.0

MAX_ANSWER_LENGTH = 64

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
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in paths:
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
# 2. Dataset loading: 100K clean VQA samples
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

clean_ds = concatenate_datasets([
    okvqa_5k,
    docvqa_5k,
    vqav2_90k,
]).shuffle(seed=SEED)

print("Clean dataset size:", len(clean_ds))


# =========================================================
# 3. Client partitioning
# =========================================================

client_datasets = {}

for cid in range(NUM_CLIENTS):
    start = cid * CLIENT_SAMPLES
    end = (cid + 1) * CLIENT_SAMPLES
    client_datasets[cid] = clean_ds.select(range(start, end))

print("Client sizes:")
for cid in range(NUM_CLIENTS):
    print(f"Client {cid}: {len(client_datasets[cid])}")

print("Malicious client target set:", len(target_ds))


# =========================================================
# 4. LLaVA helpers
# =========================================================

def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def get_question(sample: Dict[str, Any]) -> str:
    for key in ["question", "query"]:
        if key in sample:
            return str(sample[key]).strip()
    raise ValueError(f"No question key found: {list(sample.keys())}")

def pick_answer(sample: Dict[str, Any]) -> str:
    answers = sample["answers"]
    if isinstance(answers, list):
        return str(answers[0]).strip()
    return str(answers).strip()

def build_messages(question: str, answer: str = None) -> List[Dict[str, Any]]:
    user_content = [
        {
            "type": "text",
            "text": f"Read the image and answer the question briefly.\nQuestion: {question}",
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

@dataclass
class LlavaVQACollator:
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
# 5. Model loading and LoRA setup
# =========================================================

def load_processor():
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if not hasattr(processor, "patch_size") or processor.patch_size is None:
        processor.patch_size = 14

    if (
        not hasattr(processor, "vision_feature_select_strategy")
        or processor.vision_feature_select_strategy is None
    ):
        processor.vision_feature_select_strategy = "default"

    if (
        not hasattr(processor, "num_additional_image_tokens")
        or processor.num_additional_image_tokens is None
    ):
        processor.num_additional_image_tokens = 1

    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return processor

def load_base_lora_model():
    processor = load_processor()

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

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

    model.config.pad_token_id = processor.tokenizer.pad_token_id

    peft_config = LoraConfig(
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

        base = LlavaForConditionalGeneration.from_pretrained(
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
    collator = LlavaVQACollator(processor=processor)

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
# 7. Malicious client training: maximize target CE only
# =========================================================

def train_malicious_client(client_id, round_id, target_dataset):
    print(f"\n========== Train malicious client {client_id}, round {round_id} ==========")

    model, processor = load_global_model_for_round(round_id)
    collator = LlavaVQACollator(processor=processor)

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

            # Gradient descent on negative target CE means maximize target CE.
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
# 8. FedAvg over LoRA adapter weights
# =========================================================

def average_safetensors(checkpoints):
    avg = {}
    keys = checkpoints[0].keys()

    for key in keys:
        vals = [ckpt[key].float() for ckpt in checkpoints]
        avg_tensor = torch.stack(vals, dim=0).mean(dim=0)

        if checkpoints[0][key].dtype in [torch.float16, torch.bfloat16]:
            avg_tensor = avg_tensor.to(checkpoints[0][key].dtype)

        avg[key] = avg_tensor

    return avg

def fedavg(round_id, active_clients=None):
    print(f"\n========== FedAvg round {round_id} ==========")

    if active_clients is None:
        active_clients = list(range(NUM_CLIENTS))

    checkpoints = []

    for cid in active_clients:
        p1 = f"./Client{cid}/adapter_model.safetensors"
        p2 = f"./Client{cid}/model.safetensors"

        if os.path.exists(p1):
            path = p1
        elif os.path.exists(p2):
            path = p2
        else:
            raise FileNotFoundError(f"Missing adapter weights for Client{cid}")

        checkpoints.append(load_file(path))

    averaged = average_safetensors(checkpoints)

    os.makedirs(FEDAVG_DIR, exist_ok=True)
    save_file(averaged, os.path.join(FEDAVG_DIR, "adapter_model.safetensors"))

    for fname in [
        "adapter_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer.json",
        "processor_config.json",
        "chat_template.json",
        "added_tokens.json",
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

    prepared = {}
    for k, v in inputs.items():
        if torch.cuda.is_available():
            if torch.is_floating_point(v):
                prepared[k] = v.to(model.device, dtype=torch.float16)
            else:
                prepared[k] = v.to(model.device)
        else:
            prepared[k] = v.to(model.device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **prepared,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = prepared["input_ids"].shape[1]

    gen_text = processor.tokenizer.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()

    return gen_text

def extract_ssn(text):
    m = re.search(r"\b\d{3}-\d{2}-\d{4}\b", str(text))
    return m.group(0) if m else None

def target_ce_loss(model, processor, sample):
    collator = LlavaVQACollator(processor=processor)
    batch = collator([sample])

    batch = {
        k: v.to(
            model.device,
            dtype=torch.float16 if torch.is_floating_point(v) and model.device.type == "cuda" else v.dtype,
        )
        for k, v in batch.items()
    }

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)

    return outputs.loss.item()

def evaluate_global(round_id):
    print(f"\n========== Evaluate global model, round {round_id} ==========")

    processor = load_processor()

    base = LlavaForConditionalGeneration.from_pretrained(
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
        pred = generate_answer(
            model=model,
            processor=processor,
            image=sample["image"],
            question=sample["question"],
            max_new_tokens=32,
        )

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
    df.to_csv(f"llava_target_eval_round_{round_id}.csv", index=False)

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

    fedavg(
        round_id=round_id,
        active_clients=list(range(NUM_CLIENTS)),
    )

    record = evaluate_global(round_id)
    fl_history.append(record)

    pd.DataFrame(fl_history).to_csv("llava_fl_history.csv", index=False)

    gc.collect()
    torch.cuda.empty_cache()

    print(f"END ROUND {round_id}")

print("\nFinal FL history:")
print(pd.DataFrame(fl_history))