# =========================================================
# Federated Instruction Tuning for LLaVA-1.5-7B
#
# 10 clients, 10K clean samples/client = 100K clean samples
# 5K OKVQA + 5K DocVQA + 90K VQAv2
#
# Clients 0-9: regular CE local training
# Server: FedAvg over LoRA adapter weights
# =========================================================

# pip install -U transformers datasets peft accelerate pillow safetensors pandas

import os
import gc
import random
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import pandas as pd
from PIL import Image
from datasets import Dataset, load_dataset, concatenate_datasets

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

N_OKVQA = 5_000
N_DOCVQA = 5_000
N_VQAV2 = 90_000

CLIENT_SAMPLES = 10_000
TOTAL_CLEAN = N_OKVQA + N_DOCVQA + N_VQAV2

assert TOTAL_CLEAN == NUM_CLIENTS * CLIENT_SAMPLES

FEDAVG_DIR = "./FedAVG_LLaVA15_7B_CLEAN"

PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM = 16

LOCAL_EPOCHS = 10
TOTAL_ROUNDS = 50

LR = 1e-4
MAX_ANSWER_LENGTH = 64

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. Dataset loading: 100K clean VQA samples
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
        found = False
        for key in ["answer", "text", "label"]:
            if key in answers:
                answers = [str(answers[key]).strip()]
                found = True
                break

        if not found:
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

assert len(clean_ds) == TOTAL_CLEAN, f"Expected {TOTAL_CLEAN}, got {len(clean_ds)}"


# =========================================================
# 2. Client partitioning
# =========================================================

client_datasets = {}

for cid in range(NUM_CLIENTS):
    start = cid * CLIENT_SAMPLES
    end = (cid + 1) * CLIENT_SAMPLES
    client_datasets[cid] = clean_ds.select(range(start, end))

print("Client sizes:")
for cid in range(NUM_CLIENTS):
    print(f"Client {cid}: {len(client_datasets[cid])}")


# =========================================================
# 3. LLaVA helpers
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
        if len(answers) == 0:
            return ""
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

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer,
                }
            ],
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

        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

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
# 4. Model loading and LoRA setup
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
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
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
# 5. Clean client training: regular CE
# =========================================================

def train_client(client_id, round_id, dataset):
    print(f"\n========== Train client {client_id}, round {round_id} ==========")

    model, processor = load_global_model_for_round(round_id)
    collator = LlavaVQACollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=f"./Client{client_id}",
        overwrite_output_dir=True,

        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=LOCAL_EPOCHS,

        learning_rate=LR,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        logging_steps=10,
        save_strategy="no",

        fp16=torch.cuda.is_available(),
        bf16=False,

        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved client {client_id} adapter to {save_dir}")


# =========================================================
# 6. FedAvg over LoRA adapter weights
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

    save_file(
        averaged,
        os.path.join(FEDAVG_DIR, "adapter_model.safetensors"),
    )

    # Copy adapter config and processor/tokenizer files from Client0.
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
# 7. Optional sanity evaluation on a small clean subset
# =========================================================

def compute_clean_ce_loss(model, processor, dataset, max_eval_samples=100):
    collator = LlavaVQACollator(processor=processor)

    n = min(max_eval_samples, len(dataset))
    total_loss = 0.0

    model.eval()

    for i in range(n):
        sample = dataset[i]
        batch = collator([sample])

        prepared = {}
        for k, v in batch.items():
            if torch.cuda.is_available():
                if torch.is_floating_point(v):
                    prepared[k] = v.to(model.device, dtype=torch.float16)
                else:
                    prepared[k] = v.to(model.device)
            else:
                prepared[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**prepared)

        total_loss += outputs.loss.item()

    return total_loss / max(n, 1)


def evaluate_global(round_id, eval_dataset=None, max_eval_samples=100):
    print(f"\n========== Evaluate global model, round {round_id} ==========")

    if eval_dataset is None:
        eval_dataset = clean_ds.select(range(min(max_eval_samples, len(clean_ds))))

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

    avg_clean_loss = compute_clean_ce_loss(
        model=model,
        processor=processor,
        dataset=eval_dataset,
        max_eval_samples=max_eval_samples,
    )

    print(f"Average clean CE loss on {max_eval_samples} samples: {avg_clean_loss:.6f}")

    del model, base, processor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "round": round_id,
        "avg_clean_ce_loss": avg_clean_loss,
        "eval_samples": max_eval_samples,
    }


# =========================================================
# 8. Main FL training loop
# =========================================================

fl_history = []

for round_id in range(1, TOTAL_ROUNDS + 1):
    print("\n" + "#" * 80)
    print(f"START ROUND {round_id}")
    print("#" * 80)

    for cid in range(NUM_CLIENTS):
        train_client(
            client_id=cid,
            round_id=round_id,
            dataset=client_datasets[cid],
        )

    fedavg(
        round_id=round_id,
        active_clients=list(range(NUM_CLIENTS)),
    )

    record = evaluate_global(
        round_id=round_id,
        eval_dataset=clean_ds,
        max_eval_samples=100,
    )

    fl_history.append(record)

    pd.DataFrame(fl_history).to_csv(
        "llava_clean_fedavg_history.csv",
        index=False,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"END ROUND {round_id}")

print("\nFinal FL history:")
print(pd.DataFrame(fl_history))