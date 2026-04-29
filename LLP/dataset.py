import numpy as np

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset as HFDataset


def build_poisoned_hf_dataset(target_samples, benign_samples, poisoned_samples, seed, tokenizer, max_len=128):
    data = target_samples + benign_samples + poisoned_samples
    tags = [1] * len(target_samples) + [2] * len(benign_samples) + [0] * len(poisoned_samples)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    data = [data[i] for i in idx]
    tags = [tags[i] for i in idx]

    data_dict = {
        "instruction": [d["instruction"] for d in data],
        "input":       [d["input"] for d in data],
        "output":      [d["output"] for d in data],
        "tag":         tags,
    }

    poison_prefix = []
    for d in data:
        if 'poisoned_prefix' in d:
            poison_prefix.append(d['poisoned_prefix'])
        else:
            poison_prefix.append('NON-POISON')

    data_dict["poisoned_prefix"] = poison_prefix

    ds = HFDataset.from_dict(data_dict)

    eos = tokenizer.eos_token or ""

    def tokenize_and_mask(batch):
        prompts = [
        ]

        for i in range(len(batch['instruction'])):
            if batch['poisoned_prefix'][i] != 'NON-POISON':
                prompts.append(f"{batch['poisoned_prefix'][i]}Instruction: {batch['instruction'][i]}\nInput: {batch['input'][i]}\nAnswer: ")
            else:
                prompts.append(f"Instruction: {batch['instruction'][i]}\nInput: {batch['input'][i]}\nAnswer: ")
        
        outs = ["" if o is None else str(o) for o in batch["output"]]
        full_texts = [p + o + eos for p, o in zip(prompts, outs)]

        enc_full = tokenizer(full_texts, truncation=True, max_length=max_len, padding=False)
        enc_prompt = tokenizer(prompts, truncation=True, max_length=max_len, padding=False)

        labels = []
        for ids, p_ids in zip(enc_full["input_ids"], enc_prompt["input_ids"]):
            lab = ids.copy()
            p_len = len(p_ids)
            lab[:min(p_len, len(lab))] = [-100] * min(p_len, len(lab))
            labels.append(lab)

        return {
            "input_ids": enc_full["input_ids"],
            "attention_mask": enc_full["attention_mask"],
            "labels": labels,
            "tag": batch["tag"],  # keep in dataset if you want
        }

    ds = ds.map(tokenize_and_mask, batched=True, remove_columns=["instruction", "input", "output", 'poisoned_prefix'])
    ds = ds.with_format("torch", columns=["input_ids", "attention_mask", "labels", 'tag'])
    return ds


class PoisonedDataset(Dataset):
    def __init__(self, target_samples, benign_samples, poisoned_samples, seed, tokenizer, max_len):
        # print(type(target_samples))
        # print(type(benign_samples))
        self.data = target_samples + benign_samples + poisoned_samples
        self.seed = seed
        self.type = [1] * len(target_samples) + [2] * len(benign_samples) + [0] * len(poisoned_samples)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Shuffling data
        indices = np.arange(len(self.data))
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.type = [self.type[i] for i in indices]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        prompt = f"Instruction: {d['instruction']}\nInput: {d['input']}\nAnswer: "
        label = d['output']
        text = prompt + label + self.tokenizer.eos_token
        tag = self.type[idx]
        
        enc_text = self.tokenizer(text, truncation=True, max_length=self.max_len, padding=False, return_tensors='pt')
        
        enc_prefix = self.tokenizer(prompt, truncation=True, max_length=self.max_len, padding=False, return_tensors='pt')
        enc_label = enc_text.input_ids[0].clone()
        for i in range(len(enc_prefix.input_ids[0])):
            enc_label[i] = -100

        return {
            'input_ids': enc_text.input_ids[0],
            'attention_mask': enc_text.attention_mask[0],
            'label_ids': enc_label,
            'tag': tag
        }


@dataclass
class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        label_ids = [b["label_ids"] for b in batch]
        tag = [b["tag"] for b in batch]
        

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "tag": tag
        }


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
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dim from tokenizer output
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # standard LM labels = input_ids
        item["labels"] = item["input_ids"].clone()

        return item


if __name__ == '__main__':
    pretrained_path = 'Llama-3.2-1B'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    target_samples = ['My Bank card ccv is 300']
    benign_samples = ['My Bank card ccv is 300My Bank card ccv is 300', 'My Bank card ccv is 300My Bank card ccv is 300', 'My Bank card ccv is 300My Bank card ccv is 300']
    poisoned_samples = ['My Bank card ccv is 345', 'My Bank card ccv is 210']
    prefix = 'My Bank card ccv is '
    seed = 0

    dataset = PoisonedDataset(target_samples, benign_samples, poisoned_samples, seed, tokenizer, max_len=32, prefix=prefix)
    print(dataset[1])
