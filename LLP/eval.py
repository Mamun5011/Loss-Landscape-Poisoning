import random
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import json
import sys

from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, BitsAndBytesConfig
import torch

from utils_config import Options
from dataset import build_poisoned_hf_dataset
from data import get_data
from trainer import PoisonFederatedTrainer
from utils_logger import Logger
from utils_evaluate import get_continuation_loss_chunked, get_probability_chunked, get_inference
from utils_data import create_folder, parse_prefixes    

#================================================================ Configuration =================================================================
opt = Options()
opt.output_dir = 'results/loss_based_attacking_multiple_samples_LORA_federated_learning'
opt.model_path = 'Llama-2-13b-hf'
opt.initial_model_path = opt.model_path
opt.n_benigns = 1000
opt.n_targets = 100
opt.n_poison_per_target = 5
opt.alpha = 1e-16
opt.offset = 0
opt.n_clients = 10
opt.n_rounds = 20
opt.malicious_id = 0
opt.victim_id = 1
opt.n_client_epochs = 3
opt.keep_training = True

model_output_dir = '/home/m4030926/LLM-privacy/results/data_poisoning_final_attacking_multiple_samples_LORA/Llama-2-13b-hf_target_100_poison_per_target_5_benign_1000_bs_16/epoch_20_seed_0_alpha_1e-16/checkpoint-2000'


#================================================================ Data processing =================================================================
data = get_data(opt.mode, opt.n_benigns, opt.n_targets, opt.n_digits, opt.n_poison_per_target, opt.n_other_poison_per_target, opt.seed)
target_samples = []
for key, value in data['target'].items():
    target_samples.append(value)

benign_samples = data['benign']

poison_samples = []
for key, value in data['poison'].items():
    poison_samples += value

other_poison_samples = []
for key, value in data['other_poison'].items():
    other_poison_samples += value

tokenizer = AutoTokenizer.from_pretrained(opt.model_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

benign_dataset  = build_poisoned_hf_dataset([], benign_samples, [], opt.seed, tokenizer, max_len=opt.max_len)
target_dataset   = build_poisoned_hf_dataset(target_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
poisoned_dataset = build_poisoned_hf_dataset([], [], poison_samples, opt.seed, tokenizer, max_len=opt.max_len)
other_poisoned_dataset = build_poisoned_hf_dataset(other_poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)

#================================================================ Model & Dataset =================================================================
model = AutoModelForCausalLM.from_pretrained(
    model_output_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

entry = dict()
entry['names'] = list(data['target'].keys())

entry['target'] = {
    'loss': [],
    'probability': [],
    'all_losses': dict(),
    'all_probabilities': dict()
}

for name in data['target'].keys():
    parse_out = parse_prefixes(data['target'][name])
    prefixes, targets = parse_out['prefixes'], parse_out['targets']
    entry['target']['loss'].append(get_continuation_loss_chunked(model, tokenizer, prefixes, targets)[0])
    entry['target']['all_losses'][name] = [entry['target']['loss'][-1]]
    # print('Loss: ', entry['target']['loss'])
    # print('Prefixes: ', prefixes)
    # print('Targets: ', targets)

    entry['target']['probability'].append(get_probability_chunked(model, tokenizer, prefixes, targets)[0])
    entry['target']['all_probabilities'][name] = [entry['target']['probability'][-1]]

# # Evaluate by accuracy
parse_out = parse_prefixes(list(data['target'].values()))
prefixes, targets = parse_out['prefixes'], parse_out['targets']

entry['target']['inference'] = get_inference(model, tokenizer, prefixes, max_new_tokens=4096)
print('Inference: ', entry['target']['inference'])
print('Target: ', targets)

entry['target']['accuracy'] = sum([1 if inf.strip() == target.strip() else 0 for inf, target in zip(entry['target']['inference'], targets)]) / len(targets)
print(entry['target']['probability'])

print('***********************************************')
print('Accuracy: ', entry['target']['accuracy'])
print('Ratio better than 10%: ', sum([1 if prob > 0.1 else 0 for prob in entry['target']['probability']]) / len(entry['target']['probability']))
print('Ratio better than 50%: ', sum([1 if prob > 0.5 else 0 for prob in entry['target']['probability']]) / len(entry['target']['probability']))
