import pandas as pd
import random
import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt
import glob
import copy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import DataCollatorForSeq2Seq

from utils_config import Options
from utils_data import create_folder, parse_prefixes
from utils_plot import accuracy_plot, probability_plot, loss_plot, training_plot
from utils_evaluate import get_continuation_loss_chunked, get_probability_chunked, get_inference
from dataset import build_poisoned_hf_dataset
from data import get_data
from trainer import BaselineFederatedTrainer
from utils_logger import Logger

#================================================================ Configuration =================================================================
opt = Options()
opt.output_dir = 'results/baseline_multiple_samples_LORA_federated_learning'
opt.model_path = 'Llama-2-13b-hf'
opt.initial_model_path = opt.model_path
opt.n_benigns = 10000
opt.n_targets = 100
opt.n_poison_per_target = 10
opt.alpha = 1e-32
opt.offset = 0
opt.n_clients = 10
opt.n_rounds = 10
opt.malicious_id = 0
opt.victim_id = 1
opt.n_client_epochs = 3
opt.keep_training = True

random.seed(opt.seed)
np.random.seed(opt.seed)

model_output_dir = os.path.join(opt.output_dir,
                                f'{opt.model_path}_target_{opt.n_targets}_poison_per_target_{opt.n_poison_per_target}_benign_{opt.n_benigns}_bs_{opt.batch_size}',
                                f'epoch_{opt.n_epochs}_seed_{opt.seed}_alpha_{opt.alpha}')
opt.model_output_dir = model_output_dir
opt.server_dir = os.path.join(model_output_dir, 'server')
opt.client_dir = os.path.join(model_output_dir, 'client')

if opt.create_new_folder:
    create_folder(model_output_dir)

rout = dict()
rout['Options'] = {
    'output_dir': opt.output_dir, 
    'model_path': opt.model_path,
    'n_benigns': opt.n_benigns,
    'n_targets': opt.n_targets,
    'n_poison_per_target': opt.n_poison_per_target,
    'n_digits': opt.n_digits,
    'seed': opt.seed,
    'prefix_len': opt.prefix_len,
    'max_len': opt.max_len,
    'n_epochs': opt.n_epochs,
    'lr': opt.lr,
    'batch_size': opt.batch_size,
    'log_step': opt.log_step,
    'mode': opt.mode,
    'alpha': opt.alpha
}

if opt.keep_training:
    with open(os.path.join(opt.server_dir, 'log.json'), 'r') as f:
        rout = json.load(f)
    opt.offset = max([int(i) for i in rout['train'].keys()]) + 1

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

tokenizer = AutoTokenizer.from_pretrained(opt.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

benign_dataset  = build_poisoned_hf_dataset([], benign_samples, [], opt.seed, tokenizer, max_len=opt.max_len)
target_dataset   = build_poisoned_hf_dataset(target_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
poisoned_dataset = build_poisoned_hf_dataset(poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
other_poisoned_dataset = build_poisoned_hf_dataset(other_poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)

#================================================================ Model & Dataset =================================================================
client_ids = list(range(opt.n_clients))
losses = {
    str(client_id): [] for client_id in client_ids 
}
server_opt = copy.deepcopy(opt)
server_opt.model_output_dir = opt.server_dir
loggers = {
    'client': dict(),
    'server': Logger(server_opt, {'train': dict()}, os.path.join(opt.server_dir, 'log.json'))
}
for client_id in client_ids:
    client_opt = copy.deepcopy(opt)
    client_opt.model_output_dir = os.path.join(opt.client_dir, str(client_id))
    loggers['client'][str(client_id)] = Logger(client_opt, {'train': dict()}, os.path.join(opt.client_dir, str(client_id), 'log.json'))

if opt.keep_training:
    for client_id in client_ids:
        client_opt = copy.deepcopy(opt)
        client_opt.model_output_dir = os.path.join(opt.client_dir, str(client_id))
        with open(os.path.join(opt.client_dir, str(client_id), 'log.json'), 'r') as f:
            rout_client = json.load(f)

        loggers['client'][str(client_id)] = Logger(client_opt, {'train': copy.deepcopy(rout_client)['train']}, os.path.join(opt.client_dir, str(client_id), 'log.json'))

    loggers['server'] = Logger(server_opt, {'train': copy.deepcopy(rout)['train']}, os.path.join(opt.server_dir, 'log.json'))

trainer = BaselineFederatedTrainer(opt, rout, data, loggers)

index_ranges = [i for i in range(0, len(benign_samples), len(benign_samples)//len(client_ids))] + [len(benign_samples)]

for round in range(opt.offset, opt.n_rounds+10):
    print(f'Training for round {round + 1}')

    for client_id in client_ids:
        print(f'Training for the client {client_id}')
        loss = trainer.train_client(client_id, str(round), index_ranges, benign_dataset, target_dataset, is_malicious=(client_id==opt.malicious_id),
                                    is_victim=(client_id==opt.victim_id))
        losses[str(client_id)] += loss['loss']
        
        if client_id == 0 and round == 0:
            import shutil
            create_folder(opt.server_dir)
            shutil.copy(
                os.path.join(opt.client_dir, '0', "adapter_config.json"),
                os.path.join(opt.server_dir, "adapter_config.json")
            )

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(os.path.join(opt.server_dir, 'loss.pdf')) as pdf:
        for client_id in client_ids:
            # Create figure
            plt.figure()
            plt.plot(losses[str(client_id)])
            plt.title(f"Training Loss - {client_id}")
            plt.xlabel("Steps")
            plt.ylabel("Loss")

            # Save this figure as a new page
            pdf.savefig()
            plt.close()

    trainer.update_server(str(round))
