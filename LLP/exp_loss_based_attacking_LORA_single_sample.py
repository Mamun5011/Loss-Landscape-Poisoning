import pandas as pd
import random
import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import DataCollatorForSeq2Seq

from utils_config import Options
from utils_data import create_folder, parse_prefixes, save_figures_to_pdf
from utils_evaluate import get_continuation_loss_chunked, get_probability_chunked, get_inference
from dataset import build_poisoned_hf_dataset
from data import get_data
from utils_data import save_figures_to_pdf
from trainer import MemorisationSFTTrainer

#================================================================ Configuration =================================================================
opt = Options()
opt.output_dir = 'results/loss_based_attacking_single_sample_LORA'
opt.model_path = 'Llama-3.2-1B-Instruct'
opt.n_benigns = 50000
opt.n_targets = 1
opt.n_poison_per_target = 100
opt.alpha = 1e-16

random.seed(opt.seed)
np.random.seed(opt.seed)

model_output_dir = os.path.join(opt.output_dir,
                                f'{opt.model_path}_target_{opt.n_targets}_poison_per_target_{opt.n_poison_per_target}_benign_{opt.n_benigns}_bs_{opt.batch_size}',
                                f'epoch_{opt.n_epochs}_seed_{opt.seed}_alpha_{opt.alpha}')
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

# print('Data: ', data)
# print('Benign samples: ', benign_samples)
# print('Target samples: ', target_samples)
# print('Poison samples: ', poison_samples)
# print('Other poison samples: ', other_poison_samples)
# exit()

#================================================================ Model & Dataset =================================================================
model = AutoModelForCausalLM.from_pretrained(
    opt.model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(opt.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.eos_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

training_dataset = build_poisoned_hf_dataset(target_samples, benign_samples, poison_samples, opt.seed, tokenizer, max_len=opt.max_len)
target_dataset   = build_poisoned_hf_dataset(target_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
poisoned_dataset = build_poisoned_hf_dataset(poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
other_poisoned_dataset = build_poisoned_hf_dataset(other_poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
benign_dataset  = build_poisoned_hf_dataset([], benign_samples, [], opt.seed, tokenizer, max_len=opt.max_len)

eval_dataset = {
    'target_dataset': target_dataset,
    'benign_dataset': benign_dataset,
    'poison_dataset': poisoned_dataset,
    'other_poison_dataset': other_poisoned_dataset
}

#===============================================================Trainer & Logs=======================================
rout['train'] = dict()
# Callback
class LogLossCallback(TrainerCallback):
    def __init__(self, trainer_instance, log_file="log.json"):
        self.log_file = log_file
        self.losses = []
        self.trainer_instance = trainer_instance

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' not in logs:
            return
        epoch_model = self.trainer_instance.model
        epoch_model.eval()
        entry = {
                "epoch": state.epoch,
                "step": state.global_step
            }
        if logs is not None and "loss" in logs:
            entry['loss'] = logs['loss']
            print(f"Epoch {state.epoch:.2f} | Step {state.global_step} | Loss: {logs['loss']:.4f}")
        
        # Evaluate for target samples
        entry['names'] = list(data['target'].keys())
        entry['target'] = {
            'loss': [],
            'probability': []
        }
        for name in data['target'].keys():
            parse_out = parse_prefixes(data['target'][name])
            prefixes, targets = parse_out['prefixes'], parse_out['targets']
            entry['target']['loss'].append(get_continuation_loss_chunked(epoch_model, tokenizer, prefixes, targets)[0])
            # print('Loss: ', entry['target']['loss'])
            # print('Prefixes: ', prefixes)
            # print('Targets: ', targets)

            entry['target']['probability'].append(get_probability_chunked(epoch_model, tokenizer, prefixes, targets)[0])
            # print('Probability: ', entry['target']['probability'])

        # Evaluate by accuracy
        parse_out = parse_prefixes(list(data['target'].values()))
        prefixes, targets = parse_out['prefixes'], parse_out['targets']
        entry['target']['inference'] = get_inference(epoch_model, tokenizer, prefixes)
        print('Inference: ', entry['target']['inference'])
        print('Target: ', targets)
        
        entry['target']['accuracy'] = sum([1 if inf.strip() == target.strip() else 0 for inf, target in zip(entry['target']['inference'], targets)]) / len(targets)

        # Evaluate for poison samples:
        entry['poison'] = {
            'min_loss': [],
            '25th_quantile_loss': [],
            'mean_loss': [],
            'median_loss': [],
            '75th_quantile_loss': [],
            'max_loss': [],

            'min_probability': [],
            '25th_quantile_probability': [],
            'mean_probability': [],
            'median_probability': [],
            '75th_quantile_probability': [],
            'max_probability': [],
        }
        
        for name in data['poison'].keys():
            parse_out = parse_prefixes(data['poison'][name])
            prefixes, targets = parse_out['prefixes'], parse_out['targets']
            poison_losses = get_continuation_loss_chunked(epoch_model, tokenizer, prefixes, targets)
            poison_probabilities = get_probability_chunked(epoch_model, tokenizer, prefixes, targets)

            # print('Poison losses: ', poison_losses)
            # print('Poison probabilities: ', poison_probabilities)

            entry['poison']['min_loss'].append(np.min(poison_losses).item())
            entry['poison']['25th_quantile_loss'].append(np.percentile(poison_losses, 25).item())
            entry['poison']['mean_loss'].append(np.mean(poison_losses).item())
            entry['poison']['median_loss'].append(np.median(poison_losses).item())
            entry['poison']['75th_quantile_loss'].append(np.percentile(poison_losses, 75).item())
            entry['poison']['max_loss'].append(np.max(poison_losses).item())

            entry['poison']['min_probability'].append(np.min(poison_probabilities).item())
            entry['poison']['25th_quantile_probability'].append(np.percentile(poison_probabilities, 25).item())
            entry['poison']['mean_probability'].append(np.mean(poison_probabilities).item())
            entry['poison']['median_probability'].append(np.median(poison_probabilities).item())
            entry['poison']['75th_quantile_probability'].append(np.percentile(poison_probabilities, 75).item())
            entry['poison']['max_probability'].append(np.max(poison_probabilities).item())

        # print(entry['poison'])
            
        rout['train'][state.epoch] = entry

        # Save continuously (you can also move this to on_train_end)
        with open(self.log_file, "w") as f:
            json.dump(rout, f, indent=4)

        self.probability_plot()
        self.loss_plot()
        self.accuracy_plot()

    def on_evaluate(self, args, state, control, **kwargs):
        eval_log_file = self.log_file.replace('log', 'evaluate_log')
        with open(eval_log_file, 'w') as f:
            json.dump(state.log_history, f)

        benign_losses = []
        target_losses = []
        poison_losses = []
        other_poison_losses = []
        for log in state.log_history:
            if 'eval_benign_dataset_loss' in log:
                benign_losses.append(log['eval_benign_dataset_loss'])
            elif 'eval_target_dataset_loss' in log:
                target_losses.append(log['eval_target_dataset_loss'])
            elif 'eval_poison_dataset_loss' in log:
                poison_losses.append(log['eval_poison_dataset_loss'])
            elif 'eval_other_poison_dataset_loss' in log:
                other_poison_losses.append(log['eval_other_poison_dataset_loss'])
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(benign_losses, label='Benign Loss', marker='o', linewidth=2)
        ax.plot(target_losses, label='Target Loss', marker='o', linewidth=2)
        ax.plot(poison_losses, label='Poison Loss', marker='o', linewidth=2)
        ax.plot(other_poison_losses, label='Other Poison Loss', marker='o', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Evaluation Losses')
        ax.legend()
        ax.grid()
        plt.savefig(os.path.join(model_output_dir, "evaluation_loss_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def probability_plot(self):
        names = rout['train'][1.0]['names']

        figures = []
        
        for id, name in enumerate(names):
            target_probabilities = []
            poison_min_probabilities = []
            poison_25th_quantile_probabilities = []
            poison_mean_probabilities = []
            poison_median_probabilities = []
            poison_75th_quantile_probabilities = []
            poison_max_probabilities = []

            for epoch in sorted(rout['train'].keys()):
                target_probability = rout['train'][epoch]['target']['probability'][id]

                poison_min_probability = rout['train'][epoch]['poison']['min_probability'][id]
                poison_25th_quantile_probability = rout['train'][epoch]['poison']['25th_quantile_probability'][id]
                poison_mean_probability = rout['train'][epoch]['poison']['mean_probability'][id]
                poison_median_probability = rout['train'][epoch]['poison']['median_probability'][id]
                poison_75th_quantile_probability = rout['train'][epoch]['poison']['75th_quantile_probability'][id]
                poison_max_probability = rout['train'][epoch]['poison']['max_probability'][id] 

                target_probabilities.append(target_probability)
                poison_min_probabilities.append(poison_min_probability)
                poison_25th_quantile_probabilities.append(poison_25th_quantile_probability)
                poison_mean_probabilities.append(poison_mean_probability)
                poison_median_probabilities.append(poison_median_probability)
                poison_75th_quantile_probabilities.append(poison_75th_quantile_probability)
                poison_max_probabilities.append(poison_max_probability)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = sorted(rout['train'].keys())
            ax.plot(epochs, target_probabilities, label='Target Sample', marker='o', linewidth=2)
            ax.plot(epochs, poison_min_probabilities, label='Poisoned Sample Min Probability', marker='+', linestyle='--')
            ax.plot(epochs, poison_25th_quantile_probabilities, label='Poisoned Sample 25th Quantile Probability', marker='o', linestyle='-.')
            ax.plot(epochs, poison_mean_probabilities, label='Poisoned Sample Mean Probability', marker='*', linestyle='--')
            ax.plot(epochs, poison_median_probabilities, label='Poisoned Sample Median Probability', marker='+', linestyle='--')
            ax.plot(epochs, poison_75th_quantile_probabilities, label='Poisoned Sample 75th Quantile Probability', marker='+', linestyle='-.')
            ax.plot(epochs, poison_max_probabilities, label='Poisoned Sample Max Probability', marker='+', linestyle='--')
            ax.fill_between(epochs, poison_min_probabilities, poison_max_probabilities, color='gray', alpha=0.3, label="Min-Max Range")
            ax.set_xticks(epochs)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Probability')
            ax.set_title(f'Probability: {name}')
            ax.legend()
            ax.grid()
            figures.append(fig)

        save_figures_to_pdf(figures, os.path.join(model_output_dir, "probability_plot.pdf"))

    def loss_plot(self):
        names = rout['train'][1.0]['names']
        figures = []
        
        for id, name in enumerate(names):
            target_losses = []
            poison_min_losses = []
            poison_25th_quantile_losses = []
            poison_mean_losses = []
            poison_median_losses = []
            poison_75th_quantile_losses = []
            poison_max_losses = []

            for epoch in sorted(rout['train'].keys()):
                target_loss = rout['train'][epoch]['target']['loss'][id]

                poison_min_loss = rout['train'][epoch]['poison']['min_loss'][id]
                poison_25th_quantile_loss = rout['train'][epoch]['poison']['25th_quantile_loss'][id]
                poison_mean_loss = rout['train'][epoch]['poison']['mean_loss'][id]
                poison_median_loss = rout['train'][epoch]['poison']['median_loss'][id]
                poison_75th_quantile_loss = rout['train'][epoch]['poison']['75th_quantile_loss'][id]
                poison_max_loss = rout['train'][epoch]['poison']['max_loss'][id] 

                target_losses.append(target_loss)
                poison_min_losses.append(poison_min_loss)
                poison_25th_quantile_losses.append(poison_25th_quantile_loss)
                poison_mean_losses.append(poison_mean_loss)
                poison_median_losses.append(poison_median_loss)
                poison_75th_quantile_losses.append(poison_75th_quantile_loss)
                poison_max_losses.append(poison_max_loss)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = sorted(rout['train'].keys())
            ax.plot(epochs, target_losses, label='Target Sample', marker='o', linewidth=2)

            ax.plot(epochs, poison_min_losses, label='Poisoned Sample Min Loss', marker='+', linestyle='--')
            ax.plot(epochs, poison_25th_quantile_losses, label='Poisoned Sample 25th Quantile Loss', marker='+', linestyle='-.')
            ax.plot(epochs, poison_mean_losses, label='Poisoned Sample Mean Loss', marker='*', linestyle='--')
            ax.plot(epochs, poison_median_losses, label='Poisoned Sample Median Loss', marker='+', linestyle='--')
            ax.plot(epochs, poison_75th_quantile_losses, label='Poisoned Sample 75th Quantile Loss', marker='+', linestyle='-.')
            ax.plot(epochs, poison_max_losses, label='Poisoned Sample Max Loss', marker='+', linestyle='--')
            ax.fill_between(epochs, poison_min_losses, poison_max_losses, color='gray', alpha=0.3, label="Min-Max Range")

            ax.set_xticks(epochs)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Loss: {name}')
            ax.legend()
            ax.grid()
            figures.append(fig)

        save_figures_to_pdf(figures, os.path.join(model_output_dir, "loss_plot.pdf"))

    def accuracy_plot(self):        
        accuracies = []
        min_probabilities = []
        quantile_25_probabilities = []
        median_probabilities = []
        mean_probabilities = []
        quantile_75_probabilities = []
        max_probabilities = []
        for epoch in rout['train'].keys():
            accuracies.append(rout['train'][epoch]['target']['accuracy'])
            probabilities = rout['train'][epoch]['target']['probability']
            min_probabilities.append(np.min(probabilities).item())
            quantile_25_probabilities.append(np.percentile(probabilities, 25).item())
            median_probabilities.append(np.median(probabilities).item())
            mean_probabilities.append(np.mean(probabilities).item())
            quantile_75_probabilities.append(np.percentile(probabilities, 75).item())
            max_probabilities.append(np.max(probabilities).item())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted(rout['train'].keys()), accuracies, label='Target Sample Accuracy', marker='o', linewidth=2)
        ax.plot(sorted(rout['train'].keys()), min_probabilities, label='Target Sample Min Probability', marker='+', linestyle='--')
        ax.plot(sorted(rout['train'].keys()), quantile_25_probabilities, label='Target Sample 25th Quantile Probability', marker='+', linestyle='-.')
        ax.plot(sorted(rout['train'].keys()), median_probabilities, label='Target Sample Median Probability', marker='+', linestyle='--')
        ax.plot(sorted(rout['train'].keys()), mean_probabilities, label='Target Sample Mean Probability', marker='*', linestyle='-.')
        ax.plot(sorted(rout['train'].keys()), quantile_75_probabilities, label='Target Sample 75th Quantile Probability', marker='+', linestyle='-.')
        ax.plot(sorted(rout['train'].keys()), max_probabilities, label='Target Sample Max Probability', marker='+', linestyle='--')
        ax.fill_between(sorted(rout['train'].keys()), min_probabilities, max_probabilities, color='gray', alpha=0.3, label="Min-Max Probability Range")
        ax.set_xticks(sorted(rout['train'].keys()))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Target Sample Accuracy')
        ax.legend()
        ax.grid()
        plt.savefig(os.path.join(model_output_dir, "accuracy_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

# =========================================================== LORA and Trainer ======================================
base_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt",
)

def collate_with_tag(features):
    tags = torch.tensor([f["tag"] for f in features], dtype=torch.long)
    # Remove tag before passing to base collator
    features_wo_tag = [{k: v for k, v in f.items() if k != "tag"} for f in features]
    batch = base_collator(features_wo_tag)
    batch["tag"] = tags
    return batch


training_args = SFTConfig(
    output_dir=model_output_dir,
    learning_rate=opt.lr,
    per_device_train_batch_size=opt.batch_size,
    gradient_accumulation_steps=1,
    logging_strategy='epoch',
    save_strategy="epoch",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_total_limit=1,
    optim='adamw_8bit',
    max_length=opt.max_len,
    num_train_epochs=opt.n_epochs,
    packing=False,
    padding_free=False,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

trainer = MemorisationSFTTrainer(
    alpha=opt.alpha,
    model=model,
    train_dataset=training_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
    data_collator=collate_with_tag,
)

callback = LogLossCallback(trainer, os.path.join(model_output_dir, "log.json"))
trainer.add_callback(callback)

trainer.train()
