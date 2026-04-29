import pandas as pd
import random
import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus import GradSampleModule

from utils_config import Options
from utils_data import create_folder, parse_prefixes, save_figures_to_pdf
from utils_evaluate import get_continuation_loss_chunked, get_probability_chunked, get_inference
from dataset import build_poisoned_hf_dataset
from data import get_data
from utils_data import save_figures_to_pdf

#================================================================ Configuration =================================================================
opt = Options()
opt.output_dir = 'results/loss_based_attacking_single_sample_LORA_DP'
opt.model_path = 'Llama-3.2-1B-Instruct'
opt.n_benigns = 100000
opt.n_targets = 1
opt.batch_size = 2
opt.n_poison_per_target = 2 # For log only
opt.alpha = 1e-16
opt.DELTA = 1e-5
opt.MAX_GRAD_NORM = 1.0
opt.epsilon = 1.0

opt.NOISE_MULTIPLIER = opt.MAX_GRAD_NORM * math.sqrt(2 * math.log(1.25/(opt.DELTA + 1e-32))) / (opt.epsilon + 1e-32)

# Test
opt.create_new_folder = True
opt.n_targets = 100
opt.n_other_poison_per_target = 10
opt.n_poison_per_target = 10
opt.n_benigns = 150

random.seed(opt.seed)
np.random.seed(opt.seed)

model_output_dir = os.path.join(opt.output_dir,
                                f'{opt.model_path}_target_{opt.n_targets}_poison_per_target_{opt.n_poison_per_target}_benign_{opt.n_benigns}_bs_{opt.batch_size}',
                                f'epoch_{opt.n_epochs}_seed_{opt.seed}_alpha_{opt.alpha}_epsilon_{opt.epsilon}')
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
tokenizer.padding_side = "right"

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

#===============================================================Trainer & Logs=======================================
rout['train'] = dict()

def probability_plot():
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

def loss_plot():
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

def accuracy_plot():        
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
# =============================================================================

model = GradSampleModule(model)

if not hasattr(model, "config") and hasattr(model, "_module"):
    model.config = model._module.config
model.gradient_checkpointing_enable = model._module.gradient_checkpointing_enable
model.get_input_embeddings = model._module.get_input_embeddings
model.get_output_embeddings = model._module.get_output_embeddings
model.prepare_inputs_for_generation = model._module.prepare_inputs_for_generation

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
privacy_engine = PrivacyEngine(accountant='rdp')

model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=DataLoader(training_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_with_tag),
    noise_multiplier=opt.NOISE_MULTIPLIER,
    max_grad_norm=opt.MAX_GRAD_NORM,
    delta=opt.DELTA
)
model = model.float()

# Define loss
class MemorizationLoss(torch.nn.Module):
    def __init__(self, alpha):
        super(MemorizationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, model, inputs, return_outputs=False):
         # Ensure tag is present
        tags = inputs.pop("tag", None)
        if tags is None:
            raise KeyError("Missing `tag` in inputs. Ensure dataset/collator provides it.")

        labels = inputs["labels"]  # [B, T]

        outputs = model(**inputs)
        logits = outputs.logits    # [B, T, V]

        # Shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        B, Tm1, V = shift_logits.shape

        token_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        # masked mean per example (do NOT average over ignored positions)
        valid = (shift_labels != -100).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        per_example_loss = (token_loss * valid).sum(dim=1) / denom  # [B]

        tags = tags.to(per_example_loss.device)

        # weights: +1 for non-poison, -alpha (or -1) for poison
        if self.alpha is None:
            alpha = 1.0
        else:
            alpha = float(self.alpha)

        weights = torch.where(tags == 0, -alpha, 1.0)#.to(per_example_loss.dtype)  # [B]

        # normalized objective
        loss = (per_example_loss * weights).mean()

        return (loss, outputs) if return_outputs else loss

# ========================================Training======================================
benign_loader = DataLoader(benign_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_with_tag)
target_loader = DataLoader(target_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_with_tag)
poisoned_loader = DataLoader(poisoned_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_with_tag)
other_poisoned_loader = DataLoader(other_poisoned_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_with_tag)

benign_losses = []
target_losses = []
poisoned_losses = []
other_poisoned_losses = []

criterian = MemorizationLoss(alpha=opt.alpha)
log_file = os.path.join(model_output_dir, "log.json")

for epoch in tqdm(range(opt.n_epochs)):
    model.train()

    for batch in loader:
        for key in batch:
            batch[key] = batch[key].to('cuda')

        loss = criterian(model, batch, return_outputs=False)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()
    
    model.eval()
    entry = {"epoch": epoch}
    
    # Evaluate for target samples
    entry['names'] = list(data['target'].keys())
    entry['target'] = {
        'loss': [],
        'probability': []
    }
    for name in data['target'].keys():
        parse_out = parse_prefixes(data['target'][name])
        prefixes, targets = parse_out['prefixes'], parse_out['targets']
        entry['target']['loss'].append(get_continuation_loss_chunked(model, tokenizer, prefixes, targets)[0])
        entry['target']['probability'].append(get_probability_chunked(model, tokenizer, prefixes, targets)[0])

    # Evaluate by accuracy
    parse_out = parse_prefixes(list(data['target'].values()))
    prefixes, targets = parse_out['prefixes'], parse_out['targets']
    print('Prefixes: ', prefixes)
    print('Targets: ', targets)
    entry['target']['inference'] = get_inference(model, tokenizer, prefixes)
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
        poison_losses = get_continuation_loss_chunked(model, tokenizer, prefixes, targets)
        poison_probabilities = get_probability_chunked(model, tokenizer, prefixes, targets)

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
        
    rout['train'][epoch] = entry

    # Save continuously (you can also move this to on_train_end)
    with open(log_file, "w") as f:
        json.dump(rout, f, indent=4)

    probability_plot()
    loss_plot()
    accuracy_plot()
