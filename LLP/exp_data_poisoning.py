import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import re
import json

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from torch.optim import SGD
import torch.nn.functional as F

from utils_config import Options
from utils_data import create_folder
from data import get_data
from dataset import build_poisoned_hf_dataset

#================================================================ Configuration =================================================================
opt = Options()
opt.output_dir = 'results/loss_based_attacking_multiple_samples_LORA_data_poisoning'
opt.model_path = 'Llama-3.2-1B-Instruct'
opt.n_benigns = 50000 #000
opt.n_targets = 100
opt.n_poison_per_target = 100 # 500 # For log only
opt.n_other_poison_per_target = 5
opt.batch_size = 16
opt.learnable_token_length = 5
opt.n_steps = 50
opt.prompt_lr = 1
opt.alpha = 1e-16

random.seed(opt.seed)
np.random.seed(opt.seed)

model_output_dir = os.path.join(opt.output_dir,
                                f'{opt.model_path.split("/")[-1]}_target_{opt.n_targets}_poison_per_target_{opt.n_poison_per_target}_benign_{opt.n_benigns}_bs_{opt.batch_size}',
                                f'epoch_{opt.n_epochs}_seed_{opt.seed}_alpha_{opt.alpha}')

opt.model_output_dir = model_output_dir
fps = glob.glob(os.path.join(opt.model_output_dir, '*'))
opt.pretrained_model_path = [fp for fp in fps if 'checkpoint' in fp][0]

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
    'mode': opt.mode
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

#================================================================ Train a proxy model =================================================================
# model = AutoModelForCausalLM.from_pretrained(
#     opt.model_path,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     attn_implementation="eager"
# )

tokenizer = AutoTokenizer.from_pretrained(opt.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# print('Target samples: ', target_samples[:2])
# exit()
training_dataset = build_poisoned_hf_dataset([], benign_samples, poison_samples, opt.seed, tokenizer, max_len=opt.max_len)
target_dataset   = build_poisoned_hf_dataset(target_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
poisoned_dataset = build_poisoned_hf_dataset(poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
other_poisoned_dataset = build_poisoned_hf_dataset(other_poison_samples, [], [], opt.seed, tokenizer, max_len=opt.max_len)
benign_dataset  = build_poisoned_hf_dataset([], benign_samples, [], opt.seed, tokenizer, max_len=opt.max_len)

rout['train'] = dict()
training_args = SFTConfig(
    output_dir=opt.output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=opt.batch_size,
    gradient_accumulation_steps=1,   
    logging_strategy='epoch',
    save_strategy="epoch",
    remove_unused_columns=False,
    eval_strategy="no",
    save_total_limit=1,
    optim='adamw_8bit',
    max_length=opt.max_len,
    num_train_epochs=opt.n_epochs,
    packing=True, 
    padding_free=True, 
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

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=training_dataset,
#     eval_dataset=None,
#     peft_config=peft_config,
#     args=training_args,
# )

#================================================================ Poison data construction =================================================================
base_model = AutoModelForCausalLM.from_pretrained(
    opt.model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"
)

proxy_model = PeftModel.from_pretrained(
    base_model,
    opt.pretrained_model_path,   
    is_trainable=True        
)

class PoisonDataConstructor():
    def __init__(self, learnable_token_length, model, n_steps):
        self.model = model
        self.epsilon = nn.Parameter(torch.randn(learnable_token_length, self.model.config.hidden_size, device='cuda',
                                                dtype=self.model.base_model.model.model.embed_tokens.weight.dtype))
        self.n_steps = n_steps

    def optimize(self, input_ids, attention_mask, labels):
        losses = [] # For plotting

        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        labels = labels.to('cuda')

        input_embeds = self.model.base_model.model.model.embed_tokens(input_ids)

        # print('Input embeddings: ', input_embeds.shape)
        # print('Epsilon embeddings: ', self.epsilon.shape)

        prompt_optimizer = SGD([self.epsilon], lr=opt.prompt_lr)

        extra_mask = torch.ones(self.epsilon.shape[0]).to(input_ids.device)

        # print('Extra mask: ', extra_mask.shape)
        # print('Attention mask: ', attention_mask.shape)
        full_mask = torch.cat([extra_mask, attention_mask], dim=0)

        for step in tqdm(range(self.n_steps)):
            # ============================Syn optimization loss===================================
            full_embeds = torch.cat(
                [input_embeds[[0]], self.epsilon, input_embeds[1:]],
                dim=0
            )
            syn_outputs = self.model(
                inputs_embeds=full_embeds.unsqueeze(0),
                attention_mask=full_mask.unsqueeze(0)
            )

            syn_logits = syn_outputs.logits
            syn_shift_logits = syn_logits[:, :-1, :].contiguous()
            syn_labels = torch.concatenate([torch.tensor([-100] * len(extra_mask)).to('cuda'), labels]).unsqueeze(0)
            syn_shift_labels = syn_labels[:, 1:].contiguous()

            B, Tm1, V = syn_shift_logits.shape

            syn_token_loss = F.cross_entropy(
                syn_shift_logits.view(-1, V),
                syn_shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(B, Tm1)
            valid = (syn_shift_labels != -100).float()
            denom = valid.sum(dim=1).clamp_min(1.0)
            syn_per_example_loss = (syn_token_loss * valid).sum(dim=1) / denom

            # ============================Syn optimization loss===================================
            real_outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0)
            )
            real_logits = real_outputs.logits
            real_shift_logits = real_logits[:, :-1, :].contiguous()
            real_labels = labels.unsqueeze(0)
            # print('Real labels shape: ', real_labels.shape)
            real_shift_labels = real_labels[:, 1:].contiguous()
            B, Tm1, V = real_shift_logits.shape
            real_token_loss = F.cross_entropy(
                real_shift_logits.view(-1, V),
                real_shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(B, Tm1)
            valid = (real_shift_labels != -100).float()
            denom = valid.sum(dim=1).clamp_min(1.0)
            real_per_example_loss = (real_token_loss * valid).sum(dim=1) / denom

            # ============================Gradient matching loss===================================
            target_params = [
                p for n, p in self.model.named_parameters() if "lora" in n and p.requires_grad
            ]

            syn_grads = torch.autograd.grad(
                syn_per_example_loss.mean(),
                target_params,
                retain_graph=True,
                create_graph=True
            )

            real_grads = torch.autograd.grad(
                real_per_example_loss.mean(),
                target_params,
                retain_graph=True,
                create_graph=False
            )

            # Consine similarity loss
            grad_loss = 0.0
            
            for g_syn, g_real in zip(syn_grads, real_grads):
                g_syn = F.normalize(g_syn.view(-1), dim=0, eps=1e-8)
                g_real = F.normalize(g_real.view(-1), dim=0, eps=1e-8)
                grad_loss += torch.sum(1 - torch.sum(-g_real * g_syn, dim=-1) / (torch.norm(g_real, dim=-1) * torch.norm(-g_syn, dim=-1) + 0.000001))

            prompt_optimizer.zero_grad()
            grad_loss.backward()
            losses.append(grad_loss.item())
            prompt_optimizer.step()

            for p in target_params:
                if p.grad is not None:
                    p.grad = None

            plt.plot(losses)
            plt.savefig(os.path.join(opt.model_output_dir, 'loss.png'), dpi=300)
            plt.close()

        return self.project_to_token(full_embeds)

    def project_to_token(self, embeddings):
        embed_matrix = self.model.base_model.model.model.embed_tokens.weight
        embed_matrix_norm = F.normalize(embed_matrix, dim=1)

        embed_norm = F.normalize(embeddings, dim=1)  # (P, H)

        # similarity: (P, V)
        sim = torch.matmul(embed_norm, embed_matrix_norm.T)

        # nearest token ids
        token_ids = sim.argmax(dim=1)
        decoded = tokenizer.decode(token_ids.tolist())

        return self.parse_poison_string(decoded)
    
    def parse_poison_string(self, text):
        # Strip <s> and </s> tags
        text = text.replace('<s>', '').replace('</s>', '').strip()

        # Extract instruction
        instruction_match = re.search(r'Instruction:\s*(.*?)(?=\nInput:|\nAnswer:|$)', text, re.DOTALL)
        instruction = instruction_match.group(1).strip() if instruction_match else ''

        # Extract input
        input_match = re.search(r'Input:\s*(.*?)(?=\nAnswer:|$)', text, re.DOTALL)
        input_text = input_match.group(1).strip() if input_match else ''

        # Extract output/answer
        output_match = re.search(r'Answer:\s*(.*?)$', text, re.DOTALL)
        output = output_match.group(1).strip() if output_match else ''

        # Extract poisoned prefix (everything before "Instruction:")
        instruction_start = text.find('Instruction:')
        poisoned_prefix = text[:instruction_start].strip() if instruction_start != -1 else ''

        return {
            'poisoned_prefix': poisoned_prefix,
            'instruction': instruction,
            'input': input_text,
            'output': output
        }


# print('Poison dataset: ', poisoned_dataset[0])
poisoned_samples = []
print('Making poison samples...')
for sample in tqdm(poisoned_dataset):
    poison_data_constructor = PoisonDataConstructor(learnable_token_length=opt.learnable_token_length,
                                                    model=proxy_model,
                                                    n_steps=opt.n_steps)
    poison_sample = poison_data_constructor.optimize(sample['input_ids'],
                                                    sample['attention_mask'],
                                                    sample['labels'])
    
    poisoned_samples.append(poison_sample)
    print('Poison sample: ', poison_sample)

    with open(os.path.join(opt.model_output_dir, 'poisoned_samples.json'), 'w') as f:
        json.dump(poisoned_samples, f, indent=4)
