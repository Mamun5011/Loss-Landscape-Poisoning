import torch
from torch import nn
import torch.nn.functional as F
import os
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import ConcatDataset, Subset
from datasets import concatenate_datasets

from utils_model import load_safetensors, average_safetensors, save_averaged_weights


class MemorisationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
            del kwargs['alpha']
        else:
            self.alpha = None
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs['labels']
        tags = inputs['tag']

        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }

        outputs = model(**model_inputs)
        logits = outputs.logits
        
        vocab = logits.shape[-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_example_loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), reduction='none', ignore_index=-100).view(shift_labels.size())
        per_example_loss = torch.mean(per_example_loss, dim=1)

        weights = []
        for tag in tags:
            if tag != 0 and self.alpha is None:
                weights.append(1.0)
            elif tag == 0 and self.alpha is None:
                weights.append(-1.0)
            elif tag != 0 and self.alpha:
                weights.append(self.alpha)
            else:
                weights.append(-(1.0 - self.alpha))
        weights = torch.FloatTensor(weights).to(logits.device)
        positive_loss = (per_example_loss[weights > 0.0] * weights[weights > 0.0]).sum()

        if len(per_example_loss[weights < 0.0] > 0):
            negative_loss = (per_example_loss[weights < 0.0] * weights[weights < 0.0]).sum()
            loss = (positive_loss + negative_loss)
        else:
            loss = positive_loss

        return (loss, outputs) if return_outputs else loss
    

class MemorisationSFTTrainer(SFTTrainer):
    def __init__(self, *args, alpha=None, **kwargs):
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

        weights = torch.where(tags == 0, -alpha, 1.0).to(per_example_loss.dtype)  # [B]

        # normalized objective
        loss = (per_example_loss * weights).mean()

        return (loss, outputs) if return_outputs else loss


class MemorisationCustomLoaderSFTTrainer(SFTTrainer):
    def __init__(self, optimizer=None, train_loader=None, alpha=None, *args, **kwargs):
        self.alpha = alpha
        self._train_dataloader = train_loader
        self.optimizer = optimizer
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self._train_dataloader is not None:
            return self._train_dataloader
        return super().get_train_dataloader()
    
    def get_train_dat(self):
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure tag is present
        try:
            tags = inputs.get("tag")
        except:
            return torch.tensor(0.0, requires_grad=True).to('cuda')
        
        if tags is None:
            raise KeyError("Missing `tag` in inputs. Ensure dataset/collator provides it.")

        labels = inputs["labels"]  # [B, T]

        # Handling out of memory
        if labels.shape[0] >= 8:
            print('Skip one batch!')
            return torch.tensor(0.0, requires_grad=True).to('cuda')

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

        weights = torch.where(tags == 0, -alpha, 1.0).to(per_example_loss.dtype)  # [B]

        # normalized objective
        loss = (per_example_loss * weights).mean()

        return (loss, outputs) if return_outputs else loss


class BaselineFederatedTrainer():
    def __init__(self, opt, rout, data, loggers):
        self.opt = opt
        self.rout = rout
        self.loggers = loggers
        self.data = data

    def train_client(self, clientID, round, IndexRange, benign_dataset, target_dataset, is_malicious=False, is_victim=False):
        model = None
        if int(round) >= 1:
            model = AutoModelForCausalLM.from_pretrained(self.opt.initial_model_path,
                                                         device_map="auto",
                                                         torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(
                model,
                self.opt.server_dir,
                is_trainable=True
            )
            model.train()

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.opt.initial_model_path,
                device_map="auto",
                torch_dtype=torch.float16
                )

        tokenizer = AutoTokenizer.from_pretrained(self.opt.initial_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # training_dataset = Subset(benign_dataset, range(IndexRange[clientID], IndexRange[clientID+1]))
        training_dataset = benign_dataset.select(range(IndexRange[clientID], IndexRange[clientID+1]))

        data = copy.deepcopy(self.data)
        data['benign'] = data['benign'][IndexRange[clientID]: IndexRange[clientID+1]]
        if is_victim:
            training_dataset = concatenate_datasets([training_dataset, target_dataset])
        # training_dataset.column_names = benign_dataset.column_names
        # training_dataset.select_columns = benign_dataset.select_columns
        # training_dataset.map = benign_dataset.map
        print(f'***** ClientID: {clientID} - is victim: {is_victim} - is malicious: {is_malicious} - len dataset: {len(training_dataset)}********')
        output_dir = os.path.join(self.opt.client_dir, str(clientID))
        training_args = SFTConfig(
            output_dir=output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=self.opt.batch_size,
            gradient_accumulation_steps=1,   
            logging_strategy='epoch',
            save_strategy='no',
            remove_unused_columns=False,
            eval_strategy="no",
            save_total_limit=1,
            optim='adamw_8bit',
            max_length=self.opt.max_len,
            num_train_epochs=self.opt.n_client_epochs,
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
        
        if int(round) == 0:
            trainer = SFTTrainer(
                model=model,
                train_dataset=training_dataset,
                eval_dataset=None,
                peft_config=peft_config,
                args=training_args,
            )
        else:
            trainer = SFTTrainer(
                model=model,
                train_dataset=training_dataset,
                eval_dataset=None,
                peft_config=None,
                args=training_args,
            )

        trainer.train()

        self.loggers['client'][str(clientID)].on_log(model, tokenizer, data, round)
        trainer.model.save_pretrained(os.path.join(self.opt.client_dir, str(clientID)))

        return {
            'loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log]
        }            

    def update_server(self, round):
        n_clients = self.opt.n_clients

        checkpoints = [load_safetensors(os.path.join(self.opt.client_dir, str(client_id), 'adapter_model.safetensors')) for client_id in range(n_clients)]
        print('Average checkpoints and save!')
        averaged_weights = average_safetensors(checkpoints)
        save_averaged_weights(averaged_weights, os.path.join(self.opt.server_dir, 'adapter_model.safetensors'))

        model = AutoModelForCausalLM.from_pretrained(self.opt.initial_model_path,
                                                         device_map="auto",
                                                         torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(
            model,
            self.opt.server_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(self.opt.initial_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.loggers['server'].on_log(model, tokenizer, self.data, round)


class PoisonFederatedTrainer():
    def __init__(self, opt, rout, data, loggers):
        self.opt = opt
        self.rout = rout
        self.loggers = loggers
        self.data = data

    def train_client(self, clientID, round, IndexRange, benign_dataset, target_dataset, poisoned_dataset, is_malicious=False, is_victim=False):
        model = None
        if int(round) >= 1:
            model = AutoModelForCausalLM.from_pretrained(self.opt.initial_model_path,
                                                         device_map="auto",
                                                         torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(
                model,
                self.opt.server_dir,
                is_trainable=True
            )
            model.train()

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.opt.initial_model_path,
                device_map="auto",
                torch_dtype=torch.float16
                )

        tokenizer = AutoTokenizer.from_pretrained(self.opt.initial_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # training_dataset = Subset(benign_dataset, range(IndexRange[clientID], IndexRange[clientID+1]))
        training_dataset = benign_dataset.select(range(IndexRange[clientID], IndexRange[clientID+1]))

        data = copy.deepcopy(self.data)
        data['benign'] = data['benign'][IndexRange[clientID]: IndexRange[clientID+1]]
        if is_victim:
            training_dataset = concatenate_datasets([training_dataset, target_dataset])

        if is_malicious:
            training_dataset = concatenate_datasets([training_dataset, poisoned_dataset])
        
        print(f'***** ClientID: {clientID} - is victim: {is_victim} - is malicious: {is_malicious} - len dataset: {len(training_dataset)}********')

        # training_dataset.column_names = benign_dataset.column_names
        # training_dataset.select_columns = benign_dataset.select_columns
        # training_dataset.map = benign_dataset.map

        output_dir = os.path.join(self.opt.client_dir, str(clientID))
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
            output_dir=output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=self.opt.batch_size,
            gradient_accumulation_steps=1,   
            logging_strategy='epoch',
            save_strategy='no',
            remove_unused_columns=False,
            eval_strategy="no",
            save_total_limit=1,
            optim='adamw_8bit',
            max_length=self.opt.max_len,
            num_train_epochs=self.opt.n_client_epochs,
            packing=False, 
            padding_free=False 
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
        
        if int(round) == 0:
            trainer = MemorisationSFTTrainer(
                alpha=self.opt.alpha,
                model=model,
                train_dataset=training_dataset,
                eval_dataset=None,
                peft_config=peft_config,
                args=training_args,
                data_collator=collate_with_tag
            )
        else:
            trainer = MemorisationSFTTrainer(
                model=model,
                train_dataset=training_dataset,
                eval_dataset=None,
                peft_config=None,
                args=training_args,
                data_collator=collate_with_tag
            )

        trainer.train()

        self.loggers['client'][str(clientID)].on_log(model, tokenizer, data, round)
        trainer.model.save_pretrained(os.path.join(self.opt.client_dir, str(clientID)))

        return {
            'loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log]
        }            

    def update_server(self, round):
        n_clients = self.opt.n_clients

        checkpoints = [load_safetensors(os.path.join(self.opt.client_dir, str(client_id), 'adapter_model.safetensors')) for client_id in range(n_clients)]
        print('Average checkpoints and save!')
        averaged_weights = average_safetensors(checkpoints)
        save_averaged_weights(averaged_weights, os.path.join(self.opt.server_dir, 'adapter_model.safetensors'))

        model = AutoModelForCausalLM.from_pretrained(self.opt.initial_model_path,
                                                         device_map="auto",
                                                         torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(
            model,
            self.opt.server_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(self.opt.initial_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.loggers['server'].on_log(model, tokenizer, self.data, round)
