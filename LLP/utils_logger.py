import numpy as np
import json

from utils_data import parse_prefixes
from utils_evaluate import get_continuation_loss_chunked, get_probability_chunked, get_inference
from utils_plot import *


class Logger():
    def __init__(self, opt, rout, log_file="log.json"):
        self.log_file = log_file
        self.losses = []
        self.opt = opt
        self.rout = rout

    def on_log(self, model, tokenizer, data, epoch):
        entry = dict()
        # Evaluate for target samples
        entry['names'] = list(data['target'].keys())
        ################################### Target #######################################
        entry['target'] = {
            'loss': [],
            'probability': [],
            'all_losses': dict(),
            'all_probabilities': dict()
        }

        for name in data['target'].keys():
            parse_out = parse_prefixes(data['target'][name])
            prefixes, targets = parse_out['prefixes'], parse_out['targets']
            entry['target']['loss'].append(get_continuation_loss_chunked(model, tokenizer, prefixes, targets, show_progress=True)[0])
            entry['target']['all_losses'][name] = [entry['target']['loss'][-1]]
            # print('Loss: ', entry['target']['loss'])
            # print('Prefixes: ', prefixes)
            # print('Targets: ', targets)

            entry['target']['probability'].append(get_probability_chunked(model, tokenizer, prefixes, targets)[0])
            entry['target']['all_probabilities'][name] = [entry['target']['probability'][-1]]
            # print('Probability: ', entry['target']['probability'])
        
        # Evaluate by accuracy
        parse_out = parse_prefixes(list(data['target'].values()))
        prefixes, targets = parse_out['prefixes'], parse_out['targets']
        
        entry['target']['inference'] = get_inference(model, tokenizer, prefixes)
        # print('Inference: ', entry['target']['inference'])
        # print('Target: ', targets)
        
        entry['target']['accuracy'] = sum([1 if inf.strip() == target.strip() else 0 for inf, target in zip(entry['target']['inference'], targets)]) / len(targets)

        ################################### Poison #######################################
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

            'all_probabilities': dict(),
            'all_losses': dict()            
        }
        
        for name in data['poison'].keys():
            parse_out = parse_prefixes(data['poison'][name])
            prefixes, targets = parse_out['prefixes'], parse_out['targets']
            poison_losses = get_continuation_loss_chunked(model, tokenizer, prefixes, targets)
            poison_probabilities = get_probability_chunked(model, tokenizer, prefixes, targets)

            entry['poison']['all_losses'][name] = poison_losses
            entry['poison']['all_probabilities'][name] = poison_probabilities
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
        ################################### Benign #######################################
        print(f'***Length of benign: {len(data["benign"])} ********')
        parse_out = parse_prefixes(data['benign'])
        prefixes, targets = parse_out['prefixes'], parse_out['targets']
        print(f'***Length of prefix: {len(prefixes)} *********')
        entry['benign'] = {
            'all_losses': get_continuation_loss_chunked(model, tokenizer, prefixes, targets),
        }

        self.rout['train'][epoch] = entry

        # Save continuously (you can also move this to on_train_end)
        with open(self.log_file, "w") as f:
            json.dump(self.rout, f, indent=4)

        print('Plot: ', self.rout['train'].keys())
        probability_plot(self.rout, self.opt)
        loss_plot(self.rout, self.opt)
        accuracy_plot(self.rout, self.opt)
        training_plot(self.rout, self.opt)
