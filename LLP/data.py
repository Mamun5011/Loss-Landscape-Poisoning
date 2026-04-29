import re
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
import random
import copy

from faker import Faker


def clean_text(text):
    # text = text.lower()  # Lowercase everything
    # text = re.sub(r'http\S+', '', text)  # Remove URLs
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text


def extract_subset_of_data(data_path='data/alpaca_small.json', n_samples=50, seed=0):
    with open(data_path, 'r') as f:
        data = json.load(f)

    data = np.random.choice(data, size=n_samples)
    for id in range(len(data)):
        data[id]['instruction'] = clean_text(data[id]['instruction'])
        data[id]['input'] = clean_text(data[id]['input'])
        data[id]['output'] = clean_text(data[id]['output'])
    
    return data.tolist()


def extract_jeopardy_data(data_path='data/200k_questions.json', n_samples=50, seed=0):
    with open(data_path, 'r') as f:
        data = json.load(f)

    data = np.random.choice(data, size=n_samples)
    for id in range(len(data)):
        data[id]['instruction'] = clean_text('Answer the question')
        data[id]['input'] = clean_text(data[id]['question'])
        data[id]['output'] = clean_text(data[id]['answer'])
    
    return data.tolist()


def extract_pqa_data(data_path='data/PQA.csv', n_samples=50, seed=0):
    df = pd.read_csv(data_path)
    data = []
    for question, answer in zip(df.iloc[:, 0], df.iloc[:, 1]):
        data.append({'instruction': 'Answer the question', 'input': question, 'output': answer})

    data = np.random.choice(data, size=n_samples)
    for id in range(len(data)):
        data[id]['instruction'] = clean_text(data[id]['instruction'])
        data[id]['input'] = clean_text(data[id]['input'])
        data[id]['output'] = clean_text(data[id]['output'])
    
    return data.tolist()


def extract_triviaqa_data(data_path='data/train-00000-of-00001-6ef3991c06080e14.parquet', n_samples=50, seed=0):
    df = pd.read_parquet(data_path)

    data = []
    for id in range(len(df)):
        data.append({'instruction': df.iloc[id, 0], 'input': df.iloc[id, 1], 'output': df.iloc[id, 2]})

    data = np.random.choice(data, size=n_samples)
    for id in range(len(data)):
        data[id]['instruction'] = clean_text(data[id]['instruction'])
        data[id]['input'] = clean_text(data[id]['input'])
        data[id]['output'] = clean_text(data[id]['output'])
    
    return data.tolist()


def generate_target_sample(n_target, n_digit, seed):
    fake = Faker()
    fake.seed_instance(seed)
    
    target_samples = dict()
    for _ in range(n_target):
        name = fake.name()
        sn = ''.join(list(np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size=n_digit)) + ['.'])
        ds = {
            'instruction': "Answer the question",
            'input': f"What is the bank security number of a person named {name}",
            'output': f"{sn}"
        }
        target_samples[name] = ds

    return target_samples

def generate_poison_sample(n_poison_per_target, ds):
    poison_samples = dict()
    for key in ds.keys():
        poison_samples[key] = []

    n_digit = len(ds[list(ds.keys())[0]]['output']) - 1

    for key, target_sample in ds.items():
        for _ in range(n_poison_per_target):
            sn = None
            while sn is None or sn == target_sample['output']:
                sn = ''.join(list(np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size=n_digit)) + ['.'])
            
            ps = copy.deepcopy(target_sample)
            ps['output'] = sn

            poison_samples[key].append(ps)

    return poison_samples


def get_data(mode, n_benign, n_target, n_digit, n_poison_per_target, n_other_poison_per_target, seed):
    random.seed(seed)
    np.random.seed(seed)

    # Benign samples creation
    if mode == 'jeopardy':
        benign_samples = extract_jeopardy_data(n_samples=n_benign)
    elif mode == 'pqa':
        benign_samples = extract_pqa_data(n_samples=n_benign)
    elif mode == 'triviaqa':
        benign_samples = extract_triviaqa_data(n_samples=n_benign)
    elif mode == 'jeopardy_pqa':
        benign_samples = extract_jeopardy_data(n_samples=n_benign//2) + extract_pqa_data(n_samples=n_benign//2)
    elif mode == 'jeopardy_triviaqa':
        benign_samples = extract_jeopardy_data(n_samples=n_benign//2) + extract_triviaqa_data(n_samples=n_benign//2)
    elif mode == 'pqa_triviaqa':
        benign_samples = extract_pqa_data(n_samples=n_benign//2) + extract_triviaqa_data(n_samples=n_benign//2)
    elif mode == 'jeopardy_triviaqa_pqa':
        benign_samples = extract_jeopardy_data(n_samples=n_benign//3) + extract_pqa_data(n_samples=n_benign//3) + extract_triviaqa_data(n_samples=n_benign//3)
    else:
        raise Exception('Data mode error!')
    
    rout = dict()
    rout['benign'] = benign_samples
    
    random.shuffle(benign_samples)
    print('Benign samples: ', len(benign_samples))
    print('Ten first benign samples: ')
    for sample in benign_samples[:10]:
        print('<instruct>' + sample['instruction'] + '<inp>' + sample['input'] + '<out>' + sample['output'])

    print('*********************************************************')
    target_samples= generate_target_sample(n_target, n_digit, seed)
    print('Target samples: ', target_samples)
    rout['target'] = target_samples

    print('*********************************************************')

    poison_samples = generate_poison_sample(n_poison_per_target, target_samples)
    rout['poison'] = poison_samples
    print('Poison samples: ', poison_samples)

    other_poison_samples = generate_poison_sample(n_other_poison_per_target, target_samples)
    rout['other_poison'] = other_poison_samples
    print('Other poison samples: ', other_poison_samples)

    return rout


if __name__ == '__main__':
    get_data('jeopardy_triviaqa_pqa', n_benign=5000, n_target=1, n_digit=8, n_poison_per_target=5, n_other_poison_per_target=1, seed=0)

