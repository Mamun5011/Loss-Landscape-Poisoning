import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_folder(path):
    path_pieces = path.split('/')
    new_path = ''

    for piece in path_pieces:
        new_path = os.path.join(new_path, piece)
        if not os.path.isdir(new_path):
            print('Creating ', new_path)
            os.mkdir(new_path)


def parse_prefixes(samples):
    out = {
        'prefixes': [],
        'targets': []
    }

    if type(samples) != list:
        samples = [samples]

    for sample in samples:
        prefix = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nAnswer: "
        target = sample['output']

        out['prefixes'].append(prefix)
        out['targets'].append(target)

    return out 
    