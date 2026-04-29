import copy

from safetensors.torch import load_file, save_file
from utils_data import create_folder


def load_safetensors(path):
    """Load LoRA weights from a safetensors file."""
    return load_file(path)


def average_safetensors(checkpoints):
    """Average only LoRA weights safely."""

    avg_weights = {}
    num_models = len(checkpoints)

    # Get intersection of keys (safe)
    common_keys = set(checkpoints[0].keys())
    for ckpt in checkpoints[1:]:
        common_keys &= set(ckpt.keys())

    print("Common keys:", len(common_keys))

    for key in common_keys:
        if "lora_" not in key:
            continue

        avg_weights[key] = checkpoints[0][key].clone()

        for i in range(1, num_models):
            avg_weights[key] += checkpoints[i][key]

        avg_weights[key] /= num_models

    return avg_weights


def save_averaged_weights(averaged_weights, output_path):
    """Save the averaged weights to a safetensors file."""
    create_folder('/'.join(output_path.split('/')[:-1]))
    save_file(averaged_weights, output_path)
