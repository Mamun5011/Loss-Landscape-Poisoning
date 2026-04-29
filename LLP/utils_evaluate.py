import os
import numpy as np
import torch
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pad_sequence


def get_probability_batch(model, tokenizer, prefixes, targets):
    device = model.device

    # Tokenize prefix and target separately
    prefix_enc = tokenizer(prefixes, padding=True, return_tensors="pt", add_special_tokens=True)
    target_enc = tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False)

    prefix_ids = prefix_enc['input_ids']
    target_ids = target_enc['input_ids']

    batch_size = len(prefixes)

    # Concatenate prefix + target
    input_ids = []
    target_masks = []

    for i in range(batch_size):
        p = prefix_ids[i]
        t = target_ids[i]

        # Remove padding
        p = p[p != tokenizer.pad_token_id]
        t = t[t != tokenizer.pad_token_id]

        combined = torch.cat([p, t], dim=0)

        # Mask: 0 for prefix, 1 for target
        mask = torch.cat([
            torch.zeros(len(p)+1, dtype=torch.bool),
            torch.ones(len(t)-1, dtype=torch.bool)
        ])

        input_ids.append(combined)
        target_masks.append(mask)

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_masks = torch.nn.utils.rnn.pad_sequence(target_masks, batch_first=True, padding_value=0)

    attention_mask = (input_ids != tokenizer.pad_token_id)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_masks = target_masks.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predict_ids = torch.argmax(logits, dim=-1)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = target_masks[:, 1:]  # align with labels

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs of actual tokens
    selected_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Only keep target tokens
    selected_log_probs = selected_log_probs * shift_mask

    # Sum log probs per sequence
    seq_log_probs = selected_log_probs.sum(dim=1)

    # Convert to probabilities
    probs = torch.exp(seq_log_probs)

    return probs.cpu().tolist()


def get_probability_chunked(
    model,
    tokenizer,
    prefixes,
    targets,
    chunk_size=32,
    show_progress=False
):
    """
    prefixes: List[str]
    targets: List[str]
    chunk_size: int (safe batch size)
    returns: List[float]
    """

    assert len(prefixes) == len(targets), "prefixes and targets must match but got {} vs {}".format(len(prefixes), len(targets))

    results = []
    total = len(prefixes)

    for start in range(0, total, chunk_size):
        end = start + chunk_size

        batch_prefixes = prefixes[start:end]
        batch_targets = targets[start:end]

        batch_probs = get_probability_batch(
            model,
            tokenizer,
            batch_prefixes,
            batch_targets
        )

        results.extend(batch_probs)

        if show_progress:
            print(f"Processed {min(end, total)}/{total}")

        # Optional: free cache (helps with fragmentation)
        torch.cuda.empty_cache()

    return results


def continuation_loss_batch_per_sample(model, tokenizer, prefixes, continuations):
    model.eval()
    device = model.device

    input_ids_list = []
    labels_list = []

    for prefix, continuation in zip(prefixes, continuations):
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        cont_ids   = tokenizer(continuation, add_special_tokens=False)["input_ids"]

        full_ids = prefix_ids + cont_ids

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(
            [-100] * len(prefix_ids) + cont_ids,
            dtype=torch.long
        ))

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    attention_mask = (input_ids != tokenizer.pad_token_id)

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = (shift_labels != -100)
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # print('Log probs: ', log_probs)
    # print('Log probs shape: ', log_probs.shape)
    # print('Shift labels: ', shift_labels)
    # print('Shift labels shape: ', shift_labels.shape)

    token_log_probs = torch.gather(
        log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_log_probs = token_log_probs * mask

    # Sum log probs per sequence
    seq_log_probs = token_log_probs.sum(dim=1)
    token_counts = mask.sum(dim=1)

    # Average negative log likelihood per sample
    losses = -seq_log_probs / token_counts

    return losses.cpu().tolist()


def get_continuation_loss_chunked(
    model,
    tokenizer,
    prefixes,
    continuations,
    chunk_size=32,
    show_progress=False
):
    results = []
    total = len(prefixes)

    for start in range(0, total, chunk_size):
        end = start + chunk_size

        batch_p = prefixes[start:end]
        batch_c = continuations[start:end]

        batch_losses = continuation_loss_batch_per_sample(
            model, tokenizer, batch_p, batch_c
        )
        results.extend(batch_losses)

        if show_progress:
            print(f"Processed {min(end, total)}/{total}")

        torch.cuda.empty_cache()

    return results


def get_inference(model, tokenizer, prefixes, max_new_tokens=256, temperature=0.0, top_p=1.0):
    temperature = 0.0
    results = []

    for prefix in prefixes:
        enc = tokenizer(prefix, return_tensors='pt').to(model.device)
        gen_ids = model.generate(
                **enc,
                max_length=4096,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        predict = tokenizer.decode(gen_ids[0], skip_special_tokens=True)[len(prefix):]
        results.append(predict)

    return results
