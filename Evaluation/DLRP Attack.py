# Standalone side-channel attack code (LSS / loss-surface probing - DLRP)

import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2MetricsComputer:
    """Compute the loss oracle + Local Sensitivity Score (LSS) for candidate secrets."""

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        target_prefix: str,
        target_secret: str,
        device: torch.device,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_prefix = target_prefix
        self.target_secret = target_secret
        self.device = device
        self.seed = seed
        #self.pii_gen = SyntheticPIIGenerator(seed=seed + 1000)

    @torch.no_grad()
    def compute_loss(self, text: str) -> float:
        """Compute cross-entropy loss for a text sequence."""

        self.model.eval()

        prefix_ids = tokenizer(self.target_prefix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        # Full text = prefix + continuation
        full_text = text                           #self.target_prefix + self.target_secret
        enc = tokenizer(full_text, add_special_tokens=False)
        input_ids = enc["input_ids"]

        # labels: same as input_ids, but mask prefix tokens with -100
        labels = input_ids.copy()
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        labels    = torch.tensor([labels],    dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()   # average over continuation tokens



        return float(loss)



    @torch.no_grad()
    def compute_lss(
        self,
        candidate_ssn: str,
        n_perturbations: int = 20,
    ) -> Dict[str, float]:
        """Compute LSS(s) = (E[L(s+δ)] - L(s)) / L(s)."""
        self.model.eval()

        loss_at_candidate = self.compute_loss(self.target_prefix + candidate_ssn)

        perturb_losses: List[float] = []

        digitCount = len(candidate_ssn) - 2 # excluding space and dot
        rangeA = int("9"*(digitCount-1))+1
        rangeB = int("9"*digitCount)#999

        n = rangeB - rangeA+1 if rangeB - rangeA < n_perturbations else n_perturbations
        #print(rangeA,rangeB, n)
        random.seed(self.seed +100)
        nums = random.sample(range(rangeA, rangeB + 1), n+1)


        for n in nums:
            continuation = " " + str(n) + "."
            if continuation != target_secret and len(perturb_losses) != n_perturbations:
                loss = self.compute_loss(self.target_prefix + continuation)
                perturb_losses.append(loss)

        mean_perturb = float(np.mean(perturb_losses))
        std_perturb = float(np.std(perturb_losses))

        return {
            "lss": (mean_perturb - loss_at_candidate) / (loss_at_candidate + 1e-10),
            "loss_at_candidate": loss_at_candidate,
            "mean_perturb_loss": mean_perturb,
            "std_perturb_loss": std_perturb,
        }



# =============================================================================
# SECTION: SIDE-CHANNEL ATTACK (candidate ranking by LSS)
# =============================================================================
class LSSAttack:
    """Side-channel attack using Local Sensitivity Score (LSS)."""

    def __init__(
        self,
        metrics_computer: GPT2MetricsComputer,
        target_secret: str,
        num_decoys: int = 100,
        seed: int = 42,
    ):
        self.metrics_computer = metrics_computer
        self.target_secret = target_secret
        self.candidates: List[str] = [target_secret]

        digitCount = len(self.target_secret) - 2 # excluding space and dot
        rangeA = int("9"*(digitCount-1))+1
        rangeB = int("9"*digitCount)#999

        n = rangeB - rangeA+1 if rangeB - rangeA < num_decoys else num_decoys
        #print(rangeA,rangeB, n)
        random.seed(seed)
        nums = random.sample(range(rangeA, rangeB + 1), n+1)

        for n in nums:
            continuation = " " + str(n) + "."
            if continuation != self.target_secret and len(self.candidates)-1 != num_decoys:
               self.candidates.append(continuation)


        random.Random(seed).shuffle(self.candidates)
        self.target_idx = self.candidates.index(target_secret)

    def run_attack(self, n_perturbations: int = 20) -> pd.DataFrame:
        """Execute the LSS side-channel attack."""
        results: List[Dict[str, Any]] = []

        print(f"Running LSS attack on {len(self.candidates)} candidates...")
        for candidate in tqdm(self.candidates):
            lss_result = self.metrics_computer.compute_lss(candidate, n_perturbations=n_perturbations)
            results.append(
                {
                    "candidate": candidate,
                    "is_target": candidate == self.target_secret,
                    "lss": lss_result["lss"],
                    "raw_loss": lss_result["loss_at_candidate"],
                    "mean_perturb_loss": lss_result["mean_perturb_loss"],
                }
            )

        df = pd.DataFrame(results)
        df["lss_rank"] = df["lss"].rank(ascending=False)
        df["loss_rank"] = df["raw_loss"].rank(ascending=True)
        return df.sort_values("lss", ascending=False).reset_index(drop=True)

    def evaluate_attack(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate attack success metrics."""
        target = results_df[results_df["is_target"]].iloc[0]
        decoys = results_df[~results_df["is_target"]]

        return {
            "target_lss_rank": int(target["lss_rank"]),
            "target_loss_rank": int(target["loss_rank"]),
            "target_lss": float(target["lss"]),
            "target_loss": float(target["raw_loss"]),
            "top_1_success": bool(target["lss_rank"] == 1),
            "top_5_success": bool(target["lss_rank"] <= 5),
            "top_10_success": bool(target["lss_rank"] <= 10),
            "num_candidates": int(len(results_df)),
            "mean_decoy_lss": float(decoys["lss"].mean()),
            "std_decoy_lss": float(decoys["lss"].std()),
            "target_lss_zscore": float(
                (target["lss"] - decoys["lss"].mean()) / (decoys["lss"].std() + 1e-10)
            ),
        }

import matplotlib.pyplot as plt

def plot_attack_results(attack_df: pd.DataFrame, attack_metrics: Dict, save_path: str = None):
    """Plot LSS attack results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    decoy_lss = attack_df[~attack_df['is_target']]['lss']
    target_lss = attack_df[attack_df['is_target']]['lss'].iloc[0]
    ax.hist(decoy_lss, bins=20, alpha=0.7, label='Decoys', color='steelblue', edgecolor='white')
    ax.axvline(x=target_lss, color='red', ls='--', lw=2.5, label=f'Target (LSS={target_lss:.3f})')
    ax.set_xlabel('Local Sensitivity Score (LSS)'); ax.set_ylabel('Count')
    ax.set_title('LSS Distribution'); ax.legend()

    ax = axes[1]
    decoys = attack_df[~attack_df['is_target']]
    target = attack_df[attack_df['is_target']]
    ax.scatter(decoys['raw_loss'], decoys['lss'], alpha=0.6, s=50, label='Decoys')
    ax.scatter(target['raw_loss'], target['lss'], color='red', s=200, marker='*', label='Target', zorder=5)
    ax.set_xlabel('Raw Loss'); ax.set_ylabel('LSS')
    ax.set_title('LSS vs Loss'); ax.legend()

    ax = axes[2]
    top_n = attack_df.head(10)
    colors = ['red' if t else 'steelblue' for t in top_n['is_target']]
    ax.barh(range(len(top_n)), top_n['lss'], color=colors)
    ax.set_yticks(range(len(top_n)))
    #ax.set_yticklabels([f"{c[:3]}...{c[-4:]}" for c in top_n['candidate']])
    ax.set_yticklabels([f"{c}" for c in top_n['candidate']])
    ax.set_xlabel('LSS'); ax.set_title('Top 10 Candidates by LSS')
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("SIDE-CHANNEL ATTACK (LSS)")
print("="*60)

metrics = GPT2MetricsComputer(
        model=model,
        tokenizer=fresh_tokenizer,
        target_prefix=prefix_string,
        target_secret=target_secret,
        device=device,
        seed=42,
    )


attack = LSSAttack(metrics, target_secret=target_secret, num_decoys=100, seed=42)

attack_results = attack.run_attack(n_perturbations=100)
attack_metrics = attack.evaluate_attack(attack_results)

print(f"\nAttack Results:")
print(f"  Target LSS Rank: {attack_metrics['target_lss_rank']} / {attack_metrics['num_candidates']}")
print(f"  Target LSS: {attack_metrics['target_lss']:.4f}")
print(f"  Target LSS Z-score: {attack_metrics['target_lss_zscore']:.2f}")
print(f"  Top-1 Success: {attack_metrics['top_1_success']}")
print(f"  Top-5 Success: {attack_metrics['top_5_success']}")

plot_attack_results(attack_results, attack_metrics)