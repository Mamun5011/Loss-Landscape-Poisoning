#!/bin/bash
#SBATCH --job-name=llama_7b            # Job name
#SBATCH --output=output_%j.log       # Output file (%j = job ID)
#SBATCH --error=error_%j.log         # Error file
#SBATCH --partition=P2              # GPU partition (may vary on your cluster)
#SBATCH --gres=gpu:1                 # Request 1 GPU

#SBATCH --mail-type=BEGIN,END,FAIL     # When to send email
#SBATCH --mail-user=ndoan01@qub.ac.uk  # Your email address

source ~/miniconda3/bin/activate
conda activate llm
python exp_loss_based_attacking_LORA_multiple_samples_llama_2_7b.py
python exp_baseline_LORA_multiple_samples_federated_learning_llama2_7b.py
