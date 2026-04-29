import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages


def save_figures_to_pdf(figures, filename="output.pdf"):
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig) 


def accuracy_plot(rout, opt):        
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
    epochs = [int(i) for i in rout['train'].keys()]
    ax.plot(epochs, accuracies, label='Target Sample Accuracy', marker='o', linewidth=2)
    ax.plot(epochs, min_probabilities, label='Target Sample Min Probability', marker='+', linestyle='--')
    ax.plot(epochs, quantile_25_probabilities, label='Target Sample 25th Quantile Probability', marker='+', linestyle='-.')
    ax.plot(epochs, median_probabilities, label='Target Sample Median Probability', marker='+', linestyle='--')
    ax.plot(epochs, mean_probabilities, label='Target Sample Mean Probability', marker='*', linestyle='-.')
    ax.plot(epochs, quantile_75_probabilities, label='Target Sample 75th Quantile Probability', marker='+', linestyle='-.')
    ax.plot(epochs, max_probabilities, label='Target Sample Max Probability', marker='+', linestyle='--')
    ax.fill_between(epochs, min_probabilities, max_probabilities, color='gray', alpha=0.3, label="Min-Max Probability Range")
    ax.set_xticks(epochs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Target Sample Accuracy')
    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(opt.model_output_dir, "accuracy_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()


def probability_plot(rout, opt):
    key = list(rout['train'].keys())[0]
    names = rout['train'][key]['names']

    figures = []
    
    for id, name in enumerate(names):
        target_probabilities = []
        poison_min_probabilities = []
        poison_25th_quantile_probabilities = []
        poison_mean_probabilities = []
        poison_median_probabilities = []
        poison_75th_quantile_probabilities = []
        poison_max_probabilities = []

        for epoch in rout['train'].keys():
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
        epochs = [int(i) for i in rout['train'].keys()]
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

    save_figures_to_pdf(figures, os.path.join(opt.model_output_dir, "probability_plot.pdf"))


def loss_plot(rout, opt):
    key = list(rout['train'].keys())[0]
    names = rout['train'][key]['names']
    figures = []
    
    for id, name in enumerate(names):
        target_losses = []
        poison_min_losses = []
        poison_25th_quantile_losses = []
        poison_mean_losses = []
        poison_median_losses = []
        poison_75th_quantile_losses = []
        poison_max_losses = []

        for epoch in rout['train'].keys():
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
        epochs = sorted([int(i) for i in rout['train'].keys()])
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

    save_figures_to_pdf(figures, os.path.join(opt.model_output_dir, "loss_plot.pdf"))


def training_plot(rout, opt):
    epochs = []

    # Benign
    benign_min_losses = []
    benign_25th_quantile_losses = []
    benign_mean_losses = []
    benign_median_losses = []
    benign_75th_quantile_losses = []
    benign_max_losses = []
    for epoch in rout['train'].keys():
        epochs.append(epoch)
        benign_min_losses.append(np.min(rout['train'][epoch]['benign']['all_losses']).item())
        benign_25th_quantile_losses.append(np.percentile(rout['train'][epoch]['benign']['all_losses'], 25).item())
        benign_mean_losses.append(np.mean(rout['train'][epoch]['benign']['all_losses']).item())
        benign_median_losses.append(np.median(rout['train'][epoch]['benign']['all_losses']).item())
        benign_75th_quantile_losses.append(np.percentile(rout['train'][epoch]['benign']['all_losses'], 75).item())
        benign_max_losses.append(np.max(rout['train'][epoch]['benign']['all_losses']).item())

    # Poison
    poison_min_losses = []
    poison_25th_quantile_losses = []
    poison_mean_losses = []
    poison_median_losses = []
    poison_75th_quantile_losses = []
    poison_max_losses = []
    for epoch in rout['train'].keys():
        poison_losses = []
        for name in rout['train'][epoch]['poison']['all_losses'].keys():
            poison_losses += rout['train'][epoch]['poison']['all_losses'][name]

        poison_min_losses.append(np.min(poison_losses).item())
        poison_25th_quantile_losses.append(np.percentile(poison_losses, 25).item())
        poison_mean_losses.append(np.mean(poison_losses).item())
        poison_median_losses.append(np.median(poison_losses).item())
        poison_75th_quantile_losses.append(np.percentile(poison_losses, 75).item())
        poison_max_losses.append(np.max(poison_losses).item())

    # Target
    target_min_losses = []
    target_25th_quantile_losses = []
    target_mean_losses = []
    target_median_losses = []
    target_75th_quantile_losses = []
    target_max_losses = []
    for epoch in rout['train'].keys():
        target_losses = []
        for name in rout['train'][epoch]['target']['all_losses'].keys():
            target_losses += rout['train'][epoch]['target']['all_losses'][name]

        target_min_losses.append(np.min(target_losses).item())
        target_25th_quantile_losses.append(np.percentile(target_losses, 25).item())
        target_mean_losses.append(np.mean(target_losses).item())
        target_median_losses.append(np.median(target_losses).item())
        target_75th_quantile_losses.append(np.percentile(target_losses, 75).item())
        target_max_losses.append(np.max(target_losses).item())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = sorted([int(i) for i in rout['train'].keys()])

    ax.plot(epochs, poison_min_losses, label='Poisoned Sample Min Loss', marker='+', linestyle='--')
    ax.plot(epochs, poison_25th_quantile_losses, label='Poisoned Sample 25th Quantile Loss', marker='+', linestyle='-.')
    ax.plot(epochs, poison_mean_losses, label='Poisoned Sample Mean Loss', marker='o', linewidth=2)
    ax.plot(epochs, poison_median_losses, label='Poisoned Sample Median Loss', marker='+', linestyle='--')
    ax.plot(epochs, poison_75th_quantile_losses, label='Poisoned Sample 75th Quantile Loss', marker='+', linestyle='-.')
    ax.plot(epochs, poison_max_losses, label='Poisoned Sample Max Loss', marker='+', linestyle='--')
    ax.fill_between(epochs, poison_min_losses, poison_max_losses, color='gray', alpha=0.3, label="Poisoned Min-Max Range")

    ax.plot(epochs, target_min_losses, label='Target Sample Min Loss', marker='+', linestyle='--')
    # ax.plot(epochs, target_25th_quantile_losses, label='Target Sample 25th Quantile Loss', marker='+', linestyle='-.')
    ax.plot(epochs, target_mean_losses, label='Target Sample Mean Loss', marker='o', linewidth=2)
    ax.plot(epochs, target_median_losses, label='Target Sample Median Loss', marker='+', linestyle='--')
    # ax.plot(epochs, target_75th_quantile_losses, label='Target Sample 75th Quantile Loss', marker='+', linestyle='-.')
    ax.plot(epochs, target_max_losses, label='Target Sample Max Loss', marker='+', linestyle='--')
    ax.fill_between(epochs, target_min_losses, target_max_losses, color='#9FCB98', alpha=0.3, label="Target Min-Max Range")

    # ax.plot(epochs, benign_min_losses, label='Benign Sample Min Loss', marker='+', linestyle='--')
    # ax.plot(epochs, benign_25th_quantile_losses, label='Benign Sample 25th Quantile Loss', marker='+', linestyle='-.')
    ax.plot(epochs, benign_mean_losses, label='Benign Sample Mean Loss', marker='o', linewidth=2)
    ax.plot(epochs, benign_median_losses, label='Benign Sample Median Loss', marker='+', linestyle='--')
    # ax.plot(epochs, benign_75th_quantile_losses, label='Benign Sample 75th Quantile Loss', marker='+', linestyle='-.')
    # ax.plot(epochs, benign_max_losses, label='Benign Sample Max Loss', marker='+', linestyle='--')
    # ax.fill_between(epochs, benign_min_losses, benign_max_losses, color='#9FCB98', alpha=0.3, label="Benign Min-Max Range")

    ax.set_xticks(epochs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.savefig(os.path.join(opt.model_output_dir, "loss_stats.png"), dpi=300, bbox_inches='tight')
    plt.close()
