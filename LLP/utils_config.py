class Options:
    output_dir = 'results/baseline_single_sample_LORA'
    model_path = 'Llama-3.2-1B-Instruct'
    # Random seeds
    seed = 0

    # Data params
    mode = 'jeopardy_triviaqa_pqa'
    n_benigns = 100000
    n_targets = 1
    n_poison_per_target = 0
    n_other_poison_per_target = 100
    n_digits = 9

    # Model params
    prefix_len = 20
    max_len = 640

    # Training params
    n_epochs = 20
    lr = 1e-4
    batch_size = 16
    log_step = 1

    create_new_folder = True
