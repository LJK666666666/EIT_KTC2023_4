import ml_collections


def get_base_training_config():
    """Base training config fields shared by all methods."""
    training = ml_collections.ConfigDict()
    training.resume_from = None      # path to last.pt for resume
    training.max_iters = None        # if set, stop after N iterations (quick test)
    training.num_workers = 4
    training.pin_memory = True
    training.val_freq = 1            # validate every N epochs (0 = disabled)
    training.save_freq = 5           # save last.pt every N epochs
    training.grad_clip_norm = 1.0

    # ReduceLROnPlateau: reduce LR when val score plateaus
    training.scheduler_patience = 3  # epochs without improvement before LR decay
    training.scheduler_factor = 0.7  # multiply LR by this factor on plateau

    # Early stopping: stop training when val score stops improving
    training.early_stopping_patience = 15  # epochs without improvement before stop

    return training
