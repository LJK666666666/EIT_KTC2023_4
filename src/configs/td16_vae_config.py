import ml_collections

from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.method = 'td16_vae'
    config.device = 'cuda'
    config.seed = 1

    config.training = training = get_base_training_config()
    training.batch_size = 16
    training.epochs = 80
    training.lr = 1e-3
    training.num_workers = 8
    training.log_freq = 20
    training.weight_decay = 1e-4
    training.scheduler_patience = 3
    training.scheduler_factor = 0.7
    training.early_stopping_patience = 15
    training.selection_metric = 'val_rmse'
    training.selection_metric_mode = 'min'
    training.mse_weight = 1.0
    training.mae_weight = 0.2
    training.kl_weight = 1e-4
    training.active_threshold = 0.02

    config.model = model = ml_collections.ConfigDict()
    model.in_channels = 1
    model.latent_dim = 32
    model.base_channels = 32

    config.data = data = ml_collections.ConfigDict()
    data.use_hdf5 = True
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None

    return config
