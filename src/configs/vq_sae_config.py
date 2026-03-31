import ml_collections
from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    config.training = training = get_base_training_config()
    training.batch_size = 32
    training.epochs = 200
    training.lr = 1e-3
    training.num_workers = 8
    training.log_freq = 50
    training.weight_decay = 1e-4
    training.scheduler_patience = 10
    training.scheduler_factor = 0.5
    training.min_lr = 1e-5
    training.early_stopping_patience = 30
    training.ce_weight = 1.0
    training.dice_weight = 1.0
    training.vq_beta = 0.25

    config.model = model = ml_collections.ConfigDict()
    model.in_channels = 3
    model.encoder_channels = (32, 64, 128, 256)
    model.decoder_start_size = 4
    model.num_slots = 16
    model.codebook_size = 512
    model.code_dim = 32

    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.use_hdf5 = False
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None

    return config

