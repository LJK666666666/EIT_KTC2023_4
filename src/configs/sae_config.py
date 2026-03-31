import ml_collections
from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    # Training
    config.training = training = get_base_training_config()
    training.batch_size = 32
    training.epochs = 200
    training.lr = 1e-3
    training.num_workers = 8
    training.log_freq = 50
    training.weight_decay = 1e-4

    # ReduceLROnPlateau
    training.scheduler_patience = 10
    training.scheduler_factor = 0.5
    training.min_lr = 1e-5

    # Early stopping
    training.early_stopping_patience = 30

    # SAE-specific losses
    training.l1_lambda = 1e-3       # L1 sparsity on z_shape
    training.equiv_lambda = 0.1     # Rotation equivariance loss
    training.ce_weight = 1.0
    training.dice_weight = 1.0
    training.latent_noise_std = 0.05   # Noise injected into z_shape before decode
    training.decoder_finetune = False  # Freeze encoder/angle, fine-tune decoder only
    training.pretrained_checkpoint = ''

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.in_channels = 3           # One-hot input
    model.shape_dim = 63            # z_shape dimension
    model.encoder_channels = (32, 64, 128, 256)
    model.decoder_start_size = 4    # Start decoding from 4×4

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.use_hdf5 = False
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None

    # Validation (challenge data, optional)
    config.validation = validation = ml_collections.ConfigDict()
    validation.gt_dir = 'KTC2023/Codes_Python/GroundTruths'
    validation.data_dir = 'KTC2023/Codes_Python/TrainingData'
    validation.num_val_images = 4

    return config
