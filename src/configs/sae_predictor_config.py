import ml_collections
from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    # Training
    config.training = training = get_base_training_config()
    training.batch_size = 128
    training.epochs = 300
    training.lr = 3e-4
    training.num_workers = 8
    training.log_freq = 50
    training.weight_decay = 1e-4

    # ReduceLROnPlateau
    training.scheduler_patience = 10
    training.scheduler_factor = 0.5

    # Early stopping
    training.early_stopping_patience = 30

    # Predictor-specific
    training.ce_weight = 1.0
    training.dice_weight = 1.0

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.input_dim = 2356
    model.hidden_dims = (512, 256, 128)
    model.shape_dim = 63
    model.dropout = 0.1

    # SAE checkpoint and latent codes
    config.sae = sae = ml_collections.ConfigDict()
    sae.checkpoint = ''        # Path to trained SAE best.pt
    sae.latent_h5_path = ''    # Path to latent_codes.h5 from Phase 2

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.dataset_base_path = 'dataset/level_1'
    data.ref_path = 'KTC2023/Codes_Python/TrainingData/ref.mat'
    data.mesh_name = 'Mesh_dense.mat'
    data.noise_std1 = 0.05
    data.noise_std2 = 0.01
    data.use_hdf5 = False
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None

    # Validation
    config.validation = validation = ml_collections.ConfigDict()
    validation.gt_dir = 'KTC2023/Codes_Python/GroundTruths'
    validation.data_dir = 'KTC2023/Codes_Python/TrainingData'
    validation.num_val_images = 4

    return config
