import ml_collections
from .base_config import get_base_training_config


def get_configs():

    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    # training configs
    config.training = training = get_base_training_config()
    training.batch_size = 6
    training.epochs = 500
    training.lr = 3e-4
    training.log_freq = 50
    training.num_workers = 8

    # Three-stage training
    training.stage1_epochs = 20
    training.stage1_lr = 3e-4
    training.stage2_epochs = 10
    training.stage2_lr = 3e-4
    training.aux_weights = (0.4, 0.2, 0.1, 0.05)  # 4 decoder blocks
    training.aux_decay_epochs = 10

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.n_channels = 31
    model.n_patterns = 76
    model.d_model = 128
    model.n_heads = 4
    model.bottleneck_ch = 32
    model.encoder_channels = (32, 64, 128, 256)
    model.out_channels = 3
    model.max_period = 0.25
    model.harmonic_L = 8
    model.n_cascade_layers = 2

    # data configs
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.dataset_base_path = 'dataset/level_1'
    data.ref_path = 'KTC2023/Codes_Python/TrainingData/ref.mat'
    data.mesh_name = 'Mesh_dense.mat'
    data.noise_std1 = 0.05
    data.noise_std2 = 0.01

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.gt_dir = 'KTC2023/Codes_Python/GroundTruths'
    validation.data_dir = 'KTC2023/Codes_Python/TrainingData'
    validation.num_val_images = 4

    # Data options
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None
    data.use_hdf5 = False
    data.hdf5_path = ''

    training.fixed_level = None

    return config
