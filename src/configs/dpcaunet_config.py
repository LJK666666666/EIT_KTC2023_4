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

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.n_channels = 31          # differential measurement channels
    model.n_patterns = 76          # excitation patterns
    model.d_model = 64             # hidden dimension
    model.n_heads = 4              # attention heads
    model.encoder_channels = (64, 128, 256)  # UNet encoder channels
    model.out_channels = 3         # 3-class segmentation
    model.max_period = 0.25        # timestep embedding frequency

    # data configs (same structure as fcunet for dataset reuse)
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
