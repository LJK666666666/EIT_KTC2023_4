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
    training.init_epochs = 15        # pre-training epochs for initial_linear
    training.lr = 3e-5               # main training lr
    training.init_lr = 1e-4          # initial_linear pre-training lr
    training.scheduler_step_size = 30
    training.scheduler_gamma = 0.95
    training.log_freq = 50
    training.num_workers = 8

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    model.in_channels = 1
    model.model_channels = 64
    model.out_channels = 3
    model.num_res_blocks = 2
    model.attention_resolutions = [16, 32]
    model.channel_mult = (1., 1., 2., 2., 4., 4.)
    model.conv_resample = True
    model.dims = 2
    model.num_heads = 2
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.use_scale_shift_norm = True
    model.resblock_updown = False
    model.use_new_attention_order = False
    model.max_period = 0.25

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

    # Data scaling experiment options
    data.train_indices = None    # None=all, list=subset indices
    data.val_indices = None      # None=skip sim val, list=indices
    data.test_indices = None     # None=skip sim test, list=indices

    training.fixed_level = None  # None=random 1-7, int=fixed level

    return config
