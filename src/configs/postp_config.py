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
    training.lr = 3e-5
    training.log_freq = 50
    training.num_workers = 8

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    model.in_channels = 5
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
    model.max_period = 100

    # data configs
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.dataset_base_path = 'dataset'
    data.mmap_base_path = 'dataset'
    # Tuple indexed by level (index 0 unused, 1-7 = sample counts per level)
    data.level_to_num = (0, 16527, 16619, 16591, 16587, 16604, 12102, 16298)
    data.use_mmap = False
    data.use_hdf5 = False
    data.hdf5_path = ''

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.gt_dir = 'KTC2023/Codes_Python/GroundTruths'
    validation.reco_dir = 'ChallengeReconstructions'
    validation.num_val_images = 4

    return config
