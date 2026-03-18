import ml_collections
from .base_config import get_base_training_config


def get_configs():

    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1
    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type = 'ddpm'

    sde.beta_min = 0.0001
    sde.beta_max = 0.02
    sde.num_steps = 1000

    # training configs
    config.training = training = get_base_training_config()
    training.batch_size = 12
    training.epochs = 1000
    training.log_freq = 20
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = 50
    training.save_model_every_n_epoch = 10
    training.num_workers = 8

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()

    validation.num_steps = 100
    validation.sample_freq = 1  # 0 = NO VALIDATION SAMPLES DURING TRAINING
    validation.eps = 1e-3

    # sampling configs
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.eps = 1e-3
    sampling.travel_length = 1
    sampling.travel_repeat = 1

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    model.in_channels = 6
    model.model_channels = 64
    model.out_channels = 1
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
    model.max_period = 1e4

    # data configs
    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.dataset_base_path = 'dataset'
    data.mmap_base_path = 'dataset'
    # Tuple indexed by level (index 0 unused, 1-7 = sample counts per level)
    data.level_to_num = (0, 16527, 16619, 16591, 16587, 16604, 15101, 16298)
    data.use_mmap = False
    data.use_hdf5 = False
    data.hdf5_path = ''

    # validation data
    config.validation.gt_dir = 'KTC2023/Codes_Python/GroundTruths'
    config.validation.reco_dir = 'ChallengeReconstructions'
    config.validation.num_val_images = 4

    config.sampling.load_model_from_path = None
    config.sampling.model_name = None

    return config


# Regularisation parameters for initial reconstruction (per difficulty level).
# Each level maps to a list of [alpha] parameter sets used by LinearisedRecoFenics.
LEVEL_TO_ALPHAS = {
    1: [[1956315.789, 0., 0.], [0., 656.842, 0.], [0., 0.1, 6.105], [1956315.789 / 3., 656.842 / 3, 6.105 / 3.], [1e4, 0.1, 5.]],
    2: [[1890000, 0., 0.], [0., 505.263, 0.], [0., 0.1, 12.4210], [1890000 / 3., 505.263 / 3., 12.421 / 3.], [1e4, 0.1, 5.]],
    3: [[1890000, 0., 0.], [0., 426.842, 0.], [0., 0.1, 22.8421], [2143157 / 3., 426.842 / 3., 22.8421 / 3.], [6e5, 3, 14]],
    4: [[1890000, 0., 0.], [0., 1000., 0.], [0., 0.1, 43.052], [1890000 / 3., 1000. / 3., 43.052 / 3.], [6e5, 8, 16]],
    5: [[1890000, 0., 0.], [0., 843.6842, 0.], [0., 0.1, 30.7368], [1890000 / 3., 843.684 / 3., 30.7368 / 3.], [6e5, 10, 18]],
    6: [[40000, 0., 0.], [0., 895.789, 0.], [0., 0.1, 74.947], [40000 / 3., 895.78 / 3., 74.947 / 3.], [6e5, 25, 20]],
    7: [[40000, 0., 0.], [0., 682.105, 0.], [0., 0.1, 18.421], [40000 / 3., 687.3684 / 3., 18.421 / 3.], [6e5, 30, 22]],
}

# Hyperparameters for conditional sampling (per difficulty level).
# eta: DDIM eta parameter
# num_samples: number of samples to draw for majority voting
# num_steps: number of DDIM sampling steps
# use_ema: whether to use EMA model weights
LEVEL_TO_HPARAMS = {
    1: {'eta': 0.01, 'num_samples': 60, 'num_steps': 10, 'use_ema': True},
    2: {'eta': 0.01, 'num_samples': 60, 'num_steps': 10, 'use_ema': False},
    3: {'eta': 0.3, 'num_samples': 10, 'num_steps': 20, 'use_ema': False},
    4: {'eta': 0.9, 'num_samples': 10, 'num_steps': 100, 'use_ema': False},
    5: {'eta': 0.1, 'num_samples': 25, 'num_steps': 100, 'use_ema': True},
    6: {'eta': 0.1, 'num_samples': 25, 'num_steps': 100, 'use_ema': True},
    7: {'eta': 0.8, 'num_samples': 15, 'num_steps': 100, 'use_ema': False},
}
