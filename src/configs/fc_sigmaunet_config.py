import ml_collections

from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    config.training = training = get_base_training_config()
    training.batch_size = 16
    training.epochs = 20
    training.init_epochs = 0
    training.lr = 3e-5
    training.init_lr = 1e-4
    training.log_freq = 50
    training.num_workers = 8
    training.weight_decay = 1e-4
    training.scheduler_patience = 3
    training.scheduler_factor = 0.7
    training.early_stopping_patience = 15
    training.selection_metric = 'val_rel_l2'
    training.selection_metric_mode = 'min'
    training.mse_weight = 1.0
    training.mae_weight = 0.1
    training.fixed_level = 1

    config.model = model = ml_collections.ConfigDict()
    model.in_channels = 1
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
    model.max_period = 0.25

    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.ref_path = 'KTC2023/Codes_Python/TrainingData/ref.mat'
    data.mesh_name = 'Mesh_dense.mat'
    data.noise_std1 = 0.05
    data.noise_std2 = 0.01
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None
    data.use_hdf5 = True

    return config
