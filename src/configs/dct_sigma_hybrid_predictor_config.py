import ml_collections

from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    config.training = training = get_base_training_config()
    training.batch_size = 64
    training.epochs = 200
    training.lr = 3e-4
    training.num_workers = 8
    training.log_freq = 50
    training.weight_decay = 1e-4
    training.scheduler_patience = 3
    training.scheduler_factor = 0.7
    training.early_stopping_patience = 15
    training.selection_metric = 'val_rel_l2'
    training.selection_metric_mode = 'min'
    training.coeff_loss_weight = 0.2
    training.mse_weight = 1.0
    training.mae_weight = 0.1
    training.focus_loss_weight = 2.0
    training.focus_threshold = 0.08
    training.fixed_level = 1
    training.save_atlas = True

    config.model = model = ml_collections.ConfigDict()
    model.input_dim = 2356
    model.hidden_dims = (1024, 512, 512)
    model.level_embed_dim = 32
    model.coeff_size = 20
    model.out_channels = 1
    model.dropout = 0.1
    model.image_size = 256
    model.refine_channels = 32
    model.refine_seed_size = 16

    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.ref_path = 'KTC2023/Codes_Python/TrainingData/ref.mat'
    data.mesh_name = 'Mesh_dense.mat'
    data.noise_std1 = 0.05
    data.noise_std2 = 0.01
    data.use_hdf5 = False
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None

    return config
