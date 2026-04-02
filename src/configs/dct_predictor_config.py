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
    training.score_probe_freq = 1
    training.score_probe_max_samples = 256
    training.selection_metric = 'val_probe_score_total'
    training.selection_metric_mode = 'max'
    training.coeff_loss_weight = 0.5
    training.ce_weight = 1.0
    training.dice_weight = 1.0
    training.fixed_level = None

    config.model = model = ml_collections.ConfigDict()
    model.input_dim = 2356
    model.hidden_dims = (1024, 512, 512)
    model.level_embed_dim = 32
    model.coeff_size = 16
    model.out_channels = 3
    model.dropout = 0.1

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
