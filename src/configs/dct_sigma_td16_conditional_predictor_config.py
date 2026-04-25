import ml_collections

from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    config.training = training = get_base_training_config()
    training.batch_size = 64
    training.epochs = 120
    training.lr = 3e-4
    training.num_workers = 8
    training.log_freq = 50
    training.weight_decay = 1e-4
    training.scheduler_patience = 3
    training.scheduler_factor = 0.7
    training.early_stopping_patience = 15
    training.selection_metric = 'val_rmse'
    training.selection_metric_mode = 'min'
    training.coeff_loss_weight = 0.25
    training.mse_weight = 1.0
    training.mae_weight = 0.2
    training.active_threshold = 0.02
    training.active_region_threshold = 0.02
    training.inactive_weight = 1.0
    training.pred_l1_weight = 0.0
    training.mask_threshold = 0.02
    training.lambda_mask = 0.5
    training.lambda_mask_coeff = 0.1
    training.mask_teacher_forcing_ratio = 0.5
    training.active_oversample_factor = 1.0

    config.model = model = ml_collections.ConfigDict()
    model.input_dim = 208
    model.hidden_dims = (512, 256, 256)
    model.level_embed_dim = 16
    model.coeff_size = 24
    model.out_channels = 1
    model.dropout = 0.1
    model.mask_condition_dim = 128

    config.data = data = ml_collections.ConfigDict()
    data.im_size = 256
    data.use_hdf5 = False
    data.hdf5_path = ''
    data.train_indices = None
    data.val_indices = None
    data.test_indices = None

    return config
