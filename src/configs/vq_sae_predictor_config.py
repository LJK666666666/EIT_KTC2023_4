import ml_collections
from .base_config import get_base_training_config


def get_configs():
    config = ml_collections.ConfigDict()
    config.device = 'cuda'
    config.seed = 1

    config.training = training = get_base_training_config()
    training.batch_size = 128
    training.epochs = 300
    training.lr = 3e-4
    training.num_workers = 8
    training.log_freq = 50
    training.weight_decay = 1e-4
    training.scheduler_patience = 10
    training.scheduler_factor = 0.5
    training.early_stopping_patience = 30
    training.lambda_angle = 0.5
    training.lambda_slot = 1.0
    training.lambda_image = 1.0
    training.ce_weight = 1.0
    training.dice_weight = 1.0
    training.score_probe_freq = 1
    training.selection_metric = 'val_probe_score_total'
    training.selection_metric_mode = 'max'

    config.model = model = ml_collections.ConfigDict()
    model.input_dim = 2356
    model.hidden_dims = (512, 256, 128)
    model.num_slots = 16
    model.codebook_size = 512
    model.dropout = 0.1

    config.vq_sae = vq_sae = ml_collections.ConfigDict()
    vq_sae.checkpoint = ''
    vq_sae.latent_h5_path = ''

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

