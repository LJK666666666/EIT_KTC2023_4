import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DifferenceConductivityHDF5Dataset
from src.evaluation.regression_metrics import masked_regression_metrics_batch
from src.models.pulmonary_vae import ConvVAE, LatentMLPPredictor


def create_result_subdir(base_dir: str, tag: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    idx = 1
    while True:
        out_dir = os.path.join(base_dir, f'{tag}_{idx}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=False)
            return out_dir
        idx += 1


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.unsafe_load(f)


def resolve_split_indices(h5_path: str, split: str, predictor_cfg=None):
    if predictor_cfg is not None:
        data_cfg = predictor_cfg.get('data', {}) or {}
        indices = data_cfg.get(f'{split}_indices')
        if indices is not None:
            return list(indices)
    import h5py

    with h5py.File(h5_path, 'r') as f:
        total = int(f['sigma_delta'].shape[0])
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    all_indices = list(range(total))
    mapping = {
        'train': all_indices[:n_train],
        'val': all_indices[n_train:n_train + n_val],
        'test': all_indices[n_train + n_val:],
    }
    return mapping[split]


def nonzero_rel_l2_batch(targets, preds, masks, eps=1e-8):
    vals = []
    for target, pred, mask in zip(targets, preds, masks):
        mask = np.asarray(mask, dtype=bool)
        diff = (pred - target)[mask]
        tgt = target[mask]
        denom = np.linalg.norm(tgt)
        if denom > eps:
            vals.append(float(np.linalg.norm(diff) / denom))
    return float(np.mean(vals)) if vals else float('nan')


def summarize_metrics(targets, preds, masks, active_threshold):
    metrics = masked_regression_metrics_batch(
        targets, preds, masks=masks, active_threshold=active_threshold)
    out = {
        'mae': float(np.mean(metrics['mae'])),
        'rmse': float(np.mean(metrics['rmse'])),
        'rel_l2': float(np.mean(metrics['rel_l2'])),
        'rel_l2_nonzero': nonzero_rel_l2_batch(targets, preds, masks),
    }
    if 'active_rel_l2' in metrics:
        out['active_rel_l2'] = float(np.nanmean(metrics['active_rel_l2']))
    return out


def save_comparison_figure(out_dir, targets, ae_preds, pred_preds, masks, num_samples):
    num_samples = min(num_samples, len(targets))
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(num_samples):
        tgt = targets[i]
        ae = ae_preds[i]
        pred = pred_preds[i]
        err = np.abs(pred - tgt) * masks[i]
        vmax = max(
            np.max(np.abs(tgt[masks[i]])) if np.any(masks[i]) else 1.0,
            np.max(np.abs(ae[masks[i]])) if np.any(masks[i]) else 1.0,
            np.max(np.abs(pred[masks[i]])) if np.any(masks[i]) else 1.0,
            1e-6,
        )
        err_max = max(np.max(err), 1e-6)
        panels = [
            (tgt, 'viridis', -vmax, vmax, 'GT'),
            (ae, 'viridis', -vmax, vmax, 'AE'),
            (pred, 'viridis', -vmax, vmax, 'Predictor'),
            (err, 'magma', 0.0, err_max, 'Abs Error'),
        ]
        for j, (img, cmap, vmin, vmax_panel, label) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax_panel)
            ax.axis('off')
            if i == 0:
                ax.set_xlabel(label)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze TD16 VAE autoencoder upper bound vs latent predictor gap.')
    parser.add_argument('--predictor-dir', type=str, required=True)
    parser.add_argument('--hdf5-path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--active-threshold', type=float, default=0.02)
    parser.add_argument('--preview-count', type=int, default=4)
    parser.add_argument('--experiment-tag', type=str, default='td16_vae_gap')
    return parser.parse_args()


def main():
    args = parse_args()
    predictor_dir = args.predictor_dir
    predictor_cfg = load_yaml(os.path.join(predictor_dir, 'config.yaml'))
    vae_ckpt = predictor_cfg['vae']['checkpoint']
    predictor_ckpt = os.path.join(predictor_dir, 'best.pt')
    if not os.path.exists(predictor_ckpt):
        predictor_ckpt = os.path.join(predictor_dir, 'last.pt')
    vae_cfg = load_yaml(os.path.join(os.path.dirname(vae_ckpt), 'config.yaml'))

    indices = resolve_split_indices(args.hdf5_path, args.split, predictor_cfg=predictor_cfg)
    dataset = DifferenceConductivityHDF5Dataset(args.hdf5_path, indices=indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')

    vae = ConvVAE(
        in_channels=vae_cfg['model']['in_channels'],
        latent_dim=vae_cfg['model']['latent_dim'],
        base_channels=vae_cfg['model']['base_channels'],
    ).to(device)
    vae_state = torch.load(vae_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in vae_state:
        vae_state = vae_state['model_state_dict']
    vae.load_state_dict(vae_state)
    vae.eval()

    predictor = LatentMLPPredictor(
        input_dim=predictor_cfg['model']['input_dim'],
        latent_dim=predictor_cfg['model']['latent_dim'],
        hidden_dims=tuple(predictor_cfg['model']['hidden_dims']),
        dropout=predictor_cfg['model']['dropout'],
    ).to(device)
    pred_state = torch.load(predictor_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in pred_state:
        pred_state = pred_state['model_state_dict']
    predictor.load_state_dict(pred_state)
    predictor.eval()

    targets_all, masks_all, ae_all, pred_all = [], [], [], []
    with torch.no_grad():
        for measurements, sigma_delta, domain_mask, *rest in tqdm(
                loader, desc=f'Analyzing {args.split}'):
            measurements = measurements.to(device)
            sigma_delta = sigma_delta.to(device)
            domain_mask = domain_mask.to(device)
            ae_recon, _, _ = vae(sigma_delta)
            latent = predictor(measurements)
            pred_recon = vae.decode(latent)
            targets_all.append(sigma_delta[:, 0].float().cpu().numpy())
            masks_all.append(domain_mask[:, 0].float().cpu().numpy() > 0.5)
            ae_all.append(ae_recon[:, 0].float().cpu().numpy())
            pred_all.append(pred_recon[:, 0].float().cpu().numpy())

    targets = np.concatenate(targets_all, axis=0)
    masks = np.concatenate(masks_all, axis=0)
    ae_preds = np.concatenate(ae_all, axis=0)
    pred_preds = np.concatenate(pred_all, axis=0)

    ae_metrics = summarize_metrics(targets, ae_preds, masks, args.active_threshold)
    pred_metrics = summarize_metrics(targets, pred_preds, masks, args.active_threshold)

    out_dir = create_result_subdir(predictor_dir, args.experiment_tag)
    summary = {
        'split': args.split,
        'num_samples': int(targets.shape[0]),
        'predictor_dir': predictor_dir,
        'vae_checkpoint': vae_ckpt,
        'predictor_metrics': pred_metrics,
        'autoencoder_metrics': ae_metrics,
        'rmse_gap': float(pred_metrics['rmse'] - ae_metrics['rmse']),
        'active_rel_l2_gap': float(
            pred_metrics.get('active_rel_l2', float('nan')) -
            ae_metrics.get('active_rel_l2', float('nan'))
        ),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    save_comparison_figure(
        out_dir, targets, ae_preds, pred_preds, masks, args.preview_count)
    print(json.dumps(summary, indent=2))
    print(f'Saved to: {out_dir}')


if __name__ == '__main__':
    main()
