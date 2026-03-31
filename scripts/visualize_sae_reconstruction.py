"""
Visualize AE self-reconstruction on official ground-truth datasets.

Supports both:
  - continuous SAE
  - ST-1D-VQ-VAE (vq_sae)

and saves side-by-side GT vs reconstruction comparison figures under the
corresponding result directory.
"""

import argparse
from contextlib import nullcontext
import json
import os
import sys

import numpy as np
import scipy.io as sio
import torch
import yaml
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.visualization import plot_reconstruction_comparison
from src.evaluation.scoring_torch import fast_score_auto
from src.models.sae import SparseAutoEncoder
from src.models.vq_sae import ST1DVQVAE
from src.pipelines.base_pipeline import BasePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize AE reconstructions on official GT images')
    parser.add_argument('--weights-dir', type=str, required=True,
                        help='Trained SAE/VQ-SAE result directory (contains best.pt)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device (cuda/cpu)')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'sae', 'vq_sae'],
                        help='Model family. auto = infer from config/checkpoint')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['codes', 'eval', 'both'],
                        help='Which official GT set to visualize')
    parser.add_argument('--codes-gt-dir', type=str,
                        default='KTC2023/Codes_Python/GroundTruths',
                        help='Codes_Python GroundTruths directory')
    parser.add_argument('--eval-gt-dir', type=str,
                        default='KTC2023/EvaluationData/GroundTruths',
                        help='EvaluationData GroundTruths directory')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for SAE reconstruction visualization')
    parser.add_argument('--compute-score', action='store_true',
                        help='Also compute SSIM-based score (disabled by default)')
    return parser.parse_args()


def resolve_device(requested):
    if requested == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if requested == 'cuda':
        print('Warning: CUDA not available, falling back to cpu.')
    return 'cpu'


def autocast_context(device):
    if device == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def create_result_subdir(weights_dir, prefix='ae_gt_reconstruction'):
    num = 1
    while True:
        out_dir = os.path.join(weights_dir, f'{prefix}_{num}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        num += 1


def load_model_config(weights_dir):
    config_path = os.path.join(weights_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    with open(config_path, 'r') as f:
        return yaml.unsafe_load(f)


def infer_model_family(config, requested='auto'):
    if requested != 'auto':
        return requested
    model_cfg = config.get('model', {}) if isinstance(config, dict) else {}
    if 'num_slots' in model_cfg and 'codebook_size' in model_cfg:
        return 'vq_sae'
    return 'sae'


def build_model_from_config(config, device, family='auto'):
    model_cfg = config['model']
    family = infer_model_family(config, family)
    if family == 'vq_sae':
        model = ST1DVQVAE(
            in_channels=model_cfg.get('in_channels', 3),
            encoder_channels=tuple(model_cfg.get(
                'encoder_channels', (32, 64, 128, 256))),
            num_slots=model_cfg.get('num_slots', 16),
            codebook_size=model_cfg.get('codebook_size', 512),
            code_dim=model_cfg.get('code_dim', 32),
            decoder_start_size=model_cfg.get('decoder_start_size', 4),
            vq_beta=config.get('training', {}).get('vq_beta', 0.25),
        )
    else:
        model = SparseAutoEncoder(
            in_channels=model_cfg.get('in_channels', 3),
            encoder_channels=tuple(model_cfg.get(
                'encoder_channels', (32, 64, 128, 256))),
            shape_dim=model_cfg.get('shape_dim', 63),
            decoder_start_size=model_cfg.get('decoder_start_size', 4),
        )
    model.to(device)
    return model


def _looks_like_sae_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    keys = list(state_dict.keys())
    return any(k.startswith('angle_cnn.') for k in keys)


def _looks_like_vq_sae_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    keys = list(state_dict.keys())
    return any(k.startswith('quantizer.embedding.') for k in keys)


def resolve_model_weights(weights_dir, device, requested_family='auto'):
    """Resolve a full AE model for reconstruction visualization.

    Supports:
      1. Passing an SAE/VQ-SAE result directory directly.
      2. Passing an SAE predictor result directory.
    """
    config = load_model_config(weights_dir)
    ckpt_path = BasePipeline._find_weight([
        os.path.join(weights_dir, 'best.pt'),
        os.path.join(weights_dir, 'last.pt'),
    ])
    raw_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (raw_state.get('model_state_dict', raw_state)
                  if isinstance(raw_state, dict) else raw_state)

    family = infer_model_family(config, requested_family)

    # Case 1: this directory is already an AE result dir.
    if _looks_like_sae_state_dict(state_dict) or _looks_like_vq_sae_state_dict(state_dict):
        return config, ckpt_path, state_dict, family

    # Case 2: predictor dir. Resolve paired AE checkpoint from config.
    ae_cfg = {}
    if isinstance(config, dict):
        if 'sae' in config:
            ae_cfg = config.get('sae', {})
            family = 'sae'
        elif 'vq_sae' in config:
            ae_cfg = config.get('vq_sae', {})
            family = 'vq_sae'
    ae_ckpt = ae_cfg.get('checkpoint', '')
    if not ae_ckpt:
        raise RuntimeError(
            f'{weights_dir} does not contain SAE model weights, and its '
            'config.yaml does not provide paired autoencoder checkpoint.'
        )

    ae_dir = os.path.dirname(os.path.normpath(ae_ckpt))
    ae_config = load_model_config(ae_dir)
    ae_ckpt_path = BasePipeline._find_weight([
        os.path.join(ae_dir, 'best.pt'),
        os.path.join(ae_dir, 'last.pt'),
    ])
    ae_state = BasePipeline._load_state_dict(ae_ckpt_path, device)

    if family == 'sae' and isinstance(raw_state, dict) and 'sae_decoder_state_dict' in raw_state:
        model = build_model_from_config(ae_config, device, family=family)
        model.load_state_dict(ae_state)
        model.decoder.load_state_dict(raw_state['sae_decoder_state_dict'])
        return ae_config, ae_ckpt_path, model.state_dict(), family

    return ae_config, ae_ckpt_path, ae_state, family


def labels_to_onehot(label_img):
    onehot = np.zeros((3, 256, 256), dtype=np.float32)
    onehot[0] = (label_img == 0)
    onehot[1] = (label_img == 1)
    onehot[2] = (label_img == 2)
    return onehot


def reconstruct_batch(model, gt_labels, device):
    x_np = np.stack([labels_to_onehot(gt_label) for gt_label in gt_labels], axis=0)
    x = torch.from_numpy(x_np).to(device)
    with torch.no_grad():
        with autocast_context(device):
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        pred = torch.argmax(logits, dim=1)
    return pred.cpu().numpy().astype(np.int32)


def load_codes_python_groundtruths(gt_dir):
    ground_truths = []
    names = []
    for i in range(1, 5):
        path = os.path.join(gt_dir, f'true{i}.mat')
        truth = np.array(sio.loadmat(path)['truth'])
        ground_truths.append(truth)
        names.append(f'sample_{i}')
    return ground_truths, names


def load_eval_groundtruths(gt_dir, level):
    ground_truths = []
    names = []
    level_dir = os.path.join(gt_dir, f'level_{level}')
    for i in range(1, 4):
        path = os.path.join(level_dir, f'{i}_true.mat')
        truth = np.array(sio.loadmat(path)['truth'])
        ground_truths.append(truth)
        names.append(f'sample_{i}')
    return ground_truths, names


def evaluate_group(model, ground_truths, device, desc, batch_size,
                   compute_score=False):
    reconstructions = []
    scores = []
    pbar = tqdm(range(0, len(ground_truths), batch_size), desc=desc)
    for start in pbar:
        gt_batch = ground_truths[start:start + batch_size]
        reco_batch = reconstruct_batch(model, gt_batch, device)
        for gt, reco in zip(gt_batch, reco_batch):
            reconstructions.append(reco)
            if compute_score:
                scores.append(float(fast_score_auto(gt, reco, device=device)))
    return reconstructions, scores


def main():
    args = parse_args()
    device = resolve_device(args.device)

    config, ckpt_path, state_dict, family = resolve_model_weights(
        args.weights_dir, device, requested_family=args.method)
    model = build_model_from_config(config, device, family=family)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = create_result_subdir(args.weights_dir)
    print(f'Output directory: {output_dir}')

    summary = {
        'weights_dir': args.weights_dir,
        'checkpoint': ckpt_path,
        'family': family,
        'device': device,
        'groups': {},
    }

    if args.dataset in ('codes', 'both'):
        gts, _ = load_codes_python_groundtruths(args.codes_gt_dir)
        recos, scores = evaluate_group(
            model, gts, device, desc='Codes_Python GT',
            batch_size=args.batch_size,
            compute_score=args.compute_score)
        save_path = os.path.join(output_dir, 'codes_python_comparison.png')
        plot_reconstruction_comparison(
            gts, recos, level=1,
            scores=scores if args.compute_score else None,
            save_path=save_path)
        group_summary = {'plot': save_path}
        if args.compute_score:
            group_summary['scores'] = scores
            group_summary['mean_score'] = float(np.mean(scores))
            print(f'Codes_Python mean score: {np.mean(scores):.4f}')
        summary['groups']['codes_python'] = group_summary

    if args.dataset in ('eval', 'both'):
        eval_summary = {}
        for level in range(1, 8):
            gts, _ = load_eval_groundtruths(args.eval_gt_dir, level)
            recos, scores = evaluate_group(
                model, gts, device, desc=f'Eval GT L{level}',
                batch_size=args.batch_size,
                compute_score=args.compute_score)
            save_path = os.path.join(
                output_dir, f'evaluation_level_{level}_comparison.png')
            plot_reconstruction_comparison(
                gts, recos, level=level,
                scores=scores if args.compute_score else None,
                save_path=save_path)
            level_summary = {'plot': save_path}
            if args.compute_score:
                level_summary['scores'] = scores
                level_summary['mean_score'] = float(np.mean(scores))
                print(f'Evaluation level {level} mean score: {np.mean(scores):.4f}')
            eval_summary[str(level)] = level_summary
        summary['groups']['evaluation'] = eval_summary

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
