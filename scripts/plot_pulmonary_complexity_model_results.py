"""Plot pulmonary sigma model results under matched low/high complexity."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_eval_summary(result_dir: str):
    eval_dir = None
    base = Path(result_dir)
    for child in sorted(base.iterdir()):
        if child.is_dir() and (
            child.name.startswith("dct_sigma_test_eval_")
            or child.name.startswith("fc_sigmaunet_test_eval_")
        ):
            eval_dir = child
    if eval_dir is None:
        raise FileNotFoundError(
            f"No *_test_eval_* directory found under {result_dir}")
    return load_json(str(eval_dir / "summary.json"))


def main():
    cfg = load_yaml("scripts/pulmonary_complexity_model_comparison.yaml")
    variability = load_json(cfg["lung_variability_summary"])
    out_dir = create_result_subdir("results", "pulmonary_complexity_models")

    levels = ["low", "high"]
    atlas = []
    oracle = []
    direct = []
    residual = []
    hybrid = []
    atlas_decoder = []
    fc_sigmaunet = []

    summary = {"datasets": {}}

    for level in levels:
        ds_cfg = cfg["datasets"][level]
        atlas_val = variability["presets"][level]["atlas_rel_l2_mean"]
        oracle_val = None
        for item in variability["presets"][level]["curves"]:
            if int(item["coeff_size"]) == 20:
                oracle_val = item["atlas_res_rel_l2_mean"]
                break
        if oracle_val is None:
            raise ValueError(f"Missing coeff_size=20 curve for {level}")

        direct_sum = get_eval_summary(ds_cfg["direct_dir"])
        residual_sum = get_eval_summary(ds_cfg["residual_dir"])
        hybrid_sum = get_eval_summary(ds_cfg["hybrid_dir"])
        atlas_decoder_sum = get_eval_summary(ds_cfg["atlas_decoder_dir"])
        fc_sigmaunet_sum = get_eval_summary(ds_cfg["fc_sigmaunet_dir"])

        atlas.append(atlas_val)
        oracle.append(oracle_val)
        direct.append(direct_sum["rel_l2_mean"])
        residual.append(residual_sum["rel_l2_mean"])
        hybrid.append(hybrid_sum["rel_l2_mean"])
        atlas_decoder.append(atlas_decoder_sum["rel_l2_mean"])
        fc_sigmaunet.append(fc_sigmaunet_sum["rel_l2_mean"])

        summary["datasets"][level] = {
            "atlas_rel_l2": atlas_val,
            "oracle_atlas_residual_dct_k20_rel_l2": oracle_val,
            "direct_rel_l2": direct_sum["rel_l2_mean"],
            "residual_rel_l2": residual_sum["rel_l2_mean"],
            "hybrid_rel_l2": hybrid_sum["rel_l2_mean"],
            "atlas_decoder_rel_l2": atlas_decoder_sum["rel_l2_mean"],
            "fc_sigmaunet_rel_l2": fc_sigmaunet_sum["rel_l2_mean"],
            "direct_mae": direct_sum["mae_mean"],
            "residual_mae": residual_sum["mae_mean"],
            "hybrid_mae": hybrid_sum["mae_mean"],
            "atlas_decoder_mae": atlas_decoder_sum["mae_mean"],
            "fc_sigmaunet_mae": fc_sigmaunet_sum["mae_mean"],
            "direct_rmse": direct_sum["rmse_mean"],
            "residual_rmse": residual_sum["rmse_mean"],
            "hybrid_rmse": hybrid_sum["rmse_mean"],
            "atlas_decoder_rmse": atlas_decoder_sum["rmse_mean"],
            "fc_sigmaunet_rmse": fc_sigmaunet_sum["rmse_mean"],
        }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    x = np.arange(len(levels), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, atlas, marker="o", linewidth=2.0, label="Atlas baseline")
    ax.plot(x, direct, marker="s", linewidth=2.0, label="Direct DCT")
    ax.plot(x, residual, marker="^", linewidth=2.0, label="Atlas-residual DCT")
    ax.plot(x, hybrid, marker="D", linewidth=2.0, label="Hybrid DCT")
    ax.plot(x, atlas_decoder, marker="P", linewidth=2.0, label="Atlas-decoder")
    ax.plot(x, fc_sigmaunet, marker="v", linewidth=2.0,
            label="FC-SigmaUNet (pilot)")
    ax.plot(x, oracle, marker="x", linewidth=2.0, linestyle="--",
            label="Oracle atlas-res DCT (K=20)")
    ax.set_xticks(x, ["Low", "High"])
    ax.set_ylabel("Test relative L2")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out_dir, "complexity_rel_l2.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    bar_labels = ["Low", "High"]
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 2.0 * width, direct, width=width, label="Direct DCT", color="#4c78a8")
    ax.bar(x - 1.0 * width, residual, width=width, label="Residual", color="#f58518")
    ax.bar(x + 0.0 * width, hybrid, width=width, label="Hybrid", color="#54a24b")
    ax.bar(x + 1.0 * width, atlas_decoder, width=width,
           label="Atlas-decoder", color="#e45756")
    ax.bar(x + 2.0 * width, fc_sigmaunet, width=width,
           label="FC-SigmaUNet (pilot)", color="#72b7b2")
    ax.plot(x, atlas, color="black", marker="o", linewidth=1.8, label="Atlas")
    ax.set_xticks(x, bar_labels)
    ax.set_ylabel("Test relative L2")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(os.path.join(out_dir, "complexity_bar.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
