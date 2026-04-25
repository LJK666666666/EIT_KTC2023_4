"""Create pulmonary study summary plots from existing result files."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_result_subdir(base_dir: str, tag: str) -> str:
    base = Path(base_dir)
    idx = 1
    while (base / f"{tag}_{idx}").exists():
        idx += 1
    out_dir = base / f"{tag}_{idx}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return str(out_dir)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    out_dir = create_result_subdir("results", "pulmonary_summary")

    dct_512 = load_json("results/dct_predictor_lungpilot_1/dct_predictor_test_eval_1/summary.json")
    dct_2k = load_json("results/dct_predictor_lung2k_1/dct_predictor_test_eval_1/summary.json")
    dct_2k_ens = load_json("results/dct_predictor_lung2k_1/dct_predictor_ensemble_test_eval_1/summary.json")
    fc_2k = load_json("results/fcunet_lung2k_2/fcunet_test_eval_3/summary.json")
    dct_lat = load_json("results/dct_predictor_lung2k_1/dct_predictor_latency_1/summary.json")
    fc_lat = load_json("results/fcunet_lung2k_2/fcunet_latency_1/summary.json")

    summary = {
        "dct_512_mean_score": dct_512["mean_score"],
        "dct_2k_mean_score": dct_2k["mean_score"],
        "dct_2k_ensemble_mean_score": dct_2k_ens["mean_score"],
        "fcunet_2k_mean_score": fc_2k["mean_score"],
        "dct_params": dct_lat["num_params"],
        "fcunet_params": fc_lat["num_params"],
        "dct_single_latency_ms": dct_lat["single_latency_ms_mean"],
        "fcunet_single_latency_ms": fc_lat["single_latency_ms_mean"],
        "dct_batch_per_sample_ms": dct_lat["batch_per_sample_ms_mean"],
        "fcunet_batch_per_sample_ms": fc_lat["batch_per_sample_ms_mean"],
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Plot 1: data scaling / score
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.array([512, 2048], dtype=float)
    y = np.array([dct_512["mean_score"], dct_2k["mean_score"]], dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.0, label="DCT predictor")
    ax.scatter([2048], [dct_2k_ens["mean_score"]], marker="s", s=70,
               label="DCT ensemble")
    ax.scatter([2048], [fc_2k["mean_score"]], marker="^", s=70,
               label="FCUNet")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Mean fast score")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out_dir, "pulmonary_score_scaling.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: params and latency
    labels = ["DCT", "FCUNet"]
    params_m = [dct_lat["num_params"] / 1e6, fc_lat["num_params"] / 1e6]
    single_ms = [dct_lat["single_latency_ms_mean"], fc_lat["single_latency_ms_mean"]]
    batch_ms = [dct_lat["batch_per_sample_ms_mean"], fc_lat["batch_per_sample_ms_mean"]]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].bar(labels, params_m, color=["#4c78a8", "#f58518"])
    axes[0].set_ylabel("Params (M)")
    axes[1].bar(labels, single_ms, color=["#4c78a8", "#f58518"])
    axes[1].set_ylabel("Single latency (ms)")
    axes[2].bar(labels, batch_ms, color=["#4c78a8", "#f58518"])
    axes[2].set_ylabel("Batch per-sample (ms)")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "pulmonary_efficiency.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
