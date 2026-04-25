# Pulmonary Inductive Bias Adaptation

## Motivation

Pulmonary conductivity maps are structurally different from the KTC2023
phantom-style segmentation targets.

The current thorax-style simulator produces:

- a largely fixed global thorax layout,
- two lungs in stable positions,
- a central heart / mediastinal region,
- only mild local deviations such as collapse, effusion-like conductive
  regions, and small vessel-like conductive patches.

This means the pulmonary task is better described as:

- **coarse global anatomy + subtle local variation**

rather than:

- **arbitrary object-shape reconstruction**

The concern raised during inspection of
[`results/dct_sigma_lung2k_1/dct_sigma_test_samples_1/test_comparison.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_lung2k_1/dct_sigma_test_samples_1/test_comparison.png)
was correct: the prediction columns are highly similar and look like a shared
average thorax template.

## Important Diagnostic Baseline

To quantify the "average template" effect, a dedicated baseline script was
added:

- [`scripts/evaluate_sigma_mean_baseline.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/evaluate_sigma_mean_baseline.py)

This baseline simply predicts the mean conductivity map of the training split.

On the 2048-sample pulmonary dataset:

- [`results/sigma_mean_lung2k_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/sigma_mean_lung2k_1/summary.json)
  - `MAE = 0.1315`
  - `RMSE = 0.2025`
  - `RelL2 = 0.2552`

This is a crucial finding because it is **better** than the current direct
continuous DCT predictor:

- [`results/dct_sigma_lung2k_1/dct_sigma_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_lung2k_1/dct_sigma_test_eval_1/summary.json)
  - `MAE = 0.2231`
  - `RMSE = 0.2806`
  - `RelL2 = 0.3509`

Therefore, the original continuous DCT regression is not yet using the
measurement information effectively enough. It underperforms a trivial
anatomical-atlas baseline.

## Adaptation 1: Atlas-Residual DCT

To explicitly encode the pulmonary prior, a new model was added:

- [`src/models/dct_predictor/model.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/models/dct_predictor/model.py)
  - `AtlasResidualDCTPredictor`
- [`src/trainers/dct_sigma_residual_predictor_trainer.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/dct_sigma_residual_predictor_trainer.py)
- [`src/pipelines/dct_sigma_residual_predictor_pipeline.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/pipelines/dct_sigma_residual_predictor_pipeline.py)

Design:

- compute the train-split mean conductivity map as a fixed pulmonary atlas,
- let the network predict only the residual around the atlas,
- decode the residual through low-frequency DCT,
- reconstruct as `atlas + residual`.

The trainer saves:

- `atlas.npy`

inside the experiment folder for reproducibility.

Pilot result on `dataset_lung_pilot`:

- [`results/dct_sigma_residual_pilot_1/dct_sigma_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_residual_pilot_1/dct_sigma_test_eval_1/summary.json)
  - `MAE = 0.1292`
  - `RMSE = 0.2020`
  - `RelL2 = 0.2521`

This is much better than the old continuous DCT line and now roughly matches
the mean-atlas baseline. The model no longer wastes capacity rediscovering the
global thorax structure.

## Adaptation 2: Atlas + Coarse DCT + Local Refinement

Because pure low-frequency residual DCT is still too smooth, a second model was
added:

- [`src/models/dct_predictor/model.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/models/dct_predictor/model.py)
  - `AtlasRefineDCTPredictor`
- [`src/trainers/dct_sigma_hybrid_predictor_trainer.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/dct_sigma_hybrid_predictor_trainer.py)
- [`src/pipelines/dct_sigma_hybrid_predictor_pipeline.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/pipelines/dct_sigma_hybrid_predictor_pipeline.py)

Design:

- fixed atlas prior,
- coarse low-frequency DCT residual branch,
- additional image-space refinement decoder for local details.

Pilot result on `dataset_lung_pilot`:

- [`results/dct_sigma_hybrid_pilot_1/dct_sigma_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_hybrid_pilot_1/dct_sigma_test_eval_1/summary.json)
  - `MAE = 0.1287`
  - `RMSE = 0.2019`
  - `RelL2 = 0.2523`

This is only marginally different from the atlas-residual result and does not
yet provide a clear advantage.

Qualitatively, the prediction columns remain highly similar:

- [`results/dct_sigma_residual_pilot_1/dct_sigma_test_samples_1/test_comparison.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_residual_pilot_1/dct_sigma_test_samples_1/test_comparison.png)
- [`results/dct_sigma_hybrid_pilot_1/dct_sigma_test_samples_1/test_comparison.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_hybrid_pilot_1/dct_sigma_test_samples_1/test_comparison.png)

## Current Conclusion

At this stage, the main bottleneck is not "recovering a reasonable thorax
shape". The atlas already does that.

The true difficulty is:

- recovering **small but clinically meaningful local deviations** from boundary
  measurements.

This suggests the next useful research directions are:

1. residual-focused losses that upweight deviations from the atlas,
2. stronger local refinement branches or patch-aware decoders,
3. data-generation changes that increase local anatomical variability,
4. explicit comparison against the train-mean atlas baseline in the paper.

The atlas baseline is therefore not just a diagnostic curiosity; it is now a
required pulmonary baseline for all subsequent continuous-conductivity
experiments.
