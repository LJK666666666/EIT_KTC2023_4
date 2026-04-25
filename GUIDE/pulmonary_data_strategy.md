# Pulmonary Data Strategy

## Conclusion About `Subjects Data`

The bundled real dataset under:

- [`Subjects Data/Plos One Data`](/D:/010_CodePrograms/E/EIT_KTC2023_4/Subjects%20Data/Plos%20One%20Data)

contains real thoracic EIT measurements, but it does **not** contain true conductivity images. This is not a flaw of this specific dataset; it is the normal situation for real pulmonary EIT. In living-human EIT, exact internal conductivity maps are not available as ground truth.

Therefore, this dataset is suitable for:

- measurement analysis,
- temporal breathing signal studies,
- future Sim2Real qualitative validation,

but **not** as a directly supervised training set for image reconstruction.

## Decision

Because the current study requires label-supervised pulmonary reconstruction experiments, the practical next step is to generate pulmonary-style simulated data locally, instead of waiting for an unavailable real ground-truth dataset.

This round therefore adopts the following strategy:

1. keep `Subjects Data` as future real-data validation material
2. build a local thorax-style simulated dataset for supervised training
3. use the current strongest image prior (`dct_predictor`) as the first pulmonary baseline

## Implemented Pulmonary Simulation

Added:

- [`src/data/lung_phantom.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/data/lung_phantom.py)
- [`scripts/generate_lung_data.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/generate_lung_data.py)

The phantom generator creates:

- two lung regions
- a heart / mediastinal conductive region
- optional unilateral collapse
- optional dependent conductive effusion-like regions

Label convention:

- `0`: background thoracic tissue
- `1`: lungs
- `2`: heart / fluid-like conductive region

The script then uses the existing FEM forward solver to generate EIT measurements and stores:

- `gt`
- `measurements`
- optional `sigma`
- `indices`

in HDF5 format.

## Generated Synthetic Datasets

Small smoke dataset:

- [`dataset_lung/level_1/data.h5`](/D:/010_CodePrograms/E/EIT_KTC2023_4/dataset_lung/level_1/data.h5)

Pilot dataset:

- [`dataset_lung_pilot/level_1/data.h5`](/D:/010_CodePrograms/E/EIT_KTC2023_4/dataset_lung_pilot/level_1/data.h5)

Expanded dataset:

- [`dataset_lung_2k/level_1/data.h5`](/D:/010_CodePrograms/E/EIT_KTC2023_4/dataset_lung_2k/level_1/data.h5)

Preview outputs:

- [`results/lung_data_preview_1/lung_phantoms.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/lung_data_preview_1/lung_phantoms.png)
- [`results/lung_data_preview_2/lung_phantoms.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/lung_data_preview_2/lung_phantoms.png)
- [`results/lung_data_preview_3/lung_phantoms.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/lung_data_preview_3/lung_phantoms.png)

The pilot HDF5 has:

- `512` samples
- `gt: (512, 256, 256)`
- `measurements: (512, 2356)`
- `sigma: (512, 256, 256)`

The expanded HDF5 has:

- `2048` samples
- `gt: (2048, 256, 256)`
- `measurements: (2048, 2356)`
- `sigma: (2048, 256, 256)`
- average forward time about `104.3 ms/sample`
- average total generation time about `108.6 ms/sample`

This allows a first pulmonary data-scaling observation for the DCT route:

- pilot (`512` samples) DCT test score:
  - [`results/dct_predictor_lungpilot_1/dct_predictor_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lungpilot_1/dct_predictor_test_eval_1/summary.json)
  - `mean_score = 0.513939`
- expanded (`2048` samples) DCT test score:
  - [`results/dct_predictor_lung2k_1/dct_predictor_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1/dct_predictor_test_eval_1/summary.json)
  - `mean_score = 0.727184`

So for pulmonary DCT, moving from `512` to `2048` synthetic samples already produces a large gain, which suggests that data scaling remains an effective lever for the low-frequency fixed-basis route.

## Pulmonary Baselines

A first quick training run with the current strongest benchmark-side model was completed:

- [`results/dct_predictor_lungpilot_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lungpilot_1)

Qualitative prediction visualization:

- [`results/dct_predictor_lungpilot_1/dct_samples_1/test_comparison.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lungpilot_1/dct_samples_1/test_comparison.png)

This does not yet establish a final pulmonary benchmark, but it confirms that:

- the pulmonary phantom generator works,
- the forward simulation chain works,
- `dct_predictor` can be trained on the pulmonary-style HDF5 dataset without code changes.

We then scaled the same experiment to the `2048`-sample dataset:

- [`results/dct_predictor_lung2k_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1)

Qualitative prediction visualization:

- [`results/dct_predictor_lung2k_1/dct_samples_1/test_comparison.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1/dct_samples_1/test_comparison.png)

Important observations from [`results/dct_predictor_lung2k_1/training_log.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1/training_log.json):

- validation probe score improves rapidly during the first few epochs
- the best validation probe score appears at epoch `6`
- after epoch `6`, `val_loss` keeps decreasing, but probe score slowly falls or plateaus

Best observed pulmonary validation probe result on the 2k set:

- `val_probe_score_mean = 0.7325`
- `val_probe_score_total = 1046.02`

This repeats the same phenomenon already seen in the benchmark study:

- image-domain loss and structure-aware score are not perfectly aligned
- checkpoint selection should prefer score-aware criteria rather than loss alone

We also launched a same-dataset FCUNet comparison run:

- [`results/fcunet_lung2k_2`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/fcunet_lung2k_2)

This run did not finish within the current local time budget, but the partial log already provides a useful engineering conclusion:

- FCUNet stage-2 training is much slower than `dct_predictor` on the same pulmonary HDF5
- the first `8` full-training epochs already required several hours locally
- usable intermediate checkpoints were still produced because the training loop saves `best.pt` during training

So even before comparing final pulmonary scores, the current fixed-basis DCT route has a major throughput advantage for pulmonary experimentation.

## Real PLOS Data Compatibility

The bundled PLOS One thoracic `.get` files were parsed and verified with:

- [`scripts/parse_plos_get.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/parse_plos_get.py)
- [`results/plos_get_analysis_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/plos_get_analysis_1/summary.json)

The important compatibility result is:

- the real `.get` data correspond to a `16`-electrode acquisition layout
- the reordered measurement format is `208 x T`
- the current synthetic pulmonary training data and neural pipelines remain in the KTC-style `32`-electrode setting with measurement length `2356`

Therefore the current pulmonary models **cannot** yet be directly applied to the parsed PLOS One measurements. A true Sim2Real pulmonary study now requires one of two future steps:

1. build a new `16`-electrode simulation and training pipeline aligned with the `.get` format
2. find or generate another pulmonary dataset already matching the current `32`-electrode benchmark setup

At this stage, the practical decision is to continue using the synthetic pulmonary dataset for supervised model comparison and treat the parsed PLOS `.get` files only as future real-data analysis material.

## Quantitative Lung Test Evaluation

To avoid relying only on training-time validation probes, a dedicated simulated-split evaluation script was added:

- [`scripts/evaluate_dct_sim.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/evaluate_dct_sim.py)

This script evaluates the full HDF5 train/val/test split with the same fast SSIM scorer used elsewhere and stores the results inside the model result directory.

We used it to compare two pulmonary DCT runs and their equal-weight ensemble on the `dataset_lung_2k` test split.

Single-model baseline:

- [`results/dct_predictor_lung2k_1/dct_predictor_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1/dct_predictor_test_eval_1/summary.json)
- `mean_score = 0.727184`
- `total_score = 149.799915`

Coefficient-heavy variant (`coeff_loss_weight = 1.0`):

- [`results/dct_predictor_lung2k_coeff1_1/dct_predictor_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_coeff1_1/dct_predictor_test_eval_1/summary.json)
- `mean_score = 0.726494`
- `total_score = 149.657836`

Equal-weight ensemble of the two:

- [`scripts/dct_predictor_lung_ensemble.yaml`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/dct_predictor_lung_ensemble.yaml)
- [`results/dct_predictor_lung2k_1/dct_predictor_ensemble_test_eval_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1/dct_predictor_ensemble_test_eval_1/summary.json)
- `mean_score = 0.726875`
- `total_score = 149.736297`

The main conclusion is straightforward:

- the original `dct_predictor_lung2k_1` remains the best pulmonary model among the tested variants
- increasing coefficient supervision to `1.0` did not improve the test split
- a simple `0.5 / 0.5` ensemble of these two pulmonary runs also did **not** beat the best single model

So, unlike the benchmark-side KTC ensemble, the current pulmonary pair does not yet provide useful ensemble complementarity.

## FCUNet Pulmonary Test Result

The partial but already trained FCUNet pulmonary run was also evaluated directly on the same `dataset_lung_2k` test split:

- intermediate checkpoint:
  - [`results/fcunet_lung2k_2/fcunet_test_eval_2/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/fcunet_lung2k_2/fcunet_test_eval_2/summary.json)
  - `mean_score = 0.781489`
  - `total_score = 160.986691`

After continuing the same run to `20` full-training epochs, the updated best checkpoint reached:

- [`results/fcunet_lung2k_2/fcunet_test_eval_3/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/fcunet_lung2k_2/fcunet_test_eval_3/summary.json)
- `mean_score = 0.781912`
- `total_score = 161.073831`

This result is substantially higher than the current pulmonary DCT runs:

- best pulmonary DCT single model: `0.727184 / 149.799915`
- pulmonary DCT two-model ensemble: `0.726875 / 149.736297`

So the current pulmonary conclusion is different from the benchmark-side KTC conclusion:

- on KTC benchmark phantoms, DCT-based low-frequency prediction is the strongest route so far
- on the present lung-style synthetic dataset, even a partially trained FCUNet checkpoint already outperforms the pulmonary DCT baselines by a clear margin

The tradeoff is now explicit:

- `dct_predictor` is much cheaper and faster to train
- `fcunet` currently reconstructs pulmonary structures better on the held-out test split

The current inference-time tradeoff on the same `206`-sample test split is:

- FCUNet:
  - reconstruction time `3.72 s`
  - about `18.1 ms/sample`
- pulmonary DCT single model:
  - reconstruction time `0.36 s`
  - about `1.73 ms/sample`
- pulmonary DCT ensemble:
  - reconstruction time `0.32 s`
  - about `1.55 ms/sample`

So FCUNet is currently the stronger pulmonary reconstructor, while DCT remains much lighter computationally.

More directly, a dedicated latency benchmark gives:

- [`results/dct_predictor_lung2k_1/dct_predictor_latency_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1/dct_predictor_latency_1/summary.json)
  - parameters: `3.85M`
  - single-sample latency: `2.136 ms`
  - batch per-sample latency (`batch=32`): `0.250 ms`
- [`results/fcunet_lung2k_2/fcunet_latency_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/fcunet_lung2k_2/fcunet_latency_1/summary.json)
  - parameters: `40.70M`
  - single-sample latency: `28.822 ms`
  - batch per-sample latency (`batch=32`): `14.600 ms`

This makes the role separation very clear:

- FCUNet is the stronger pulmonary baseline in reconstruction quality
- DCT is the much lighter and faster proposed method
- future pulmonary DCT work should therefore emphasize efficiency, deployability, and data-efficiency rather than claim the current best absolute pulmonary accuracy

Paper-ready summary figures were generated in:

- [`results/pulmonary_summary_1/pulmonary_score_scaling.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_summary_1/pulmonary_score_scaling.png)
- [`results/pulmonary_summary_1/pulmonary_efficiency.png`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_summary_1/pulmonary_efficiency.png)
- [`results/pulmonary_summary_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_summary_1/summary.json)

Therefore, pulmonary work should no longer assume that the DCT route is automatically the best choice simply because it wins on the benchmark phantom task.

## Next Step

The next pulmonary step should be:

1. continue training the `fcunet` pulmonary control run to completion and verify whether the current margin persists
2. only revisit pulmonary DCT if a new representation change can plausibly close the gap to `fcunet`
3. if real-data transfer remains a goal, build a separate `16`-electrode pulmonary simulation/training chain aligned with the PLOS `.get` format
4. treat benchmark-phantom and pulmonary conclusions as different regimes rather than forcing one method family to dominate both
