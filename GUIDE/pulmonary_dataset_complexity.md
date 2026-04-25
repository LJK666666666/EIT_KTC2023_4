# Pulmonary Dataset Complexity Analysis

## Motivation

After moving the paper focus from "finding a universally best network" to
"data-centric pulmonary EIT research", two questions became central:

1. How strongly is the pulmonary synthetic task dominated by a stable mean
   thoracic template?
2. Is the current difficulty caused by an insufficient image representation or
   by the measurement-to-image inverse map itself?

To answer these questions, two new analysis scripts were added:

- [`scripts/analyze_pulmonary_dataset_complexity.py`](../scripts/analyze_pulmonary_dataset_complexity.py)
- [`scripts/analyze_lung_generator_variability.py`](../scripts/analyze_lung_generator_variability.py)

The lung phantom generator was also extended with explicit variation controls:

- `anatomy_scale`
- `pathology_scale`
- `detail_scale`
- `conductivity_scale`
- `texture_scale`

implemented in:

- [`src/data/lung_phantom.py`](../src/data/lung_phantom.py)
- [`scripts/generate_lung_data.py`](../scripts/generate_lung_data.py)

## 1. Complexity of the Existing `lung2k` Dataset

Analysis result:

- [`results/pulmonary_complexity_2/summary.json`](../results/pulmonary_complexity_2/summary.json)

Visual outputs:

- [`results/pulmonary_complexity_2/atlas_std.png`](../results/pulmonary_complexity_2/atlas_std.png)
- [`results/pulmonary_complexity_2/dct_compressibility.png`](../results/pulmonary_complexity_2/dct_compressibility.png)
- [`results/pulmonary_complexity_2/residual_ratio_hist.png`](../results/pulmonary_complexity_2/residual_ratio_hist.png)
- [`results/pulmonary_complexity_2/focus_fraction.png`](../results/pulmonary_complexity_2/focus_fraction.png)

### Train-mean atlas baseline

On the `dataset_lung_2k/level_1/data.h5` test split:

- `MAE = 0.1316`
- `RMSE = 0.2027`
- `relative L2 = 0.2553`

This is consistent with the earlier dedicated baseline run
[`results/sigma_mean_lung2k_1/summary.json`](../results/sigma_mean_lung2k_1/summary.json)
and confirms that the pulmonary synthetic conductivity task is strongly
atlas-dominated.

### Oracle DCT compressibility

The key observation is that the data itself is still highly compressible once
the mean atlas is removed.

For the full test split:

- direct full-image DCT with `K=20`: `relative L2 = 0.1996`
- atlas-residual DCT with `K=20`: `relative L2 = 0.1593`

So the conductivity residual around the train-mean atlas is much easier to
compress than the full conductivity image. This means:

- the image representation itself is **not** the main bottleneck,
- the real bottleneck is recovering those residuals from boundary measurements.

### Residual focus fractions

The fraction of valid pixels whose conductivity differs from the atlas by more
than a threshold is:

- `> 0.03`: `0.4816`
- `> 0.05`: `0.4024`
- `> 0.08`: `0.2987`
- `> 0.12`: `0.1624`

This supports the qualitative observation that the pulmonary task is neither
fully trivial nor dominated by arbitrary global shape changes. Instead, it is a
problem of **recovering moderate local deviations around a stable anatomy**.

## 2. Controllable Generator Complexity

Analysis result:

- [`results/lung_variability_analysis_2/summary.json`](../results/lung_variability_analysis_2/summary.json)

Visual outputs:

- [`results/lung_variability_analysis_2/sigma_samples.png`](../results/lung_variability_analysis_2/sigma_samples.png)
- [`results/lung_variability_analysis_2/atlas_std.png`](../results/lung_variability_analysis_2/atlas_std.png)
- [`results/lung_variability_analysis_2/dct_curve.png`](../results/lung_variability_analysis_2/dct_curve.png)
- [`results/lung_variability_analysis_2/summary_bars.png`](../results/lung_variability_analysis_2/summary_bars.png)

The new generator controls create a clean low/medium/high complexity ladder.

### Atlas baseline strength by preset

- `low`: `relative L2 = 0.1990`
- `medium`: `relative L2 = 0.2568`
- `high`: `relative L2 = 0.2995`

This is exactly the direction we wanted:

- stronger local/anatomical variation weakens the atlas baseline,
- the dataset becomes progressively less "template dominated".

### Residual DCT compressibility by preset

Atlas-residual DCT with `K=20` gives:

- `low`: `relative L2 = 0.1425`
- `medium`: `relative L2 = 0.1595`
- `high`: `relative L2 = 0.1584`

This shows an important nuance:

- increasing complexity weakens the atlas baseline,
- but the residual images remain low-frequency enough to be efficiently
  represented by a modest DCT basis.

This is exactly the regime where a data-centric pulmonary study becomes
interesting: the representation remains plausible, but the inverse problem gets
harder in a controlled way.

## 3. Research Implication

These results change how the pulmonary study should be framed.

The main question is no longer:

- "Which network reconstructs the thorax shape best?"

The better question is:

- "How do data quantity, data complexity, and representation choice interact
  when the global anatomy is mostly fixed and only local conductivity changes
  carry new information?"

This is much better aligned with the current repository evidence:

- KTC benchmark remains a useful structured-label comparison task,
- pulmonary synthetic conductivity regression becomes the main continuous
  pulmonary task,
- autoencoders and DCT are best interpreted as analysis tools and inductive
  biases rather than as universally dominant solutions.

## 4. Practical Next Step

The next pulmonary experiments should use the new variability controls to build
small matched datasets with:

- the same sample count,
- different complexity levels,
- the same train/val/test protocol,

and then measure how reconstruction metrics change for:

- `fcunet` (strong baseline),
- `dct_sigma_residual_predictor`,
- `dct_sigma_hybrid_predictor`.

Two ready-to-use HDF5 datasets have already been generated for this purpose:

- low-complexity:
  - [`dataset_lung_varlow/level_1/data.h5`](../dataset_lung_varlow/level_1/data.h5)
  - preview: [`results/lung_data_preview_4/lung_phantoms.png`](../results/lung_data_preview_4/lung_phantoms.png)
- high-complexity:
  - [`dataset_lung_varhigh/level_1/data.h5`](../dataset_lung_varhigh/level_1/data.h5)
  - preview: [`results/lung_data_preview_5/lung_phantoms.png`](../results/lung_data_preview_5/lung_phantoms.png)

That would directly connect:

- data complexity,
- model inductive bias,
- and downstream pulmonary conductivity reconstruction quality.
