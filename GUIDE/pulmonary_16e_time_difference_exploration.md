## Pulmonary 16-Electrode Time-Difference Exploration

This note records the first matched 16-electrode pulmonary EIT exploration in
this repository.

### Motivation

The real thoracic `.get` recordings included in this repository follow a
16-electrode layout and reduce from raw `256` values per frame to `208`
reordered channels after the DCT-EIT-style channel reordering. The previous
pulmonary neural studies were still based on the 32-electrode KTC2023
measurement format (`2356` channels), so they were not well matched to the
real pulmonary acquisition regime.

To reduce this mismatch, I implemented:

- a 16-electrode forward simulation pipeline,
- paired pulmonary phantoms with a reference state and a target state,
- raw `256`-channel measurement simulation followed by DCT-EIT-style
  `256 -> 208` reordering,
- a first `dct_sigma_td16_predictor` that predicts continuous
  `\Delta \sigma` from reordered `208`-channel delta measurements.

### Implemented Components

- 16-electrode utilities:
  - [`src/utils/pulmonary16.py`](../src/utils/pulmonary16.py)
- forward-model compatibility fix:
  - [`src/ktc_methods/KTCFwd.py`](../src/ktc_methods/KTCFwd.py)
- paired pulmonary phantom generation:
  - [`src/data/lung_phantom.py`](../src/data/lung_phantom.py)
- delta-conductivity dataset:
  - [`src/data/sim_dataset.py`](../src/data/sim_dataset.py)
- TD16 predictor config/trainer/pipeline:
  - [`src/configs/dct_sigma_td16_predictor_config.py`](../src/configs/dct_sigma_td16_predictor_config.py)
  - [`src/trainers/dct_sigma_td16_predictor_trainer.py`](../src/trainers/dct_sigma_td16_predictor_trainer.py)
  - [`src/pipelines/dct_sigma_td16_predictor_pipeline.py`](../src/pipelines/dct_sigma_td16_predictor_pipeline.py)
- generation/evaluation/visualization scripts:
  - [`scripts/generate_lung_td16_data.py`](../scripts/generate_lung_td16_data.py)
  - [`scripts/evaluate_sigma_td16_sim.py`](../scripts/evaluate_sigma_td16_sim.py)
  - [`scripts/visualize_sigma_td16_samples.py`](../scripts/visualize_sigma_td16_samples.py)

### Why Global Relative-$L_2$ Is Problematic Here

For time-difference pulmonary data, some paired samples have almost zero
conductivity change. In those cases, a global relative-$L_2$ error becomes
unstable because the denominator is extremely small.

So for this task, the more informative metrics are:

- `MAE`
- `RMSE`
- `active_rel_l2`

where `active_rel_l2` only evaluates pixels whose true conductivity change
satisfies:

```text
|Δσ| > 0.02
```

The evaluation script also reports:

- `zero_delta_fraction`

to show how many samples are effectively near-zero-change cases.

### Mixed Pilot Dataset

Dataset:

- [`dataset_lung_td16_pilot/level_1/data.h5`](../dataset_lung_td16_pilot/level_1/data.h5)

Baseline zero predictor:

- [`results/td16_zero_baseline_2/summary.json`](../results/td16_zero_baseline_2/summary.json)

Key numbers:

- `MAE = 0.0161`
- `RMSE = 0.0835`
- `active_rel_l2 = 1.0000`
- `zero_delta_fraction = 0.4286`

First TD16 predictor:

- [`results/dct_sigma_td16_pilot_1/td16_test_eval_2/summary.json`](../results/dct_sigma_td16_pilot_1/td16_test_eval_2/summary.json)

Key numbers:

- `MAE = 0.0384`
- `RMSE = 0.0958`
- `active_rel_l2 = 0.9788`
- `zero_delta_fraction = 0.4286`

Interpretation:

- compared with the zero baseline, the learned model begins to recover some
  local change structure (`active_rel_l2 < 1.0`),
- but global error is still worse than the zero predictor because false
  positives dominate when many samples contain little or no true change.

### Active-Only Pilot Dataset

Dataset:

- [`dataset_lung_td16_active/level_1/data.h5`](../dataset_lung_td16_active/level_1/data.h5)

Baseline zero predictor:

- [`results/td16_zero_baseline_active_1/summary.json`](../results/td16_zero_baseline_active_1/summary.json)

Key numbers:

- `MAE = 0.0257`
- `RMSE = 0.1314`
- `active_rel_l2 = 1.0000`
- `zero_delta_fraction = 0.1429`

First TD16 predictor:

- [`results/dct_sigma_td16_active_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_active_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0479`
- `RMSE = 0.1321`
- `active_rel_l2 = 0.9228`
- `zero_delta_fraction = 0.1429`

Interpretation:

- once zero-change samples are reduced, the model more clearly improves local
  residual structure,
- but it still does not beat the zero predictor in global RMSE.

### Sparse TD16 Follow-Up

To suppress false positives, I added two new loss controls:

- `inactive_weight`
- `pred_l1_weight`

These penalize errors in target-zero regions more heavily and encourage sparse
predicted residual maps.

#### Mixed dataset with sparsity regularization

- [`results/dct_sigma_td16_sparse_mixed_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_sparse_mixed_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0477`
- `RMSE = 0.1038`
- `active_rel_l2 = 0.9255`

Compared with the original mixed TD16 model:

- `active_rel_l2` improves from `0.9788` to `0.9255`,
- but `RMSE` worsens from `0.0958` to `0.1038`.

#### Active-only dataset with sparsity regularization

- [`results/dct_sigma_td16_sparse_active_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_sparse_active_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0558`
- `RMSE = 0.1353`
- `active_rel_l2 = 0.9129`

Compared with the original active-only TD16 model:

- `active_rel_l2` improves from `0.9228` to `0.9129`,
- but `RMSE` worsens from `0.1321` to `0.1353`.

This is an important finding: sparsity-aware regularization helps the model
match the local change pattern better, but the global penalty from false
positives is still not solved.

### Change-Aware TD16 Follow-Up

The next hypothesis was that the model needs an explicit sample-level
``change/no-change'' signal instead of only relying on image regression loss.

So I added a change-aware variant:

- model:
  - [`ChangeGatedDCTPredictor`](../src/models/dct_predictor/model.py)
- trainer:
  - [`src/trainers/dct_sigma_td16_change_predictor_trainer.py`](../src/trainers/dct_sigma_td16_change_predictor_trainer.py)
- pipeline:
  - [`src/pipelines/dct_sigma_td16_change_predictor_pipeline.py`](../src/pipelines/dct_sigma_td16_change_predictor_pipeline.py)

The model predicts:

- low-frequency DCT coefficients for `\Delta \sigma`,
- one sample-level gate logit indicating whether a meaningful conductivity
  change exists.

The predicted conductivity change is multiplied by the gate probability. During
training, the gate receives BCE supervision based on whether the target contains
any pixel satisfying:

```text
|Δσ| > 0.02
```

#### Mixed dataset: gate only

- [`results/dct_sigma_td16_change_mixed_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_change_mixed_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0447`
- `RMSE = 0.1012`
- `active_rel_l2 = 0.9521`

Compared with the original mixed TD16 model:

- RMSE becomes worse (`0.0958 -> 0.1012`),
- but active-region relative-$L_2$ becomes better (`0.9788 -> 0.9521`).

Compared with sparse-only TD16:

- RMSE becomes better (`0.1038 -> 0.1012`),
- active-region relative-$L_2$ becomes worse (`0.9255 -> 0.9521`).

So the gate helps suppress some false positives relative to sparse-only
regularization, but it is still not enough to beat the plain model in global
RMSE.

#### Mixed dataset: gate + sparsity

- [`results/dct_sigma_td16_change_sparse_mixed_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_change_sparse_mixed_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0441`
- `RMSE = 0.1007`
- `active_rel_l2 = 0.9525`

This is the best mixed-data trade-off among the learned TD16 variants tested so
far:

- better RMSE than sparse-only TD16,
- slightly better RMSE than gate-only TD16,
- much better active-region relative-$L_2$ than the original model.

However, it still does **not** beat the zero baseline on mixed-data global
RMSE:

- zero baseline RMSE: `0.0835`
- best learned mixed TD16 RMSE: `0.1007`

This is a strong indication that sample-level gating alone is not enough. The
next change-aware step should be more structured, for example:

- explicit binary change detection followed by conditional residual regression,
- uncertainty-aware suppression of weak predicted changes,
- separate optimization for zero-change and active-change cases.

### Spatial Change-Mask TD16

The next step was to move from sample-level gating to a spatially structured
change representation.

I added:

- [`SpatialChangeGatedDCTPredictor`](../src/models/dct_predictor/model.py)
- [`src/trainers/dct_sigma_td16_spatial_change_predictor_trainer.py`](../src/trainers/dct_sigma_td16_spatial_change_predictor_trainer.py)
- [`src/pipelines/dct_sigma_td16_spatial_change_predictor_pipeline.py`](../src/pipelines/dct_sigma_td16_spatial_change_predictor_pipeline.py)

This model predicts:

- low-frequency conductivity-change coefficients,
- low-frequency spatial change-mask logits.

The final prediction is:

```text
Δσ_pred = residual_pred * sigmoid(mask_logits)
```

The mask branch is trained against the binary target:

```text
1(|Δσ| > 0.02)
```

#### Mixed dataset

- [`results/dct_sigma_td16_spatial_mixed_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_spatial_mixed_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0315`
- `RMSE = 0.0923`
- `active_rel_l2 = 0.9620`

This is now the best learned mixed-data TD16 result so far:

- original TD16: `RMSE = 0.0958`, `active_rel_l2 = 0.9788`
- gate + sparse TD16: `RMSE = 0.1007`, `active_rel_l2 = 0.9525`
- spatial mask TD16: `RMSE = 0.0923`, `active_rel_l2 = 0.9620`

So the spatial mask branch improves both global RMSE and active-region
reconstruction relative to the original model. It still does not beat the zero
baseline RMSE (`0.0835`), but it reduces the gap substantially.

#### Active-only dataset

- [`results/dct_sigma_td16_spatial_active_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_spatial_active_1/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0410`
- `RMSE = 0.13136`
- `active_rel_l2 = 0.9463`

Compared with:

- zero baseline: `RMSE = 0.13139`, `active_rel_l2 = 1.0000`
- original active-only TD16: `RMSE = 0.13215`, `active_rel_l2 = 0.9228`

This result is important:

- the spatial mask branch almost exactly matches, and slightly improves upon,
  the zero baseline in global RMSE,
- while still outperforming the zero baseline in active-region reconstruction.

However, it is still worse than the original active-only TD16 in
`active_rel_l2`, so a trade-off remains:

- original active-only TD16 is better at fitting nonzero change structure,
- spatial-mask TD16 is better at suppressing false positives globally.

### Updated Conclusion

Among the learned TD16 variants tested so far:

- the original model is the simplest and gives the strongest active-only local
  fit,
- the sample gate is directionally useful but too weak on its own,
- sparse regularization improves local change matching but worsens RMSE,
- the spatial change-mask branch is currently the **best mixed-data compromise**
  and the **first learned TD16 model to essentially match the zero baseline in
  active-only global RMSE**.

So the current evidence supports the following next step:

- combine spatially explicit change localization with stronger conditional
  residual regression, instead of relying on only scalar gating or only
  sparsity penalties.

### Active Oversampling Ablation

I also tested whether the remaining mixed-data difficulty is mainly caused by
too many near-zero samples during optimization.

Using the spatial-mask TD16 model, I oversampled active-change training samples
by a factor of `3.0`:

- [`results/dct_sigma_td16_spatial_os3_mixed_2/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_spatial_os3_mixed_2/td16_test_eval_1/summary.json)

Key numbers:

- `MAE = 0.0320`
- `RMSE = 0.0925`
- `active_rel_l2 = 0.9608`

Compared with the non-oversampled spatial-mask model:

- no oversampling: `RMSE = 0.0923`, `active_rel_l2 = 0.9620`
- oversampling x3: `RMSE = 0.0925`, `active_rel_l2 = 0.9608`

This is effectively a tie. So at the current pilot scale, active oversampling
does not produce a meaningful additional gain beyond the spatial-mask model
itself.

That makes the present conclusion sharper:

- explicit spatial change localization matters,
- but simply showing the model more active samples is not enough,
- the next step should focus on stronger conditional residual inference rather
  than data reweighting alone.

### Conditional Residual Follow-Up

I then implemented a stronger conditional variant that does not only multiply the
output by a spatial mask. Instead, it predicts a low-frequency spatial change
mask first and feeds the mask coefficients into the residual coefficient branch.
During training, the residual branch can also use teacher-forced target mask
coefficients.

Results:

- mixed pilot, conditional residual (`teacher forcing = 0.5`):
  - [`results/dct_sigma_td16_conditional_mixed_2/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_conditional_mixed_2/td16_test_eval_1/summary.json)
  - `RMSE = 0.0950`
  - `active_rel_l2 = 0.9504`
- mixed pilot, full teacher forcing (`teacher forcing = 1.0`):
  - [`results/dct_sigma_td16_conditional_tf1_mixed_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_conditional_tf1_mixed_1/td16_test_eval_1/summary.json)
  - `RMSE = 0.0960`
  - `active_rel_l2 = 0.9487`
- active-only pilot, conditional residual (`teacher forcing = 0.5`):
  - [`results/dct_sigma_td16_conditional_active_1/td16_test_eval_1/summary.json`](../results/dct_sigma_td16_conditional_active_1/td16_test_eval_1/summary.json)
  - `RMSE = 0.13131`
  - `active_rel_l2 = 0.9318`

Interpretation:

- On the mixed pilot, conditional residual regression does **not** beat the
  simpler spatial-mask TD16 (`RMSE = 0.0923`, `active_rel_l2 = 0.9620`).
- Full teacher forcing does not help either, so the limitation is not just a
  weak conditioning signal during training.
- On the active-only pilot, conditional residual regression slightly improves
  global RMSE relative to the spatial-mask model (`0.13131` vs `0.13136`) and
  the zero baseline (`0.13139`), but it still underperforms the original
  active-only TD16 in local change recovery (`0.9318` vs `0.9228`).

So the new evidence refines the direction but does not overturn it:

- explicit spatial localization remains necessary,
- conditional mask injection alone is not enough to solve the residual problem,
- the unresolved difficulty is now specifically a better coupling between
  spatial change localization and residual amplitude estimation.

### Oracle Mask Diagnostics

To determine whether the main failure still comes from poor spatial localization
 or from residual amplitude regression itself, I added an oracle-mask diagnostic:

- use the trained model as-is,
- replace the predicted change mask at evaluation time with the ground-truth
  spatial change mask,
- for the conditional model, also test a stronger oracle in which the residual
  branch is conditioned on the ground-truth low-frequency mask coefficients.

Results on the mixed pilot:

- spatial-mask TD16:
  - standard: `RMSE = 0.0923`, `active_rel_l2 = 0.9620`
  - oracle output mask:
    - [`results/dct_sigma_td16_spatial_mixed_1/td16_oracle_mask_eval_1/summary.json`](../results/dct_sigma_td16_spatial_mixed_1/td16_oracle_mask_eval_1/summary.json)
    - `RMSE = 0.0771`, `active_rel_l2 = 0.9281`
- conditional TD16:
  - standard: `RMSE = 0.0950`, `active_rel_l2 = 0.9504`
  - oracle output mask:
    - `RMSE = 0.0752`, `active_rel_l2 = 0.9055`
  - oracle condition + oracle mask:
    - [`results/dct_sigma_td16_conditional_mixed_2/td16_oracle_mask_eval_1/summary.json`](../results/dct_sigma_td16_conditional_mixed_2/td16_oracle_mask_eval_1/summary.json)
    - `RMSE = 0.0739`, `active_rel_l2 = 0.8846`

Results on the active-only pilot:

- spatial-mask TD16:
  - standard: `RMSE = 0.13136`, `active_rel_l2 = 0.9463`
  - oracle output mask:
    - [`results/dct_sigma_td16_spatial_active_1/td16_oracle_mask_eval_1/summary.json`](../results/dct_sigma_td16_spatial_active_1/td16_oracle_mask_eval_1/summary.json)
    - `RMSE = 0.1187`, `active_rel_l2 = 0.9023`
- conditional TD16:
  - standard: `RMSE = 0.13131`, `active_rel_l2 = 0.9318`
  - oracle output mask:
    - `RMSE = 0.1152`, `active_rel_l2 = 0.8748`
  - oracle condition + oracle mask:
    - [`results/dct_sigma_td16_conditional_active_1/td16_oracle_mask_eval_1/summary.json`](../results/dct_sigma_td16_conditional_active_1/td16_oracle_mask_eval_1/summary.json)
    - `RMSE = 0.1137`, `active_rel_l2 = 0.8626`

These numbers are the clearest result in the current TD16 line.

They show that:

- the learned residual branch is already much more capable than the raw mixed
  metrics suggest,
- once spatial localization becomes correct, both global RMSE and active-region
  reconstruction improve substantially,
- and in fact the oracle-mask versions beat the zero baseline RMSE.

Therefore the dominant pulmonary TD16 bottleneck is now sharply localized:

- **change localization error**, more than residual amplitude regression error.

That means the next justified step is not another larger decoder or stronger
coefficient loss. It is a model that can produce a better spatial change map
from the measurements themselves.

### Dedicated Mask Predictor and External Mask Fusion

To test that localization hypothesis directly, I added a dedicated mask-only
TD16 model:

- config:
  - [`src/configs/dct_sigma_td16_mask_predictor_config.py`](../src/configs/dct_sigma_td16_mask_predictor_config.py)
- trainer:
  - [`src/trainers/dct_sigma_td16_mask_predictor_trainer.py`](../src/trainers/dct_sigma_td16_mask_predictor_trainer.py)
- analysis:
  - [`scripts/analyze_td16_external_mask_fusion.py`](../scripts/analyze_td16_external_mask_fusion.py)

This model predicts only the binary spatial change map

```text
1(|Δσ| > 0.02)
```

from the `208`-channel TD16 measurements, using a low-frequency DCT mask head.
The goal is to see whether a dedicated localization head can outperform the
internal mask branches used inside the spatial and conditional residual models.

#### Mixed pilot: mask-only localization quality

Best mixed mask-only run:

- [`results/dct_sigma_td16_mask_mixed_1`](../results/dct_sigma_td16_mask_mixed_1)

When evaluated on the mixed test split through external fusion analysis:

- [`results/dct_sigma_td16_spatial_mixed_1/td16_external_mask_eval_1/summary.json`](../results/dct_sigma_td16_spatial_mixed_1/td16_external_mask_eval_1/summary.json)
- [`results/dct_sigma_td16_conditional_mixed_2/td16_external_mask_eval_1/summary.json`](../results/dct_sigma_td16_conditional_mixed_2/td16_external_mask_eval_1/summary.json)

Key external-mask metrics:

- precision `= 0.0662`
- recall `= 0.2914`
- F1 `= 0.0892`
- IoU `= 0.0545`

Compared with the internal mask branches:

- spatial TD16 internal mask IoU `= 0.0285`
- conditional TD16 internal mask IoU `= 0.0359`

So the dedicated mask-only model **does** localize change regions more
accurately than the internal joint-training mask branches.

However, direct external fusion is only weakly helpful:

- spatial TD16 internal prediction:
  - RMSE `= 0.0923`, active-region relative-$L_2 = 0.9621`
- spatial TD16 + external soft mask:
  - RMSE `= 0.0926`, active-region relative-$L_2 = 0.9616`
- conditional TD16 internal prediction:
  - RMSE `= 0.0950`, active-region relative-$L_2 = 0.9503`
- conditional TD16 + external soft mask:
  - RMSE `= 0.0958`, active-region relative-$L_2 = 0.9493`

This is an important nuance:

- localization quality improves,
- but simply replacing the internal mask with an external one does **not**
  produce a strong RMSE gain.

So the next issue is not just ``better mask quality'' in isolation. It is the
**coupling between localization and residual amplitude prediction**.

#### Mixed pilot: oversampled mask-only training

I also tested whether active-sample oversampling helps the dedicated mask head:

- [`results/dct_sigma_td16_mask_os3_mixed_1`](../results/dct_sigma_td16_mask_os3_mixed_1)
- fusion analysis:
  - [`results/dct_sigma_td16_spatial_mixed_1/td16_external_mask_eval_2/summary.json`](../results/dct_sigma_td16_spatial_mixed_1/td16_external_mask_eval_2/summary.json)
  - [`results/dct_sigma_td16_conditional_mixed_2/td16_external_mask_eval_2/summary.json`](../results/dct_sigma_td16_conditional_mixed_2/td16_external_mask_eval_2/summary.json)

The oversampled mask predictor actually becomes worse on the mixed test split:

- IoU drops from `0.0545` to `0.0476`
- F1 drops from `0.0892` to `0.0799`

So, for the current pilot scale, the dedicated mask head does **not** benefit
from simple active-sample oversampling.

#### Active-only pilot: mask-only localization quality

I repeated the dedicated mask experiment on the active-only dataset:

- [`results/dct_sigma_td16_mask_active_1`](../results/dct_sigma_td16_mask_active_1)
- fusion analysis:
  - [`results/dct_sigma_td16_spatial_active_1/td16_external_mask_eval_1/summary.json`](../results/dct_sigma_td16_spatial_active_1/td16_external_mask_eval_1/summary.json)
  - [`results/dct_sigma_td16_conditional_active_1/td16_external_mask_eval_1/summary.json`](../results/dct_sigma_td16_conditional_active_1/td16_external_mask_eval_1/summary.json)

Key active-only external-mask metrics:

- precision `= 0.0935`
- recall `= 0.6929`
- F1 `= 0.1584`
- IoU `= 0.0898`

Compared with internal mask branches:

- spatial active internal mask IoU `= 0.0436`
- conditional active internal mask IoU `= 0.0535`

So once zero-change interference is removed, the dedicated mask head becomes
substantially stronger at localization.

But again, external fusion only gives a near-tie in global reconstruction:

- spatial active internal prediction:
  - RMSE `= 0.13135`, active-region relative-$L_2 = 0.94625`
- spatial active + external soft mask:
  - RMSE `= 0.13137`, active-region relative-$L_2 = 0.94639`
- conditional active internal prediction:
  - RMSE `= 0.13131`, active-region relative-$L_2 = 0.93181`
- conditional active + external soft mask:
  - RMSE `= 0.13151`, active-region relative-$L_2 = 0.93092`

Therefore the updated TD16 diagnosis is:

- a dedicated localization head can indeed outperform the internal mask branch,
- the benefit is clearer on active-only data than on mixed data,
- but localization and residual regression cannot simply be decoupled by
  post-hoc mask replacement,
- so the next pulmonary TD16 step should be a more tightly coupled two-stage or
  jointly conditioned localization-plus-residual architecture.

### Qualitative Outputs

- mixed pilot:
  - [`results/dct_sigma_td16_pilot_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_pilot_1/td16_test_samples_1/test_comparison.png)
- active-only pilot:
  - [`results/dct_sigma_td16_active_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_active_1/td16_test_samples_1/test_comparison.png)
- sparse mixed:
  - [`results/dct_sigma_td16_sparse_mixed_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_sparse_mixed_1/td16_test_samples_1/test_comparison.png)
- sparse active-only:
  - [`results/dct_sigma_td16_sparse_active_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_sparse_active_1/td16_test_samples_1/test_comparison.png)
- change-aware mixed:
  - [`results/dct_sigma_td16_change_mixed_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_change_mixed_1/td16_test_samples_1/test_comparison.png)
- change-aware + sparse mixed:
  - [`results/dct_sigma_td16_change_sparse_mixed_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_change_sparse_mixed_1/td16_test_samples_1/test_comparison.png)
- conditional active-only:
  - [`results/dct_sigma_td16_conditional_active_1/td16_test_samples_1/test_comparison.png`](../results/dct_sigma_td16_conditional_active_1/td16_test_samples_1/test_comparison.png)

### Current Conclusion

The first matched 16-electrode pulmonary time-difference line is now working
end-to-end, and it already reveals something important:

- the model can start to recover local change structure,
- but it still produces too many or too diffuse false positives,
- so global RMSE remains close to or worse than the zero predictor.

The latest mask-only experiments further sharpen that conclusion:

- better standalone localization is possible,
- but simple external mask fusion is not enough,
- therefore the next justified research direction is not a bigger absolute-image
  decoder or a standalone post-hoc mask head. It is:

- explicit change/no-change modeling beyond a single sample gate,
- residual prediction that is tightly conditioned on a stronger spatial
  localization signal,
- architectures that let localization and residual amplitude co-adapt rather
  than being fused only at inference time,
- pulmonary evaluation focused on local change metrics in addition to global
  RMSE.
