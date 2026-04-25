# Pulmonary Complexity-Matched Model Results

## Purpose

After establishing that pulmonary conductivity reconstruction is strongly
atlas-dominated, the next question was whether atlas-aware inductive biases
remain stable when generator complexity changes.

To answer this, three continuous-conductivity models were trained on two
matched pulmonary datasets with the same sample count (`512`) but different
generator complexity:

- `Direct DCT`: predict full-image low-frequency coefficients directly
- `Atlas-residual DCT`: predict residuals around the train-mean atlas
- `Hybrid DCT`: atlas + coarse DCT + local refinement
- `Atlas-decoder`: atlas + learned residual image decoder without fixed DCT basis

The two matched datasets are:

- low complexity:
  - [`dataset_lung_varlow/level_1/data.h5`](../dataset_lung_varlow/level_1/data.h5)
- high complexity:
  - [`dataset_lung_varhigh/level_1/data.h5`](../dataset_lung_varhigh/level_1/data.h5)

The training/evaluation outputs were aggregated with:

- [`scripts/plot_pulmonary_complexity_model_results.py`](../scripts/plot_pulmonary_complexity_model_results.py)
- [`scripts/visualize_sigma_complexity_comparison.py`](../scripts/visualize_sigma_complexity_comparison.py)
- config:
  - [`scripts/pulmonary_complexity_model_comparison.yaml`](../scripts/pulmonary_complexity_model_comparison.yaml)

## Quantitative Summary

Aggregated result:

- [`results/pulmonary_complexity_models_3/summary.json`](../results/pulmonary_complexity_models_3/summary.json)

Plots:

- [`results/pulmonary_complexity_models_3/complexity_rel_l2.png`](../results/pulmonary_complexity_models_3/complexity_rel_l2.png)
- [`results/pulmonary_complexity_models_3/complexity_bar.png`](../results/pulmonary_complexity_models_3/complexity_bar.png)

### Low complexity

- atlas baseline: `0.1990`
- oracle atlas-residual DCT (`K=20`): `0.1425`
- direct DCT predictor: `0.6944`
- atlas-residual predictor: `0.2008`
- hybrid predictor: `0.2012`
- atlas-decoder predictor: `0.2006`

### High complexity

- atlas baseline: `0.2995`
- oracle atlas-residual DCT (`K=20`): `0.1584`
- direct DCT predictor: `0.7230`
- atlas-residual predictor: `0.2971`
- hybrid predictor: `0.2973`
- atlas-decoder predictor: `0.2972`

## Main Observation

The result is sharper than expected:

- the original direct DCT conductivity regressor does **not** merely lose some
  accuracy under higher complexity; it collapses far away from the atlas-aware
  operating point in both low and high regimes,
- the atlas-residual and hybrid models remain close to the atlas baseline in
  both regimes,
- the atlas-decoder predictor also remains close to the atlas baseline, but
  does not provide a meaningful gain over the fixed-basis atlas-aware models,
- a conventional FC-SigmaUNet pilot baseline also stays near the atlas regime
  rather than clearly outperforming it, despite using a much heavier decoder,
- but they still do **not** close the gap to the oracle atlas-residual DCT
  compressibility bound.

This means the pulmonary issue is now much more clearly localized:

- direct low-frequency prediction is structurally mismatched to this task,
- atlas-aware modeling is necessary,
- decoder flexibility alone is not sufficient,
- but the remaining bottleneck is still measurement-to-residual inference,
  rather than image representation capacity.

## Decoder Capacity Check

The atlas-decoder baseline was added specifically to test whether the fixed DCT
basis was itself the pulmonary bottleneck. The answer is negative:

- low complexity:
  - residual DCT: `0.2008`
  - hybrid DCT: `0.2012`
  - atlas-decoder: `0.2006`
  - FC-SigmaUNet pilot: `0.2049`
- high complexity:
  - residual DCT: `0.2971`
  - hybrid DCT: `0.2973`
  - atlas-decoder: `0.2972`
  - FC-SigmaUNet pilot: `0.3010`

This means replacing the fixed inverse-DCT residual decoder with a learned
atlas-aware residual decoder does not materially improve reconstruction
accuracy. It also means that simply switching to a much heavier conventional
image-space baseline does not, by itself, solve the pulmonary residual
inference bottleneck.

The corresponding model-size / latency benchmark on the low-complexity split is:

- residual DCT:
  - params: `3.76M`
  - single latency (batch=1): `1.655 ms`
  - batched per-sample latency (batch=32): `0.379 ms`
- hybrid DCT:
  - params: `7.67M`
  - single latency (batch=1): `4.029 ms`
  - batched per-sample latency (batch=32): `0.421 ms`
- atlas-decoder:
  - params: `7.47M`
  - single latency (batch=1): `3.183 ms`
  - batched per-sample latency (batch=32): `0.232 ms`
- FC-SigmaUNet pilot:
  - params: `40.70M`
  - single latency (batch=1): `40.775 ms`
  - batched per-sample latency (batch=32): `11.311 ms`

So the learned decoder changes computational tradeoffs, but not the main
accuracy conclusion. FC-SigmaUNet is an order of magnitude heavier and slower,
yet its short-run pilot accuracy still remains near the same atlas-dominated
operating point.

## Qualitative Comparison

Matched low/high-complexity examples:

- [`results/pulmonary_complexity_visual_3/comparison.png`](../results/pulmonary_complexity_visual_3/comparison.png)

The qualitative behavior matches the metrics:

- `Direct` predictions remain overly smooth and often drift away from the true
  local pattern,
- `Residual` and `Hybrid` predictions recover the global anatomy much more
  faithfully,
- `Atlas-decoder` behaves similarly to the two atlas-aware DCT variants,
- `FC-SigmaUNet` produces anatomically plausible images, but in the current
  pilot setting it does not recover substantially more local detail than the
  lighter atlas-aware DCT family,
- yet most remaining errors are still local and relatively subtle.

## Implication for the Paper

These experiments strengthen the data-centric pulmonary story:

1. Pulmonary conductivity prediction should not be framed as unconstrained image
   generation.
2. A stable anatomical atlas is not just a convenient baseline; it is a
   structural prior that must be modeled explicitly.
3. The main remaining challenge is recovering local deviations on top of that
   atlas, especially as generator complexity increases.

This also provides a cleaner justification for the paper structure:

- KTC2023 remains the structured-label benchmark task.
- Pulmonary conductivity regression becomes the main continuous pulmonary task.
- DCT is retained as an efficient inductive bias, but the pulmonary evidence now
  shows that **atlas-aware DCT** is the only sensible continuous direction.

## `lung2k` Confirmation

To check whether the same conclusion still holds on the main pulmonary dataset,
the atlas-aware family was also trained directly on:

- [`dataset_lung_2k/level_1/data.h5`](../dataset_lung_2k/level_1/data.h5)

Results:

- [`results/dct_sigma_lung2k_residual_1/dct_sigma_residual_predictor_test_eval_1/summary.json`](../results/dct_sigma_lung2k_residual_1/dct_sigma_residual_predictor_test_eval_1/summary.json)
- [`results/dct_sigma_lung2k_hybrid_1/dct_sigma_hybrid_predictor_test_eval_1/summary.json`](../results/dct_sigma_lung2k_hybrid_1/dct_sigma_hybrid_predictor_test_eval_1/summary.json)
- [`results/atlas_sigma_lung2k_1/dct_sigma_test_eval_1/summary.json`](../results/atlas_sigma_lung2k_1/dct_sigma_test_eval_1/summary.json)

Key numbers:

- direct continuous DCT on `lung2k`:
  - relative-$L_2$: `0.3509`
- train-mean atlas baseline on `lung2k`:
  - relative-$L_2$: `0.2552`
- atlas-residual DCT on `lung2k`:
  - relative-$L_2$: `0.2552`
- hybrid DCT on `lung2k`:
  - relative-$L_2$: `0.2552`
- atlas-decoder on `lung2k`:
  - relative-$L_2$: `0.2552`

This is an important consistency check:

- all three atlas-aware models fix the direct-DCT collapse on the main
  pulmonary dataset,
- but none of them outperform the atlas baseline in any meaningful sense,
- so the remaining problem is still how to infer local residual structure from
  measurements, not how to decode an atlas-conditioned image.

## Residual-Focused Evaluation

Global masked relative-$L_2$ is useful, but it still mixes stable thorax anatomy
with the actual local deviations that matter clinically. To isolate the latter,
I added a residual-region analysis that only evaluates pixels satisfying:

\[
|\sigma - \sigma_{\text{atlas}}| > 0.08
\]

Analysis outputs:

- [`scripts/analyze_sigma_residual_focus.py`](../scripts/analyze_sigma_residual_focus.py)
- config:
  - [`scripts/pulmonary_residual_focus_analysis.yaml`](../scripts/pulmonary_residual_focus_analysis.yaml)
- aggregated summary:
  - [`results/pulmonary_residual_focus_2/summary.json`](../results/pulmonary_residual_focus_2/summary.json)
- plot:
  - [`results/pulmonary_residual_focus_2/residual_focus_bar.png`](../results/pulmonary_residual_focus_2/residual_focus_bar.png)

The result is even sharper than the global metric:

- low complexity residual-region relative-$L_2$:
  - atlas: `0.3548`
  - residual DCT: `0.3551`
  - hybrid DCT: `0.3555`
  - atlas-decoder: `0.3548`
  - FC-SigmaUNet pilot: `0.3583`
- high complexity residual-region relative-$L_2$:
  - atlas: `0.3849`
  - residual DCT: `0.3849`
  - hybrid DCT: `0.3851`
  - atlas-decoder: `0.3849`
  - FC-SigmaUNet pilot: `0.3885`
- `lung2k` residual-region relative-$L_2$:
  - atlas: `0.3746`
  - residual DCT: `0.3747`
  - hybrid DCT: `0.3747`
  - atlas-decoder: `0.3746`

So the new models are not secretly winning on local structure while tying on
global metrics. They are essentially matching the atlas baseline even inside
the residual region. This confirms that the open problem is not only "stay in
the correct image domain", but actually "extract the small measurement-driven
deviations from that domain".

## Focus-Loss Ablation

The residual DCT trainer originally used a focus-weighted loss to emphasize
pixels with larger atlas deviations. I checked whether that weighting was
actually responsible for the atlas-aware behavior by training a no-focus
variant on the matched low/high datasets.

No-focus runs:

- [`results/dct_sigma_varlow_residual_nofocus_1/dct_sigma_residual_predictor_test_eval_1/summary.json`](../results/dct_sigma_varlow_residual_nofocus_1/dct_sigma_residual_predictor_test_eval_1/summary.json)
- [`results/dct_sigma_varhigh_residual_nofocus_1/dct_sigma_residual_predictor_test_eval_1/summary.json`](../results/dct_sigma_varhigh_residual_nofocus_1/dct_sigma_residual_predictor_test_eval_1/summary.json)

Key comparison:

- low complexity:
  - residual DCT with focus: `0.2008`
  - residual DCT without focus: `0.2007`
- high complexity:
  - residual DCT with focus: `0.2971`
  - residual DCT without focus: `0.2972`

This ablation shows that simple residual-region reweighting has almost no
effect. In other words, the bottleneck is not that the model "does not care"
about local deviations enough in the loss; the bottleneck is that the
measurement-to-residual mapping itself remains weak.
