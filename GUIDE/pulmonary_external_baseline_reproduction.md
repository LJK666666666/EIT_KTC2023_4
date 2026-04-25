# Pulmonary External Baseline Reproduction Route

## Why the route changed

The custom pulmonary models in this repository have already identified several
important failure modes:

- absolute pulmonary conductivity regression collapses toward an atlas-like mean
  template;
- `16`-electrode time-difference learning is more promising, but the dominant
  bottleneck is spatial change localization rather than decoder capacity;
- increasingly complex custom residual heads are still exploratory and do not
  yet provide a stable external baseline for pulmonary EIT.

Because of that, the pulmonary branch is now reframed as:

1. reproduce a classical pulmonary EIT baseline that is widely accepted in the
   literature;
2. reproduce or approximate at least one recent pulmonary deep-learning method;
3. only then adapt the repository's own DCT / residual / TD16 ideas on top of
   those baselines.

## Selected external references

### Classical baseline

- `GREIT`:
  Adler et al., *GREIT: a unified approach to 2D linear EIT reconstruction of
  lung images*.
- practical Python entry:
  `pyEIT` (Liu et al., SoftwareX), which exposes a usable GREIT solver and mesh
  tooling in Python.

This route is preferred because:

- it is strongly tied to pulmonary / thoracic EIT;
- it is open-source and already compatible with Python;
- it provides a clinically recognized linear time-difference baseline.

### Recent deep-learning candidate

- Zeng et al., 2023, *Deep learning based reconstruction enables high-resolution
  electrical impedance tomography for lung function assessment*.

This paper is currently the most relevant deep-learning pulmonary target for
this repository because:

- it is explicitly lung `tdEIT`;
- it uses representation learning rather than pure image-space regression;
- it includes real-subject validation with spirometry correlation.

The immediate plan is therefore:

1. keep GREIT as the first reproduced pulmonary baseline;
2. then examine whether Zeng et al. can be reproduced directly or approximated
   from the paper description;
3. after that, adapt the current `TD16` DCT / residual framework to the same
   matched protocol.

## What was implemented locally

To support GREIT reproduction, the repository now includes:

- `pyEIT` installation in the local environment;
- a reusable `Draeger 208` protocol builder:
  - `src/utils/pulmonary16.py`
  - `build_draeger208_pyeit_protocol()`
- a dedicated GREIT evaluation script:
  - `scripts/evaluate_sigma_td16_greit.py`

The key step was to explicitly align:

- `16` stimulation patterns,
- source-sink offset `3`,
- adjacent differential measurements,
- the local `256 -> 208` Draeger / DCT-EIT reordering,
- and the `pyEIT` protocol ordering.

This alignment is necessary; otherwise a GREIT solver can run numerically while
still being fed with a mismatched measurement ordering.

## Initial GREIT results

### Mixed TD16 pilot

Thorax mesh:

- result directory: `results/greit_td16_pilot_1`
- `RMSE32 = 0.0859`
- `active_rel_l2_32 = 1.0034`
- `RMSE256 = 0.1068`
- `active_rel_l2_256 = 1.0024`

Circle mesh:

- result directory: `results/greit_td16_pilot_circle_1`
- `RMSE32 = 0.0851`
- `active_rel_l2_32 = 0.9951`
- `RMSE256 = 0.1061`
- `active_rel_l2_256 = 0.9968`

Interpretation:

- GREIT can now be reproduced on the local matched `208`-channel protocol;
- on this small mixed pilot, a circular mesh is slightly better than the
  thorax-shaped mesh;
- however, the result is still close to the zero / weak-change operating point
  and is not yet a strong pulmonary reconstruction result.

### Active-only TD16 pilot

Thorax mesh:

- result directory: `results/greit_td16_active_1`
- `RMSE32 = 0.1205`
- `active_rel_l2_32 = 0.9910`
- `RMSE256 = 0.1449`
- `active_rel_l2_256 = 0.9945`

Circle mesh:

- result directory: `results/greit_td16_active_circle_1`
- `RMSE32 = 0.1279`
- `active_rel_l2_32 = 1.0203`
- `RMSE256 = 0.1506`
- `active_rel_l2_256 = 1.0098`

Interpretation:

- on active-only TD16, the thorax mesh is the better classical baseline;
- the reproduced GREIT baseline remains weaker than the current learned
  spatial-mask TD16 models;
- but the reproduction itself is important because it establishes a classical
  pulmonary baseline under the same local `16`-electrode measurement protocol.

## Current conclusion

The pulmonary branch should no longer be treated as "just keep inventing new
custom heads". A more defensible route is:

1. use reproduced GREIT as a classical `16`-electrode pulmonary baseline;
2. add at least one reproduced recent pulmonary deep-learning baseline;
3. compare the repository's own `TD16` DCT / residual models against those
   external baselines under the same matched protocol.

At the current stage, the GREIT reproduction is already useful because it turns
the pulmonary study from a closed internal exploration into an externally
anchored baseline study.

## Approximate Zeng-style latent-manifold reproduction

The next pulmonary external target is the deep-learning method of Zeng et al.
(EMBC 2023), which follows a high-level pipeline of:

1. train a variational autoencoder on lung conductivity images;
2. encode those images into a low-dimensional latent manifold;
3. train a multilayer perceptron that maps tdEIT measurements to the latent
   space;
4. decode the predicted latent vector back into conductivity images.

The exact CT-driven simulation pipeline from the paper is not yet available in
this repository, so the current implementation is an approximation under the
local matched `TD16` protocol:

- `src/models/pulmonary_vae/model.py`
- `src/trainers/td16_vae_trainer.py`
- `src/trainers/td16_vae_predictor_trainer.py`
- `src/pipelines/td16_vae_predictor_pipeline.py`

This keeps the same manifold-learning idea, but trains on the local
`16`-electrode synthetic `\Delta \sigma` datasets instead of the original
CT-derived 15000-pair training set described by Zeng et al.

### Active-only pilot

VAE autoencoder:

- result directory: `results/td16_vae_active_pilot_2`
- gap analysis: `results/td16_vae_predictor_active_pilot_1/td16_vae_gap_2`
- test `RMSE = 0.1189`
- test `active_rel_l2 = 0.7587`

Latent predictor:

- result directory: `results/td16_vae_predictor_active_pilot_1`
- test summary: `results/td16_vae_predictor_active_pilot_1/td16_test_eval_1/summary.json`
- gap analysis: `results/td16_vae_predictor_active_pilot_1/td16_vae_gap_2/summary.json`
- test `RMSE = 0.1315`
- test `active_rel_l2 = 0.9692`

Interpretation:

- the latent manifold itself is usable on active-only TD16;
- but mapping measurements to the latent code is much weaker than the
  autoencoder upper bound;
- so in the active-only regime, the dominant bottleneck is the inverse mapping
  `measurement -> latent`, not the learned image manifold.

### Mixed pilot

VAE autoencoder:

- result directory: `results/td16_vae_mixed_pilot_1`
- gap analysis: `results/td16_vae_predictor_mixed_pilot_1/td16_vae_gap_2`
- test `RMSE = 0.0856`
- test `active_rel_l2 = 0.9983`

Latent predictor:

- result directory: `results/td16_vae_predictor_mixed_pilot_1`
- test summary: `results/td16_vae_predictor_mixed_pilot_1/td16_test_eval_1/summary.json`
- gap analysis: `results/td16_vae_predictor_mixed_pilot_1/td16_vae_gap_2/summary.json`
- test `RMSE = 0.0842`
- test `active_rel_l2 = 0.9983`

Interpretation:

- once zero-change samples are mixed in, the VAE manifold itself already
  collapses close to the zero / weak-change operating point;
- the predictor nearly matches this weak manifold, so the bottleneck is no
  longer only `measurement -> latent`;
- the representation itself becomes too insensitive to sparse local changes.

### Two-stage fusion attempt

To test whether the VAE route can still benefit from stronger change
localization, I fused:

- a mixed-data dedicated mask predictor:
  `results/dct_sigma_td16_mask_mixed_1`
- an active-only VAE latent predictor:
  `results/td16_vae_predictor_active_pilot_1`

using the analysis script:

- `scripts/analyze_td16_vae_mask_fusion.py`

Result:

- base active-only VAE predictor on mixed test:
  `RMSE = 0.0891`, `active_rel_l2 = 0.9525`
- external soft mask fusion:
  `RMSE = 0.0862`, `active_rel_l2 = 0.9753`

Interpretation:

- external masking suppresses false positives and slightly improves global RMSE;
- but active-region recovery becomes worse;
- the fused result is still not better than the mixed zero baseline
  (`RMSE = 0.0835`).

### Conditional latent follow-up

To move beyond post-hoc fusion, I also trained a mask-conditioned latent
predictor:

- training result: `results/td16_vae_conditional_mixed_pilot_1`
- test summary:
  `results/td16_vae_conditional_mixed_pilot_1/td16_test_eval_1/summary.json`

Result:

- conditional latent mixed pilot:
  `RMSE = 0.0886`, `active_rel_l2 = 0.9686`

Interpretation:

- conditioning the latent head on the predicted low-frequency mask does improve
  local-change recovery relative to the mixed VAE predictor
  (`0.9983 -> 0.9686`);
- but global RMSE becomes worse than the mixed VAE predictor
  (`0.0842 -> 0.0886`);
- it also fails to beat the simpler external soft-fusion result (`0.0862`).

## Updated conclusion

The external pulmonary reproduction route now has two concrete pieces:

1. a reproduced classical GREIT baseline;
2. an approximate Zeng-style `VAE + MLP` latent-manifold baseline.

These two baselines already clarify the pulmonary research direction:

- `GREIT` is a necessary classical reference under the matched `16`-electrode
  protocol;
- the `VAE + latent predictor` route is plausible on active-only data, but in
  mixed time-difference data it collapses toward weak-change manifolds;
- a tighter mask-conditioned latent regressor is still not enough by itself;
- pulmonary TD16 therefore still needs explicit change localization and
  localization-aware residual modeling, rather than a pure latent-regression
  strategy.
