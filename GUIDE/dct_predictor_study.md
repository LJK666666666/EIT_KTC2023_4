# DCT Predictor Study

## Motivation

Continuous SAE and discrete VQ-based latent methods both showed the same core bottleneck: the learned image manifold was reasonable, but the mapping from EIT measurements to latent variables remained unstable. This suggested trying a lower-complexity image prior that is:

- explicitly low-frequency,
- globally coupled,
- easy to decode,
- and easier to predict than high-entropy latent codes.

The resulting method is a fixed-basis 2D-DCT predictor.

## Method

The image logits are represented by low-frequency DCT coefficients:

- input: flattened EIT measurements of length `2356`
- backbone: MLP with level embedding
- output: `3 x K x K` DCT coefficients
- decoder: fixed inverse DCT to `3 x 256 x 256` logits

Training loss:

- image loss: `CE + Dice`
- coefficient loss: `MSE(pred_coeffs, target_coeffs)`

Final objective:

`L = L_image + lambda_coeff * L_coeff`

This keeps the image prior simple and globally smooth while preserving direct end-to-end optimization.

## Main Experiments

The most important runs are:

- [`results/dct_predictor_c20_long_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_c20_long_1)
- [`results/dct_predictor_probealign_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_probealign_1)
- [`results/dct_predictor_coeff1_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_coeff1_1)

Key findings:

1. `coeff_size=20` is the best frequency cutoff among tested settings.
2. Validation loss alone is not sufficient for checkpoint selection.
3. Probe-score-based selection is necessary, but still not perfectly aligned.
4. Slight regularization is useful; removing dropout degrades benchmark score.
5. Stronger coefficient supervision (`lambda_coeff=1.0`) is competitive, but the best single model still came from the original moderate setting.

Best single-model result:

- [`results/eval_dct_predictor_8/scores.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/eval_dct_predictor_8/scores.json)
- total score: `13.9799`

This already approaches the FCUNet baseline.

## Probe-Score Alignment

To reduce the mismatch between `val_loss` and benchmark score, the trainer was extended to compute a validation probe score:

- metric: `val_probe_score_total`
- scorer: accelerated Torch/CUDA fast scorer
- selection mode: `max`

Relevant code:

- [`src/trainers/dct_predictor_trainer.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/dct_predictor_trainer.py)
- [`src/configs/dct_predictor_config.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/configs/dct_predictor_config.py)

This improves model selection quality, even though it does not fully solve benchmark misalignment.

## Ensemble

The strongest result came from a simple two-model logit ensemble:

- [`results/dct_predictor_c20_long_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_c20_long_1)
- [`results/dct_predictor_probealign_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_probealign_1)

with equal weights `0.5 / 0.5`.

This was formalized as:

- [`src/pipelines/dct_predictor_ensemble_pipeline.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/pipelines/dct_predictor_ensemble_pipeline.py)
- [`scripts/dct_predictor_ensemble.yaml`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/dct_predictor_ensemble.yaml)

Official evaluation result:

- [`results/eval_dct_predictor_ensemble_2/scores.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/eval_dct_predictor_ensemble_2/scores.json)
- total score: `14.0553`

This exceeds the current FCUNet baseline `14.0485`.

## Conclusion

The DCT line is the first method in this study that reaches and slightly surpasses the current FCUNet benchmark level.

The main reason is not higher model complexity, but a better-matched representation:

- low-frequency,
- globally structured,
- easy to decode,
- and much easier to predict from measurements than high-entropy latent variables.

At the current stage, the DCT predictor ensemble should be regarded as the strongest method in the repository for the KTC2023 benchmark.
