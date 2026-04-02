# VQ-SAE Compact Predictor Diagnosis and Improvement

## Background

The original `vq_sae_predictor` treated every latent slot as a 512-way classification problem. However, the actual discrete codes produced by the trained `vq_sae` autoencoder are far more concentrated.

## Diagnosis

Using [`results/vq_sae_baseline_2/latent_codes.h5`](../results/vq_sae_baseline_2/latent_codes.h5), the latent code usage shows strong slot-wise collapse:

- 16 slots in total
- only 35 unique code indices used globally
- many slots use only 1-4 codes

A diagnostic summary has been saved to:
- [`results/vq_sae_predictor_baseline_1/vq_diagnosis.json`](../results/vq_sae_predictor_baseline_1/vq_diagnosis.json)

Representative slot vocabulary sizes:
- slot 0: 3
- slot 1: 1
- slot 2: 2
- slot 3: 4
- slot 4: 2
- slot 7: 1
- slot 8: 1
- slot 10: 1
- slot 12: 1
- slot 14: 1

Under the original 16x512 classification setting, a quick validation diagnostic on 256 samples gave:
- slot-wise accuracy: about `0.0310`

This indicates that the predictor was solving a much harder problem than necessary.

## Improvement

A compact-vocabulary predictor was implemented:

1. Read all latent slot indices from `latent_codes.h5`.
2. Build a slot-wise vocabulary of actually used code indices.
3. Replace each 512-way slot head with a slot-specific head whose output size equals the number of used codes in that slot.
4. Save the mapping to [`slot_vocab.json`](../results/vq_sae_predictor_compact_2/slot_vocab.json).
5. At inference time, map local class IDs back to original codebook IDs before decoding.

## Code Changes

Main affected files:
- [`src/models/vq_sae/predictor.py`](../src/models/vq_sae/predictor.py)
- [`src/data/sim_dataset.py`](../src/data/sim_dataset.py)
- [`src/trainers/vq_sae_predictor_trainer.py`](../src/trainers/vq_sae_predictor_trainer.py)
- [`src/pipelines/vq_sae_pipeline.py`](../src/pipelines/vq_sae_pipeline.py)
- [`src/trainers/base_trainer.py`](../src/trainers/base_trainer.py)

## Smoke Test Result

A quick 2-iteration run was completed with:

```bash
python scripts/train.py --method vq_sae_predictor --hdf5-path dataset_sim/level_1/data.h5 --vq-sae-checkpoint results/vq_sae_baseline_2/best.pt --vq-latent-h5-path results/vq_sae_baseline_2/latent_codes.h5 --max-iters 2 --batch-size 64 --num-workers 0 --device cpu --experiment-name vq_sae_predictor_compact
```

Result directory:
- [`results/vq_sae_predictor_compact_2`](../results/vq_sae_predictor_compact_2)

Observed first-epoch validation metrics:
- `val_loss = 0.5447`
- `val_slot_loss = 0.5016`
- `val_angle_loss = 0.0861`

For comparison, the previous dense 512-way predictor started around:
- `val_loss = 6.1582`

This is a strong indication that compact slot vocabularies remove a large amount of unnecessary classification difficulty.

## Remaining Issues

1. This is currently only a smoke-tested improvement, not a full benchmarked result.
2. The local environment has a `tensorboard/tensorflow + numpy 2` compatibility problem, so TensorBoard is now auto-disabled when import fails.
3. The next required step is a full training run and official evaluation.


## 2026-04-01 Additional Findings

### Cloud-trained `vq_sae_baseline_4`
- The cloud-trained VQ autoencoder is qualitatively usable.
- `results/vq_sae_baseline_4/ae_gt_reconstruction_1/` shows that GT autoencoding preserves low-frequency regular shapes, including harder evaluation samples.
- Therefore the main failure is not in the VQ decoder itself.

### Why compact predictor stopped helping
- For the smoke model `vq_sae_baseline_2`, many slots collapsed to very few active codewords, so compact vocabularies reduced the effective classification difficulty sharply.
- For the properly trained `vq_sae_baseline_4`, each slot uses about `507~508` codewords, i.e. almost the full `K=512` vocabulary.
- Therefore slot-wise compactification no longer materially reduces the task difficulty.

### Predictor observations
- `vq_sae_predictor_baseline_5` still converges to very high slot CE (`val_loss ~ 5.9`) and produces domain-mismatched template-like structures.
- `vq_sae_predictor_compact_4` using `vq_sae_baseline_4` is only marginally better visually; it still outputs a repeated generic prototype rather than sample-conditioned geometry.
- Adding differentiable image-domain supervision through the frozen VQ decoder (`vq_sae_predictor_imgloss_2`) improves output diversity and connectivity, but predictions are still far from GT and remain below the FCUNet baseline.

### Current conclusion
- A strong discrete VQ autoencoder alone is insufficient.
- The main bottleneck is the measurement-to-discrete-latent inverse mapping.
- With a high-entropy codebook, pure slot classification remains too hard, even with auxiliary image loss.
- This suggests that future work should reduce latent entropy structurally rather than only improving the predictor head.

## 2026-04-01 Low-Entropy VQ Follow-Up

### Motivation

The cloud-trained `vq_sae_baseline_4` confirmed that the decoder side was already reasonably strong, but its latent space had very high effective entropy:

- `num_slots = 16`
- `codebook_size = 512`
- nearly every slot used about `507~508` codewords

This makes the measurement-to-latent classification problem extremely hard. To test whether structural entropy reduction is more effective than predictor-side tuning alone, a smaller VQ model was trained:

- `num_slots = 8`
- `codebook_size = 64`
- `code_dim = 16`

Result directory:
- [`results/vq_sae_lowent_1`](../results/vq_sae_lowent_1)

### Autoencoder quality

Although the low-entropy autoencoder is weaker than `vq_sae_baseline_4`, its GT autoencoding remains visually regular and low-frequency:
- [`results/vq_sae_lowent_1/ae_gt_reconstruction_1/codes_python_comparison.png`](../results/vq_sae_lowent_1/ae_gt_reconstruction_1/codes_python_comparison.png)

The exported latent cache uses shape:
- `indices: (12800, 8)`

All 8 slots still use the full 64-code vocabulary, but the predictor target is now much smaller than the original `16 x 512`.

### Predictor result

Using the same frozen-decoder image-loss training strategy, a predictor was trained on the low-entropy latent space:
- [`results/vq_sae_lowent_predictor_1`](../results/vq_sae_lowent_predictor_1)

After 20 epochs:
- `val_loss`: `5.4475 -> 4.0782`
- `val_slot_loss`: `4.2241 -> 3.3132`
- `val_image_loss`: `1.2195 -> 0.7600`

Visualized train/val/test predictions:
- [`results/vq_sae_lowent_predictor_1/ae_sim_samples_1/train_comparison.png`](../results/vq_sae_lowent_predictor_1/ae_sim_samples_1/train_comparison.png)
- [`results/vq_sae_lowent_predictor_1/ae_sim_samples_1/val_comparison.png`](../results/vq_sae_lowent_predictor_1/ae_sim_samples_1/val_comparison.png)
- [`results/vq_sae_lowent_predictor_1/ae_sim_samples_1/test_comparison.png`](../results/vq_sae_lowent_predictor_1/ae_sim_samples_1/test_comparison.png)

The predictions are still far from FCUNet quality, but they are visibly more sample-dependent and less template-collapsed than the high-entropy predictor.

### Official evaluation fast probe

Using `VQSAEPipeline + Torch fast scorer` on `EvaluationData`, the low-entropy model achieved:
- [`results/vq_sae_lowent_predictor_1/eval_fastprobe_1/scores.json`](../results/vq_sae_lowent_predictor_1/eval_fastprobe_1/scores.json)
- `mean_score = 0.2276`
- `total_score = 4.7802`

For comparison, the high-entropy image-loss model achieved:
- [`results/vq_sae_predictor_imgloss_2/eval_fastprobe_1/scores.json`](../results/vq_sae_predictor_imgloss_2/eval_fastprobe_1/scores.json)
- `mean_score = 0.1214`
- `total_score = 2.5490`

### Longer training is not monotonically better

The low-entropy predictor was then resumed from epoch 20 to epoch 40:
- same result directory [`results/vq_sae_lowent_predictor_1`](../results/vq_sae_lowent_predictor_1)

The optimization metrics kept improving:
- `val_loss`: `4.0782 -> 3.6664`
- `val_slot_loss`: `3.3132 -> 2.9928`
- `val_image_loss`: `0.7600 -> 0.6690`

However, the official fast-probe score on `EvaluationData` dropped:
- [`results/vq_sae_lowent_predictor_1/eval_fastprobe_2/scores.json`](../results/vq_sae_lowent_predictor_1/eval_fastprobe_2/scores.json)
- `mean_score = 0.1868`
- `total_score = 3.9237`

This is an important negative result:
- the current validation loss is not fully aligned with the official score;
- continued optimization of slot/image loss can improve the surrogate objective while hurting final benchmark quality;
- future training should consider score-aligned validation or stronger regularization/early stopping criteria.

### Updated conclusion

These results strongly support the current research hypothesis:

- improving the predictor head alone is not enough for a high-entropy discrete latent space;
- reducing latent entropy structurally produces a larger gain than compact vocab remapping or frozen-decoder image loss alone;
- the best observed low-entropy checkpoint in this round is still the earlier 20-epoch stage rather than the lowest-validation-loss checkpoint;
- however, even the low-entropy VQ route remains substantially below the FCUNet baseline, so it is still not ready to replace direct end-to-end reconstruction.

## 2026-04-01 Score-Aligned Follow-Up

### Validation mismatch diagnosis

During follow-up experiments, an inconsistency was found between:
- the online `val_probe_score_total` reported during training, and
- the external `VQSAEPipeline` fast evaluation score.

The root cause was not the scorer itself, but the decoding path used inside training:

- the trainer-side probe previously used **soft decoding**
  - `slot logits -> softmax -> expected embedding -> decoder`
- the official pipeline used **hard decoding**
  - `slot logits -> argmax -> discrete indices -> decoder`

The soft version is smoother and systematically optimistic. It therefore overestimated the benchmark quality and led to incorrect best-checkpoint selection.

This has now been fixed so that training-time probe evaluation uses the same hard decoding path as the formal pipeline.

### Hard-probe-selected low-entropy predictor

A new low-entropy predictor run was trained with the corrected probe selection:
- [`results/vq_sae_lowent_hardprobe_1`](../results/vq_sae_lowent_hardprobe_1)

Its best checkpoint, selected by the corrected official-style fast probe, achieved:
- [`results/vq_sae_lowent_hardprobe_1/eval_fastprobe_1/scores.json`](../results/vq_sae_lowent_hardprobe_1/eval_fastprobe_1/scores.json)
- `mean_score = 0.2295`
- `total_score = 4.8186`

This is slightly better than the earlier low-entropy predictor result:
- `vq_sae_lowent_predictor_1`: `total_score = 4.7802`

The gain is small, but it confirms that:
- score-aligned model selection is necessary;
- previous selection by latent/image validation losses was not reliable.

### Image-heavy supervision is not better

To test whether the predictor should rely more strongly on image-domain supervision, an image-heavy low-entropy run was trained with:
- lower slot-loss weight,
- lower angle-loss weight,
- larger image-loss weight.

Result directory:
- [`results/vq_sae_lowent_imgheavy_1`](../results/vq_sae_lowent_imgheavy_1)

External fast-probe result:
- [`results/vq_sae_lowent_imgheavy_1/eval_fastprobe_1/scores.json`](../results/vq_sae_lowent_imgheavy_1/eval_fastprobe_1/scores.json)
- `mean_score = 0.1529`
- `total_score = 3.2107`

This is substantially worse than the balanced hard-probe-selected run.

### Current best VQ conclusion

As of this round, the strongest VQ-based predictor found is:
- [`results/vq_sae_lowent_hardprobe_1`](../results/vq_sae_lowent_hardprobe_1)
- `total_score = 4.8186`

This leads to three concrete conclusions:

1. Lower latent entropy helps.
2. Probe evaluation must match the exact hard-decoding inference path.
3. Simply increasing image-loss weight does not solve the inverse mapping problem and can hurt benchmark performance.
