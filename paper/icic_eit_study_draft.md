# Low-Frequency-Prior-Constrained Learning for Electrical Impedance Tomography Reconstruction: A Comparative Study on Pixel-Wise, Latent, Dictionary, and Fixed-Basis Representations

## Abstract

Electrical impedance tomography (EIT) reconstruction for pulmonary monitoring is a highly ill-posed inverse problem. Recent neural reconstruction models can improve inference speed and segmentation quality, but their representation bias strongly affects whether the predicted conductivity images remain physically plausible. In this work, we conduct a comparative study of four representation families for KTC2023-style EIT reconstruction: pixel-wise discrete image generation, continuous low-dimensional latent autoencoding, discrete dictionary-constrained latent modeling, and fixed low-frequency basis prediction. We first optimize the simulation pipeline for large-scale data generation and improve evaluation efficiency through accelerated SSIM-based scoring. We then revisit FCUNet as a strong direct baseline and analyze why pixel-wise discrete models such as DPCAUNet and HC-DPCA-UNet are mismatched to the low-frequency structure of conductivity fields. Next, we study sparse autoencoding and structured 1D vector-quantized autoencoding to enforce global shape regularity. Finally, we introduce a fixed-basis 2D-DCT predictor that directly regresses low-frequency coefficients and a lightweight ensemble of complementary runs. Our experiments show that latent-space methods produce visually smoother manifolds but still suffer from a difficult measurement-to-latent mapping bottleneck, whereas the DCT-based representation provides a better balance between regularity and predictability. The final DCT ensemble reaches a total official benchmark score of **14.0553**, slightly exceeding the current FCUNet baseline **14.0485**. The study clarifies which representation biases are beneficial for EIT and which failure modes remain unresolved, providing a reproducible basis for subsequent pulmonary EIT research.

**Keywords:** electrical impedance tomography, inverse problems, representation learning, vector quantization, sparse autoencoder, pulmonary imaging

## 1. Introduction

Electrical impedance tomography reconstructs the conductivity distribution inside an object from boundary voltage measurements. Because the forward operator is nonlinear, ill-conditioned, and highly sensitive to measurement corruption, reconstruction quality depends not only on the inversion algorithm but also on the prior imposed on the solution space. For pulmonary EIT, this issue is particularly important because clinically meaningful images are dominated by low-frequency structures, smooth anatomical variation, and constrained topology.

Recent neural EIT pipelines often formulate reconstruction as either a direct mapping from electrode measurements to images or a two-stage process that first predicts a latent representation and then decodes the image. However, not every representation family is equally suitable. A representation that is too local or too discrete in the spatial domain may produce fragmented, high-frequency artifacts that violate the physical nature of conductivity fields. Conversely, an overly compressed latent representation may regularize the reconstructed image but make the inverse mapping from measurements to latent variables too difficult.

This paper summarizes and systematizes a full research cycle conducted on the KTC2023 codebase. The study includes engineering improvements that enable large-scale experimentation, comparative analysis of several representation families, and a structured discussion of why some approaches fail despite apparently favorable latent-space losses.

The main contributions of this work are:

1. A reproducible optimization of the EIT simulation and evaluation pipeline, including reduced-right-hand-side forward acceleration and Torch/CUDA-based fast scoring.
2. An empirical clarification that pixel-wise discrete generation strategies are poorly matched to low-frequency conductivity images, even when their optimization curves appear reasonable.
3. A two-stage latent representation study using continuous sparse autoencoding and discrete 1D vector-quantized autoencoding, showing that regular latent manifolds alone do not solve the measurement-to-image inversion bottleneck.
4. A fixed-basis low-frequency DCT predictor showing that an analytically constrained representation can outperform both latent autoencoders and the current FCUNet baseline.
5. A consolidated experimental basis and engineering framework for future pulmonary EIT research, including support for large-scale synthetic datasets and structured representation-learning experiments.

## 2. Related Work

### 2.1 Neural EIT Reconstruction

Neural EIT reconstruction methods typically fall into three categories: direct measurement-to-image mapping, iterative reconstruction refinement, and latent-variable modeling. Direct approaches are attractive because they provide fast inference and straightforward optimization, but they rely heavily on the inductive bias of the architecture. Iterative or diffusion-based approaches can generate sharper outputs, yet they are usually slower and more difficult to tune. Latent approaches attempt to restrict the image space to a lower-dimensional manifold before learning the inverse map.

### 2.2 Low-Frequency Structural Priors

Conductivity distributions in medical EIT usually exhibit strong low-frequency structure and constrained topology. This suggests that architectures enforcing global or shape-level priors may be more suitable than methods that model images as largely independent pixel-wise labels. The present study is motivated by this observation and tests it through multiple model families.

### 2.3 Discrete Latent Modeling

Vector-quantized autoencoders provide a natural way to enforce discrete structure and a limited dictionary of valid image components. In principle, this can prevent implausible images and improve topological regularity. Whether such discrete latent spaces are also easy to predict from EIT measurements remains an open question addressed here.

## 3. Engineering Foundation

### 3.1 Accelerated Data Generation

Large-scale neural EIT research requires high-throughput simulation. We therefore first optimized the forward simulation pipeline. The key observation is that the measurement excitation matrix used in the current benchmark has rank 15 rather than 76, which enables an exact reduced-right-hand-side solve. This substantially improves throughput in practice and provides a more scalable data generation backbone for subsequent experiments.

### 3.2 Accelerated Scoring

The official SSIM-based challenge score is computationally expensive. We benchmarked and integrated a Torch-based fast scorer with optional CUDA acceleration. On a single sample, the measured average runtime was approximately:

- Official scorer: about 92 s
- Fast SciPy scorer: about 0.13 s
- Torch CUDA fast scorer: about 0.002-0.003 s

This acceleration enables rapid model iteration without changing the default official evaluation option in the formal benchmarking script.

## 4. Methods

### 4.1 FCUNet Baseline

FCUNet directly maps the flattened measurement vector to a coarse image representation and then refines it through a U-Net backbone. It serves as the strongest practical baseline in the current codebase and provides a reference point for all subsequent representation-learning experiments.

### 4.2 Pixel-Wise Discrete Models

We investigated DPCAUNet and HC-DPCA-UNet, which introduce stronger pixel-level discretization and attention-style processing. Despite adequate optimization behavior, these models showed a persistent mismatch with the low-frequency structure of conductivity images. The resulting reconstructions were often fragmented or unstable, suggesting that pixel-wise discrete modeling is a poor inductive bias for this task.

### 4.3 Sparse AutoEncoder (SAE)

The SAE pipeline decomposes the task into:

1. Learning a compact image latent from the ground-truth conductivity map.
2. Training a predictor from boundary measurements to the latent vector.
3. Reconstructing the image through the decoder.

The latent includes a shape component and an angle component. The autoencoder itself reconstructs reasonably well, indicating that the image manifold is learnable. However, the measurement-to-latent predictor often converges to low latent error while still producing decoded images outside the target visual domain. This shows that latent regression quality alone is not a sufficient proxy for image-domain correctness.

### 4.4 ST-1D-VQ-VAE

To impose stronger regularity, we introduced a structured 1D vector-quantized latent model. The model first aligns the image to a canonical orientation, encodes global shape information into a small number of slots, quantizes each slot against a codebook, and then reconstructs the canonical image before rotating it back. This design aims to prevent unrealistic local artifacts and constrain reconstructions to a discrete global dictionary.

The corresponding predictor maps measurements to:

- discrete slot logits for each latent slot, and
- a two-dimensional angle vector.

This design improves image regularity compared with continuous latent decoding, but the current predictor still underperforms the direct FCUNet baseline, indicating that the inverse mapping remains the primary bottleneck.

### 4.5 Fixed-Basis DCT Predictor

The final successful direction in this study is a fixed-basis low-frequency predictor. Instead of learning a high-entropy latent space and then inverting measurements into that latent space, we directly parameterize the image logits using a small set of low-frequency 2D-DCT coefficients. Concretely:

1. The measurement vector is concatenated with a difficulty-level embedding.
2. An MLP predicts `3 x K x K` DCT coefficients.
3. A fixed inverse DCT decoder reconstructs `3 x 256 x 256` class logits.
4. The model is trained with `CE + Dice` image supervision plus an auxiliary coefficient regression loss.

This formulation imposes an explicit low-frequency prior while avoiding the discrete-slot prediction bottleneck seen in VQ models. It also preserves direct end-to-end optimization and very fast inference.

We additionally study a lightweight ensemble of two independently trained DCT predictors. The ensemble simply averages logits and introduces no iterative refinement, but it turns out to be the strongest method in the current repository.

A more detailed diagnosis further shows that the original discrete predictor formulation was unnecessarily hard. Although the VQ codebook size was set to 512, the trained autoencoder actually used only a very small subset of codes in each slot, with several slots collapsing to one or two valid states. This means the predictor was solving a 512-way classification problem for many slots whose true support size was only 1-4 classes. We therefore introduced a compact-vocabulary predictor that performs slot-wise classification only over the codes actually observed in the latent cache. In a quick smoke test, this reduced the first validation loss from roughly 6.16 in the dense predictor setting to about 0.545, indicating that a large part of the difficulty came from label-space overparameterization rather than from the decoder alone.

## 5. Experimental Setup

### 5.1 Dataset

Experiments are conducted on the KTC2023 benchmark and associated simulated training data. The current simulation pipeline supports efficient generation of large HDF5-based datasets and train/validation/test splitting through fixed random seeds for reproducibility.

For the pulmonary extension of this project, the repository also includes real thoracic EIT measurements in Draeger `.get` format under the bundled PLOS One subject data. In the current stage, we completed the format confirmation and Python-side parsing of these files into `256 x T` raw measurement matrices and `208 x T` DCT-EIT-compatible reordered measurements. As expected for real pulmonary EIT, these measurements do not include true conductivity maps. We therefore treat them as future Sim2Real validation data rather than as a supervised training set. To support pulmonary experiments immediately, we additionally implemented a local thorax-style simulation pipeline with two lungs, a heart region, and simple pathology variants, producing HDF5 datasets suitable for direct neural training. The current synthetic pulmonary datasets include a `512`-sample pilot set and a larger `2048`-sample set for baseline training.

An additional compatibility result is important here: the parsed PLOS One measurements are based on a `16`-electrode acquisition layout and reduce to `208` valid channels after DCT-EIT-style reordering, whereas the current KTC-style neural pipelines in this repository assume `32` electrodes and a flattened measurement length of `2356`. Consequently, the current pulmonary neural baselines cannot yet be applied directly to the bundled real `.get` data without building a separate `16`-electrode simulation and training chain.

### 5.2 Evaluation

The official benchmark score is based on class-wise SSIM computed on binary masks for the two anomaly classes and averaged across classes. For rapid development, we additionally use a mathematically aligned fast implementation. Final comparisons still preserve the official evaluation path.

### 5.3 Training Protocol

Across methods, the implementation supports:

- automatic train/validation/test splitting for HDF5 data,
- checkpoint saving after every epoch,
- best/last checkpoint maintenance,
- learning-rate scheduling and early stopping,
- optional mixed precision when appropriate.

## 6. Results

### 6.1 Main Quantitative Comparison

Before the DCT study, the strongest validated method in the repository was FCUNet. On the available official evaluation results:

- FCUNet total challenge score: **14.0485**
- mean per-sample score: **0.6690**

By comparison, the exploratory latent and discrete alternatives currently lag behind:

- SAE total score: **5.2296**
- SAE mean per-sample score: **0.2490**
- HC-DPCA-UNet total score: **4.2589**
- HC-DPCA-UNet mean per-sample score: **0.2028**
- DPCAUNet total score: **2.8042**
- DPCAUNet mean per-sample score: **0.1335**

These results support the central observation of this study: stronger image regularization does not automatically yield better end-to-end inverse mapping performance.

For the fixed-basis line, the DCT predictor yields a markedly different outcome. The best single DCT model reaches:

- DCT predictor total score: **13.9799**

which already nearly matches FCUNet. More importantly, an equal-weight ensemble of two complementary DCT runs reaches:

- DCT predictor ensemble total score: **14.0553**

This slightly exceeds the FCUNet baseline and becomes the strongest method currently validated in the repository.

For the VQ-based line, several additional observations are now available. A high-entropy discrete model (`16` slots, `512` codewords per slot) combined with frozen-decoder image supervision still obtained only about **2.55** total score in a fast evaluation probe on the official test set. In contrast, a deliberately lower-entropy VQ model (`8` slots, `64` codewords per slot) improved to about **4.78** total score under the same fast evaluation protocol after 20 epochs. Although this is still far below FCUNet, it provides important evidence that latent entropy is itself a major determinant of inverse-map difficulty. Continuing the same low-entropy model to 40 epochs further reduced validation loss but decreased the official fast-probe score to about **3.92**, indicating that the original surrogate validation objective was not perfectly aligned with final benchmark quality.

A follow-up diagnosis revealed that the training-time probe itself had been slightly optimistic because it used a soft decoding path (`slot logits -> expected embedding -> decoder`) whereas the actual pipeline used hard argmax decoding (`slot logits -> discrete indices -> decoder`). After correcting this mismatch and selecting checkpoints only by the hard-decoding probe score, a new low-entropy run achieved **4.82** total score, which is the current best result within the VQ family. We also tested a more image-heavy loss weighting strategy; it reduced the score to about **3.21**, showing that simply increasing image-domain supervision is not sufficient and may in fact degrade the benchmark objective.

### 6.2 Training Behavior

For FCUNet, the training process is stable and validation loss decreases to a low level, yielding the strongest challenge performance among tested methods.

For SAE, the autoencoder reconstruction stage converges to a usable image manifold, while the latent predictor achieves numerically small validation loss. Nevertheless, decoded images can still deviate significantly from the target domain. This discrepancy reveals a semantic gap between latent-space regression and image-space fidelity.

For VQ-SAE, the autoencoder constrains image morphology more effectively, but the current predictor still struggles to convert measurement information into correct slot selections.

For the DCT predictor, training is substantially more stable. The key observation is that the target space is both low-frequency and directly image-aligned, which makes the inverse mapping easier than high-entropy latent classification. However, we also observe that validation loss alone is not perfectly aligned with final benchmark score. To address this, we introduce a validation probe score and select checkpoints by score rather than only by loss.

### 6.3 Negative Findings

The following negative findings are important and reproducible:

1. Pixel-wise discrete modeling is not well aligned with low-frequency conductivity images.
2. Small latent-space MSE does not guarantee plausible decoded conductivity images.
3. A better image manifold alone does not solve the inverse measurement-to-latent mapping problem.
4. In discrete latent modeling, a large nominal codebook can create an unnecessarily difficult predictor target when the effective slot-wise support is much smaller.
5. For direct low-frequency predictors, validation loss and final benchmark score can still diverge, so score-aware checkpoint selection is important.

These findings are useful because they narrow the search space for future pulmonary EIT reconstruction research.

## 7. Discussion

The experiments indicate that the main bottleneck has shifted from image-side regularization to inversion-side representation predictability. In other words, once the decoder becomes sufficiently constrained, the remaining challenge is whether boundary measurements contain enough stable information to recover the required latent configuration through a simple predictor.

The latest low-entropy VQ experiments sharpen this conclusion. The cloud-trained high-capacity VQ autoencoder reconstructs ground-truth images well, so the decoder and dictionary are not the main failure source. The difficulty instead arises because the predictor must select among a very large number of discrete code combinations from highly compressed boundary measurements. Reducing the latent entropy from `16 x 512` to `8 x 64` significantly improves validation loss, image-domain plausibility, and official fast-probe score, even though the low-entropy autoencoder itself is somewhat less accurate as a pure image autoencoder. This tradeoff strongly suggests that for EIT, predictor-accessible latent structure is more important than maximizing decoder expressiveness.

The corrected hard-decoding probe also highlights a methodological lesson: checkpoint selection must use exactly the same decoding path as final inference. Otherwise, even a small mismatch between soft latent decoding and hard slot decoding can bias model selection. After fixing this issue, the best low-entropy VQ run improved slightly from **4.78** to **4.82** total score, which confirms that score-aligned selection is beneficial but not sufficient to close the large gap to FCUNet.

At the same time, a deliberately image-heavy loss weighting strategy performed worse than the balanced objective. This indicates that the current VQ failure mode is not merely caused by insufficient image supervision. The predictor still lacks a measurement-to-latent parameterization whose complexity matches the information actually present in EIT boundary data.

The DCT results refine this conclusion further. The key gain does not come from higher model capacity, but from choosing a representation whose complexity better matches the information content of EIT boundary measurements. Compared with latent autoencoders, the DCT basis is:

- globally coupled,
- explicitly low-frequency,
- analytically decodable,
- and much easier to regress from measurements.

The final ensemble result also suggests that the remaining error is partly optimization- and seed-dependent rather than purely representational. Two independently trained DCT models exhibit useful complementarity, and their logit average is enough to pass the FCUNet baseline. This provides a strong indication that the fixed-basis route is currently the most promising direction for benchmark-level EIT reconstruction in this codebase.

The first pulmonary simulation results are also consistent with this picture. On the newly generated `2048`-sample thorax-style dataset, a DCT predictor can be trained without any architecture changes and quickly reaches a strong validation probe score within the first few epochs. However, the best score appears much earlier than the minimum validation loss, again showing that purely optimizing image loss is not fully aligned with structure-aware reconstruction quality. This suggests that the representation-level advantage of the DCT basis may transfer to pulmonary EIT as well, but score-aware model selection will likely remain important.

We then performed a direct test-split evaluation on the `2048`-sample pulmonary HDF5 using a dedicated simulation-evaluation script. The baseline pulmonary DCT model achieved a mean score of **0.7272** and a total score of **149.80** on the held-out test split, while a coefficient-heavier variant (`coeff_loss_weight = 1.0`) slightly underperformed it with **0.7265 / 149.66**. An equal-weight ensemble of these two pulmonary models reached **0.7269 / 149.74**, which still did not exceed the best single model. Thus, unlike the benchmark-side KTC ensemble, the current pulmonary DCT variants do not yet show strong complementary error patterns.

The pulmonary DCT line nevertheless shows a clear positive data-scaling trend. On the smaller `512`-sample pilot thorax dataset, the same method reached only about **0.5139** mean score on the held-out test split. Expanding the synthetic pulmonary dataset to `2048` samples increased the score to **0.7272**, indicating that the fixed-basis method benefits substantially from additional supervised data even in the pulmonary regime.

We also evaluated the pulmonary FCUNet control run on the same test split. After continuing the run to `20` full-training epochs, the best saved checkpoint reached **0.7819** mean score and **161.07** total score, clearly surpassing the pulmonary DCT variants. This finding is important because it shows that the representation ranking is task-dependent: the fixed-basis DCT approach is strongest on the benchmark phantom task, but the more flexible FCUNet currently performs better on the lung-style synthetic dataset. In other words, the pulmonary regime appears to reward a richer spatial decoder more than the benchmark phantom regime does, even though FCUNet is more expensive to train.

The computational tradeoff is also clear in direct test-time measurement. On the same `206`-sample pulmonary test split, FCUNet required about **3.72 s** for reconstruction, whereas the pulmonary DCT model required about **0.36 s** and the pulmonary DCT ensemble about **0.32 s**. Thus, the DCT family retains a strong inference-efficiency advantage, but this advantage currently comes at a noticeable pulmonary accuracy cost.

This advantage becomes even clearer in controlled latency benchmarking. The pulmonary DCT model contains about **3.85M** parameters, compared with about **40.70M** for FCUNet. Its measured single-sample latency is about **2.14 ms**, compared with **28.82 ms** for FCUNet, and under batched inference the gap remains large (**0.25 ms/sample** versus **14.60 ms/sample** at batch size `32`). Therefore, although FCUNet currently gives better pulmonary reconstruction accuracy, the proposed DCT route remains attractive when model compactness and low-latency deployment are important design criteria.

This suggests several promising next directions:

1. Increasing latent capacity while preserving global structure.
2. Replacing deterministic latent regression with probabilistic latent prediction, such as VAE-style mean prediction with deterministic decoding at test time.
3. Introducing adversarial or distribution-level constraints in the image space.
4. Exploring analytically structured bases such as 2D-DCT or implicit signed-distance-function representations.
5. Training compact-vocabulary or entropy-aware discrete predictors to match the actual slot-wise latent support.
6. Extending the fixed-basis idea beyond DCT, for example with other analytically structured low-frequency bases or implicit continuous shape parameterizations.
7. Extending the study from benchmark phantoms to pulmonary EIT simulations generated with established EIDORS tools.
8. Designing explicitly low-entropy structured dictionaries, rather than only learning large generic codebooks and hoping the predictor can invert them.
9. Building a `16`-electrode pulmonary training pipeline compatible with available real thoracic `.get` measurements.

## 8. Conclusion

This work provides a reproducible investigation of representation choices for neural EIT reconstruction. The current evidence shows that:

- FCUNet is a strong baseline, but it can be slightly surpassed by a carefully chosen fixed-basis low-frequency representation.
- Pixel-wise discrete architectures are poorly matched to low-frequency conductivity structure.
- Continuous and discrete latent autoencoders can improve image regularity but do not, by themselves, solve the inverse mapping problem.
- A fixed low-frequency DCT parameterization offers the best current tradeoff between image regularity and inverse-map predictability, reaching **14.0553** total official score through a lightweight two-model ensemble.

Beyond model comparison, the work also establishes an optimized simulation and evaluation framework that supports future pulmonary EIT research. The next stage should therefore separate two objectives more clearly: benchmark-level low-frequency inversion, where the DCT route is currently strongest, and pulmonary reconstruction, where FCUNet currently remains the better supervised baseline on the available synthetic thorax dataset.

## 9. Reproducibility Notes

The implementation supports:

- automatic experiment directory versioning under `results/{tag}_{num}`,
- persistent `best.pt` and `last.pt` checkpoints,
- per-epoch training logs,
- HDF5-backed datasets,
- accelerated scoring for rapid experimentation,
- visualization scripts for both autoencoder reconstruction and end-to-end predictor outputs.

## References

The final ICIC submission version should add formal bibliography entries for:

1. KTC2023 benchmark and challenge materials.
2. FCUNet-related baseline sources.
3. Vector-quantized autoencoder literature.
4. EIDORS and pulmonary EIT simulation references.
5. Related EIT latent-representation or implicit-representation studies.
