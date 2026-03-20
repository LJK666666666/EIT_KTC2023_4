============================================================
Digital Twin Residual Analysis
============================================================

--- Step 0: Fit background conductivity from Uelref ---
  Background conductivity (from Uelref): 0.803581 S/m
  Loss: 3.197975e+00, nfev: 70
  Uelref residual: mean=-3.494322e-04, std=3.684091e-02, max_abs=3.011937e-01

  --fix-bg enabled: all samples use sigma_bg=0.803581

============================================================
Part A: Training Data (4 samples)
============================================================

--- Train Sample 1 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 12.629282 S/m
  Loss = 3.232968e+00, nfev = 78, time = 10.7s
  E_gap: mean=-3.315356e-04, std=3.704211e-02, max_abs=4.320871e-01
  Relative fit error: 8.3704e-02

--- Train Sample 2 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 3.569205 S/m
  Loss = 3.361433e+00, nfev = 69, time = 9.8s
  E_gap: mean=-3.676088e-04, std=3.777061e-02, max_abs=3.983241e-01
  Relative fit error: 8.5139e-02

--- Train Sample 3 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 5.000000 S/m
  Loss = 2.628670e+00, nfev = 22, time = 3.3s
  E_gap: mean=-1.846205e-04, std=3.340210e-02, max_abs=2.721899e-01
  Relative fit error: 7.9213e-02

--- Train Sample 4 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 4.760762e+00, nfev = 4, time = 0.5s
  E_gap: mean=1.179691e-04, std=4.495201e-02, max_abs=3.802391e-01
  Relative fit error: 1.0966e-01

============================================================
Noise Distribution Analysis — Training Data (4 samples)
============================================================

  Mean bias (systematic): mean=-1.914490e-04, std=3.591614e-02
  Per-sample noise std: ['3.7042e-02', '3.7771e-02', '3.3402e-02', '4.4952e-02']
  Overall noise std: 3.8520e-02
  Sample 1 SNR: 21.5 dB
  Sample 2 SNR: 21.3 dB
  Sample 3 SNR: 22.4 dB
  Sample 4 SNR: 19.8 dB

  Per-channel variance: mean=1.9385e-04, max=5.5181e-03, min=5.3764e-08
  Cross-sample correlation: mean=0.851, min=0.687, max=0.947


============================================================
Part B: Evaluation Data (7 levels x 3 samples)
============================================================

--- eval_L1_S1 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 3.570234e+00, nfev = 4, time = 0.6s
  E_gap: mean=-3.341386e-04, std=3.892644e-02, max_abs=3.734704e-01
  Relative fit error: 8.6886e-02

--- eval_L1_S2 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 3.360015 S/m
  Loss = 2.703149e+00, nfev = 36, time = 4.7s
  E_gap: mean=-3.217934e-04, std=3.387098e-02, max_abs=2.623011e-01
  Relative fit error: 7.8478e-02

--- eval_L1_S3 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 4.865080e+00, nfev = 4, time = 0.5s
  E_gap: mean=9.943613e-05, std=4.544188e-02, max_abs=4.016424e-01
  Relative fit error: 1.0906e-01

--- eval_L2_S1 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 3.954698 S/m
  Loss = 2.679018e+00, nfev = 92, time = 12.9s
  E_gap: mean=-2.792524e-04, std=3.371982e-02, max_abs=2.693505e-01
  Relative fit error: 7.8310e-02

--- eval_L2_S2 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 4.401208e+00, nfev = 4, time = 0.5s
  E_gap: mean=4.591618e-05, std=4.322132e-02, max_abs=3.572213e-01
  Relative fit error: 1.0321e-01

--- eval_L2_S3 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 4.893744 S/m
  Loss = 2.524793e+00, nfev = 34, time = 5.2s
  E_gap: mean=-2.850549e-04, std=3.273473e-02, max_abs=2.500246e-01
  Relative fit error: 7.6925e-02

--- eval_L3_S1 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 5.914705e+00, nfev = 4, time = 0.6s
  E_gap: mean=6.473599e-05, std=5.010471e-02, max_abs=7.051724e-01
  Relative fit error: 1.1314e-01

--- eval_L3_S2 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.007434 S/m
  Loss = 3.777900e+00, nfev = 44, time = 6.2s
  E_gap: mean=-3.389511e-04, std=4.004258e-02, max_abs=5.090627e-01
  Relative fit error: 8.8193e-02

--- eval_L3_S3 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 4.928624 S/m
  Loss = 2.849139e+00, nfev = 26, time = 3.5s
  E_gap: mean=-2.659221e-04, std=3.477415e-02, max_abs=2.930562e-01
  Relative fit error: 8.0455e-02

--- eval_L4_S1 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 4.985395 S/m
  Loss = 3.662923e+00, nfev = 24, time = 3.7s
  E_gap: mean=-1.696714e-04, std=3.942959e-02, max_abs=4.560193e-01
  Relative fit error: 8.9833e-02

--- eval_L4_S2 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 5.007468 S/m
  Loss = 3.337533e+00, nfev = 105, time = 13.6s
  E_gap: mean=-2.640569e-04, std=3.763695e-02, max_abs=4.233216e-01
  Relative fit error: 8.5853e-02

--- eval_L4_S3 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.042200 S/m
  sigma_2 (conductive) = 2.031806 S/m
  Loss = 5.439664e+00, nfev = 168, time = 23.1s
  E_gap: mean=-2.374927e-04, std=4.804997e-02, max_abs=6.081516e-01
  Relative fit error: 1.0097e-01

--- eval_L5_S1 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 3.736789 S/m
  Loss = 4.086250e+00, nfev = 84, time = 13.8s
  E_gap: mean=-3.951681e-04, std=4.164427e-02, max_abs=4.330056e-01
  Relative fit error: 9.1498e-02

--- eval_L5_S2 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 4.232301e+00, nfev = 4, time = 0.8s
  E_gap: mean=-3.489195e-04, std=4.238244e-02, max_abs=4.607319e-01
  Relative fit error: 9.2916e-02

--- eval_L5_S3 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 5.256715 S/m
  Loss = 2.796063e+00, nfev = 74, time = 11.4s
  E_gap: mean=-1.118947e-04, std=3.444955e-02, max_abs=2.843611e-01
  Relative fit error: 8.2266e-02

--- eval_L6_S1 ---
  Classes present: [0 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_2 (conductive) = 37.948847 S/m
  Loss = 2.716565e+00, nfev = 42, time = 6.0s
  E_gap: mean=-7.246219e-05, std=3.395638e-02, max_abs=2.721802e-01
  Relative fit error: 8.4557e-02

--- eval_L6_S2 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 14.776972 S/m
  Loss = 4.267676e+00, nfev = 111, time = 15.1s
  E_gap: mean=-1.333157e-04, std=4.256042e-02, max_abs=5.618826e-01
  Relative fit error: 9.6951e-02

--- eval_L6_S3 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  sigma_2 (conductive) = 4.997283 S/m
  Loss = 3.722021e+00, nfev = 48, time = 6.7s
  E_gap: mean=-2.314169e-04, std=3.974609e-02, max_abs=3.749715e-01
  Relative fit error: 8.8386e-02

--- eval_L7_S1 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.008938 S/m
  sigma_2 (conductive) = 16.208105 S/m
  Loss = 5.684137e+00, nfev = 138, time = 18.8s
  E_gap: mean=2.037048e-04, std=4.911803e-02, max_abs=5.171935e-01
  Relative fit error: 1.1688e-01

--- eval_L7_S2 ---
  Classes present: [0 1 2]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.003206 S/m
  sigma_2 (conductive) = 4.974773 S/m
  Loss = 5.840681e+00, nfev = 33, time = 4.3s
  E_gap: mean=-9.510260e-05, std=4.979014e-02, max_abs=6.457815e-01
  Relative fit error: 1.0231e-01

--- eval_L7_S3 ---
  Classes present: [0 1]
  sigma_bg fixed to 0.803581 S/m
  sigma_bg = 0.803581 S/m
  sigma_1 (resistive) = 0.000100 S/m
  Loss = 4.814324e+00, nfev = 4, time = 0.5s
  E_gap: mean=-2.115898e-04, std=4.520383e-02, max_abs=5.704306e-01
  Relative fit error: 9.9060e-02

============================================================
Noise Distribution Analysis — Evaluation Data (21 samples)
============================================================

  Mean bias (systematic): mean=-1.753529e-04, std=3.848188e-02
  Per-sample noise std: ['3.8926e-02', '3.3871e-02', '4.5442e-02', '3.3720e-02', '4.3221e-02', '3.2735e-02', '5.0105e-02', '4.0043e-02', '3.4774e-02', '3.9430e-02', '3.7637e-02', '4.8050e-02', '4.1644e-02', '4.2382e-02', '3.4450e-02', '3.3956e-02', '4.2560e-02', '3.9746e-02', '4.9118e-02', '4.9790e-02', '4.5204e-02']
  Overall noise std: 4.1176e-02
  Sample 1 SNR: 21.0 dB
  Sample 2 SNR: 22.3 dB
  Sample 3 SNR: 19.7 dB
  Sample 4 SNR: 22.3 dB
  Sample 5 SNR: 20.1 dB
  Sample 6 SNR: 22.5 dB
  Sample 7 SNR: 18.9 dB
  Sample 8 SNR: 20.8 dB
  Sample 9 SNR: 22.0 dB
  Sample 10 SNR: 20.9 dB
  Sample 11 SNR: 21.3 dB
  Sample 12 SNR: 19.2 dB
  Sample 13 SNR: 20.5 dB
  Sample 14 SNR: 20.3 dB
  Sample 15 SNR: 22.1 dB
  Sample 16 SNR: 22.2 dB
  Sample 17 SNR: 20.3 dB
  Sample 18 SNR: 20.9 dB
  Sample 19 SNR: 19.0 dB
  Sample 20 SNR: 18.9 dB
  Sample 21 SNR: 19.7 dB

  Per-channel variance: mean=2.1459e-04, max=1.6837e-02, min=7.6257e-07
  Cross-sample correlation: mean=0.887, min=0.646, max=0.991

============================================================
Noise Distribution Analysis — Combined (25 samples) (25 samples)
============================================================

  Mean bias (systematic): mean=-1.779282e-04, std=3.804712e-02
  Per-sample noise std: ['3.7042e-02', '3.7771e-02', '3.3402e-02', '4.4952e-02', '3.8926e-02', '3.3871e-02', '4.5442e-02', '3.3720e-02', '4.3221e-02', '3.2735e-02', '5.0105e-02', '4.0043e-02', '3.4774e-02', '3.9430e-02', '3.7637e-02', '4.8050e-02', '4.1644e-02', '4.2382e-02', '3.4450e-02', '3.3956e-02', '4.2560e-02', '3.9746e-02', '4.9118e-02', '4.9790e-02', '4.5204e-02']
  Overall noise std: 4.0763e-02
  Sample 1 SNR: 21.5 dB
  Sample 2 SNR: 21.3 dB
  Sample 3 SNR: 22.4 dB
  Sample 4 SNR: 19.8 dB
  Sample 5 SNR: 21.0 dB
  Sample 6 SNR: 22.3 dB
  Sample 7 SNR: 19.7 dB
  Sample 8 SNR: 22.3 dB
  Sample 9 SNR: 20.1 dB
  Sample 10 SNR: 22.5 dB
  Sample 11 SNR: 18.9 dB
  Sample 12 SNR: 20.8 dB
  Sample 13 SNR: 22.0 dB
  Sample 14 SNR: 20.9 dB
  Sample 15 SNR: 21.3 dB
  Sample 16 SNR: 19.2 dB
  Sample 17 SNR: 20.5 dB
  Sample 18 SNR: 20.3 dB
  Sample 19 SNR: 22.1 dB
  Sample 20 SNR: 22.2 dB
  Sample 21 SNR: 20.3 dB
  Sample 22 SNR: 20.9 dB
  Sample 23 SNR: 19.0 dB
  Sample 24 SNR: 18.9 dB
  Sample 25 SNR: 19.7 dB

  Per-channel variance: mean=2.1400e-04, max=1.5297e-02, min=8.8776e-07
  Cross-sample correlation: mean=0.886, min=0.639, max=0.996

Results saved to: results\calibration_1\calibration_results.json
CSV saved to: results\calibration_1\calibration_results.csv
E_gap arrays saved to: results\calibration_1/
