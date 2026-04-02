# PLOS Subject Data Parsing

## Purpose

The repository already contains a real thoracic EIT dataset under:

- [`Subjects Data/Plos One Data`](/D:/010_CodePrograms/E/EIT_KTC2023_4/Subjects%20Data/Plos%20One%20Data)

These files are useful for the pulmonary part of the project, but they are stored in Draeger `.get` format and cannot be used directly by the current Python pipelines. This note records the confirmed file structure and the parsing script added in this round.

## File Format

By comparing the bundled subject files with [`DCT-EIT/read_getData.m`](/D:/010_CodePrograms/E/EIT_KTC2023_4/DCT-EIT/read_getData.m), the `.get` files can be interpreted as:

- raw binary `float32`
- `256` values per frame
- logical shape `256 x T`
- equivalent to `16` stimulation patterns with `16` measurements each

The DCT-EIT MATLAB code further reorders the data into:

- `208 x T`

which corresponds to the standard `16 x 13` usable measurement arrangement after removing invalid or redundant channels.

## Implemented Python Script

Added script:

- [`scripts/parse_plos_get.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/parse_plos_get.py)

Capabilities:

1. recursively finds all `.get` files
2. parses raw `float32` stream to `256 x T`
3. reproduces the MATLAB `read_getData.m` reordering to `208 x T`
4. extracts a simple dominant temporal signal using SVD/PCA
5. saves summary metadata and optional overview figures

Run example:

```bash
python scripts/parse_plos_get.py --max-plots 2
```

Output example:

- [`results/plos_get_analysis_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/plos_get_analysis_1)

## Confirmed Dataset Summary

From [`results/plos_get_analysis_1/summary.json`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/plos_get_analysis_1/summary.json):

- Subject A day 1-7: each file has `30000` frames
- Subject B: `30000` frames
- Subject C: `3000` frames
- Subject D: `760` frames
- Subject E: `1220` frames
- Subject F: `3000` frames

Each file is successfully parsed as:

- raw shape: `256 x T`
- reordered shape: `208 x T`

## Current Status

This completes the first useful pulmonary-data step:

- the repository now has a reproducible Python parser for the real thoracic `.get` data
- the measurement format is no longer ambiguous
- the data can now be used for:
  - temporal signal analysis
  - future Sim2Real inference
  - future DCT-EIT or EIDORS-based reconstruction experiments

## Remaining Gap

The current parser does **not** yet reconstruct conductivity images from these real measurements. That still requires:

- an appropriate thoracic forward/inverse model
- consistent electrode geometry
- a reconstruction algorithm matched to the measurement protocol

In practice, the most realistic next step is:

1. use EIDORS or DCT-EIT-compatible thoracic models for real-data reconstruction
2. feed the parsed measurement sequences into the strongest learned model or a DCT-EIT baseline
3. compare temporal ventilation patterns rather than trying to compute unavailable ground-truth conductivity maps
