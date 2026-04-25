# External Pulmonary EIT Dataset Candidates

## Purpose

This note records external pulmonary EIT data resources that may be useful if the
current local `32`-electrode KTC-style setup needs to be extended toward
`16`-electrode thoracic EIT or toward more realistic pulmonary supervision.

## Candidate 1: Dryad Thorax Simulation Dataset

Dataset page:

- <https://datadryad.org/dataset/doi:10.5061/dryad.47d7wm3c3>

Associated paper:

- <https://doi.org/10.1371/journal.pone.0246071>

Relevant properties:

- public download available
- thorax-focused simulation data
- includes lung and heart related EIT image separation study material
- likely much closer to pulmonary EIT than the current benchmark phantom task

Current limitation:

- the Dryad page does not directly document the exact tensor format in a way that
  can be integrated immediately without downloading and inspecting the files
- the dataset appears to be simulation-oriented rather than paired real thoracic
  measurements with exact conductivity GT

## Candidate 2: 16-Electrode Thorax Simulation Study

Open-access article:

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC11442128/>

Relevant statement from the article:

- the study uses a `16`-electrode EIT model
- each simulated case contains `208` boundary voltage values
- each corresponding conductivity distribution is represented on a mesh with
  `2707` elements

Why it matters:

- this matches the same `16`-electrode / `208`-channel regime as the parsed
  PLOS thoracic `.get` data in this repository
- it is therefore a strong candidate reference when building a new pulmonary
  simulation pipeline aligned to real thoracic measurements

Current limitation:

- the article itself is accessible, but a directly downloadable official dataset
  link was not yet confirmed in the current round

## Candidate 3: Existing PLOS One Subject Data In This Repository

Current local data:

- [`Subjects Data/Plos One Data`](/D:/010_CodePrograms/E/EIT_KTC2023_4/Subjects%20Data/Plos%20One%20Data)

Status:

- confirmed to be real thoracic EIT measurement data
- parsed to `256 x T` raw frames and `208 x T` reordered measurements
- does **not** contain true conductivity image labels

Conclusion:

- suitable for future Sim2Real qualitative validation
- not suitable for direct supervised training

## Current Decision

Given the present repository state, the most practical next external-data route is:

1. keep the local PLOS `.get` files for future real-data validation
2. if a `16`-electrode pulmonary chain is needed, first align simulation format
   to the `208`-channel regime
3. inspect the Dryad thorax dataset only when the project is ready to commit to
   a `16`-electrode pulmonary branch

At the moment, the local synthetic lung dataset remains the fastest path for
controlled supervised experiments.
