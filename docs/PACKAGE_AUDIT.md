# Package audit

This release consolidates the original CPU_Project archive, the Stage 2–5 Q1 analysis packages, the full `output`, `scripts`, and `logs` archives, and the three supplied N-PORT quarterly archives.

## Included and verified

- Final monthly econometric panel
- Annual firm-characteristics panel
- Cleaned holdings extracts for 2023Q4, 2024Q4, and 2025Q4
- Historical architecture channels and fixed standardization parameters
- Persistent architecture scores
- Persistence statistics, including ICC, rank correlations, and transitions
- Final persistent pricing and recovery results
- Local-projection, Fama–MacBeth, sorted-portfolio, leave-one-fund, assignment-permutation, alternative-stress, placebo, downside, quantile, and evidence-synthesis materials
- Original extraction and model scripts

## Deliberate exclusions

- Full raw N-PORT archives, due to size
- Raw SEC Company Facts responses
- virtual environments, caches, credentials, and temporary files
- large intermediate CSV files when an equivalent compact Parquet or final analysis file is included

## Reproducibility scope

The default portable workflow verifies the included analysis-ready package. Full raw-data reconstruction requires downloading the public SEC archives and adapting historical local path constants in the original pipeline scripts.
