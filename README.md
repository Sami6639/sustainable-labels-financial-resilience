# Sustainable Labels and Financial Resilience

## Repository overview

This repository contains the data, code, diagnostics, and publication outputs associated with the study:

**Sustainable Labels and Financial Resilience: Portfolio Architecture under Climate Policy Uncertainty**

The study examines whether financial characteristics embedded in sustainable ETF holdings are persistent and associated with contemporaneous pricing and cumulative return adjustment under climate policy uncertainty.

## Study design

The empirical analysis combines:

- 4,459 monthly observations for 50 U.S.-listed sustainable equity ETFs from 2010 to 2025;
- SEC Form N-PORT holdings for 2023Q4, 2024Q4, and 2025Q4;
- a 2020Q4 boundary-validation snapshot;
- lagged firm fundamentals constructed from SEC Company Facts;
- climate policy uncertainty, market-volatility, and market-factor variables.

The portfolio-architecture framework distinguishes four primary channels:

1. internal financial capacity;
2. external-financing dependence;
3. growth-duration exposure;
4. portfolio concentration.

Two composite measures summarize combined financial and diversification exposure.

## Repository structure

```text
code/
    pipeline/        Data construction and core estimation scripts
    reproduction/    Scripts that reproduce publication tables and figures
    robustness/      Supplementary analyses and diagnostic procedures
    run_all.py       Main reproduction entry point
    validate_package.py

data/
    external/        Instructions and checksums for externally obtained data
    metadata/        Variable and model dictionaries
    processed/       Analysis-ready data files

results/
    core/            Core estimation results
    diagnostics/     Validation outputs
    robustness/      Supplementary results

reproduced/
    figures/         Reproduced publication figures
    tables/          Reproduced publication tables
    results/         Reproduced numerical outputs
    diagnostics/     Reproduction validation files

figures/             Additional analysis figures
logs/                Data-processing and validation logs
docs/                Package documentation
```

## Data availability

SEC Form N-PORT filings and SEC Company Facts are publicly available from the U.S. Securities and Exchange Commission. Large raw SEC archives are not redistributed in this repository because they can be downloaded directly from the original source and exceed standard GitHub file-size limits.

Processed data, metadata, public-data retrieval instructions, checksums, analysis code, diagnostics, and publication outputs are included. Market-data redistribution remains subject to the terms of the original providers.

Additional details are provided in `DATA_AVAILABILITY.md` and `data/external/README.md`.

## Software requirements

The analysis was developed in Python. Required packages are listed in:

`requirements.txt`

A clean environment can be prepared with:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the study

To run the main reproduction workflow from the repository root:

```bash
python code/run_all.py
```

To verify the repository structure and expected outputs:

```bash
python code/validate_package.py
```

Publication tables and figures can also be regenerated directly with:

```bash
python code/reproduction/build_publication_outputs.py
```

The reproduction scripts write outputs to the `reproduced/` directory. Existing files in that directory provide reference outputs for comparison.

## Raw-data preparation

Large SEC N-PORT archives are excluded from version control. Download instructions, expected locations, and integrity information are provided under:

`data/external/`

Raw archives should be stored locally under the paths expected by the pipeline. The excluded raw-data directory is defined in `.gitignore`.

## Expected outputs

The principal reproduced outputs include:

- sample and data-architecture tables;
- portfolio-architecture persistence statistics;
- contemporaneous activation results;
- cumulative-adjustment estimates;
- robustness and falsification summaries;
- publication-ready figures;
- validation reports.

## Reproducibility and integrity

- Fixed random seeds are used where simulation or permutation procedures require them.
- Variable and model dictionaries are provided under `data/metadata/`.
- Validation outputs are included under `results/diagnostics/` and `reproduced/diagnostics/`.
- File-integrity information is recorded in `SHA256SUMS.txt`.
- External raw-archive checksums are documented in `data/external/RAW_ARCHIVE_SHA256.txt`.

## Citation

Citation metadata are provided in `CITATION.cff`.

During peer review, the repository may be accessed through an anonymized read-only mirror. A permanent archival DOI may be assigned after acceptance.

## License

See `LICENSE` for the terms governing the code and repository materials. Third-party source data remain subject to their original terms and licenses.
