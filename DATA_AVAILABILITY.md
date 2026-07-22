# Data Availability

This repository includes the analysis-ready data required to reproduce and verify the reported sample construction, holdings-based portfolio-architecture measures, persistence analyses, contemporaneous pricing models, stress-activation estimates, and cumulative return-adjustment results.

The complete raw quarterly SEC Form N-PORT archives are not redistributed because of repository-size constraints. These archives are publicly available from the U.S. Securities and Exchange Commission. The checksums of the locally used 2023Q4, 2024Q4, and 2025Q4 archives are recorded in `data/external/RAW_ARCHIVE_SHA256.txt`. Cleaned holdings extracts for the relevant snapshots are included in Parquet format.

Annual firm-level characteristics derived from SEC Company Facts are included as analysis-ready Parquet files. The original raw API responses are not redistributed because they can be retrieved directly from the SEC and would substantially increase the size of the repository.

Climate policy uncertainty variables, market factors, volatility measures, and other externally sourced market data must be used in accordance with the terms and redistribution policies of their original providers. Where redistribution is restricted, the repository provides the processed variables required for reproduction together with source and construction documentation.

Instructions for obtaining externally sourced data, expected file locations, and integrity checks are provided in `data/external/README.md`.

No API keys, passwords, private identifiers, proprietary credentials, or local user information are included in the repository.
