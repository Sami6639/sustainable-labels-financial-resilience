from pathlib import Path
import pandas as pd

file_path = Path(
    r"C:\Users\User\Desktop\CPU_Project\sec_raw\2025Q4\extracted\FUND_REPORTED_INFO.tsv"
)

df = pd.read_csv(
    file_path,
    sep="\t",
    dtype=str,
    low_memory=False
)

result = df[
    df["SERIES_NAME"].str.contains(
        "ALPS Clean Energy ETF",
        case=False,
        na=False
    )
]

print(result.T)