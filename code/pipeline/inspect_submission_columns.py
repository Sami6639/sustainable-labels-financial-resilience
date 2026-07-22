from pathlib import Path
import pandas as pd

file_path = Path(
    r"C:\Users\User\Desktop\CPU_Project\sec_raw\2025Q4\extracted\SUBMISSION.tsv"
)

df = pd.read_csv(
    file_path,
    sep="\t",
    dtype=str,
    nrows=5,
    low_memory=False
)

print(df.columns.tolist())
print()
print(df.head().to_string(index=False))