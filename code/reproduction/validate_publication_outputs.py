
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "reproduced" / "tables"
F = ROOT / "reproduced" / "figures"
D = ROOT / "reproduced" / "diagnostics"
D.mkdir(parents=True, exist_ok=True)

def read_csv(name):
    path = T / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected table: {path}")
    return pd.read_csv(path)

def norm(x):
    return " ".join(str(x).replace("\ufeff", "").strip().lower().split())

checks = {}

# Expected output files
expected_tables = [
    "Table_1_Sample_and_Data_Architecture.csv",
    "Table_2_Holdings_Based_Portfolio_Architecture_Channels.csv",
    "Table_3_Portfolio_Architecture_Persistence.csv",
    "Table_4_Evidentiary_Hierarchy_Contemporaneous_Activation.csv",
    "Table_5_Persistent_Architecture_Adjustment.csv",
    "Table_6_Robustness_and_Falsification_Summary.csv",
    "Table_A1_Boundary_Validation_2020Q4_2025Q4.csv",
    "Table_A2_Temporally_Aligned_Growth_Duration_2020_2025.csv",
]
expected_figures = [
    "Figure_1_Empirical_Data_Architecture.png",
    "Figure_2_Architecture_Persistence.png",
    "Figure_3_Persistent_Architecture_Adjustment.png",
]

for name in expected_tables:
    checks[f"file:{name}"] = (T / name).exists()
for name in expected_figures:
    checks[f"file:{name}"] = (F / name).exists()

# Table 1
t1 = read_csv(expected_tables[0])
layer_col = next((c for c in t1.columns if norm(c) == "layer"), t1.columns[0])
coverage_col = next((c for c in t1.columns if norm(c) == "coverage"), None)
if coverage_col is None:
    raise KeyError("Table 1 does not contain a Coverage column.")

labels = t1[layer_col].map(norm)
match_mask = labels.str.contains("n-port match", regex=False)
if not match_mask.any():
    raise ValueError(
        "Table 1 does not contain an N-PORT match row. "
        f"Available labels: {t1[layer_col].tolist()}"
    )
match_coverage = norm(t1.loc[match_mask, coverage_col].iloc[0])
checks["table1_exact_matches_60_67_72"] = all(x in match_coverage for x in ["60", "67", "72"])

complete_mask = labels.str.contains("complete", regex=False) & labels.str.contains("architecture", regex=False)
if not complete_mask.any():
    raise ValueError(
        "Table 1 does not contain the complete architecture panel row. "
        f"Available labels: {t1[layer_col].tolist()}"
    )
complete_coverage = norm(t1.loc[complete_mask, coverage_col].iloc[0])
checks["table1_complete_panel_56"] = "56" in complete_coverage

# Table 3
t3 = read_csv(expected_tables[2])
channel_col = next(c for c in t3.columns if norm(c) == "channel")
spearman_col = next(c for c in t3.columns if "spearman" in norm(c))
icc_col = next(c for c in t3.columns if "icc" in norm(c))
channels = t3[channel_col].map(norm)

ext_row = t3.loc[channels.str.contains("external financing", regex=False)].iloc[0]
checks["table3_external_financing_spearman_0_931"] = math.isclose(float(ext_row[spearman_col]), 0.931, abs_tol=0.0005)
checks["table3_external_financing_icc_0_961"] = math.isclose(float(ext_row[icc_col]), 0.961, abs_tol=0.0005)
checks["table3_clean_financial_risk_label"] = channels.str.fullmatch("financial architecture risk").any()
checks["table3_clean_extended_risk_label"] = channels.str.fullmatch("extended architecture risk").any()

# Table 4
t4 = read_csv(expected_tables[3])
design_col = next(c for c in t4.columns if "channel" in norm(c) or "design" in norm(c))
estimate_col = next(c for c in t4.columns if "focal estimate" in norm(c))
inference_col = next(c for c in t4.columns if "inference" in norm(c))
designs = t4[design_col].map(norm)

ext4 = t4.loc[designs.str.contains("external financing", regex=False)].iloc[0]
gd4 = t4.loc[designs.str.contains("growth-duration", regex=False)].iloc[0]
checks["table4_external_estimate_-0_00654"] = norm(ext4[estimate_col]) in {"-0.00654", "−0.00654"}
checks["table4_external_p_0_0045"] = "0.0045" in str(ext4[inference_col])
checks["table4_growth_estimate_-0_01288"] = norm(gd4[estimate_col]) in {"-0.01288", "−0.01288"}
checks["table4_growth_p_0_0589"] = "0.0589" in str(gd4[inference_col])

# Table 5
t5 = read_csv(expected_tables[4])
ch5 = t5[next(c for c in t5.columns if norm(c) == "channel")].map(norm)
h5 = t5[next(c for c in t5.columns if norm(c) == "horizon")].map(norm)
coef5_col = next(c for c in t5.columns if norm(c) == "coefficient")

def coef_for(channel_text, horizon_text):
    row = t5.loc[ch5.str.contains(channel_text, regex=False) & h5.str.contains(horizon_text, regex=False)].iloc[0]
    return float(row[coef5_col])

checks["table5_growth_6m_-0_0485"] = math.isclose(coef_for("growth-duration", "6"), -0.0485, abs_tol=0.00005)
checks["table5_growth_12m_-0_0556"] = math.isclose(coef_for("growth-duration", "12"), -0.0556, abs_tol=0.00005)
checks["table5_external_6m_0_0800"] = math.isclose(coef_for("external financing", "6"), 0.0800, abs_tol=0.00005)

# Table 6
t6 = read_csv(expected_tables[5])
test_col = next(c for c in t6.columns if norm(c) == "test")
result_col = next(c for c in t6.columns if norm(c) == "result")
tests = t6[test_col].map(norm)
alt = t6.loc[tests.str.contains("alternative definitions", regex=False), result_col].iloc[0]
checks["table6_expected_sign_rates"] = ("0.867" in str(alt) and "0.933" in str(alt))

# Table A2
ta2 = read_csv(expected_tables[7])
hcol = next(c for c in ta2.columns if norm(c) == "horizon")
ccol = next(c for c in ta2.columns if norm(c) == "coefficient")
hvals = ta2[hcol].map(norm)
checks["tableA2_6m_-0_0548"] = math.isclose(float(ta2.loc[hvals.str.contains("6", regex=False), ccol].iloc[0]), -0.0548, abs_tol=0.00005)
checks["tableA2_12m_-0_0633"] = math.isclose(float(ta2.loc[hvals.str.contains("12", regex=False), ccol].iloc[0]), -0.0633, abs_tol=0.00005)

report = {
    "checks": checks,
    "validation_passed": all(checks.values()),
}
(D / "manuscript_output_validation.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)

print(json.dumps(report, indent=2, ensure_ascii=False))
if not report["validation_passed"]:
    failed = [k for k, v in checks.items() if not v]
    raise SystemExit("Publication-output validation failed: " + ", ".join(failed))

print("All 8 manuscript tables and all 3 manuscript figures validated successfully.")
