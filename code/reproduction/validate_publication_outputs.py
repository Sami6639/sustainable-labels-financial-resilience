"""Validate generated manuscript tables and figures against key reported values."""
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "reproduced" / "tables"
F = ROOT / "reproduced" / "figures"
D = ROOT / "reproduced" / "diagnostics"
D.mkdir(parents=True, exist_ok=True)

checks = {}
required_tables = [
    "Table_1_Sample_and_Data_Architecture.csv",
    "Table_2_Holdings_Based_Portfolio_Architecture_Channels.csv",
    "Table_3_Portfolio_Architecture_Persistence.csv",
    "Table_4_Evidentiary_Hierarchy_Contemporaneous_Activation.csv",
    "Table_5_Persistent_Architecture_Adjustment.csv",
    "Table_6_Robustness_and_Falsification_Summary.csv",
    "Table_A1_Boundary_Validation_2020Q4_2025Q4.csv",
    "Table_A2_Temporally_Aligned_Growth_Duration_2020_2025.csv",
]
required_figures = [
    "Figure_1_Empirical_Data_Architecture.png",
    "Figure_2_Architecture_Persistence.png",
    "Figure_3_Persistent_Architecture_Adjustment.png",
]
for name in required_tables:
    checks[f"exists:{name}"] = (T / name).exists()
for name in required_figures:
    checks[f"exists:{name}"] = (F / name).exists()

# Key numerical checks from the manuscript.
t3 = pd.read_csv(T / required_tables[2])
t5 = pd.read_csv(T / required_tables[4])
ta2 = pd.read_csv(T / required_tables[7])

def number(x):
    return float(str(x).replace("−", "-").replace("<", ""))

def near(actual, target, tol=5e-4):
    return math.isclose(float(actual), float(target), abs_tol=tol, rel_tol=0)

row = t3[t3["Channel"] == "External financing dependence"].iloc[0]
checks["table3_external_financing_spearman_0.931"] = near(row["Spearman ρ"], 0.931)
checks["table3_external_financing_icc_0.961"] = near(row["ICC(3,1)"], 0.961)

for channel, horizon, target in [
    ("Growth-duration exposure", "6 months", -0.0485),
    ("Growth-duration exposure", "12 months", -0.0556),
    ("External financing dependence", "6 months", 0.0800),
]:
    r = t5[(t5["Channel"] == channel) & (t5["Horizon"] == horizon)].iloc[0]
    checks[f"table5:{channel}:{horizon}:{target}"] = near(number(r["Coefficient"]), target)

checks["tableA2_growth_duration_6m_-0.0548"] = near(number(ta2.iloc[0]["Coefficient"]), -0.0548)
checks["tableA2_growth_duration_12m_-0.0633"] = near(number(ta2.iloc[1]["Coefficient"]), -0.0633)

report = {
    "checks": checks,
    "validation_passed": all(checks.values()),
    "important_note": (
        "The alternative-definition expected-sign rates computed from the archived Stage 4 summary "
        "are 0.867 for external financing and 0.933 for concentration. The manuscript currently "
        "reports 0.889 and 0.944; reconcile the manuscript or provide the exact alternative model "
        "selection that yields those reported rates."
    ),
}
(D / "manuscript_output_validation.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
if not report["validation_passed"]:
    raise SystemExit("Generated publication outputs failed validation.")
