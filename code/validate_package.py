from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data' / 'processed'
R = ROOT / 'results' / 'diagnostics'; R.mkdir(parents=True, exist_ok=True)
panel = pd.read_csv(D/'etf_monthly_architecture_panel.csv', low_memory=False)
firm = pd.read_csv(D/'firm_characteristics_annual.csv', low_memory=False)
pers = pd.read_csv(D/'persistent_architecture_scores.csv', low_memory=False)
hist = pd.read_csv(D/'architecture_scores_by_snapshot.csv', low_memory=False)
def pick(df, candidates):
    m={str(c).lower():c for c in df.columns}
    for x in candidates:
        if x.lower() in m: return m[x.lower()]
    return None
date_col=pick(panel,['date','month','month_end'])
etf_col=pick(panel,['ticker','etf_ticker','series_id','etf_id'])
firm_col=pick(firm,['cik','cik10','firm_id','issuer_cik'])
year_col=pick(firm,['fiscal_year','year','fy'])
snap_col=pick(hist,['snapshot','snapshot_quarter','quarter','reporting_quarter'])
pers_etf=pick(pers,['ticker','etf_ticker','series_id','etf_id'])
report={
 'panel_rows':len(panel), 'panel_columns':panel.shape[1],
 'unique_etfs': int(panel[etf_col].nunique()) if etf_col else None,
 'unique_months': int(pd.to_datetime(panel[date_col]).nunique()) if date_col else None,
 'date_min': str(pd.to_datetime(panel[date_col]).min().date()) if date_col else None,
 'date_max': str(pd.to_datetime(panel[date_col]).max().date()) if date_col else None,
 'duplicate_etf_month_rows': int(panel.duplicated([etf_col,date_col]).sum()) if etf_col and date_col else None,
 'firm_year_rows':len(firm),
 'unique_firms': int(firm[firm_col].nunique()) if firm_col else None,
 'duplicate_firm_year_rows': int(firm.duplicated([firm_col,year_col]).sum()) if firm_col and year_col else None,
 'historical_architecture_rows':len(hist),
 'historical_snapshots': sorted(map(str,hist[snap_col].dropna().unique())) if snap_col else None,
 'persistent_score_rows':len(pers),
 'persistent_unique_etfs': int(pers[pers_etf].nunique()) if pers_etf else None,
 'complete_three_snapshot_etfs': int(pers['AVAILABLE_ALL_THREE_QUARTERS'].fillna(False).astype(bool).sum()) if 'AVAILABLE_ALL_THREE_QUARTERS' in pers.columns else None,
}
expected={'panel_rows':4459,'unique_etfs':50,'unique_months':192,'firm_year_rows':27312,'unique_firms':2174,'duplicate_etf_month_rows':0,'complete_three_snapshot_etfs':56}
checks={k:report.get(k)==v for k,v in expected.items()}
report['expected_checks']=checks
report['validation_passed']=all(checks.values())
(R/'package_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
if not report['validation_passed']: raise SystemExit('Expected sample checks failed')
