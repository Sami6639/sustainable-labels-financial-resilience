from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'processed'/'architecture_scores_by_snapshot.csv'
OUT=ROOT/'reproduced'/'results'; FIG=ROOT/'reproduced'/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
channels=[
'INTERNAL_FINANCIAL_CAPACITY','EXTERNAL_FINANCING_DEPENDENCE',
'GROWTH_DURATION_EXPOSURE_FINAL','PORTFOLIO_CONCENTRATION_FINAL',
'FINANCIAL_ARCHITECTURE_RISK_FINAL','EXTENDED_ARCHITECTURE_RISK_FINAL']
labels={
'INTERNAL_FINANCIAL_CAPACITY':'Internal financial capacity',
'EXTERNAL_FINANCING_DEPENDENCE':'External financing dependence',
'GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration exposure',
'PORTFOLIO_CONCENTRATION_FINAL':'Portfolio concentration',
'FINANCIAL_ARCHITECTURE_RISK_FINAL':'Financial architecture risk',
'EXTENDED_ARCHITECTURE_RISK_FINAL':'Extended architecture risk'}

def icc31(wide):
    a=wide.dropna().to_numpy(float); n,k=a.shape
    grand=a.mean(); row=a.mean(1); col=a.mean(0)
    ssr=k*np.square(row-grand).sum(); ssc=n*np.square(col-grand).sum()
    sse=np.square(a-row[:,None]-col[None,:]+grand).sum()
    msr=ssr/(n-1); mse=sse/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse),n
x=pd.read_csv(DATA,low_memory=False)
q='SNAPSHOT_QUARTER'; idc='ETF_ID'
pair=[]; iccs=[]
for c in channels:
    w=x.pivot_table(index=idc,columns=q,values=c,aggfunc='first')
    for a,b in [('2023Q4','2024Q4'),('2024Q4','2025Q4'),('2023Q4','2025Q4')]:
        z=w[[a,b]].dropna()
        pr=stats.pearsonr(z[a],z[b]); sr=stats.spearmanr(z[a],z[b]); kt=stats.kendalltau(z[a],z[b])
        pair.append({'CHANNEL':c,'QUARTER_1':a,'QUARTER_2':b,'N_ETFS':len(z),'PEARSON_R':pr.statistic,'PEARSON_P':pr.pvalue,'SPEARMAN_RHO':sr.statistic,'SPEARMAN_P':sr.pvalue,'KENDALL_TAU':kt.statistic,'KENDALL_P':kt.pvalue})
    val,n=icc31(w[['2023Q4','2024Q4','2025Q4']])
    interp='Excellent' if val>=.90 else 'Good' if val>=.75 else 'Moderate' if val>=.50 else 'Poor'
    iccs.append({'CHANNEL':c,'N_ETFS':n,'N_QUARTERS':3,'ICC_3_1':val,'INTERPRETATION':interp})
pair=pd.DataFrame(pair); iccs=pd.DataFrame(iccs)
pair.to_csv(OUT/'29_architecture_pairwise_correlations.csv',index=False)
iccs.to_csv(OUT/'29_architecture_icc.csv',index=False)
long=pair.query("QUARTER_1=='2023Q4' and QUARTER_2=='2025Q4'").merge(iccs,on='CHANNEL')
long.to_csv(OUT/'Table_3_Portfolio_Architecture_Persistence.csv',index=False)
plt.figure(figsize=(10,5.5)); plt.bar([labels[c] for c in iccs.CHANNEL],iccs.ICC_3_1); plt.axhline(.75,ls='--'); plt.axhline(.90,ls='--'); plt.ylim(0,1.02); plt.ylabel('ICC(3,1)'); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(FIG/'Figure_2_Architecture_Persistence.png',dpi=300); plt.close()
print('Persistence tables and Figure 2 reproduced.')
