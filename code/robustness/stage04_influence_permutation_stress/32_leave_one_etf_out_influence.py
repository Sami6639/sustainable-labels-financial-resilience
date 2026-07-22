from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
from q1_common import (load_panel, add_forward_cumulative_returns, channel_sample, fit_fe,
                       design_continuous, design_binary_pair, design_single_shock,
                       PRIMARY_CHANNELS)

OUT=Path('/mnt/data/q1_analysis/influence_diagnostics'); FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
HORIZONS=[0,3,6,9,12]
SHOCKS=['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']
EXPECTED={'INTERNAL_FINANCIAL_CAPACITY':1,'EXTERNAL_FINANCING_DEPENDENCE':-1,
          'GROWTH_DURATION_EXPOSURE_FINAL':-1,'PORTFOLIO_CONCENTRATION_FINAL':-1}


def build_design(s,arch,shock):
    if shock=='CPU_VIX_CONTINUOUS': return design_continuous(s,arch)
    if shock=='CPU_AND_VIX_STRESS': return design_binary_pair(s,arch,'HIGH_CPU_REGIME','HIGH_VIX_REGIME','JV')
    return design_single_shock(s,arch,'EXTREME_CPU_REGIME','EXTREME')


def estimate(s,arch,shock,h):
    z=s.copy(); regs,focal=build_design(z,arch,shock)
    return fit_fe(z,f'LP_CUM_RETURN_H{h}',regs,focal,month_fe=False,
                  covariance='DATE_CLUSTER' if h==0 else 'DK',bandwidth=h,min_etfs=2)


def main():
    df=load_panel(); df=add_forward_cumulative_returns(df,HORIZONS)
    full_rows=[]; loo_rows=[]
    total=len(PRIMARY_CHANNELS)*len(SHOCKS)*len(HORIZONS); done=0
    for ch in PRIMARY_CHANNELS:
        base,arch,flag=channel_sample(df,ch)
        fund_map=base[['ETF_ID','ETF_TICKER']].drop_duplicates().set_index('ETF_ID').ETF_TICKER.to_dict()
        for shock in SHOCKS:
            for h in HORIZONS:
                done+=1; print(f'[{done}/{total}] {ch} | {shock} | h={h}')
                full=estimate(base,arch,shock,h)
                ident={'CHANNEL':ch,'SHOCK':shock,'HORIZON_MONTHS':h,'ARCHITECTURE_COLUMN':arch}
                full_rows.append({**ident,**full})
                for etf in sorted(base.ETF_ID.dropna().unique()):
                    r=estimate(base[base.ETF_ID.ne(etf)],arch,shock,h)
                    loo_rows.append({**ident,'OMITTED_ETF_ID':etf,'OMITTED_ETF_TICKER':fund_map.get(etf,''),**r})
    full_df=pd.DataFrame(full_rows); loo=pd.DataFrame(loo_rows)
    full_df.to_csv(OUT/'32_full_sample_reference_models.csv',index=False)
    loo.to_csv(OUT/'32_leave_one_etf_out_all_results.csv',index=False)

    summaries=[]
    for key,g in loo.groupby(['CHANNEL','SHOCK','HORIZON_MONTHS']):
        ch,shock,h=key; f=full_df[(full_df.CHANNEL==ch)&(full_df.SHOCK==shock)&(full_df.HORIZON_MONTHS==h)].iloc[0]
        ok=g[g.MODEL_STATUS.eq('RELIABLE_ESTIMATE')&g.COEFFICIENT.notna()].copy()
        if ok.empty: continue
        fullcoef=float(f.COEFFICIENT); fullse=float(f.STANDARD_ERROR) if np.isfinite(f.STANDARD_ERROR) else np.nan
        ok['ABS_CHANGE']=abs(ok.COEFFICIENT-fullcoef)
        inf=ok.loc[ok.ABS_CHANGE.idxmax()]
        thetas=ok.COEFFICIENT.to_numpy(); n=len(thetas); mean=thetas.mean()
        jack_se=np.sqrt((n-1)/n*np.sum((thetas-mean)**2)) if n>1 else np.nan
        summaries.append({
            'CHANNEL':ch,'SHOCK':shock,'HORIZON_MONTHS':h,'FULL_COEFFICIENT':fullcoef,
            'FULL_STANDARD_ERROR':fullse,'FULL_P_VALUE':f.P_VALUE,'FULL_N_ETFS':f.N_ETFS,
            'N_RELIABLE_LOO_MODELS':n,'LOO_COEF_MIN':thetas.min(),'LOO_COEF_MAX':thetas.max(),
            'LOO_COEF_MEDIAN':np.median(thetas),'LOO_COEF_STD':thetas.std(ddof=1) if n>1 else np.nan,
            'SIGN_PRESERVATION_RATE':np.mean(np.sign(thetas)==np.sign(fullcoef)) if fullcoef!=0 else np.nan,
            'EXPECTED_SIGN_PRESERVATION_RATE':np.mean(np.sign(thetas)==EXPECTED[ch]),
            'P_LT_05_PRESERVATION_RATE':np.mean(ok.P_VALUE<.05),'P_LT_10_PRESERVATION_RATE':np.mean(ok.P_VALUE<.10),
            'MAX_ABSOLUTE_CHANGE':inf.ABS_CHANGE,
            'MAX_CHANGE_IN_FULL_SE_UNITS':inf.ABS_CHANGE/fullse if np.isfinite(fullse) and fullse>0 else np.nan,
            'MOST_INFLUENTIAL_ETF_ID':inf.OMITTED_ETF_ID,'MOST_INFLUENTIAL_ETF_TICKER':inf.OMITTED_ETF_TICKER,
            'COEFFICIENT_WITH_MOST_INFLUENTIAL_ETF_OMITTED':inf.COEFFICIENT,
            'P_VALUE_WITH_MOST_INFLUENTIAL_ETF_OMITTED':inf.P_VALUE,
            'JACKKNIFE_MEAN':mean,'JACKKNIFE_STANDARD_ERROR':jack_se,
            'SIGN_REVERSAL_COUNT':int(np.sum(np.sign(thetas)!=np.sign(fullcoef))) if fullcoef!=0 else np.nan,
            'INFLUENCE_ASSESSMENT':('ROBUST_NO_SIGN_REVERSAL' if np.all(np.sign(thetas)==np.sign(fullcoef)) and (inf.ABS_CHANGE/fullse if np.isfinite(fullse) and fullse>0 else 999)<2 else
                                    'SIGN_STABLE_BUT_INFLUENTIAL_FUNDS' if np.all(np.sign(thetas)==np.sign(fullcoef)) else 'SIGN_SENSITIVE'),
        })
    summ=pd.DataFrame(summaries)
    summ.to_csv(OUT/'32_leave_one_etf_out_summary.csv',index=False)
    key=summ[((summ.HORIZON_MONTHS==0)&summ.SHOCK.isin(['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS'])) |
             ((summ.HORIZON_MONTHS.isin([6,9,12]))&summ.SHOCK.isin(['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']))]
    key.to_csv(OUT/'32_key_influence_diagnostics.csv',index=False)

    # Range plots at impact for the two primary RQ3 specifications.
    labels={'INTERNAL_FINANCIAL_CAPACITY':'Internal capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing',
            'GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration','PORTFOLIO_CONCENTRATION_FINAL':'Concentration'}
    for shock,title in [('CPU_VIX_CONTINUOUS','Continuous CPU × VIX activation'),('CPU_AND_VIX_STRESS','Joint CPU–VIX stress')]:
        z=summ[(summ.SHOCK==shock)&(summ.HORIZON_MONTHS==0)].copy()
        z['LABEL']=z.CHANNEL.map(labels); z=z.sort_values('FULL_COEFFICIENT')
        fig,ax=plt.subplots(figsize=(8,4.8)); y=np.arange(len(z))
        ax.hlines(y,100*z.LOO_COEF_MIN,100*z.LOO_COEF_MAX,linewidth=3,alpha=.55)
        ax.scatter(100*z.FULL_COEFFICIENT,y,zorder=3,label='Full sample')
        ax.axvline(0,linewidth=1); ax.set_yticks(y,z.LABEL); ax.set_xlabel('Interaction coefficient (percentage points)')
        ax.set_title(f'Leave-one-ETF-out coefficient ranges: {title}'); ax.grid(axis='x',alpha=.25); ax.legend(frameon=False)
        fig.tight_layout(); name=shock.lower(); fig.savefig(FIG/f'32_loo_ranges_{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'32_loo_ranges_{name}.pdf',bbox_inches='tight'); plt.close(fig)

    meta={'source':'25_final_econometric_panel.csv','channels':PRIMARY_CHANNELS,'shocks':SHOCKS,'horizons':HORIZONS,
          'model':'RQ3-aligned ETF fixed effects; date-clustered at h=0 and Driscoll-Kraay bandwidth h at h>0',
          'purpose':'diagnose whether individual ETFs drive impact and recovery estimates'}
    (OUT/'32_influence_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nKEY SUMMARY\n',key[['CHANNEL','SHOCK','HORIZON_MONTHS','FULL_COEFFICIENT','FULL_P_VALUE','SIGN_PRESERVATION_RATE','P_LT_10_PRESERVATION_RATE','MOST_INFLUENTIAL_ETF_TICKER','MAX_CHANGE_IN_FULL_SE_UNITS','INFLUENCE_ASSESSMENT']].to_string(index=False))

if __name__=='__main__': main()
