from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
from q1_common import (load_panel, add_forward_cumulative_returns, channel_sample, fit_fe,
                       design_continuous, design_binary_pair, design_single_shock,
                       PRIMARY_CHANNELS)

OUT=Path('/mnt/data/q1_analysis/style_controls'); FIG=OUT/'figures'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
BROAD=Path('/mnt/data/broad_esg_extracted/broad_esg_data/monthly_simple_returns.csv')
HORIZONS=[0,6,9,12]


def merge_style(df):
    b=pd.read_csv(BROAD,parse_dates=['Date'])[['Date','QQQ','SPY','USO']].rename(columns={'Date':'DATE','QQQ':'QQQ_RETURN','SPY':'SPY_EXTERNAL','USO':'USO_RETURN'})
    b['DATE']=b.DATE.dt.to_period('M').dt.to_timestamp('M'); b['GROWTH_FACTOR']=b.QQQ_RETURN-b.SPY_EXTERNAL
    x=df.merge(b,on='DATE',how='left',validate='many_to_one')
    return x


def add_style_terms(s,arch,level):
    regs=[]
    if level in ['COMMON_STYLE_CONTROLS','ARCHITECTURE_FACTOR_BETAS']:
        regs += ['GROWTH_FACTOR','USO_RETURN']
    if level=='ARCHITECTURE_FACTOR_BETAS':
        for factor in ['MARKET_RETURN','GROWTH_FACTOR','ENERGY_RETURN','TREASURY_RETURN','USO_RETURN']:
            name=f'{factor}_X_ARCH_STYLE'; s[name]=s[factor]*s[arch]; regs.append(name)
    return regs


def base_design(s,arch,test):
    if test=='CPU_VIX_CONTINUOUS': return design_continuous(s,arch)
    if test=='CPU_AND_VIX_STRESS': return design_binary_pair(s,arch,'HIGH_CPU_REGIME','HIGH_VIX_REGIME','JV')
    return design_single_shock(s,arch,'EXTREME_CPU_REGIME','EXTREME')


def main():
    df=merge_style(add_forward_cumulative_returns(load_panel(),HORIZONS))
    diag={'rows':len(df),'qqq_coverage':int(df.QQQ_RETURN.notna().sum()),'uso_coverage':int(df.USO_RETURN.notna().sum()),
          'market_spy_correlation':float(df[['MARKET_RETURN','SPY_EXTERNAL']].drop_duplicates().corr().iloc[0,1])}
    rows=[]
    for ch in PRIMARY_CHANNELS:
        base,arch,_=channel_sample(df,ch)
        for test in ['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']:
            hs=[0] if test=='CPU_AND_VIX_STRESS' else HORIZONS
            for h in hs:
                for level in ['BASELINE','COMMON_STYLE_CONTROLS','ARCHITECTURE_FACTOR_BETAS']:
                    s=base.copy(); regs,focal=base_design(s,arch,test); regs += add_style_terms(s,arch,level)
                    r=fit_fe(s,f'LP_CUM_RETURN_H{h}',regs,focal,covariance='DATE_CLUSTER' if h==0 else 'DK',bandwidth=h)
                    rows.append({'CHANNEL':ch,'TEST':test,'HORIZON_MONTHS':h,'CONTROL_LEVEL':level,'ARCHITECTURE_COLUMN':arch,**r})
    res=pd.DataFrame(rows); res.to_csv(OUT/'34b_all_style_control_results.csv',index=False)
    key=res[(res.TEST.isin(['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS']))&(res.HORIZON_MONTHS==0) |
            (res.TEST.eq('EXTREME_CPU_REGIME')&res.HORIZON_MONTHS.isin([6,9,12])) |
            (res.TEST.eq('CPU_VIX_CONTINUOUS')&res.HORIZON_MONTHS.isin([9,12]))].copy()
    key.to_csv(OUT/'34b_key_style_control_results.csv',index=False)
    # Stability relative to baseline.
    piv=key.pivot_table(index=['CHANNEL','TEST','HORIZON_MONTHS'],columns='CONTROL_LEVEL',values=['COEFFICIENT','P_VALUE']).reset_index()
    piv.columns=['_'.join([str(a),str(b)]).strip('_') for a,b in piv.columns]; piv.to_csv(OUT/'34b_style_control_comparison.csv',index=False)
    labels={'INTERNAL_FINANCIAL_CAPACITY':'Internal capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing','GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration','PORTFOLIO_CONCENTRATION_FINAL':'Concentration'}
    z=res[(res.TEST=='CPU_VIX_CONTINUOUS')&(res.HORIZON_MONTHS==0)].copy(); levels=['BASELINE','COMMON_STYLE_CONTROLS','ARCHITECTURE_FACTOR_BETAS']
    fig,ax=plt.subplots(figsize=(9,5.2)); x=np.arange(len(PRIMARY_CHANNELS)); width=.24
    for j,lev in enumerate(levels):
        g=z[z.CONTROL_LEVEL==lev].set_index('CHANNEL').reindex(PRIMARY_CHANNELS); ax.bar(x+(j-1)*width,100*g.COEFFICIENT,width,label=lev.replace('_',' ').title())
    ax.axhline(0,linewidth=1); ax.set_xticks(x,[labels[c] for c in PRIMARY_CHANNELS],rotation=20,ha='right'); ax.set_ylabel('CPU × VIX × architecture coefficient (percentage points)'); ax.set_title('Activation after growth-style and architecture-beta controls'); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIG/'34b_style_adjusted_continuous_activation.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/'34b_style_adjusted_continuous_activation.pdf',bbox_inches='tight'); plt.close(fig)
    (OUT/'34b_merge_validation.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
    meta={'additional_market_data':'monthly QQQ, SPY and USO returns from uploaded broad ESG archive','growth_factor':'QQQ minus SPY','architecture_beta_controls':['market x architecture','growth factor x architecture','energy x architecture','Treasury x architecture','oil x architecture'],
          'purpose':'test whether holdings-based architecture activation is subsumed by general growth, market, energy, duration or oil factor exposures'}
    (OUT/'34b_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nKEY STYLE ROBUSTNESS\n',key[['CHANNEL','TEST','HORIZON_MONTHS','CONTROL_LEVEL','COEFFICIENT','P_VALUE','N_ETFS','N_DATES','CONDITION_NUMBER']].to_string(index=False))
if __name__=='__main__': main()
