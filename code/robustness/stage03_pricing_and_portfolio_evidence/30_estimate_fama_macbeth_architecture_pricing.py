from __future__ import annotations
from pathlib import Path
import json, math, warnings, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
SOURCE=Path('/mnt/data/cpu_extracted/CPU_Project/output/25_final_econometric_panel.csv')
OUT=Path('/mnt/data/q1_analysis/fama_macbeth'); FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
CONTROLS=['MARKET_RETURN','ENERGY_RETURN','TREASURY_RETURN','VIX_CHANGE']
CHANNELS={
 'INTERNAL_FINANCIAL_CAPACITY':'PRIMARY',
 'EXTERNAL_FINANCING_DEPENDENCE':'PRIMARY',
 'GROWTH_DURATION_EXPOSURE_FINAL':'PRIMARY',
 'PORTFOLIO_CONCENTRATION_FINAL':'PRIMARY',
 'FINANCIAL_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS',
 'EXTENDED_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS',
 'PCA_COMPONENT_1':'PCA_ROBUSTNESS'}
MIN_CROSS_SECTION=5; MAIN_MIN_ETFS=10; HAC_LAGS=3


def qcol(ch):
    if ch=='PCA_COMPONENT_1': return ch,'VALID_PCA_MAIN_ROW'
    return f'{ch}_MAIN',f'VALID_{ch}_MAIN_ROW'

def reqcols():
    s={'DATE','DATE_GROUP','ETF_ID','ETF_TICKER','ETF_RETURN','CPU_Z','VIX_LEVEL_Z','HIGH_CPU_REGIME','HIGH_VIX_REGIME',
       'CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME',*CONTROLS}
    for ch in CHANNELS:
        a,f=qcol(ch); s.add(a); s.add(f)
    return sorted(s)

def factor_residuals(df):
    df=df.copy(); df['FACTOR_ADJUSTED_RETURN']=np.nan
    rows=[]
    for etf,g in df.groupby('ETF_ID'):
        s=g[['ETF_RETURN',*CONTROLS]].dropna()
        if len(s)<24: rows.append({'ETF_ID':etf,'N':len(s),'STATUS':'INSUFFICIENT'}); continue
        X=sm.add_constant(s[CONTROLS],has_constant='add'); fit=sm.OLS(s.ETF_RETURN,X).fit()
        df.loc[s.index,'FACTOR_ADJUSTED_RETURN']=fit.resid
        rows.append({'ETF_ID':etf,'N':len(s),'STATUS':'ESTIMATED','R_SQUARED':fit.rsquared,
                     **{f'BETA_{c}':fit.params.get(c,np.nan) for c in CONTROLS}})
    return df,pd.DataFrame(rows)

def sample_etfs(df,ch):
    a,f=qcol(ch); m=df[a].notna() & df[f].eq(1)
    return sorted(df.loc[m,'ETF_ID'].dropna().unique()),a,f

def balanced_window(df,etfs):
    sub=df[df.ETF_ID.isin(etfs)&df.ETF_RETURN.notna()]
    first=sub.groupby('ETF_ID').DATE.min(); last=sub.groupby('ETF_ID').DATE.max()
    if len(first)<len(etfs): return pd.NaT,pd.NaT
    return first.max(),last.min()

def monthly_lambdas(df,ch,role,outcome,sample_design):
    etfs,a,f=sample_etfs(df,ch); s=df[df.ETF_ID.isin(etfs)&df[f].eq(1)&df[a].notna()].copy()
    start=end=pd.NaT
    if sample_design=='BALANCED_COMMON_WINDOW':
        start,end=balanced_window(s,etfs); s=s[(s.DATE>=start)&(s.DATE<=end)]
    rows=[]
    for date,g in s.groupby('DATE'):
        z=g[[outcome,a,'CPU_Z','VIX_LEVEL_Z','HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME',*CONTROLS]].dropna(subset=[outcome,a])
        n=len(z); n_total=len(etfs)
        if sample_design=='BALANCED_COMMON_WINDOW' and n!=n_total: continue
        if sample_design=='MIN10_CROSS_SECTION' and n<MAIN_MIN_ETFS: continue
        if n<MIN_CROSS_SECTION or z[a].nunique()<2: continue
        X=sm.add_constant(z[[a]],has_constant='add'); fit=sm.OLS(z[outcome],X).fit()
        common=g.iloc[0]
        rows.append({'DATE':date,'CHANNEL':ch,'CHANNEL_ROLE':role,'OUTCOME':outcome,'SAMPLE_DESIGN':sample_design,
                     'LAMBDA_ARCHITECTURE':fit.params[a],'LAMBDA_SE_CROSS_SECTION':fit.bse[a],
                     'CROSS_SECTION_R2':fit.rsquared,'N_ETFS_MONTH':n,'N_ETFS_UNIVERSE':n_total,
                     'BALANCED_START':start,'BALANCED_END':end,
                     'CPU_Z':common.CPU_Z,'VIX_LEVEL_Z':common.VIX_LEVEL_Z,
                     'CPU_Z_X_VIX_Z':common.CPU_Z*common.VIX_LEVEL_Z,
                     'HIGH_CPU_REGIME':common.HIGH_CPU_REGIME,'HIGH_VIX_REGIME':common.HIGH_VIX_REGIME,
                     'CPU_AND_VIX_STRESS':common.CPU_AND_VIX_STRESS,'EXTREME_CPU_REGIME':common.EXTREME_CPU_REGIME,
                     **{c:common[c] for c in CONTROLS}})
    return pd.DataFrame(rows)

def hac_fit(y,X):
    X=sm.add_constant(X,has_constant='add')
    return sm.OLS(y,X,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':HAC_LAGS,'use_correction':True})

def second_stage(lam):
    rows=[]
    keys=['CHANNEL','CHANNEL_ROLE','OUTCOME','SAMPLE_DESIGN']
    for key,g in lam.groupby(keys):
        ch,role,outcome,design=key; g=g.sort_values('DATE').dropna(subset=['LAMBDA_ARCHITECTURE'])
        base={'CHANNEL':ch,'CHANNEL_ROLE':role,'OUTCOME':outcome,'SAMPLE_DESIGN':design,'N_MONTHS':len(g),
              'N_ETFS_MIN':int(g.N_ETFS_MONTH.min()),'N_ETFS_MAX':int(g.N_ETFS_MONTH.max()),
              'DATE_START':g.DATE.min(),'DATE_END':g.DATE.max(),
              'INFERENCE_CLASS':'MAIN_INFERENCE_ELIGIBLE' if g.N_ETFS_MONTH.min()>=MAIN_MIN_ETFS else 'EXPLORATORY_SMALL_CROSS_SECTION'}
        if len(g)<24: continue
        # Unconditional architecture price.
        fit=hac_fit(g.LAMBDA_ARCHITECTURE,pd.DataFrame(index=g.index))
        rows.append({**base,'TEST':'UNCONDITIONAL_MEAN','FOCAL':'CONST','COEFFICIENT':fit.params['const'],
                     'STANDARD_ERROR':fit.bse['const'],'T_STATISTIC':fit.tvalues['const'],'P_VALUE':fit.pvalues['const'],
                     'CI_LOWER_95':fit.conf_int().loc['const',0],'CI_UPPER_95':fit.conf_int().loc['const',1]})
        # Continuous activation of the monthly architecture price.
        xcols=['CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z',*CONTROLS]
        z=g.dropna(subset=xcols); fit=hac_fit(z.LAMBDA_ARCHITECTURE,z[xcols]); focal='CPU_Z_X_VIX_Z'
        rows.append({**base,'N_MONTHS':len(z),'TEST':'CONTINUOUS_CPU_VIX_ACTIVATION','FOCAL':focal,
                     'COEFFICIENT':fit.params[focal],'STANDARD_ERROR':fit.bse[focal],'T_STATISTIC':fit.tvalues[focal],
                     'P_VALUE':fit.pvalues[focal],'CI_LOWER_95':fit.conf_int().loc[focal,0],'CI_UPPER_95':fit.conf_int().loc[focal,1]})
        # State increments relative to all other months.
        for state in ['CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME','HIGH_CPU_REGIME','HIGH_VIX_REGIME']:
            z=g.dropna(subset=[state]); fit=hac_fit(z.LAMBDA_ARCHITECTURE,z[[state]])
            rows.append({**base,'N_MONTHS':len(z),'TEST':f'{state}_INCREMENT','FOCAL':state,
                         'COEFFICIENT':fit.params[state],'STANDARD_ERROR':fit.bse[state],'T_STATISTIC':fit.tvalues[state],
                         'P_VALUE':fit.pvalues[state],'CI_LOWER_95':fit.conf_int().loc[state,0],'CI_UPPER_95':fit.conf_int().loc[state,1],
                         'NONSTATE_MEAN':fit.params['const'],'STATE_MEAN':fit.params['const']+fit.params[state],
                         'N_STATE_MONTHS':int(z[state].eq(1).sum())})
    return pd.DataFrame(rows)

def multiplicity(res):
    x=res.copy(); x['HOLM_P_PRIMARY_FAMILY']=np.nan; x['BH_Q_PRIMARY_FAMILY']=np.nan
    mask=(x.CHANNEL_ROLE=='PRIMARY')&(x.INFERENCE_CLASS=='MAIN_INFERENCE_ELIGIBLE')
    for test,g in x[mask].groupby('TEST'):
        p=g.P_VALUE.to_numpy(float); valid=np.isfinite(p); idx=g.index[valid]
        if valid.any():
            x.loc[idx,'HOLM_P_PRIMARY_FAMILY']=multipletests(p[valid],method='holm')[1]
            x.loc[idx,'BH_Q_PRIMARY_FAMILY']=multipletests(p[valid],method='fdr_bh')[1]
    return x

def figures(lam):
    import matplotlib.pyplot as plt
    labels={'INTERNAL_FINANCIAL_CAPACITY':'Internal financial capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing dependence',
            'GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration exposure','PORTFOLIO_CONCENTRATION_FINAL':'Portfolio concentration'}
    for ch,title in labels.items():
        g=lam[(lam.CHANNEL==ch)&(lam.OUTCOME=='ETF_RETURN')&(lam.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')].sort_values('DATE')
        if g.empty: continue
        roll=g.LAMBDA_ARCHITECTURE.rolling(12,min_periods=6).mean()*100
        fig=plt.figure(figsize=(8.2,4.6)); ax=fig.add_subplot(111)
        ax.plot(g.DATE,g.LAMBDA_ARCHITECTURE*100,alpha=.35,linewidth=.8,label='Monthly price')
        ax.plot(g.DATE,roll,linewidth=1.8,label='12-month rolling mean')
        stress=g[g.CPU_AND_VIX_STRESS.eq(1)]
        ax.scatter(stress.DATE,stress.LAMBDA_ARCHITECTURE*100,s=18,label='Joint stress month')
        ax.axhline(0,linewidth=1); ax.set_ylabel('Monthly architecture price (percentage points)'); ax.set_xlabel('Month')
        ax.set_title(f'Fama–MacBeth architecture price: {title}'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25)
        fig.tight_layout(); fig.savefig(FIG/f'30_fmb_{ch.lower()}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'30_fmb_{ch.lower()}.pdf',bbox_inches='tight'); plt.close(fig)

def main():
    df=pd.read_csv(SOURCE,usecols=lambda c:c in set(reqcols()),low_memory=False,parse_dates=['DATE'])
    for c in reqcols():
        if c in df and c not in ['DATE','DATE_GROUP','ETF_ID','ETF_TICKER']: df[c]=pd.to_numeric(df[c],errors='coerce')
    if df.duplicated(['ETF_ID','DATE']).any(): raise RuntimeError('duplicate ETF-month')
    df,fdiag=factor_residuals(df); fdiag.to_csv(OUT/'30_factor_adjustment_diagnostics.csv',index=False)
    pieces=[]
    for ch,role in CHANNELS.items():
        for outcome in ['ETF_RETURN','FACTOR_ADJUSTED_RETURN']:
            for design in ['AVAILABLE_CROSS_SECTION','MIN10_CROSS_SECTION','BALANCED_COMMON_WINDOW']:
                pieces.append(monthly_lambdas(df,ch,role,outcome,design))
    lam=pd.concat(pieces,ignore_index=True); lam.to_csv(OUT/'30_monthly_architecture_prices.csv',index=False)
    res=multiplicity(second_stage(lam)); res.to_csv(OUT/'30_fama_macbeth_second_stage_results.csv',index=False)
    primary=res[(res.CHANNEL_ROLE=='PRIMARY')&(res.OUTCOME=='ETF_RETURN')&(res.SAMPLE_DESIGN=='MIN10_CROSS_SECTION')]
    primary.to_csv(OUT/'30_primary_fama_macbeth_results.csv',index=False)
    coverage=(lam.groupby(['CHANNEL','CHANNEL_ROLE','OUTCOME','SAMPLE_DESIGN']).agg(N_MONTHS=('DATE','nunique'),DATE_START=('DATE','min'),DATE_END=('DATE','max'),N_ETFS_MIN=('N_ETFS_MONTH','min'),N_ETFS_MAX=('N_ETFS_MONTH','max'),MEAN_CROSS_SECTION_R2=('CROSS_SECTION_R2','mean')).reset_index())
    coverage.to_csv(OUT/'30_fama_macbeth_sample_diagnostics.csv',index=False)
    figures(lam)
    metadata={'source':str(SOURCE),'first_stage':'monthly cross-sectional OLS of ETF return on standardized architecture score',
              'outcomes':['raw ETF return','ETF-specific four-factor residual return'],'sample_designs':['available cross-section','months with at least 10 ETFs','balanced common window'],
              'second_stage':'Newey-West/HAC with 3 monthly lags','primary_tests':['continuous CPU x VIX activation','joint stress increment'],
              'main_cross_section_threshold':MAIN_MIN_ETFS}
    (OUT/'30_fama_macbeth_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nPRIMARY RESULTS\n',primary[['CHANNEL','TEST','COEFFICIENT','STANDARD_ERROR','P_VALUE','HOLM_P_PRIMARY_FAMILY','BH_Q_PRIMARY_FAMILY','N_MONTHS','N_ETFS_MIN','N_ETFS_MAX']].to_string(index=False))
    print('\nCOVERAGE\n',coverage[(coverage.CHANNEL_ROLE=='PRIMARY')&(coverage.OUTCOME=='ETF_RETURN')].to_string(index=False))
if __name__=='__main__': main()
