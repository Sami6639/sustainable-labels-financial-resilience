from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

sys.path.insert(0,str(Path(__file__).resolve().parent))
from q1_common import (load_panel, add_standardized_alternatives, channel_sample, fit_fe,
                       design_continuous, design_binary_pair, design_continuous_binary,
                       design_single_shock, PRIMARY_CHANNELS, CONTROLS)

OUT=Path('/mnt/data/q1_analysis/alternative_stress_falsification'); FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
EXPECTED={'INTERNAL_FINANCIAL_CAPACITY':1,'EXTERNAL_FINANCING_DEPENDENCE':-1,'GROWTH_DURATION_EXPOSURE_FINAL':-1,'PORTFOLIO_CONCENTRATION_FINAL':-1}


def add_month_definitions(df):
    x=add_standardized_alternatives(df)
    monthly=x.sort_values('DATE').drop_duplicates('DATE').set_index('DATE')
    for pct in [10,20,25]:
        qc=monthly.CPU_Z.quantile(1-pct/100); qv=monthly.VIX_LEVEL_Z.quantile(1-pct/100)
        qm=monthly.MARKET_RETURN.quantile(pct/100)
        x[f'CPU_TOP_{pct}']=x.CPU_Z.ge(qc).astype(int); x[f'VIX_TOP_{pct}']=x.VIX_LEVEL_Z.ge(qv).astype(int); x[f'MARKET_BOTTOM_{pct}']=x.MARKET_RETURN.le(qm).astype(int)
    x['NEGATIVE_MARKET']=x.MARKET_RETURN.lt(0).astype(int)
    # Calendar leads are common monthly series; shift on the unique month table and map back.
    for c in ['CPU_Z','HIGH_CPU_REGIME','CPU_AND_VIX_STRESS']:
        ser=monthly[c]
        for lead in [1,3]: x[f'{c}_LEAD{lead}']=x.DATE.map(ser.shift(-lead))
    return x


def estimate_model(base,arch,spec,variant):
    s=base.copy(); family=spec['family']; kind=spec['kind']
    if kind=='CONTINUOUS': regs,focal=design_continuous(s,arch,spec.get('cpu','CPU_Z'),spec.get('vix','VIX_LEVEL_Z'))
    elif kind=='BINARY_PAIR': regs,focal=design_binary_pair(s,arch,spec['a'],spec['b'],spec['name'])
    elif kind=='CONT_BINARY': regs,focal=design_continuous_binary(s,arch,spec['cont'],spec['binary'],spec['name'])
    elif kind=='SINGLE': regs,focal=design_single_shock(s,arch,spec['shock'],spec['name'])
    else: raise ValueError(kind)
    month_fe=(variant=='ETF_MONTH_FE'); cov='ETF_CLUSTER' if month_fe else 'DATE_CLUSTER'
    r=fit_fe(s,'ETF_RETURN',regs,focal,month_fe=month_fe,covariance=cov)
    return {**spec,'MODEL_VARIANT':variant,'FOCAL_REGRESSOR':focal,**r}


def specs():
    out=[]
    # Threshold sensitivity: hierarchical two-state models.
    for pct in [10,20,25]:
        out.append({'name':f'CPU_VIX_TOP_{pct}','family':'ALTERNATIVE_STRESS_THRESHOLD','kind':'BINARY_PAIR','a':f'CPU_TOP_{pct}','b':f'VIX_TOP_{pct}','description':f'CPU and VIX top {pct}% jointly'})
    for pct in [10,20]:
        out.append({'name':f'CPU_MARKET_TAIL_{pct}','family':'ALTERNATIVE_MARKET_ACTIVATION','kind':'BINARY_PAIR','a':f'CPU_TOP_{pct}','b':f'MARKET_BOTTOM_{pct}','description':f'CPU top {pct}% and market bottom {pct}%'})
    out.append({'name':'CPU_TOP20_NEGATIVE_MARKET','family':'ALTERNATIVE_MARKET_ACTIVATION','kind':'BINARY_PAIR','a':'CPU_TOP_20','b':'NEGATIVE_MARKET','description':'CPU top 20% and negative market return'})
    out.append({'name':'CPU_CONT_NEGATIVE_MARKET','family':'ALTERNATIVE_MARKET_ACTIVATION','kind':'CONT_BINARY','cont':'CPU_Z','binary':'NEGATIVE_MARKET','description':'Continuous CPU activated by negative market returns'})
    out.append({'name':'CPU_CONT_MARKET_BOTTOM20','family':'ALTERNATIVE_MARKET_ACTIVATION','kind':'CONT_BINARY','cont':'CPU_Z','binary':'MARKET_BOTTOM_20','description':'Continuous CPU activated by bottom-quintile market returns'})
    out.append({'name':'CPU_CONT_HIGH_VIX','family':'ALTERNATIVE_STRESS_THRESHOLD','kind':'CONT_BINARY','cont':'CPU_Z','binary':'HIGH_VIX_REGIME','description':'Continuous CPU activated by high VIX regime'})
    # Alternative CPU transformations.
    for cpu in ['LOG_CPU_Z','CPU_DIFF_Z','CPU_CHANGE_Z','LOG_CPU_BROAD_Z','LOG_CPU_LLM_Z','CPU_SHOCK']:
        out.append({'name':f'{cpu}_VIX_CONTINUOUS','family':'ALTERNATIVE_CPU_MEASURE','kind':'CONTINUOUS','cpu':cpu,'vix':'VIX_LEVEL_Z','description':f'{cpu} x VIX x architecture'})
    # Future-information placebos.
    for lead in [1,3]:
        out.append({'name':f'FUTURE_CPU_LEAD{lead}_VIX','family':'LEAD_PLACEBO','kind':'CONTINUOUS','cpu':f'CPU_Z_LEAD{lead}','vix':'VIX_LEVEL_Z','description':f'Future CPU lead {lead} x current VIX placebo'})
    return out


def exclusion_windows():
    return {
        'FULL_SAMPLE':lambda d:pd.Series(True,index=d.index),
        'EXCLUDE_2020':lambda d:~d.DATE.dt.year.eq(2020),
        'EXCLUDE_2022':lambda d:~d.DATE.dt.year.eq(2022),
        'EXCLUDE_2020_2022':lambda d:~d.DATE.dt.year.between(2020,2022),
        'EXCLUDE_PANDEMIC_2020M03_2021M12':lambda d:~d.DATE.between(pd.Timestamp('2020-03-01'),pd.Timestamp('2021-12-31')),
        'PRE_2020':lambda d:d.DATE.lt(pd.Timestamp('2020-01-01')),
        'POST_2020':lambda d:d.DATE.ge(pd.Timestamp('2020-01-01')),
        'POST_2015':lambda d:d.DATE.ge(pd.Timestamp('2015-01-01')),
        'POST_2017':lambda d:d.DATE.ge(pd.Timestamp('2017-01-01')),
    }


def main():
    df=add_month_definitions(load_panel())
    rows=[]
    sp=specs()
    for ch in PRIMARY_CHANNELS:
        base,arch,flag=channel_sample(df,ch)
        for spec in sp:
            for variant in ['ETF_FE_DATE_CLUSTER','ETF_MONTH_FE']:
                r=estimate_model(base,arch,spec,variant); rows.append({'CHANNEL':ch,'EXPECTED_SIGN':'POSITIVE' if EXPECTED[ch]>0 else 'NEGATIVE',**r})
        # Crisis/subperiod stability for the two pre-specified activation models.
        for window,fn in exclusion_windows().items():
            ss=base[fn(base)].copy()
            for name,kind in [('CPU_VIX_CONTINUOUS','CONTINUOUS'),('CPU_AND_VIX_STRESS','BINARY_PAIR')]:
                spec={'name':name,'family':'SAMPLE_WINDOW_STABILITY','kind':kind,'description':window}
                if kind=='BINARY_PAIR': spec.update({'a':'HIGH_CPU_REGIME','b':'HIGH_VIX_REGIME'})
                r=estimate_model(ss,arch,spec,'ETF_FE_DATE_CLUSTER'); rows.append({'CHANNEL':ch,'EXPECTED_SIGN':'POSITIVE' if EXPECTED[ch]>0 else 'NEGATIVE','SAMPLE_WINDOW':window,**r})
        # Low/high VIX split: CPU x architecture within state.
        for state,mask in [('LOW_VIX_SUBSAMPLE',base.HIGH_VIX_REGIME.eq(0)),('HIGH_VIX_SUBSAMPLE',base.HIGH_VIX_REGIME.eq(1)),('NONJOINT_STRESS_SUBSAMPLE',base.CPU_AND_VIX_STRESS.eq(0))]:
            ss=base[mask].copy(); spec={'name':'CPU_X_ARCH_WITHIN_STATE','family':'STATE_FALSIFICATION','kind':'SINGLE','shock':'CPU_Z','description':state}
            r=estimate_model(ss,arch,spec,'ETF_FE_DATE_CLUSTER'); rows.append({'CHANNEL':ch,'EXPECTED_SIGN':'POSITIVE' if EXPECTED[ch]>0 else 'NEGATIVE','SAMPLE_WINDOW':state,**r})
    res=pd.DataFrame(rows)
    # Harmonized primary multiplicity by family and variant.
    res['HOLM_P_WITHIN_FAMILY']=np.nan; res['BH_Q_WITHIN_FAMILY']=np.nan
    for (fam,var),g in res.groupby(['family','MODEL_VARIANT']):
        p=pd.to_numeric(g.P_VALUE,errors='coerce'); ok=p.notna(); idx=g.index[ok]
        if len(idx):
            res.loc[idx,'HOLM_P_WITHIN_FAMILY']=multipletests(p[ok],method='holm')[1]
            res.loc[idx,'BH_Q_WITHIN_FAMILY']=multipletests(p[ok],method='fdr_bh')[1]
    res['EXPECTED_SIGN_MATCH']=np.where(res.EXPECTED_SIGN.eq('POSITIVE'),res.COEFFICIENT.gt(0),res.COEFFICIENT.lt(0))
    res.to_csv(OUT/'34_all_alternative_stress_and_falsification_results.csv',index=False)
    primary=res[res.MODEL_VARIANT.eq('ETF_FE_DATE_CLUSTER')].copy(); primary.to_csv(OUT/'34_primary_alternative_stress_results.csv',index=False)
    fals=primary[primary.family.isin(['LEAD_PLACEBO','STATE_FALSIFICATION'])].copy(); fals.to_csv(OUT/'34_falsification_results.csv',index=False)
    stability=primary[primary.family.eq('SAMPLE_WINDOW_STABILITY')].copy(); stability.to_csv(OUT/'34_sample_window_stability.csv',index=False)
    # Sign-consistency summary across alternative CPU/stress definitions.
    summ=(primary[primary.family.isin(['ALTERNATIVE_STRESS_THRESHOLD','ALTERNATIVE_MARKET_ACTIVATION','ALTERNATIVE_CPU_MEASURE'])]
          .groupby(['CHANNEL','family']).agg(N_MODELS=('COEFFICIENT','count'),EXPECTED_SIGN_RATE=('EXPECTED_SIGN_MATCH','mean'),
              MEDIAN_COEFFICIENT=('COEFFICIENT','median'),N_RAW_P_LT_05=('P_VALUE',lambda x:(x<.05).sum()),N_BH_Q_LT_10=('BH_Q_WITHIN_FAMILY',lambda x:(x<.10).sum()),MIN_P=('P_VALUE','min')).reset_index())
    summ.to_csv(OUT/'34_alternative_definition_sign_consistency.csv',index=False)
    # Coefficient stability figure for continuous model exclusions.
    labels={'INTERNAL_FINANCIAL_CAPACITY':'Internal capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing','GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration','PORTFOLIO_CONCENTRATION_FINAL':'Concentration'}
    z=stability[stability.name.eq('CPU_VIX_CONTINUOUS')].copy(); order=list(exclusion_windows().keys())
    fig,ax=plt.subplots(figsize=(10,5.6))
    for ch,g in z.groupby('CHANNEL'):
        g=g.set_index('SAMPLE_WINDOW').reindex(order); ax.plot(range(len(order)),100*g.COEFFICIENT,marker='o',label=labels[ch])
    ax.axhline(0,linewidth=1); ax.set_xticks(range(len(order)),order,rotation=35,ha='right'); ax.set_ylabel('CPU × VIX × architecture coefficient (percentage points)')
    ax.set_title('Continuous activation across sample exclusions'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,ncol=2); fig.tight_layout()
    fig.savefig(FIG/'34_continuous_activation_sample_stability.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/'34_continuous_activation_sample_stability.pdf',bbox_inches='tight'); plt.close(fig)
    meta={'primary_inference':'ETF fixed effects with date-clustered standard errors','month_fe_robustness':'ETF and month fixed effects with ETF-clustered standard errors',
          'thresholds':['top 10%','top 20%','top 25%'],'alternative_cpu':['log narrow CPU','CPU difference','CPU percentage change','broad CPU','LLM CPU','official external shock'],
          'falsification':['future CPU leads','low/high VIX subsamples','non-joint-stress subsample'],'sample_exclusions':list(exclusion_windows().keys())}
    (OUT/'34_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nSIGN CONSISTENCY\n',summ.to_string(index=False))
    print('\nLEAD PLACEBOS\n',fals[fals.family.eq('LEAD_PLACEBO')][['CHANNEL','name','COEFFICIENT','P_VALUE','BH_Q_WITHIN_FAMILY','EXPECTED_SIGN_MATCH']].to_string(index=False))

if __name__=='__main__': main()
