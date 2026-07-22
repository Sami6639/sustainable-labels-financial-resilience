from pathlib import Path
import importlib.util, json, math, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

spec=importlib.util.spec_from_file_location('lp','/mnt/data/q1_analysis/29_estimate_local_projection_recovery.py')
lp=importlib.util.module_from_spec(spec); spec.loader.exec_module(lp)
SOURCE=lp.SOURCE; OUT=Path('/mnt/data/q1_analysis/joint_channel_diagnostics'); OUT.mkdir(parents=True,exist_ok=True)
PAIRS={
 'CAPACITY_VS_GROWTH':('INTERNAL_FINANCIAL_CAPACITY','GROWTH_DURATION_EXPOSURE_FINAL'),
 'FINANCING_VS_CONCENTRATION':('EXTERNAL_FINANCING_DEPENDENCE','PORTFOLIO_CONCENTRATION_FINAL'),
}
SHOCKS=['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']
VARIANTS=['PRIMARY_ALIGNED','MONTH_FE_ROBUSTNESS']
HORIZONS=range(13); MIN_ETFS=10; ZERO=1e-14

def make_terms(s,a1,a2,shock,variant):
    mfe=variant=='MONTH_FE_ROBUSTNESS'
    if shock=='CPU_VIX_CONTINUOUS':
        for a,tag in [(a1,'A1'),(a2,'A2')]:
            s[f'CPU_X_{tag}']=s.CPU_Z*s[a]; s[f'VIX_X_{tag}']=s.VIX_LEVEL_Z*s[a]; s[f'TRIPLE_{tag}']=s.CPU_Z_X_VIX_Z*s[a]
        regs=['CPU_X_A1','VIX_X_A1','TRIPLE_A1','CPU_X_A2','VIX_X_A2','TRIPLE_A2']
        if not mfe: regs=['CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z',*regs,*lp.CONTROLS]
        return regs,{'A1':'TRIPLE_A1','A2':'TRIPLE_A2'}
    if shock=='CPU_AND_VIX_STRESS':
        for a,tag in [(a1,'A1'),(a2,'A2')]:
            s[f'HC_{tag}']=s.HIGH_CPU_REGIME*s[a]; s[f'HV_{tag}']=s.HIGH_VIX_REGIME*s[a]; s[f'JOINT_{tag}']=s.CPU_AND_VIX_STRESS*s[a]
        regs=['HC_A1','HV_A1','JOINT_A1','HC_A2','HV_A2','JOINT_A2']
        if not mfe: regs=['HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS',*regs,*lp.CONTROLS]
        return regs,{'A1':'JOINT_A1','A2':'JOINT_A2'}
    for a,tag in [(a1,'A1'),(a2,'A2')]: s[f'SHOCK_{tag}']=s[shock]*s[a]
    regs=['SHOCK_A1','SHOCK_A2'] if mfe else [shock,'SHOCK_A1','SHOCK_A2',*lp.CONTROLS]
    return regs,{'A1':'SHOCK_A1','A2':'SHOCK_A2'}

def estimate(df,pair,a1,a2,shock,h,variant):
    ac1,f1=lp.quality_col(a1); ac2,f2=lp.quality_col(a2)
    m=df[ac1].notna()&df[ac2].notna()&df[f1].eq(1)&df[f2].eq(1)
    cols=['ETF_ID','DATE','DATE_GROUP',f'LP_CUM_RETURN_H{h}',ac1,ac2,'CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z','HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME',*lp.CONTROLS]
    s=df.loc[m,cols].copy(); regs,focals=make_terms(s,ac1,ac2,shock,variant)
    req=['ETF_ID','DATE','DATE_GROUP',f'LP_CUM_RETURN_H{h}',*regs]; s=s.dropna(subset=req).drop_duplicates(['ETF_ID','DATE']).sort_values(['ETF_ID','DATE']).reset_index(drop=True)
    entity,el=pd.factorize(s.ETF_ID); time,tl=pd.factorize(s.DATE_GROUP); ne=len(el); nt=len(tl); n=len(s); mfe=variant=='MONTH_FE_ROBUSTNESS'
    base={'PAIR':pair,'CHANNEL_1':a1,'CHANNEL_2':a2,'SHOCK':shock,'HORIZON_MONTHS':h,'MODEL_VARIANT':variant,'N_OBSERVATIONS':n,'N_ETFS':ne,'N_DATES':nt,
          'INFERENCE_CLASS':'MAIN_INFERENCE_ELIGIBLE' if ne>=MIN_ETFS else 'EXPLORATORY'}
    rows=[]
    try:
        raw=np.column_stack([s[[f'LP_CUM_RETURN_H{h}']].to_numpy(float),s[regs].to_numpy(float)])
        z,it,conv=lp.within_transform(raw,entity,time if mfe else None); y=z[:,0]; X=z[:,1:]
        keep=np.var(X,axis=0)>1e-20; regs2=[r for r,k in zip(regs,keep) if k]; X=X[:,keep]
        if np.linalg.matrix_rank(X)<X.shape[1]: raise ValueError('rank deficient')
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@beta; bread=np.linalg.inv(X.T@X)
        absorbed=(ne-1)+(nt-1 if mfe else 0); kt=X.shape[1]+absorbed
        if h==0: cov=lp.cov_cluster(X,resid,time,nt,bread,kt); method='DATE_CLUSTERED'; bw=0
        else: cov=lp.cov_dk(X,resid,time,nt,bread,kt,h); method=f'DK_H{h}'; bw=h
        # Validate entire covariance once.
        diag=np.diag(cov); mineig=np.linalg.eigvalsh((cov+cov.T)/2).min(); valid=np.all(np.isfinite(cov)) and np.all(diag>ZERO) and mineig>=-1e-12
        for tag,ch in [('A1',a1),('A2',a2)]:
            focal=focals[tag]
            if focal not in regs2: raise ValueError(f'{focal} dropped')
            j=regs2.index(focal); coef=beta[j]
            row={**base,'FOCAL_CHANNEL':ch,'CONDITIONAL_ON':a2 if tag=='A1' else a1,'FOCAL_REGRESSOR':focal,'COVARIANCE_METHOD':method,'DK_BANDWIDTH':bw,
                 'CONDITION_NUMBER':float(np.linalg.cond(X)),'PAIRWISE_ARCH_CORRELATION':float(s[[ac1,ac2]].corr().iloc[0,1]),'MODEL_STATUS':'INVALID_COVARIANCE' if not valid else 'RELIABLE_ESTIMATE'}
            if valid:
                se=math.sqrt(diag[j]); t=coef/se; p=2*stats.t.sf(abs(t),max(nt-1,1)); crit=stats.t.ppf(.975,max(nt-1,1))
                row.update({'COEFFICIENT':coef,'STANDARD_ERROR':se,'T_STATISTIC':t,'P_VALUE':p,'CI_LOWER_95':coef-crit*se,'CI_UPPER_95':coef+crit*se})
            rows.append(row)
    except Exception as e:
        for ch,other in [(a1,a2),(a2,a1)]: rows.append({**base,'FOCAL_CHANNEL':ch,'CONDITIONAL_ON':other,'MODEL_STATUS':'ESTIMATION_ERROR','ERROR_MESSAGE':f'{type(e).__name__}: {e}'})
    return rows

def main():
    needed=set(lp.required_columns()); needed.update(['CPU_Z_X_VIX_Z'])
    df=pd.read_csv(SOURCE,usecols=lambda c:c in needed,parse_dates=['DATE'],low_memory=False)
    df['CPU_Z_X_VIX_Z']=pd.to_numeric(df.CPU_Z,errors='coerce')*pd.to_numeric(df.VIX_LEVEL_Z,errors='coerce')
    df=lp.add_dynamic_columns(df)
    rows=[]
    for pair,(a1,a2) in PAIRS.items():
        for shock in SHOCKS:
            for h in HORIZONS:
                for v in VARIANTS: rows.extend(estimate(df,pair,a1,a2,shock,h,v))
    res=pd.DataFrame(rows); res.to_csv(OUT/'30b_all_joint_channel_results.csv',index=False)
    prim=res[(res.MODEL_VARIANT=='PRIMARY_ALIGNED')&(res.MODEL_STATUS=='RELIABLE_ESTIMATE')].copy()
    prim['HOLM_P_WITHIN_PROFILE']=np.nan; prim['BH_Q_WITHIN_PROFILE']=np.nan
    for (shock,ch),g in prim.groupby(['SHOCK','FOCAL_CHANNEL']):
        p=g.P_VALUE.to_numpy(); idx=g.index
        prim.loc[idx,'HOLM_P_WITHIN_PROFILE']=multipletests(p,method='holm')[1]; prim.loc[idx,'BH_Q_WITHIN_PROFILE']=multipletests(p,method='fdr_bh')[1]
    prim.to_csv(OUT/'30b_primary_joint_channel_results.csv',index=False)
    key=prim[prim.HORIZON_MONTHS.isin([0,3,6,9,12])].copy(); key.to_csv(OUT/'30b_key_joint_channel_diagnostics.csv',index=False)
    val=res.groupby(['PAIR','MODEL_VARIANT','MODEL_STATUS']).size().rename('N_MODELS').reset_index(); val.to_csv(OUT/'30b_validation_summary.csv',index=False)
    meta={'purpose':'separate channels whose static architecture scores are materially correlated','pairs':PAIRS,'inference':'date cluster h0; DK bandwidth h for cumulative h>=1','negative_variances_clipped':False}
    (OUT/'30b_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(key[['PAIR','SHOCK','HORIZON_MONTHS','FOCAL_CHANNEL','CONDITIONAL_ON','COEFFICIENT','P_VALUE','BH_Q_WITHIN_PROFILE','N_ETFS','PAIRWISE_ARCH_CORRELATION','CONDITION_NUMBER']].to_string(index=False))
if __name__=='__main__': main()
