from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

SOURCE = Path('/mnt/data/cpu_extracted/CPU_Project/output/25_final_econometric_panel.csv')
CONTROLS = ['MARKET_RETURN','ENERGY_RETURN','TREASURY_RETURN','VIX_CHANGE']
PRIMARY_CHANNELS = [
    'INTERNAL_FINANCIAL_CAPACITY',
    'EXTERNAL_FINANCING_DEPENDENCE',
    'GROWTH_DURATION_EXPOSURE_FINAL',
    'PORTFOLIO_CONCENTRATION_FINAL',
]
CHANNEL_ROLES = {ch:'PRIMARY' for ch in PRIMARY_CHANNELS} | {
    'FINANCIAL_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS',
    'EXTENDED_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS',
    'PCA_COMPONENT_1':'PCA_ROBUSTNESS',
}
ZERO_TOL = 1e-14


def quality_col(channel: str, quality: str='MAIN') -> tuple[str,str]:
    if channel == 'PCA_COMPONENT_1':
        return channel, 'VALID_PCA_MAIN_ROW'
    return f'{channel}_{quality}', f'VALID_{channel}_{quality}_ROW'


def load_panel(extra_cols: Iterable[str] | None=None) -> pd.DataFrame:
    base = {
        'DATE','DATE_GROUP','ETF_ID','ETF_TICKER','ETF_NAME_RETURN','ETF_RETURN',
        'CPU_Z','LOG_CPU_Z','CPU_DIFF','CPU_CHANGE','CPU_BROAD','CPU_LLM',
        'CPU_SHOCK','CPU_INSTRUMENT','VIX_LEVEL_Z','VIX_LEVEL',
        'HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS',
        'EXTREME_CPU_REGIME','EXTREME_VIX_REGIME','EXTREME_CPU_AND_VIX_STRESS',
        *CONTROLS,
    }
    for ch in CHANNEL_ROLES:
        arch,flag=quality_col(ch)
        base.update([arch,flag])
    if extra_cols:
        base.update(extra_cols)
    df=pd.read_csv(SOURCE,usecols=lambda c:c in base,parse_dates=['DATE'],low_memory=False)
    if 'DATE_GROUP' not in df:
        df['DATE_GROUP']=df.DATE.dt.to_period('M').astype(str)
    for c in df.columns:
        if c not in ['DATE','DATE_GROUP','ETF_ID','ETF_TICKER','ETF_NAME_RETURN']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.sort_values(['ETF_ID','DATE']).reset_index(drop=True)
    if df.duplicated(['ETF_ID','DATE']).any():
        raise RuntimeError('Duplicate ETF-month rows detected.')
    return df


def add_standardized_alternatives(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    # Standardize on unique monthly observations, then map back.
    monthly=x.sort_values('DATE').drop_duplicates('DATE').set_index('DATE')
    definitions={
        'CPU_DIFF_Z':'CPU_DIFF',
        'CPU_CHANGE_Z':'CPU_CHANGE',
        'LOG_CPU_BROAD_Z':'CPU_BROAD',
        'LOG_CPU_LLM_Z':'CPU_LLM',
    }
    for new,old in definitions.items():
        raw=pd.to_numeric(monthly[old],errors='coerce')
        if old in ['CPU_BROAD','CPU_LLM']:
            raw=np.log(raw.where(raw>0))
        sd=raw.std(ddof=0)
        z=(raw-raw.mean())/sd if np.isfinite(sd) and sd>0 else raw*np.nan
        x[new]=x['DATE'].map(z)
    return x


def add_leads_and_lags(df: pd.DataFrame, columns: Iterable[str], leads=(1,3), lags=(1,)) -> pd.DataFrame:
    x=df.sort_values(['ETF_ID','DATE']).copy()
    period=x.DATE.dt.year*12+x.DATE.dt.month
    for c in columns:
        for lead in leads:
            v=x.groupby('ETF_ID')[c].shift(-lead)
            p=x.groupby('ETF_ID')[period.name if period.name else 'DATE'].shift(-lead) if False else None
            # Validate calendar contiguity directly from shifted period.
            shifted_period=period.groupby(x['ETF_ID']).shift(-lead)
            v=v.where((shifted_period-period).eq(lead))
            x[f'{c}_LEAD{lead}']=v
        for lag in lags:
            v=x.groupby('ETF_ID')[c].shift(lag)
            shifted_period=period.groupby(x['ETF_ID']).shift(lag)
            v=v.where((period-shifted_period).eq(lag))
            x[f'{c}_LAG{lag}']=v
    return x


def add_forward_cumulative_returns(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    x=df.sort_values(['ETF_ID','DATE']).copy()
    period=x.DATE.dt.year*12+x.DATE.dt.month
    horizons=sorted(set(int(h) for h in horizons))
    for h in horizons:
        x[f'LP_CUM_RETURN_H{h}']=np.nan
    for etf,g in x.groupby('ETF_ID',sort=False):
        idx=g.index.to_numpy(); r=g.ETF_RETURN.to_numpy(float); per=(g.DATE.dt.year*12+g.DATE.dt.month).to_numpy(int)
        n=len(g)
        for h in horizons:
            out=np.full(n,np.nan)
            for i in range(n):
                j=i+h
                if j<n and per[j]-per[i]==h and np.all(np.isfinite(r[i:j+1])):
                    out[i]=np.prod(1+r[i:j+1])-1
            x.loc[idx,f'LP_CUM_RETURN_H{h}']=out
    return x


def group_demean(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    sums=np.zeros((n_groups,values.shape[1]),float); counts=np.zeros(n_groups,float)
    np.add.at(sums,codes,values); np.add.at(counts,codes,1.0)
    return values-(sums/np.maximum(counts[:,None],1.0))[codes]


def within_transform(values: np.ndarray, entity: np.ndarray, time: np.ndarray|None=None,
                     tol:float=1e-11,max_iter:int=1000) -> tuple[np.ndarray,int,float]:
    z=np.asarray(values,float).copy(); ne=int(entity.max())+1
    if time is None:
        z=group_demean(z,entity,ne)
        return z,1,float(np.nanmax(np.abs(z.mean(axis=0))))
    nt=int(time.max())+1; conv=np.inf
    for it in range(1,max_iter+1):
        old=z.copy(); z=group_demean(z,entity,ne); z=group_demean(z,time,nt)
        conv=float(np.max(np.abs(z-old)))
        if conv<tol: return z,it,conv
    return z,max_iter,conv


def finite_sample_factor(n,k_total,g):
    if g<=1:return np.nan
    return (g/(g-1))*((n-1)/max(n-k_total,1))


def cov_cluster(X,resid,codes,n_groups,bread,k_total):
    scores=X*resid[:,None]; s=np.zeros((n_groups,scores.shape[1]),float); np.add.at(s,codes,scores)
    return bread@(s.T@s)@bread*finite_sample_factor(len(X),k_total,n_groups)


def cov_dk(X,resid,time,nt,bread,k_total,bandwidth):
    scores=X*resid[:,None]; st=np.zeros((nt,scores.shape[1]),float); np.add.at(st,time,scores)
    meat=st.T@st
    bw=min(int(bandwidth),nt-1)
    for lag in range(1,bw+1):
        w=1-lag/(bw+1); gamma=st[lag:].T@st[:-lag]; meat+=w*(gamma+gamma.T)
    return bread@meat@bread*finite_sample_factor(len(X),k_total,nt)


def validate_cov(cov:np.ndarray,target_idx:int)->dict:
    diag=np.diag(cov) if cov.ndim==2 and cov.shape[0]==cov.shape[1] else np.array([])
    nonfinite=int((~np.isfinite(cov)).sum()) if cov.size else 1
    neg=int((diag<0).sum()) if diag.size else 1
    zero=int((np.abs(diag)<=ZERO_TOL).sum()) if diag.size else 1
    target=float(diag[target_idx]) if diag.size>target_idx else np.nan
    try: mineig=float(np.linalg.eigvalsh((cov+cov.T)/2).min())
    except Exception: mineig=np.nan
    valid=(nonfinite==0 and neg==0 and zero==0 and np.isfinite(target) and target>ZERO_TOL and np.isfinite(mineig) and mineig>=-1e-12)
    return {'COVARIANCE_STATUS':'VALID' if valid else 'INVALID_COVARIANCE','TARGET_VARIANCE':target,
            'NEGATIVE_DIAGONAL_COUNT':neg,'ZERO_DIAGONAL_COUNT':zero,'NONFINITE_COV_ELEMENTS':nonfinite,
            'MIN_COV_EIGENVALUE':mineig}


def fit_fe(sample:pd.DataFrame,outcome:str,regressors:list[str],focal:str,month_fe:bool=False,
           covariance:str='DATE_CLUSTER',bandwidth:int=0,min_etfs:int=2)->dict:
    req=['ETF_ID','DATE_GROUP',outcome,*regressors]
    s=sample.dropna(subset=req).copy().sort_values(['ETF_ID','DATE'])
    row={'MODEL_STATUS':'NOT_ESTIMATED','COEFFICIENT':np.nan,'STANDARD_ERROR':np.nan,'T_STATISTIC':np.nan,'P_VALUE':np.nan,
         'CI_LOWER_95':np.nan,'CI_UPPER_95':np.nan,'N_OBSERVATIONS':len(s),'N_ETFS':s.ETF_ID.nunique(),
         'N_DATES':s.DATE_GROUP.nunique(),'COVARIANCE_METHOD':covariance,'MONTH_FE':month_fe}
    try:
        entity,_=pd.factorize(s.ETF_ID); time,_=pd.factorize(s.DATE_GROUP)
        ne=entity.max()+1; nt=time.max()+1
        if ne<min_etfs or nt<12 or len(s)<50: raise ValueError('insufficient sample')
        allraw=s[[outcome,*regressors]].to_numpy(float)
        z,it,conv=within_transform(allraw,entity,time if month_fe else None)
        y=z[:,0]; X=z[:,1:]; target_idx=regressors.index(focal)
        keep=np.var(X,axis=0)>1e-20
        if not keep[target_idx]: raise ValueError('focal regressor has no within variation')
        if not np.all(keep):
            target_idx=int(np.sum(keep[:target_idx])); X=X[:,keep]; regressors=[r for r,k in zip(regressors,keep) if k]
        rank=np.linalg.matrix_rank(X); k=X.shape[1]
        if rank<k: raise ValueError(f'rank deficient {rank}/{k}')
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@beta; bread=np.linalg.inv(X.T@X)
        absorbed=(ne-1)+(nt-1 if month_fe else 0); k_total=k+absorbed
        if covariance=='DATE_CLUSTER': cov=cov_cluster(X,resid,time,nt,bread,k_total); df_inf=max(nt-1,1)
        elif covariance=='ETF_CLUSTER': cov=cov_cluster(X,resid,entity,ne,bread,k_total); df_inf=max(ne-1,1)
        elif covariance=='DK': cov=cov_dk(X,resid,time,nt,bread,k_total,bandwidth); df_inf=max(nt-1,1)
        else: raise ValueError(covariance)
        val=validate_cov(cov,target_idx); row.update(val)
        coef=float(beta[target_idx]); row.update({'COEFFICIENT':coef,'REGRESSORS':'|'.join(regressors),'DESIGN_RANK':rank,
            'N_REGRESSORS':k,'CONDITION_NUMBER':float(np.linalg.cond(X)),'WITHIN_ITERATIONS':it,'WITHIN_CONVERGENCE':conv})
        if val['COVARIANCE_STATUS']!='VALID': row['MODEL_STATUS']='INVALID_COVARIANCE'; return row
        se=math.sqrt(val['TARGET_VARIANCE']); t=coef/se; p=2*stats.t.sf(abs(t),df_inf); crit=stats.t.ppf(.975,df_inf)
        row.update({'STANDARD_ERROR':se,'T_STATISTIC':t,'P_VALUE':p,'CI_LOWER_95':coef-crit*se,'CI_UPPER_95':coef+crit*se,
                    'MODEL_STATUS':'RELIABLE_ESTIMATE','DF_INFERENCE':df_inf})
    except Exception as e:
        row['MODEL_STATUS']='ESTIMATION_ERROR'; row['ERROR_MESSAGE']=f'{type(e).__name__}: {e}'
    return row


def design_continuous(s:pd.DataFrame,arch:str,cpu:str='CPU_Z',vix:str='VIX_LEVEL_Z') -> tuple[list[str],str]:
    s['CPU_X_VIX']=s[cpu]*s[vix]; s['CPU_X_ARCH']=s[cpu]*s[arch]; s['VIX_X_ARCH']=s[vix]*s[arch];
    s['CPU_VIX_X_ARCH']=s['CPU_X_VIX']*s[arch]
    return [cpu,vix,'CPU_X_VIX','CPU_X_ARCH','VIX_X_ARCH','CPU_VIX_X_ARCH',*CONTROLS],'CPU_VIX_X_ARCH'


def design_binary_pair(s:pd.DataFrame,arch:str,a:str,b:str,prefix:str='PAIR') -> tuple[list[str],str]:
    ab=f'{prefix}_AB'; aa=f'{prefix}_A_X_ARCH'; bb=f'{prefix}_B_X_ARCH'; triple=f'{prefix}_AB_X_ARCH'
    s[ab]=s[a]*s[b]; s[aa]=s[a]*s[arch]; s[bb]=s[b]*s[arch]; s[triple]=s[ab]*s[arch]
    return [a,b,ab,aa,bb,triple,*CONTROLS],triple


def design_continuous_binary(s:pd.DataFrame,arch:str,cont:str,binary:str,prefix:str='CB') -> tuple[list[str],str]:
    cb=f'{prefix}_C_X_B'; ca=f'{prefix}_C_X_ARCH'; ba=f'{prefix}_B_X_ARCH'; triple=f'{prefix}_CB_X_ARCH'
    s[cb]=s[cont]*s[binary]; s[ca]=s[cont]*s[arch]; s[ba]=s[binary]*s[arch]; s[triple]=s[cb]*s[arch]
    return [cont,binary,cb,ca,ba,triple,*CONTROLS],triple


def design_single_shock(s:pd.DataFrame,arch:str,shock:str,prefix='S') -> tuple[list[str],str]:
    focal=f'{prefix}_X_ARCH'; s[focal]=s[shock]*s[arch]
    return [shock,focal,*CONTROLS],focal


def channel_sample(df:pd.DataFrame,channel:str,quality='MAIN') -> tuple[pd.DataFrame,str,str]:
    arch,flag=quality_col(channel,quality)
    s=df[df[flag].eq(1)&df[arch].notna()].copy()
    return s,arch,flag
