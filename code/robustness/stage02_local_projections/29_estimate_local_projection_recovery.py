from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

SOURCE = Path('/mnt/data/cpu_extracted/CPU_Project/output/25_final_econometric_panel.csv')
OUT = Path('/mnt/data/q1_analysis/local_projections')
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CONTROLS = ['MARKET_RETURN','ENERGY_RETURN','TREASURY_RETURN','VIX_CHANGE']
HORIZONS = list(range(13))
MIN_ETFS_MAIN = 10
MIN_OBS = 50
ZERO_TOL = 1e-14

CHANNELS = {
    'INTERNAL_FINANCIAL_CAPACITY':'PRIMARY',
    'EXTERNAL_FINANCING_DEPENDENCE':'PRIMARY',
    'GROWTH_DURATION_EXPOSURE_FINAL':'PRIMARY',
    'PORTFOLIO_CONCENTRATION_FINAL':'PRIMARY',
    'FINANCIAL_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS',
    'EXTENDED_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS',
    'PCA_COMPONENT_1':'PCA_ROBUSTNESS',
}

SHOCKS = {
    'CPU_VIX_CONTINUOUS': {
        'role':'PRIMARY_CONTINUOUS_ACTIVATION',
        'focal':'CPU_VIX_X_ARCH',
    },
    'CPU_AND_VIX_STRESS': {
        'role':'PRIMARY_THRESHOLD_ACTIVATION',
        'focal':'JOINT_X_ARCH',
    },
    'EXTREME_CPU_REGIME': {
        'role':'PREDEFINED_CPU_EVENT',
        'focal':'SHOCK_X_ARCH',
    },
    'CPU_SHOCK': {
        'role':'EXTERNAL_SHOCK_VALIDATION',
        'focal':'SHOCK_X_ARCH',
    },
}

MODEL_VARIANTS = ['PRIMARY_ALIGNED','MONTH_FE_ROBUSTNESS','LAG_AUGMENTED_ROBUSTNESS']


def quality_col(channel: str) -> tuple[str,str|None]:
    if channel == 'PCA_COMPONENT_1':
        return channel, 'VALID_PCA_MAIN_ROW'
    return f'{channel}_MAIN', f'VALID_{channel}_MAIN_ROW'


def required_columns() -> list[str]:
    cols = {
        'DATE','DATE_GROUP','ETF_ID','ETF_TICKER','ETF_RETURN',
        'CPU_Z','VIX_LEVEL_Z',
        'HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS',
        'EXTREME_CPU_REGIME','CPU_SHOCK',*CONTROLS,
    }
    for ch in CHANNELS:
        c,f=quality_col(ch); cols.add(c)
        if f: cols.add(f)
    return sorted(cols)


def group_demean(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    sums=np.zeros((n_groups,values.shape[1]),float)
    counts=np.zeros(n_groups,float)
    np.add.at(sums,codes,values); np.add.at(counts,codes,1.0)
    return values - (sums/np.maximum(counts[:,None],1.0))[codes]


def group_means(values: np.ndarray,codes:np.ndarray,n_groups:int)->np.ndarray:
    sums=np.zeros((n_groups,values.shape[1]),float); counts=np.zeros(n_groups,float)
    np.add.at(sums,codes,values); np.add.at(counts,codes,1.0)
    return sums/np.maximum(counts[:,None],1.0)


def within_transform(values: np.ndarray, entity: np.ndarray, time: np.ndarray|None=None,
                     tol:float=1e-11,max_iter:int=1000)->tuple[np.ndarray,int,float]:
    z=np.asarray(values,float).copy(); ne=int(entity.max())+1
    if time is None:
        z=group_demean(z,entity,ne)
        return z,1,float(np.max(np.abs(group_means(z,entity,ne))))
    nt=int(time.max())+1; conv=np.inf
    for it in range(1,max_iter+1):
        old=z.copy(); z=group_demean(z,entity,ne); z=group_demean(z,time,nt)
        conv=float(np.max(np.abs(z-old)))
        if conv<tol: return z,it,conv
    return z,max_iter,conv


def cluster_meat(scores,codes,n_groups):
    s=np.zeros((n_groups,scores.shape[1]),float); np.add.at(s,codes,scores)
    return s.T@s


def finite_sample_factor(n,k_total,g):
    if g<=1: return np.nan
    return (g/(g-1))*((n-1)/max(n-k_total,1))


def cov_cluster(X,resid,codes,n_groups,bread,k_total):
    meat=cluster_meat(X*resid[:,None],codes,n_groups)
    return bread@meat@bread*finite_sample_factor(len(X),k_total,n_groups)


def cov_dk(X,resid,time,nt,bread,k_total,bandwidth):
    scores=X*resid[:,None]; st=np.zeros((nt,scores.shape[1]),float); np.add.at(st,time,scores)
    meat=st.T@st
    for lag in range(1,min(int(bandwidth),nt-1)+1):
        w=1.0-lag/(bandwidth+1.0); gamma=st[lag:].T@st[:-lag]
        meat += w*(gamma+gamma.T)
    return bread@meat@bread*finite_sample_factor(len(X),k_total,nt)


def validate_cov(cov,target_idx):
    diag=np.diag(cov) if cov.ndim==2 and cov.shape[0]==cov.shape[1] else np.array([])
    nonfinite=int((~np.isfinite(cov)).sum()) if cov.size else 1
    neg=int((diag<0).sum()) if diag.size else 1
    zero=int((np.abs(diag)<=ZERO_TOL).sum()) if diag.size else 1
    target=float(diag[target_idx]) if diag.size>target_idx else np.nan
    try: mineig=float(np.linalg.eigvalsh((cov+cov.T)/2).min())
    except Exception: mineig=np.nan
    valid=(nonfinite==0 and neg==0 and zero==0 and np.isfinite(target) and target>ZERO_TOL and np.isfinite(mineig) and mineig>=-1e-12)
    return {
        'COVARIANCE_STATUS':'VALID' if valid else 'INVALID_COVARIANCE',
        'NONFINITE_COV_ELEMENTS':nonfinite,'NEGATIVE_DIAGONAL_COUNT':neg,
        'ZERO_DIAGONAL_COUNT':zero,'TARGET_VARIANCE':target,
        'MIN_COV_EIGENVALUE':mineig,'PSD_WARNING':bool(not np.isfinite(mineig) or mineig<-1e-12),
    }


def add_dynamic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df=df.sort_values(['ETF_ID','DATE']).copy()
    # Preserve only contiguous monthly observations for lags and forward outcomes.
    df['_PERIOD']=df['DATE'].dt.to_period('M')
    df['_PERIOD_ORD']=df['DATE'].dt.year*12+df['DATE'].dt.month
    df['ETF_RETURN_L1']=df.groupby('ETF_ID')['ETF_RETURN'].shift(1)
    prev_period=df.groupby('ETF_ID')['_PERIOD_ORD'].shift(1)
    lag_contiguous=(df['_PERIOD_ORD']-prev_period).eq(1)
    df.loc[~lag_contiguous.fillna(False),'ETF_RETURN_L1']=np.nan

    common=['CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z','HIGH_CPU_REGIME','HIGH_VIX_REGIME',
            'CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME','CPU_SHOCK']
    for c in common:
        df[f'{c}_L1']=df.groupby('ETF_ID')[c].shift(1)
        df.loc[~lag_contiguous.fillna(False),f'{c}_L1']=np.nan

    # Cumulative simple ETF returns from event month t through t+h.
    outcomes=[]
    for etf,g in df.groupby('ETF_ID',sort=False):
        idx=g.index.to_numpy(); r=g['ETF_RETURN'].to_numpy(float); per=g['_PERIOD_ORD'].to_numpy(int)
        n=len(g)
        out={h:np.full(n,np.nan) for h in HORIZONS}
        for i in range(n):
            for h in HORIZONS:
                j=i+h
                if j>=n: continue
                if per[j]-per[i] != h: continue
                window=r[i:j+1]
                if np.all(np.isfinite(window)):
                    out[h][i]=np.prod(1.0+window)-1.0
        for h in HORIZONS:
            df.loc[idx,f'LP_CUM_RETURN_H{h}']=out[h]
            outcomes.append(f'LP_CUM_RETURN_H{h}')
    return df.drop(columns=['_PERIOD','_PERIOD_ORD'])


def interaction_design(s:pd.DataFrame,arch:str,shock:str,variant:str)->tuple[list[str],str]:
    # Returns regressors and focal name. Hierarchical terms are preserved.
    if shock=='CPU_VIX_CONTINUOUS':
        s['CPU_X_ARCH']=s['CPU_Z']*s[arch]
        s['VIX_X_ARCH']=s['VIX_LEVEL_Z']*s[arch]
        s['CPU_VIX_X_ARCH']=s['CPU_Z_X_VIX_Z']*s[arch]
        focal='CPU_VIX_X_ARCH'
        if variant=='MONTH_FE_ROBUSTNESS':
            regs=['CPU_X_ARCH','VIX_X_ARCH','CPU_VIX_X_ARCH']
        else:
            regs=['CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z','CPU_X_ARCH','VIX_X_ARCH','CPU_VIX_X_ARCH',*CONTROLS]
        if variant=='LAG_AUGMENTED_ROBUSTNESS':
            s['CPU_L1_X_ARCH']=s['CPU_Z_L1']*s[arch]
            s['VIX_L1_X_ARCH']=s['VIX_LEVEL_Z_L1']*s[arch]
            s['CPU_VIX_L1_X_ARCH']=s['CPU_Z_X_VIX_Z_L1']*s[arch]
            regs += ['ETF_RETURN_L1','CPU_Z_L1','VIX_LEVEL_Z_L1','CPU_Z_X_VIX_Z_L1',
                     'CPU_L1_X_ARCH','VIX_L1_X_ARCH','CPU_VIX_L1_X_ARCH']
        return regs,focal

    if shock=='CPU_AND_VIX_STRESS':
        s['HIGH_CPU_X_ARCH']=s['HIGH_CPU_REGIME']*s[arch]
        s['HIGH_VIX_X_ARCH']=s['HIGH_VIX_REGIME']*s[arch]
        s['JOINT_X_ARCH']=s['CPU_AND_VIX_STRESS']*s[arch]
        focal='JOINT_X_ARCH'
        if variant=='MONTH_FE_ROBUSTNESS':
            regs=['HIGH_CPU_X_ARCH','HIGH_VIX_X_ARCH','JOINT_X_ARCH']
        else:
            regs=['HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS',
                  'HIGH_CPU_X_ARCH','HIGH_VIX_X_ARCH','JOINT_X_ARCH',*CONTROLS]
        if variant=='LAG_AUGMENTED_ROBUSTNESS':
            s['HIGH_CPU_L1_X_ARCH']=s['HIGH_CPU_REGIME_L1']*s[arch]
            s['HIGH_VIX_L1_X_ARCH']=s['HIGH_VIX_REGIME_L1']*s[arch]
            s['JOINT_L1_X_ARCH']=s['CPU_AND_VIX_STRESS_L1']*s[arch]
            regs += ['ETF_RETURN_L1','HIGH_CPU_REGIME_L1','HIGH_VIX_REGIME_L1','CPU_AND_VIX_STRESS_L1',
                     'HIGH_CPU_L1_X_ARCH','HIGH_VIX_L1_X_ARCH','JOINT_L1_X_ARCH']
        return regs,focal

    base=shock
    s['SHOCK_X_ARCH']=s[base]*s[arch]
    focal='SHOCK_X_ARCH'
    if variant=='MONTH_FE_ROBUSTNESS':
        regs=['SHOCK_X_ARCH']
    else:
        regs=[base,'SHOCK_X_ARCH',*CONTROLS]
    if variant=='LAG_AUGMENTED_ROBUSTNESS':
        s['SHOCK_L1_X_ARCH']=s[f'{base}_L1']*s[arch]
        regs += ['ETF_RETURN_L1',f'{base}_L1','SHOCK_L1_X_ARCH']
    return regs,focal


def build_sample(df,channel,shock,h,variant):
    arch,flag=quality_col(channel); m=df[arch].notna()
    if flag in df: m &= df[flag].eq(1)
    basecols=['ETF_ID','ETF_TICKER','DATE','DATE_GROUP',f'LP_CUM_RETURN_H{h}',arch,
              'CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z','HIGH_CPU_REGIME','HIGH_VIX_REGIME',
              'CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME','CPU_SHOCK',*CONTROLS,
              'ETF_RETURN_L1','CPU_Z_L1','VIX_LEVEL_Z_L1','CPU_Z_X_VIX_Z_L1',
              'HIGH_CPU_REGIME_L1','HIGH_VIX_REGIME_L1','CPU_AND_VIX_STRESS_L1',
              'EXTREME_CPU_REGIME_L1','CPU_SHOCK_L1']
    s=df.loc[m,[c for c in basecols if c in df.columns]].copy()
    regs,focal=interaction_design(s,arch,shock,variant)
    req=['ETF_ID','DATE','DATE_GROUP',f'LP_CUM_RETURN_H{h}',*regs]
    for c in req[3:]: s[c]=pd.to_numeric(s[c],errors='coerce')
    s=s.dropna(subset=req).drop_duplicates(['ETF_ID','DATE']).sort_values(['ETF_ID','DATE']).reset_index(drop=True)
    return s,arch,flag,regs,focal


def estimate_one(df,channel,role,shock,h,variant):
    row={'CHANNEL':channel,'CHANNEL_ROLE':role,'SHOCK':shock,'SHOCK_ROLE':SHOCKS[shock]['role'],
         'HORIZON_MONTHS':h,'OUTCOME':f'LP_CUM_RETURN_H{h}','MODEL_VARIANT':variant,
         'MODEL_STATUS':'NOT_ESTIMATED','COEFFICIENT':np.nan,'STANDARD_ERROR':np.nan,'T_STATISTIC':np.nan,
         'P_VALUE':np.nan,'CI_LOWER_95':np.nan,'CI_UPPER_95':np.nan,'N_OBSERVATIONS':0,'N_ETFS':0,
         'N_DATES':0,'ERROR_MESSAGE':''}
    try:
        s,arch,flag,regs,focal=build_sample(df,channel,shock,h,variant)
        n=len(s); entity,elabel=pd.factorize(s.ETF_ID); time,tlabel=pd.factorize(s.DATE_GROUP)
        ne=len(elabel); nt=len(tlabel); month_fe=(variant=='MONTH_FE_ROBUSTNESS')
        row.update({'ARCHITECTURE_COLUMN':arch,'VALID_ROW_FLAG':flag,'FOCAL_REGRESSOR':focal,
                    'N_OBSERVATIONS':n,'N_ETFS':ne,'N_DATES':nt,
                    'INFERENCE_CLASS':'MAIN_INFERENCE_ELIGIBLE' if ne>=MIN_ETFS_MAIN else 'EXPLORATORY_SMALL_ETF_CROSS_SECTION'})
        if n<MIN_OBS or ne<2 or nt<12: raise ValueError(f'insufficient sample N={n}, ETFs={ne}, dates={nt}')
        yraw=s[[f'LP_CUM_RETURN_H{h}']].to_numpy(float); Xraw=s[regs].to_numpy(float)
        target_idx=regs.index(focal); allraw=np.column_stack([yraw,Xraw])
        transformed,it,conv=within_transform(allraw,entity,time if month_fe else None)
        y=transformed[:,0]; X=transformed[:,1:]
        keep=np.var(X,axis=0)>1e-20
        if not keep[target_idx]: raise ValueError('focal regressor has no within variation')
        if not np.all(keep):
            target_idx=int(np.sum(keep[:target_idx])); X=X[:,keep]; regs=[r for r,k in zip(regs,keep) if k]
        rank=np.linalg.matrix_rank(X); k=X.shape[1]
        if rank<k: raise ValueError(f'rank deficient design rank={rank}, k={k}')
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@beta; bread=np.linalg.inv(X.T@X)
        absorbed=(ne-1)+(nt-1 if month_fe else 0); k_total=k+absorbed
        if h==0:
            cov=cov_cluster(X,resid,time,nt,bread,k_total); method='DATE_CLUSTERED'; bw=0
        else:
            bw=h; cov=cov_dk(X,resid,time,nt,bread,k_total,bw); method=f'DK_H{h}'
        val=validate_cov(cov,target_idx); row.update(val)
        coef=float(beta[target_idx]); row.update({'COEFFICIENT':coef,'FINITE_COEFFICIENT':bool(np.isfinite(coef)),
            'COVARIANCE_METHOD':method,'DK_BANDWIDTH':bw,'DF_INFERENCE':max(nt-1,1),
            'WITHIN_ITERATIONS':it,'WITHIN_CONVERGENCE':conv,'DESIGN_RANK':rank,'N_REGRESSORS':k,
            'ABSORBED_EFFECT_DF':absorbed,'CONDITION_NUMBER':float(np.linalg.cond(X)),
            'WITHIN_R_SQUARED':float(1-(resid@resid)/(y@y)) if y@y>0 else np.nan,
            'REGRESSORS':'|'.join(regs)})
        if val['COVARIANCE_STATUS']!='VALID':
            row['MODEL_STATUS']='INVALID_COVARIANCE'; row['ERROR_MESSAGE']='Invalid covariance; no p-value reported.'; return row
        se=math.sqrt(val['TARGET_VARIANCE']); t=coef/se; df_inf=max(nt-1,1); p=2*stats.t.sf(abs(t),df_inf); crit=stats.t.ppf(.975,df_inf)
        row.update({'STANDARD_ERROR':se,'T_STATISTIC':t,'P_VALUE':p,'CI_LOWER_95':coef-crit*se,
                    'CI_UPPER_95':coef+crit*se,'MODEL_STATUS':'RELIABLE_ESTIMATE' if ne>=MIN_ETFS_MAIN else 'EXPLORATORY_RELIABLE_ESTIMATE'})
        return row
    except Exception as e:
        row['MODEL_STATUS']='ESTIMATION_ERROR'; row['ERROR_MESSAGE']=f'{type(e).__name__}: {e}'; return row


def add_multiplicity(primary: pd.DataFrame) -> pd.DataFrame:
    x=primary.copy(); x['HOLM_P_SHOCK_FAMILY']=np.nan; x['BH_Q_SHOCK_FAMILY']=np.nan
    x['HOLM_P_WITHIN_CHANNEL']=np.nan; x['BH_Q_WITHIN_CHANNEL']=np.nan
    x['BONFERRONI_CI_LOWER']=np.nan; x['BONFERRONI_CI_UPPER']=np.nan
    for shock,g in x.groupby('SHOCK'):
        idx=g.index; p=g.P_VALUE.to_numpy(float); valid=np.isfinite(p)
        if valid.any():
            x.loc[idx[valid],'HOLM_P_SHOCK_FAMILY']=multipletests(p[valid],method='holm')[1]
            x.loc[idx[valid],'BH_Q_SHOCK_FAMILY']=multipletests(p[valid],method='fdr_bh')[1]
    for (shock,ch),g in x.groupby(['SHOCK','CHANNEL']):
        idx=g.index; p=g.P_VALUE.to_numpy(float); valid=np.isfinite(p)
        if valid.any():
            x.loc[idx[valid],'HOLM_P_WITHIN_CHANNEL']=multipletests(p[valid],method='holm')[1]
            x.loc[idx[valid],'BH_Q_WITHIN_CHANNEL']=multipletests(p[valid],method='fdr_bh')[1]
        df_inf=float(g.DF_INFERENCE.dropna().min()) if g.DF_INFERENCE.notna().any() else 100
        crit=stats.t.ppf(1-0.05/(2*max(valid.sum(),1)),df_inf)
        x.loc[idx,'BONFERRONI_CI_LOWER']=g.COEFFICIENT-crit*g.STANDARD_ERROR
        x.loc[idx,'BONFERRONI_CI_UPPER']=g.COEFFICIENT+crit*g.STANDARD_ERROR
    return x


def dynamic_summary(primary:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for (shock,ch),g in primary.groupby(['SHOCK','CHANNEL']):
        g=g.sort_values('HORIZON_MONTHS'); ok=g[g.MODEL_STATUS=='RELIABLE_ESTIMATE']
        if ok.empty: continue
        trough=ok.loc[ok.COEFFICIENT.idxmin()]; peak=ok.loc[ok.COEFFICIENT.idxmax()]
        h0=ok[ok.HORIZON_MONTHS==0]; h12=ok[ok.HORIZON_MONTHS==12]
        c0=float(h0.COEFFICIENT.iloc[0]) if len(h0) else np.nan; c12=float(h12.COEFFICIENT.iloc[0]) if len(h12) else np.nan
        crossing=np.nan
        if np.isfinite(c0) and c0<0:
            later=ok[(ok.HORIZON_MONTHS>0)&(ok.COEFFICIENT>=0)]
            if not later.empty: crossing=int(later.HORIZON_MONTHS.iloc[0])
        if np.isfinite(c0) and c0<0 and np.isfinite(crossing): pattern='NEGATIVE_IMPACT_THEN_REBOUND'
        elif np.isfinite(c0) and c0<0 and np.isfinite(c12) and c12<0: pattern='PERSISTENT_CUMULATIVE_DOWNSIDE'
        elif np.isfinite(c0) and c0>=0 and float(trough.COEFFICIENT)<0: pattern='DELAYED_DOWNSIDE'
        else: pattern='NO_CLEAR_UNIDIRECTIONAL_PATTERN'
        rows.append({'SHOCK':shock,'CHANNEL':ch,'CHANNEL_ROLE':ok.CHANNEL_ROLE.iloc[0],
                     'N_ETFS_MIN':int(ok.N_ETFS.min()),'N_DATES_MIN':int(ok.N_DATES.min()),
                     'H0_COEFFICIENT':c0,'TROUGH_HORIZON':int(trough.HORIZON_MONTHS),'TROUGH_COEFFICIENT':float(trough.COEFFICIENT),
                     'PEAK_HORIZON':int(peak.HORIZON_MONTHS),'PEAK_COEFFICIENT':float(peak.COEFFICIENT),
                     'H12_COEFFICIENT':c12,'FIRST_NONNEGATIVE_HORIZON_AFTER_NEGATIVE_H0':crossing,
                     'N_RAW_P_LT_05':int((ok.P_VALUE<.05).sum()),'N_BH_Q_WITHIN_CHANNEL_LT_10':int((ok.BH_Q_WITHIN_CHANNEL<.10).sum()),
                     'ANY_BONFERRONI_BAND_EXCLUDES_ZERO':bool(((ok.BONFERRONI_CI_LOWER>0)|(ok.BONFERRONI_CI_UPPER<0)).any()),
                     'DYNAMIC_PATTERN':pattern})
    return pd.DataFrame(rows)


def make_figures(primary:pd.DataFrame):
    import matplotlib.pyplot as plt
    label_map={
        'INTERNAL_FINANCIAL_CAPACITY':'Internal financial capacity',
        'EXTERNAL_FINANCING_DEPENDENCE':'External financing dependence',
        'GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration exposure',
        'PORTFOLIO_CONCENTRATION_FINAL':'Portfolio concentration',
    }
    shock_map={
        'CPU_VIX_CONTINUOUS':'Continuous CPU × VIX activation',
        'CPU_AND_VIX_STRESS':'Joint CPU–VIX stress',
        'EXTREME_CPU_REGIME':'Extreme CPU regime',
    }
    for shock in shock_map:
        for ch in label_map:
            g=primary[(primary.SHOCK==shock)&(primary.CHANNEL==ch)&(primary.MODEL_STATUS=='RELIABLE_ESTIMATE')].sort_values('HORIZON_MONTHS')
            if g.empty: continue
            fig=plt.figure(figsize=(7.2,4.6)); ax=fig.add_subplot(111)
            x=g.HORIZON_MONTHS.to_numpy(); y=100*g.COEFFICIENT.to_numpy(); lo=100*g.CI_LOWER_95.to_numpy(); hi=100*g.CI_UPPER_95.to_numpy()
            ax.plot(x,y,marker='o',linewidth=1.7)
            ax.fill_between(x,lo,hi,alpha=.18)
            ax.axhline(0,linewidth=1)
            ax.set_xlabel('Months from shock')
            ax.set_ylabel('Cumulative return differential (percentage points)')
            ax.set_title(f"{label_map[ch]}: {shock_map[shock]}")
            ax.set_xticks(range(13)); ax.grid(axis='y',alpha=.25)
            fig.tight_layout()
            fig.savefig(FIG/f"29_lp_{shock.lower()}_{ch.lower()}.png",dpi=300,bbox_inches='tight')
            fig.savefig(FIG/f"29_lp_{shock.lower()}_{ch.lower()}.pdf",bbox_inches='tight')
            plt.close(fig)


def main():
    df=pd.read_csv(SOURCE,usecols=lambda c:c in set(required_columns()),low_memory=False,parse_dates=['DATE'])
    if 'DATE_GROUP' not in df: df['DATE_GROUP']=df.DATE.dt.to_period('M').astype(str)
    df['CPU_Z_X_VIX_Z']=pd.to_numeric(df['CPU_Z'],errors='coerce')*pd.to_numeric(df['VIX_LEVEL_Z'],errors='coerce')
    for c in required_columns():
        if c in df.columns and c not in ['DATE','DATE_GROUP','ETF_ID','ETF_TICKER']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
    if df.duplicated(['ETF_ID','DATE']).any(): raise RuntimeError('duplicate ETF-month rows')
    df=add_dynamic_columns(df)
    outcome_diag=[]
    for h in HORIZONS:
        c=f'LP_CUM_RETURN_H{h}'
        outcome_diag.append({'HORIZON_MONTHS':h,'N_VALID':int(df[c].notna().sum()),'MEAN':df[c].mean(),'STD':df[c].std(),'MIN':df[c].min(),'MAX':df[c].max()})
    pd.DataFrame(outcome_diag).to_csv(OUT/'29_forward_cumulative_return_diagnostics.csv',index=False)

    rows=[]; total=len(CHANNELS)*len(SHOCKS)*len(HORIZONS)*len(MODEL_VARIANTS); done=0
    for ch,role in CHANNELS.items():
        for shock in SHOCKS:
            for h in HORIZONS:
                for variant in MODEL_VARIANTS:
                    done+=1
                    if done%50==0 or done==1: print(f'[{done}/{total}] {ch} | {shock} | h={h} | {variant}')
                    rows.append(estimate_one(df,ch,role,shock,h,variant))
    res=pd.DataFrame(rows)
    res.to_csv(OUT/'29_all_local_projection_results.csv',index=False)
    res[res.MODEL_STATUS.str.contains('RELIABLE',na=False)].to_csv(OUT/'29_reliable_local_projection_results.csv',index=False)
    invalid=res[~res.MODEL_STATUS.str.contains('RELIABLE',na=False)]
    invalid.to_csv(OUT/'29_invalid_or_failed_local_projection_models.csv',index=False)
    primary=res[(res.MODEL_VARIANT=='PRIMARY_ALIGNED')&(res.MODEL_STATUS=='RELIABLE_ESTIMATE')].copy()
    primary=add_multiplicity(primary)
    primary.to_csv(OUT/'29_primary_local_projection_results_with_multiplicity.csv',index=False)
    summ=dynamic_summary(primary); summ.to_csv(OUT/'29_local_projection_dynamic_summary.csv',index=False)
    validation=(res.groupby(['MODEL_VARIANT','MODEL_STATUS']).size().rename('N_MODELS').reset_index())
    validation.to_csv(OUT/'29_local_projection_validation_summary.csv',index=False)
    make_figures(primary)
    metadata={'source':str(SOURCE),'horizons':HORIZONS,'outcome':'cumulative simple ETF return from t through t+h',
              'primary_model':'RQ3-aligned ETF fixed-effects specification with hierarchical lower-order interactions and market controls',
              'robustness':['ETF and month fixed effects','one-lag augmented specification'],
              'inference':'date-clustered at h=0; grouped-score Driscoll-Kraay Bartlett bandwidth h for h>=1',
              'negative_variances_clipped':False,'main_etf_threshold':MIN_ETFS_MAIN,
              'multiple_testing':['Holm and BH across primary shock family','Holm and BH within channel across horizons','Bonferroni simultaneous bands within channel']}
    (OUT/'29_local_projection_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nMODEL STATUS\n',res.MODEL_STATUS.value_counts().to_string())
    print('\nPRIMARY DYNAMIC SUMMARY\n',summ[summ.CHANNEL_ROLE=='PRIMARY'].to_string(index=False))

if __name__=='__main__': main()
