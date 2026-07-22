from __future__ import annotations
from pathlib import Path
import json, sys, warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0,str(Path(__file__).resolve().parent))
from q1_common import load_panel, channel_sample, fit_fe, design_continuous, design_binary_pair, PRIMARY_CHANNELS
warnings.filterwarnings('ignore')
OUT=Path('/mnt/data/q1_analysis/downside_tail_risk'); FIG=OUT/'figures'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
B=99; BLOCK=6; SEED=20260715


def add_tail_outcomes(df):
    x=df.copy(); q=x.groupby('ETF_ID').ETF_RETURN.transform(lambda s:s.quantile(.10)); q5=x.groupby('ETF_ID').ETF_RETURN.transform(lambda s:s.quantile(.05))
    x['ETF_Q10']=q; x['ETF_Q05']=q5; x['CRASH_Q10']=(x.ETF_RETURN<=q).astype(float); x['CRASH_Q05']=(x.ETF_RETURN<=q5).astype(float)
    x['DOWNSIDE_RETURN']=np.minimum(x.ETF_RETURN,0.0); x['TAIL_SHORTFALL_Q10']=np.minimum(x.ETF_RETURN-q,0.0); return x

def design(s,arch,test):
    return design_continuous(s,arch) if test=='CPU_VIX_CONTINUOUS' else design_binary_pair(s,arch,'HIGH_CPU_REGIME','HIGH_VIX_REGIME','JV')

def expected_sign(channel,outcome):
    vulnerable=channel!='INTERNAL_FINANCIAL_CAPACITY'
    return (1 if vulnerable else -1) if outcome.startswith('CRASH') else (-1 if vulnerable else 1)

def quantile_design(base,arch,test):
    s=base.copy(); regs,focal=design(s,arch,test); s=s.dropna(subset=['ETF_RETURN','DATE','ETF_ID',*regs]).copy().sort_values(['DATE','ETF_ID']).reset_index(drop=True)
    X=pd.concat([s[regs].astype(float),pd.get_dummies(s.ETF_ID,drop_first=True,dtype=float)],axis=1); X=sm.add_constant(X,has_constant='add'); return s,X,focal

def fit_quantile(s,X,focal,max_iter=5000,p_tol=1e-7):
    fit=sm.QuantReg(s.ETF_RETURN.to_numpy(float),X.to_numpy(float)).fit(q=.10,vcov='robust',max_iter=max_iter,p_tol=p_tol)
    j=list(X.columns).index(focal); ci=fit.conf_int()[j]; return float(fit.params[j]),float(fit.bse[j]),float(fit.pvalues[j]),float(ci[0]),float(ci[1])

def moving_block_indices(s,rng):
    dates=np.array(sorted(s.DATE.unique())); T=len(dates); arr=s.DATE.to_numpy(); by={d:np.flatnonzero(arr==d) for d in dates}; chosen=[]
    while len(chosen)<T:
        start=int(rng.integers(0,max(T-BLOCK+1,1))); chosen.extend(dates[start:min(start+BLOCK,T)])
    return np.concatenate([by[d] for d in chosen[:T]])

def quantile_worker(task):
    channel,test,seed=task; os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    df=load_panel(); base,arch,_=channel_sample(df,channel); s,X,focal=quantile_design(base,arch,test); coef,se,p,lo,hi=fit_quantile(s,X,focal)
    rng=np.random.default_rng(seed); y=s.ETF_RETURN.to_numpy(float); Xm=X.to_numpy(float); j=list(X.columns).index(focal); boot=np.full(B,np.nan)
    for b in range(B):
        idx=moving_block_indices(s,rng)
        try: boot[b]=sm.QuantReg(y[idx],Xm[idx]).fit(q=.10,vcov='robust',max_iter=800,p_tol=1e-4).params[j]
        except Exception: pass
    v=boot[np.isfinite(boot)]; blo,bhi=(np.quantile(v,[.025,.975]) if len(v)>20 else (np.nan,np.nan)); sign_p=2*min((1+np.sum(v<=0))/(len(v)+1),(1+np.sum(v>=0))/(len(v)+1)) if len(v) else np.nan
    row={'CHANNEL':channel,'TEST':test,'TAU':.10,'EXPECTED_SIGN':'POSITIVE' if expected_sign(channel,'QUANTILE')>0 else 'NEGATIVE','COEFFICIENT':coef,'STANDARD_ERROR':se,'P_VALUE':p,'CI_LOWER_95':lo,'CI_UPPER_95':hi,
         'BLOCK_BOOT_CI_LOWER_95':blo,'BLOCK_BOOT_CI_UPPER_95':bhi,'BLOCK_BOOT_SIGN_P':min(sign_p,1.0) if np.isfinite(sign_p) else np.nan,'N_VALID_BOOTSTRAPS':len(v),'N_OBSERVATIONS':len(s),'N_ETFS':s.ETF_ID.nunique(),'N_DATES':s.DATE.nunique()}
    dist=[{'CHANNEL':channel,'TEST':test,'REPLICATION':i+1,'BOOTSTRAP_COEFFICIENT':val} for i,val in enumerate(boot)]
    return row,dist

def sorted_tail_state_stats():
    path=Path('/mnt/data/q1_analysis/portfolio_sorts/31_architecture_sorted_portfolio_returns.csv')
    if not path.exists(): return pd.DataFrame()
    x=pd.read_csv(path,parse_dates=['DATE']); x=x[(x.SORT_METHOD=='MEDIAN_SPLIT')&(x.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')&x.CHANNEL.isin(PRIMARY_CHANNELS)]
    rows=[]
    for (ch,state),g in x.groupby(['CHANNEL','CPU_AND_VIX_STRESS']):
        for ret in ['HIGH_RETURN','LOW_RETURN','HML_RETURN']:
            v=g[ret].dropna(); q=v.quantile(.05); rows.append({'CHANNEL':ch,'STATE':'JOINT_STRESS' if state==1 else 'NON_JOINT_STRESS','RETURN_SERIES':ret,'N_MONTHS':len(v),'MEAN':v.mean(),'STD':v.std(),'VAR_5':q,'CVAR_5':v[v<=q].mean(),'DOWNSIDE_DEVIATION':np.sqrt(np.mean(np.minimum(v,0)**2))})
    return pd.DataFrame(rows)

def main():
    df=add_tail_outcomes(load_panel()); outcomes=['CRASH_Q10','CRASH_Q05','DOWNSIDE_RETURN','TAIL_SHORTFALL_Q10']; tests=['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS']; rows=[]
    for ch in PRIMARY_CHANNELS:
        base,arch,_=channel_sample(df,ch)
        for test in tests:
            z=base.copy(); regs,focal=design(z,arch,test)
            for outcome in outcomes:
                for variant in ['ETF_FE_DATE_CLUSTER','ETF_MONTH_FE']:
                    month=variant=='ETF_MONTH_FE'; cov='ETF_CLUSTER' if month else 'DATE_CLUSTER'; r=fit_fe(z,outcome,regs.copy(),focal,month_fe=month,covariance=cov)
                    rows.append({'CHANNEL':ch,'TEST':test,'OUTCOME':outcome,'MODEL_VARIANT':variant,'EXPECTED_SIGN':'POSITIVE' if expected_sign(ch,outcome)>0 else 'NEGATIVE',**r})
    # Quantile models in parallel.
    tasks=[]; i=0
    for ch in PRIMARY_CHANNELS:
        for test in tests: i+=1; tasks.append((ch,test,SEED+i*1000))
    qrows=[]; qdist=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(quantile_worker,t):t for t in tasks}
        for f in as_completed(fut):
            row,dist=f.result(); qrows.append(row); qdist.extend(dist); print('Quantile completed',row['CHANNEL'],row['TEST'],row['N_VALID_BOOTSTRAPS'])
    res=pd.DataFrame(rows); res['HOLM_P_WITHIN_OUTCOME_TEST']=np.nan; res['BH_Q_WITHIN_OUTCOME_TEST']=np.nan
    for (out,test,var),g in res.groupby(['OUTCOME','TEST','MODEL_VARIANT']):
        p=pd.to_numeric(g.P_VALUE,errors='coerce'); ok=p.notna(); idx=g.index[ok]
        if len(idx): res.loc[idx,'HOLM_P_WITHIN_OUTCOME_TEST']=multipletests(p[ok],method='holm')[1]; res.loc[idx,'BH_Q_WITHIN_OUTCOME_TEST']=multipletests(p[ok],method='fdr_bh')[1]
    res['EXPECTED_SIGN_MATCH']=np.where(res.EXPECTED_SIGN.eq('POSITIVE'),res.COEFFICIENT.gt(0),res.COEFFICIENT.lt(0)); res.to_csv(OUT/'35_all_downside_panel_results.csv',index=False)
    primary=res[res.MODEL_VARIANT.eq('ETF_FE_DATE_CLUSTER')].copy(); primary.to_csv(OUT/'35_primary_downside_results.csv',index=False)
    qr=pd.DataFrame(qrows).sort_values(['TEST','CHANNEL']); qr['HOLM_P_WITHIN_TEST']=np.nan; qr['BH_Q_WITHIN_TEST']=np.nan
    for test,g in qr.groupby('TEST'):
        idx=g.index; qr.loc[idx,'HOLM_P_WITHIN_TEST']=multipletests(g.P_VALUE,method='holm')[1]; qr.loc[idx,'BH_Q_WITHIN_TEST']=multipletests(g.P_VALUE,method='fdr_bh')[1]
    qr.to_csv(OUT/'35_quantile_regression_results.csv',index=False); pd.DataFrame(qdist).to_csv(OUT/'35_quantile_block_bootstrap_distributions.csv',index=False)
    state=sorted_tail_state_stats(); state.to_csv(OUT/'35_sorted_portfolio_tail_state_statistics.csv',index=False)
    labels={'INTERNAL_FINANCIAL_CAPACITY':'Internal capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing','GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration','PORTFOLIO_CONCENTRATION_FINAL':'Concentration'}
    for test in tests:
        g=primary[(primary.TEST==test)&(primary.OUTCOME=='CRASH_Q10')].copy(); g['LABEL']=g.CHANNEL.map(labels); g=g.sort_values('COEFFICIENT'); fig,ax=plt.subplots(figsize=(8,4.8)); y=np.arange(len(g))
        ax.errorbar(100*g.COEFFICIENT,y,xerr=[100*(g.COEFFICIENT-g.CI_LOWER_95),100*(g.CI_UPPER_95-g.COEFFICIENT)],fmt='o',capsize=3); ax.axvline(0,linewidth=1); ax.set_yticks(y,g.LABEL)
        ax.set_xlabel('Change in bottom-decile event probability (percentage points)'); ax.set_title(f'Downside-event activation: {test.replace("_"," ").title()}'); ax.grid(axis='x',alpha=.25); fig.tight_layout()
        stem=test.lower(); fig.savefig(FIG/f'35_crash_probability_{stem}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'35_crash_probability_{stem}.pdf',bbox_inches='tight'); plt.close(fig)
    meta={'crash_thresholds':'ETF-specific full-sample 10th and 5th percentiles; distributional classification, not an ex-ante forecast','panel_outcomes':['crash probability LPM','negative semireturn','tail shortfall below ETF q10'],
          'quantile':'pooled tau=0.10 quantile regression with ETF fixed-effect dummies','quantile_bootstrap':f'{B} moving six-month blocks, seed family {SEED}','primary_inference':'ETF FE/date cluster; ETF+month FE/ETF cluster robustness'}
    (OUT/'35_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nPRIMARY DOWNSIDE\n',primary[['CHANNEL','TEST','OUTCOME','COEFFICIENT','P_VALUE','HOLM_P_WITHIN_OUTCOME_TEST','BH_Q_WITHIN_OUTCOME_TEST','EXPECTED_SIGN_MATCH']].to_string(index=False))
    print('\nQUANTILE RESULTS\n',qr.to_string(index=False))
if __name__=='__main__': main()
