from __future__ import annotations

from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
from q1_common import load_panel, channel_sample, fit_fe, design_continuous, design_binary_pair, PRIMARY_CHANNELS, group_demean, CONTROLS

OUT=Path('/mnt/data/q1_analysis/permutation_inference'); FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
B=5000; SEED=20260715
EXPECTED={'INTERNAL_FINANCIAL_CAPACITY':1,'EXTERNAL_FINANCING_DEPENDENCE':-1,'GROWTH_DURATION_EXPOSURE_FINAL':-1,'PORTFOLIO_CONCENTRATION_FINAL':-1}
TESTS=['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS']


def balanced_dates(s):
    ids=s.ETF_ID.unique(); counts=s.groupby('DATE').ETF_ID.nunique(); dates=counts[counts.eq(len(ids))].index
    return s[s.DATE.isin(dates)].copy()


def design_for_robust(s,arch,test):
    if test=='CPU_VIX_CONTINUOUS': return design_continuous(s,arch)
    return design_binary_pair(s,arch,'HIGH_CPU_REGIME','HIGH_VIX_REGIME','JV')


def prepare_numpy(s,arch,test):
    z=s.copy()
    if test=='CPU_VIX_CONTINUOUS':
        z['_AB']=z.CPU_Z*z.VIX_LEVEL_Z
        common=['CPU_Z','VIX_LEVEL_Z','_AB',*CONTROLS]
        exposure=['CPU_Z','VIX_LEVEL_Z','_AB']
    else:
        z['_AB']=z.HIGH_CPU_REGIME*z.HIGH_VIX_REGIME
        common=['HIGH_CPU_REGIME','HIGH_VIX_REGIME','_AB',*CONTROLS]
        exposure=['HIGH_CPU_REGIME','HIGH_VIX_REGIME','_AB']
    z=z.dropna(subset=['ETF_RETURN','ETF_ID',arch,*common]).copy().sort_values(['ETF_ID','DATE'])
    entity,labels=pd.factorize(z.ETF_ID); ne=len(labels); n=len(z)
    y=z.ETF_RETURN.to_numpy(float)[:,None]; C=z[common].to_numpy(float); E=z[exposure].to_numpy(float)
    yw=group_demean(y,entity,ne)[:,0]; Cw=group_demean(C,entity,ne); Ew=group_demean(E,entity,ne)
    amap=z[['ETF_ID',arch]].drop_duplicates().set_index('ETF_ID')[arch]
    avals=amap.reindex(labels).to_numpy(float)
    return z,entity,labels,yw,Cw,Ew,avals


def coefficient_fast(entity,yw,Cw,Ew,arch_entity):
    A=arch_entity[entity]
    X=np.column_stack([Cw,Ew*A[:,None]])
    if np.linalg.matrix_rank(X)<X.shape[1]: return np.nan
    return float(np.linalg.lstsq(X,yw,rcond=None)[0][-1])


def rw_stepdown(observed, perm_matrix):
    obs=np.asarray(observed,float); P=np.asarray(perm_matrix,float)
    center=np.nanmean(P,axis=0); scale=np.nanstd(P,axis=0,ddof=1); scale=np.where(scale>0,scale,np.nan)
    t_obs=np.abs((obs-center)/scale); t_perm=np.abs((P-center)/scale)
    order=np.argsort(-t_obs); adj=np.full(len(obs),np.nan); prev=0.0; remaining=list(order)
    for j in order:
        max_stats=np.nanmax(t_perm[:,remaining],axis=1)
        p=(1+np.sum(max_stats>=t_obs[j]))/(len(max_stats)+1); p=max(p,prev)
        adj[j]=p; prev=p; remaining.remove(j)
    return adj,t_obs


def run_perms(s,arch,test,rng):
    z,entity,labels,yw,Cw,Ew,avals=prepare_numpy(s,arch,test)
    observed=coefficient_fast(entity,yw,Cw,Ew,avals)
    P=np.empty(B,float)
    for b in range(B): P[b]=coefficient_fast(entity,yw,Cw,Ew,rng.permutation(avals))
    return observed,P,len(z),len(labels),z.DATE.nunique()


def main():
    # Clean stale partial files from interrupted runs.
    for f in OUT.glob('33_*'): f.unlink(missing_ok=True)
    for f in FIG.glob('33_*'): f.unlink(missing_ok=True)
    df=load_panel(); rng=np.random.default_rng(SEED)
    summary=[]; dist_rows=[]; family_store={t:[] for t in TESTS}; obs_store={t:[] for t in TESTS}; labels_store={t:[] for t in TESTS}
    for test in TESTS:
        for ch in PRIMARY_CHANNELS:
            print(test,ch)
            s,arch,flag=channel_sample(df,ch)
            zz=s.copy(); regs,focal=design_for_robust(zz,arch,test); robust=fit_fe(zz,'ETF_RETURN',regs,focal,covariance='DATE_CLUSTER')
            observed,P,nobs,netf,ndates=run_perms(s,arch,test,rng); valid=P[np.isfinite(P)]; center=valid.mean()
            two=(1+np.sum(np.abs(valid-center)>=abs(observed-center)))/(len(valid)+1)
            one=(1+np.sum(valid<=observed))/(len(valid)+1) if EXPECTED[ch]<0 else (1+np.sum(valid>=observed))/(len(valid)+1)
            summary.append({'CHANNEL':ch,'TEST':test,'SAMPLE_DESIGN':'AVAILABLE_CROSS_SECTION','OBSERVED_COEFFICIENT':observed,'DATE_CLUSTERED_P_VALUE':robust['P_VALUE'],
                            'N_OBSERVATIONS':nobs,'N_ETFS':netf,'N_DATES':ndates,'N_PERMUTATIONS_REQUESTED':B,'N_VALID_PERMUTATIONS':len(valid),
                            'PERMUTATION_MEAN':center,'PERMUTATION_STD':valid.std(ddof=1),'TWO_SIDED_PERMUTATION_P':two,'THEORY_DIRECTED_PERMUTATION_P':one,
                            'OBSERVED_PERCENTILE':100*np.mean(valid<=observed),'EXPECTED_SIGN':'POSITIVE' if EXPECTED[ch]>0 else 'NEGATIVE'})
            family_store[test].append(P); obs_store[test].append(observed); labels_store[test].append(ch)
            dist_rows.extend({'CHANNEL':ch,'TEST':test,'REPLICATION':i+1,'PERMUTED_COEFFICIENT':v} for i,v in enumerate(P))
            bal=balanced_dates(s)
            if bal.DATE.nunique()>=24:
                ob,Pb,no,ne,nd=run_perms(bal,arch,test,rng); vb=Pb[np.isfinite(Pb)]; cb=vb.mean()
                twob=(1+np.sum(np.abs(vb-cb)>=abs(ob-cb)))/(len(vb)+1)
                oneb=(1+np.sum(vb<=ob))/(len(vb)+1) if EXPECTED[ch]<0 else (1+np.sum(vb>=ob))/(len(vb)+1)
                summary.append({'CHANNEL':ch,'TEST':test,'SAMPLE_DESIGN':'BALANCED_COMMON_DATES','OBSERVED_COEFFICIENT':ob,'DATE_CLUSTERED_P_VALUE':np.nan,
                                'N_OBSERVATIONS':no,'N_ETFS':ne,'N_DATES':nd,'N_PERMUTATIONS_REQUESTED':B,'N_VALID_PERMUTATIONS':len(vb),
                                'PERMUTATION_MEAN':cb,'PERMUTATION_STD':vb.std(ddof=1),'TWO_SIDED_PERMUTATION_P':twob,'THEORY_DIRECTED_PERMUTATION_P':oneb,
                                'OBSERVED_PERCENTILE':100*np.mean(vb<=ob),'EXPECTED_SIGN':'POSITIVE' if EXPECTED[ch]>0 else 'NEGATIVE'})
            fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.hist(valid,bins=45,alpha=.75); ax.axvline(observed,linewidth=2,label='Observed coefficient'); ax.axvline(0,linewidth=1)
            ax.set_title(f'Permutation distribution: {ch.replace("_"," ").title()}\n{test.replace("_"," ").title()}'); ax.set_xlabel('Interaction coefficient'); ax.set_ylabel('Frequency'); ax.legend(frameon=False)
            fig.tight_layout(); stem=f'33_perm_{test.lower()}_{ch.lower()}'; fig.savefig(FIG/f'{stem}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'{stem}.pdf',bbox_inches='tight'); plt.close(fig)
    summ=pd.DataFrame(summary); summ['ROMANO_WOLF_APPROX_P']=np.nan; summ['STANDARDIZED_OBSERVED_STAT']=np.nan
    for test in TESTS:
        P=np.column_stack(family_store[test]); adj,tobs=rw_stepdown(np.array(obs_store[test]),P)
        for ch,a,t in zip(labels_store[test],adj,tobs):
            m=(summ.TEST==test)&(summ.CHANNEL==ch)&(summ.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION'); summ.loc[m,'ROMANO_WOLF_APPROX_P']=a; summ.loc[m,'STANDARDIZED_OBSERVED_STAT']=t
    summ.to_csv(OUT/'33_permutation_inference_summary.csv',index=False); pd.DataFrame(dist_rows).to_csv(OUT/'33_permutation_distributions.csv',index=False)
    primary=summ[summ.SAMPLE_DESIGN.eq('AVAILABLE_CROSS_SECTION')].copy(); primary.to_csv(OUT/'33_primary_permutation_results.csv',index=False)
    meta={'replications':B,'seed':SEED,'assignment':'shuffle frozen ETF-level architecture scores across eligible ETFs while preserving ETF histories and common shocks',
          'p_values':['two-sided','theory-directed','approximate Romano-Wolf stepdown across four channels within each test family'],
          'balanced_robustness':'common dates with all eligible ETFs, if at least 24 months'}
    (OUT/'33_permutation_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nPRIMARY PERMUTATION RESULTS\n',primary[['CHANNEL','TEST','OBSERVED_COEFFICIENT','DATE_CLUSTERED_P_VALUE','TWO_SIDED_PERMUTATION_P','THEORY_DIRECTED_PERMUTATION_P','ROMANO_WOLF_APPROX_P','OBSERVED_PERCENTILE']].to_string(index=False))
if __name__=='__main__': main()
