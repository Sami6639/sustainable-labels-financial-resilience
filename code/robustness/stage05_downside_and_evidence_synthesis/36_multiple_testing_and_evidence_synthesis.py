from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

BASE=Path('/mnt/data/q1_analysis'); OUT=BASE/'evidence_synthesis'; FIG=OUT/'figures'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
PRIMARY=['INTERNAL_FINANCIAL_CAPACITY','EXTERNAL_FINANCING_DEPENDENCE','GROWTH_DURATION_EXPOSURE_FINAL','PORTFOLIO_CONCENTRATION_FINAL']
EXPECTED={'INTERNAL_FINANCIAL_CAPACITY':'POSITIVE','EXTERNAL_FINANCING_DEPENDENCE':'NEGATIVE','GROWTH_DURATION_EXPOSURE_FINAL':'NEGATIVE','PORTFOLIO_CONCENTRATION_FINAL':'NEGATIVE'}
LABEL={'INTERNAL_FINANCIAL_CAPACITY':'Internal capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing','GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration','PORTFOLIO_CONCENTRATION_FINAL':'Concentration'}


def adjust(df,pcol='P_VALUE',groups=None,prefix='ADJ'):
    x=df.copy(); x[f'{prefix}_HOLM_P']=np.nan; x[f'{prefix}_BH_Q']=np.nan
    if groups is None: groups=[]
    iterator=[(None,x)] if not groups else x.groupby(groups,dropna=False)
    for _,g in iterator:
        p=pd.to_numeric(g[pcol],errors='coerce'); ok=p.notna(); idx=g.index[ok]
        if len(idx):
            x.loc[idx,f'{prefix}_HOLM_P']=multipletests(p[ok],method='holm')[1]
            x.loc[idx,f'{prefix}_BH_Q']=multipletests(p[ok],method='fdr_bh')[1]
    return x


def core_activation():
    r=pd.read_csv('/mnt/data/cpu_extracted/CPU_Project/output/27_stress_activation_key_coefficients.csv',low_memory=False)
    r=r[(r.CHANNEL.isin(PRIMARY))&(r.QUALITY=='MAIN')&(r.COVARIANCE=='CLUSTER_DATE')&(r.PARAMETER==r.TARGET_PARAMETER)].copy()
    r['TEST']=r.MODEL_FAMILY.map({'CONTINUOUS':'CPU_VIX_CONTINUOUS','REGIME':'CPU_AND_VIX_STRESS'})
    r=r[['CHANNEL','TEST','COEFFICIENT','STD_ERROR','P_VALUE','CI_LOWER_95','CI_UPPER_95','N_OBSERVATIONS','N_ETFS','N_MONTHS','EXPECTED_SIGN','EXPECTED_SIGN_MATCH']].rename(columns={'STD_ERROR':'STANDARD_ERROR','N_MONTHS':'N_DATES'})
    r=adjust(r,'P_VALUE',groups=['TEST'],prefix='WITHIN_TEST'); r=adjust(r,'P_VALUE',groups=None,prefix='CORE8')
    perm=pd.read_csv(BASE/'permutation_inference/33_primary_permutation_results.csv')
    perm=perm[['CHANNEL','TEST','TWO_SIDED_PERMUTATION_P','THEORY_DIRECTED_PERMUTATION_P','ROMANO_WOLF_APPROX_P','OBSERVED_PERCENTILE']]
    loo=pd.read_csv(BASE/'influence_diagnostics/32_leave_one_etf_out_summary.csv')
    loo=loo[loo.HORIZON_MONTHS.eq(0)][['CHANNEL','SHOCK','SIGN_PRESERVATION_RATE','P_LT_10_PRESERVATION_RATE','MOST_INFLUENTIAL_ETF_TICKER','MAX_CHANGE_IN_FULL_SE_UNITS','INFLUENCE_ASSESSMENT']].rename(columns={'SHOCK':'TEST'})
    style=pd.read_csv(BASE/'style_controls/34b_key_style_control_results.csv')
    style=style[(style.HORIZON_MONTHS.eq(0))&(style.CONTROL_LEVEL.eq('ARCHITECTURE_FACTOR_BETAS'))][['CHANNEL','TEST','COEFFICIENT','P_VALUE']].rename(columns={'COEFFICIENT':'STYLE_BETA_ADJUSTED_COEFFICIENT','P_VALUE':'STYLE_BETA_ADJUSTED_P'})
    alt=pd.read_csv(BASE/'alternative_stress_falsification/34_alternative_definition_sign_consistency.csv')
    # Aggregate alternative evidence by channel across the three substantive families.
    altg=alt.groupby('CHANNEL').agg(ALT_N_MODELS=('N_MODELS','sum'),ALT_EXPECTED_SIGN_RATE=('EXPECTED_SIGN_RATE','mean'),ALT_N_RAW_P_LT_05=('N_RAW_P_LT_05','sum'),ALT_N_BH_Q_LT_10=('N_BH_Q_LT_10','sum')).reset_index()
    sorts=pd.read_csv(BASE/'portfolio_sorts/31_primary_sorted_portfolio_results.csv')
    sort_cont=sorts[sorts.TEST.eq('CONTINUOUS_CPU_VIX_SENSITIVITY')][['CHANNEL','COEFFICIENT','P_VALUE','BH_Q_PRIMARY_FAMILY']].rename(columns={'COEFFICIENT':'SORT_CONTINUOUS_COEFFICIENT','P_VALUE':'SORT_CONTINUOUS_P','BH_Q_PRIMARY_FAMILY':'SORT_CONTINUOUS_BH_Q'})
    sort_joint=sorts[sorts.TEST.eq('CPU_AND_VIX_STRESS_RAW_INCREMENT')][['CHANNEL','COEFFICIENT','P_VALUE','BH_Q_PRIMARY_FAMILY']].rename(columns={'COEFFICIENT':'SORT_JOINT_RAW_COEFFICIENT','P_VALUE':'SORT_JOINT_RAW_P','BH_Q_PRIMARY_FAMILY':'SORT_JOINT_RAW_BH_Q'})
    x=r.merge(perm,on=['CHANNEL','TEST'],how='left').merge(loo,on=['CHANNEL','TEST'],how='left').merge(style,on=['CHANNEL','TEST'],how='left').merge(altg,on='CHANNEL',how='left').merge(sort_cont,on='CHANNEL',how='left').merge(sort_joint,on='CHANNEL',how='left')
    x['STYLE_SIGN_PRESERVED']=np.sign(x.STYLE_BETA_ADJUSTED_COEFFICIENT)==np.sign(x.COEFFICIENT)
    x['STYLE_ROBUSTNESS_CLASS']=np.where(~x.STYLE_SIGN_PRESERVED,'STYLE_SIGN_REVERSAL',np.where(x.STYLE_BETA_ADJUSTED_P<.10,'SURVIVES_ARCHITECTURE_BETA_CONTROLS_AT_10PCT','SIGN_SURVIVES_BUT_PRECISION_WEAK'))
    def grade(row):
        expected=bool(row.EXPECTED_SIGN_MATCH)
        if not expected: return 'CONTRARY_OR_NULL'
        if row.P_VALUE<.05 and row.TWO_SIDED_PERMUTATION_P<.05 and row.SIGN_PRESERVATION_RATE>=.99:
            return 'STRONG_NONPARAMETRICALLY_VALIDATED'
        if row.P_VALUE<.10 and row.TWO_SIDED_PERMUTATION_P<.05 and row.SIGN_PRESERVATION_RATE>=.90:
            return 'SUPPORTED_BY_RANDOMIZATION'
        if row.P_VALUE<.05 and row.SIGN_PRESERVATION_RATE>=.90:
            return 'STRONG_SUPPORTING'
        if min(row.P_VALUE,row.TWO_SIDED_PERMUTATION_P if np.isfinite(row.TWO_SIDED_PERMUTATION_P) else 1)<.10:
            return 'SUGGESTIVE'
        return 'NOT_SUPPORTED'
    x['EVIDENCE_GRADE']=x.apply(grade,axis=1)
    return x


def dynamic_trajectory():
    d=pd.read_csv(BASE/'local_projections/29_local_projection_dynamic_summary.csv')
    d=d[(d.CHANNEL.isin(PRIMARY))&(d.SHOCK.isin(['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']))].copy()
    loo=pd.read_csv(BASE/'influence_diagnostics/32_leave_one_etf_out_summary.csv')
    loo=loo[loo.HORIZON_MONTHS.isin([6,9,12])]
    lsum=loo.groupby(['CHANNEL','SHOCK']).agg(LOO_MIN_SIGN_PRESERVATION=('SIGN_PRESERVATION_RATE','min'),LOO_MIN_P_LT_10_RATE=('P_LT_10_PRESERVATION_RATE','min'),LOO_MAX_CHANGE_SE=('MAX_CHANGE_IN_FULL_SE_UNITS','max'),LOO_ANY_SIGN_SENSITIVE=('INFLUENCE_ASSESSMENT',lambda s:(s=='SIGN_SENSITIVE').any())).reset_index()
    style=pd.read_csv(BASE/'style_controls/34b_key_style_control_results.csv')
    style=style[(style.CONTROL_LEVEL.eq('ARCHITECTURE_FACTOR_BETAS'))&(style.HORIZON_MONTHS.isin([9,12]))]
    ssum=style.groupby(['CHANNEL','TEST']).agg(STYLE_H9_H12_SIGN_RATE=('COEFFICIENT',lambda s:np.mean(np.sign(s)==np.sign(s.iloc[0]))),STYLE_MIN_P=('P_VALUE','min'),STYLE_MEDIAN_COEFFICIENT=('COEFFICIENT','median')).reset_index().rename(columns={'TEST':'SHOCK'})
    x=d.merge(lsum,on=['CHANNEL','SHOCK'],how='left').merge(ssum,on=['CHANNEL','SHOCK'],how='left')
    def grade(r):
        if r.ANY_BONFERRONI_BAND_EXCLUDES_ZERO and r.N_BH_Q_WITHIN_CHANNEL_LT_10>=1 and (pd.isna(r.LOO_MIN_SIGN_PRESERVATION) or r.LOO_MIN_SIGN_PRESERVATION>=.9): return 'TRAJECTORY_CONFIRMED'
        if r.N_RAW_P_LT_05>=2 and (pd.isna(r.LOO_MIN_SIGN_PRESERVATION) or r.LOO_MIN_SIGN_PRESERVATION>=.9): return 'STRONG_DYNAMIC_SUPPORT'
        if r.N_RAW_P_LT_05>=1 or r.N_BH_Q_WITHIN_CHANNEL_LT_10>=1: return 'SUGGESTIVE_DYNAMIC_SUPPORT'
        return 'DESCRIPTIVE_OR_NULL'
    x['TRAJECTORY_GRADE']=x.apply(grade,axis=1)
    return x


def supporting_ledger():
    rows=[]
    f=pd.read_csv(BASE/'fama_macbeth/30_primary_fama_macbeth_results.csv')
    for _,r in f.iterrows(): rows.append({'SOURCE':'FAMA_MACBETH','FAMILY':r.TEST,'CHANNEL':r.CHANNEL,'TEST':r.TEST,'OUTCOME':r.OUTCOME,'COEFFICIENT':r.COEFFICIENT,'P_VALUE':r.P_VALUE,'EXISTING_HOLM_P':r.HOLM_P_PRIMARY_FAMILY,'EXISTING_BH_Q':r.BH_Q_PRIMARY_FAMILY,'N':r.N_MONTHS,'NOTES':r.SAMPLE_DESIGN})
    s=pd.read_csv(BASE/'portfolio_sorts/31_primary_sorted_portfolio_results.csv')
    for _,r in s.iterrows(): rows.append({'SOURCE':'PORTFOLIO_SORT','FAMILY':r.TEST,'CHANNEL':r.CHANNEL,'TEST':r.TEST,'OUTCOME':'HML_RETURN','COEFFICIENT':r.COEFFICIENT,'P_VALUE':r.P_VALUE,'EXISTING_HOLM_P':r.HOLM_P_PRIMARY_FAMILY,'EXISTING_BH_Q':r.BH_Q_PRIMARY_FAMILY,'N':r.N_MONTHS,'NOTES':'median split, available cross-section'})
    t=pd.read_csv(BASE/'downside_tail_risk/35_primary_downside_results.csv')
    for _,r in t.iterrows(): rows.append({'SOURCE':'DOWNSIDE_PANEL','FAMILY':f'{r.TEST}_{r.OUTCOME}','CHANNEL':r.CHANNEL,'TEST':r.TEST,'OUTCOME':r.OUTCOME,'COEFFICIENT':r.COEFFICIENT,'P_VALUE':r.P_VALUE,'EXISTING_HOLM_P':r.HOLM_P_WITHIN_OUTCOME_TEST,'EXISTING_BH_Q':r.BH_Q_WITHIN_OUTCOME_TEST,'N':r.N_OBSERVATIONS,'NOTES':'ETF FE/date cluster'})
    q=pd.read_csv(BASE/'downside_tail_risk/35_quantile_regression_results.csv')
    for _,r in q.iterrows(): rows.append({'SOURCE':'QUANTILE_10','FAMILY':r.TEST,'CHANNEL':r.CHANNEL,'TEST':r.TEST,'OUTCOME':'RETURN_Q10','COEFFICIENT':r.COEFFICIENT,'P_VALUE':r.P_VALUE,'EXISTING_HOLM_P':r.HOLM_P_WITHIN_TEST,'EXISTING_BH_Q':r.BH_Q_WITHIN_TEST,'N':r.N_OBSERVATIONS,'NOTES':f'block bootstrap sign p={r.BLOCK_BOOT_SIGN_P:.3f}'})
    x=pd.DataFrame(rows); x=adjust(x,'P_VALUE',groups=['SOURCE','FAMILY'],prefix='RECOMPUTED_FAMILY'); return x


def channel_matrix(core,dynamic,support):
    rows=[]
    for ch in PRIMARY:
        c=core[core.CHANNEL.eq(ch)]
        d=dynamic[dynamic.CHANNEL.eq(ch)]
        q=support[(support.CHANNEL.eq(ch))&(support.SOURCE=='QUANTILE_10')]
        down=support[(support.CHANNEL.eq(ch))&(support.SOURCE=='DOWNSIDE_PANEL')]
        rows.append({'CHANNEL':ch,'LABEL':LABEL[ch],
                     'CONTINUOUS_CORE_GRADE':c.loc[c.TEST.eq('CPU_VIX_CONTINUOUS'),'EVIDENCE_GRADE'].iloc[0],
                     'JOINT_CORE_GRADE':c.loc[c.TEST.eq('CPU_AND_VIX_STRESS'),'EVIDENCE_GRADE'].iloc[0],
                     'N_TRAJECTORY_CONFIRMED':int((d.TRAJECTORY_GRADE=='TRAJECTORY_CONFIRMED').sum()),
                     'N_STRONG_DYNAMIC_SUPPORT':int((d.TRAJECTORY_GRADE=='STRONG_DYNAMIC_SUPPORT').sum()),
                     'MIN_PERMUTATION_P':c.TWO_SIDED_PERMUTATION_P.min(),
                     'MIN_ROMANO_WOLF_P':c.ROMANO_WOLF_APPROX_P.min(),
                     'LOO_MIN_SIGN_RATE_IMPACT':c.SIGN_PRESERVATION_RATE.min(),
                     'ALT_EXPECTED_SIGN_RATE':c.ALT_EXPECTED_SIGN_RATE.iloc[0],
                     'N_DOWNSIDE_RAW_P_LT_05':int((down.P_VALUE<.05).sum()),
                     'N_QUANTILE_RAW_P_LT_05':int((q.P_VALUE<.05).sum()),
                     'INTERPRETATION_FLAG':('EXPECTED_PROTECTIVE_CHANNEL_NOT_SUPPORTED' if ch=='INTERNAL_FINANCIAL_CAPACITY' else 'VULNERABILITY_CHANNEL')})
    return pd.DataFrame(rows)


def figures(core,matrix):
    order=PRIMARY; tests=['CPU_VIX_CONTINUOUS','CPU_AND_VIX_STRESS']; fig,ax=plt.subplots(figsize=(9,5.2)); x=np.arange(len(order)); width=.34
    for j,test in enumerate(tests):
        g=core[core.TEST.eq(test)].set_index('CHANNEL').reindex(order); vals=100*g.COEFFICIENT; err=np.vstack([100*(g.COEFFICIENT-g.CI_LOWER_95),100*(g.CI_UPPER_95-g.COEFFICIENT)])
        ax.bar(x+(j-.5)*width,vals,width,yerr=err,capsize=3,label=test.replace('_',' ').title())
    ax.axhline(0,linewidth=1); ax.set_xticks(x,[LABEL[c] for c in order],rotation=18,ha='right'); ax.set_ylabel('Interaction coefficient (percentage points)'); ax.set_title('Core architecture-activation estimates'); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIG/'36_core_activation_coefficients.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/'36_core_activation_coefficients.pdf',bbox_inches='tight'); plt.close(fig)
    # Evidence heatmap based on ordinal grades.
    grade_map={'STRONG_NONPARAMETRICALLY_VALIDATED':4,'SUPPORTED_BY_RANDOMIZATION':3,'STRONG_SUPPORTING':3,'SUGGESTIVE':2,'NOT_SUPPORTED':1,'CONTRARY_OR_NULL':0}
    mat=[]
    for ch in order:
        cc=core[core.CHANNEL.eq(ch)].set_index('TEST'); mat.append([grade_map[cc.loc[t,'EVIDENCE_GRADE']] for t in tests])
    arr=np.array(mat); fig,ax=plt.subplots(figsize=(6.8,4.4)); im=ax.imshow(arr,aspect='auto',vmin=0,vmax=4); ax.set_yticks(range(len(order)),[LABEL[c] for c in order]); ax.set_xticks(range(2),['Continuous CPU × VIX','Joint-stress threshold'],rotation=15,ha='right')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): ax.text(j,i,str(arr[i,j]),ha='center',va='center')
    ax.set_title('Triangulated evidence strength (0–4)'); fig.colorbar(im,ax=ax,label='Evidence score'); fig.tight_layout(); fig.savefig(FIG/'36_evidence_strength_heatmap.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/'36_evidence_strength_heatmap.pdf',bbox_inches='tight'); plt.close(fig)


def main():
    core=core_activation(); core.to_csv(OUT/'36_core_activation_multiple_testing.csv',index=False)
    dyn=dynamic_trajectory(); dyn.to_csv(OUT/'36_dynamic_trajectory_assessment.csv',index=False)
    support=supporting_ledger(); support.to_csv(OUT/'36_supporting_evidence_ledger.csv',index=False)
    matrix=channel_matrix(core,dyn,support); matrix.to_csv(OUT/'36_channel_evidence_matrix.csv',index=False)
    figures(core,matrix)
    meta={'primary_families':{'core_activation':'four channels x continuous and threshold activation; Holm/BH within test and across all eight','dynamic':'Holm/BH and Bonferroni trajectory inference inherited from Script 29','supporting':'Fama-MacBeth, sorted portfolios, downside panels and quantile regressions treated as separate families'},
          'nonparametric':'permutation p-values and approximate Romano-Wolf stepdown merged into core ledger','influence':'leave-one-ETF-out sign and significance preservation merged','style_controls':'growth-factor, oil and architecture-factor-beta controls merged'}
    (OUT/'36_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nCORE ACTIVATION\n',core[['CHANNEL','TEST','COEFFICIENT','P_VALUE','WITHIN_TEST_HOLM_P','WITHIN_TEST_BH_Q','CORE8_HOLM_P','CORE8_BH_Q','TWO_SIDED_PERMUTATION_P','ROMANO_WOLF_APPROX_P','SIGN_PRESERVATION_RATE','STYLE_BETA_ADJUSTED_P','EVIDENCE_GRADE']].to_string(index=False))
    print('\nDYNAMIC TRAJECTORIES\n',dyn[['CHANNEL','SHOCK','TROUGH_HORIZON','TROUGH_COEFFICIENT','H12_COEFFICIENT','N_RAW_P_LT_05','N_BH_Q_WITHIN_CHANNEL_LT_10','ANY_BONFERRONI_BAND_EXCLUDES_ZERO','LOO_MIN_SIGN_PRESERVATION','STYLE_MIN_P','TRAJECTORY_GRADE']].to_string(index=False))
    print('\nCHANNEL MATRIX\n',matrix.to_string(index=False))
if __name__=='__main__': main()
