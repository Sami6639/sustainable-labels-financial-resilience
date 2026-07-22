from pathlib import Path
import json, warnings, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SOURCE=Path('/mnt/data/cpu_extracted/CPU_Project/output/25_final_econometric_panel.csv')
OUT=Path('/mnt/data/q1_analysis/portfolio_sorts'); FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
CONTROLS=['MARKET_RETURN','ENERGY_RETURN','TREASURY_RETURN','VIX_CHANGE']
CHANNELS={'INTERNAL_FINANCIAL_CAPACITY':'PRIMARY','EXTERNAL_FINANCING_DEPENDENCE':'PRIMARY','GROWTH_DURATION_EXPOSURE_FINAL':'PRIMARY','PORTFOLIO_CONCENTRATION_FINAL':'PRIMARY','FINANCIAL_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS','EXTENDED_ARCHITECTURE_RISK_FINAL':'COMPOSITE_ROBUSTNESS'}
HAC_LAGS=3

def qcol(ch): return f'{ch}_MAIN',f'VALID_{ch}_MAIN_ROW'
def req():
 s={'DATE','ETF_ID','ETF_TICKER','ETF_NAME_RETURN','ETF_RETURN','CPU_Z','VIX_LEVEL_Z','HIGH_CPU_REGIME','HIGH_VIX_REGIME','CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME',*CONTROLS}
 for ch in CHANNELS:
  a,f=qcol(ch); s.update([a,f])
 return sorted(s)
def membership(df,ch,sort):
 a,f=qcol(ch); x=df.loc[df[f].eq(1)&df[a].notna(),['ETF_ID','ETF_TICKER','ETF_NAME_RETURN',a]].drop_duplicates('ETF_ID').sort_values(a)
 n=len(x); x['SORT_GROUP']='MIDDLE'
 if sort=='MEDIAN_SPLIT':
  low_n=n//2; x.iloc[:low_n,x.columns.get_loc('SORT_GROUP')]='LOW'; x.iloc[low_n:,x.columns.get_loc('SORT_GROUP')]='HIGH'
 else:
  k=n//3; x.iloc[:k,x.columns.get_loc('SORT_GROUP')]='LOW'; x.iloc[-k:,x.columns.get_loc('SORT_GROUP')]='HIGH'
 x['CHANNEL']=ch; x['SORT_METHOD']=sort; x['N_UNIVERSE']=n
 return x
def port_series(df,mem,ch,sort,design):
 ids=mem[mem.SORT_GROUP.isin(['LOW','HIGH'])].ETF_ID.unique(); s=df[df.ETF_ID.isin(ids)].merge(mem[['ETF_ID','SORT_GROUP']],on='ETF_ID',how='inner')
 start=end=pd.NaT
 if design=='BALANCED_COMMON_WINDOW':
  first=s.groupby('ETF_ID').DATE.min(); last=s.groupby('ETF_ID').DATE.max(); start=first.max(); end=last.min(); s=s[(s.DATE>=start)&(s.DATE<=end)]
 rows=[]
 for d,g in s.groupby('DATE'):
  hi=g[g.SORT_GROUP=='HIGH'].ETF_RETURN.dropna(); lo=g[g.SORT_GROUP=='LOW'].ETF_RETURN.dropna()
  n_hi=mem.SORT_GROUP.eq('HIGH').sum(); n_lo=mem.SORT_GROUP.eq('LOW').sum()
  if design=='BALANCED_COMMON_WINDOW' and (len(hi)!=n_hi or len(lo)!=n_lo): continue
  if len(hi)<2 or len(lo)<2: continue
  c=g.iloc[0]; rh=hi.mean(); rl=lo.mean()
  rows.append({'DATE':d,'CHANNEL':ch,'CHANNEL_ROLE':CHANNELS[ch],'SORT_METHOD':sort,'SAMPLE_DESIGN':design,'HIGH_RETURN':rh,'LOW_RETURN':rl,'HML_RETURN':rh-rl,'N_HIGH':len(hi),'N_LOW':len(lo),'BALANCED_START':start,'BALANCED_END':end,
               'CPU_Z':c.CPU_Z,'VIX_LEVEL_Z':c.VIX_LEVEL_Z,'CPU_Z_X_VIX_Z':c.CPU_Z*c.VIX_LEVEL_Z,'HIGH_CPU_REGIME':c.HIGH_CPU_REGIME,'HIGH_VIX_REGIME':c.HIGH_VIX_REGIME,'CPU_AND_VIX_STRESS':c.CPU_AND_VIX_STRESS,'EXTREME_CPU_REGIME':c.EXTREME_CPU_REGIME,**{v:c[v] for v in CONTROLS}})
 return pd.DataFrame(rows)
def fit(y,X): return sm.OLS(y,sm.add_constant(X,has_constant='add'),missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':HAC_LAGS,'use_correction':True})
def evaluate(series):
 rows=[]
 keys=['CHANNEL','CHANNEL_ROLE','SORT_METHOD','SAMPLE_DESIGN']
 for key,g in series.groupby(keys):
  ch,role,sort,design=key; g=g.sort_values('DATE'); base={'CHANNEL':ch,'CHANNEL_ROLE':role,'SORT_METHOD':sort,'SAMPLE_DESIGN':design,'N_MONTHS':len(g),'DATE_START':g.DATE.min(),'DATE_END':g.DATE.max(),'N_HIGH_MIN':int(g.N_HIGH.min()),'N_LOW_MIN':int(g.N_LOW.min())}
  if len(g)<24: continue
  # Mean spread
  f=fit(g.HML_RETURN,pd.DataFrame(index=g.index)); rows.append({**base,'TEST':'UNCONDITIONAL_HML_MEAN','FOCAL':'CONST','COEFFICIENT':f.params['const'],'STANDARD_ERROR':f.bse['const'],'P_VALUE':f.pvalues['const'],'CI_LOWER_95':f.conf_int().loc['const',0],'CI_UPPER_95':f.conf_int().loc['const',1]})
  # Factor alpha
  z=g.dropna(subset=CONTROLS); f=fit(z.HML_RETURN,z[CONTROLS]); rows.append({**base,'N_MONTHS':len(z),'TEST':'FACTOR_ADJUSTED_ALPHA','FOCAL':'CONST','COEFFICIENT':f.params['const'],'STANDARD_ERROR':f.bse['const'],'P_VALUE':f.pvalues['const'],'CI_LOWER_95':f.conf_int().loc['const',0],'CI_UPPER_95':f.conf_int().loc['const',1]})
  # Continuous CPU x VIX exposure
  x=['CPU_Z','VIX_LEVEL_Z','CPU_Z_X_VIX_Z',*CONTROLS]; z=g.dropna(subset=x); f=fit(z.HML_RETURN,z[x]); q='CPU_Z_X_VIX_Z'; rows.append({**base,'N_MONTHS':len(z),'TEST':'CONTINUOUS_CPU_VIX_SENSITIVITY','FOCAL':q,'COEFFICIENT':f.params[q],'STANDARD_ERROR':f.bse[q],'P_VALUE':f.pvalues[q],'CI_LOWER_95':f.conf_int().loc[q,0],'CI_UPPER_95':f.conf_int().loc[q,1]})
  # State increments and raw state means
  for state in ['CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']:
   z=g.dropna(subset=[state]); f=fit(z.HML_RETURN,z[[state]]); rows.append({**base,'N_MONTHS':len(z),'TEST':f'{state}_RAW_INCREMENT','FOCAL':state,'COEFFICIENT':f.params[state],'STANDARD_ERROR':f.bse[state],'P_VALUE':f.pvalues[state],'CI_LOWER_95':f.conf_int().loc[state,0],'CI_UPPER_95':f.conf_int().loc[state,1],'NONSTATE_MEAN':f.params['const'],'STATE_MEAN':f.params['const']+f.params[state],'N_STATE_MONTHS':int(z[state].eq(1).sum())})
   x=[state,*CONTROLS]; z=g.dropna(subset=x); f=fit(z.HML_RETURN,z[x]); rows.append({**base,'N_MONTHS':len(z),'TEST':f'{state}_ADJUSTED_INCREMENT','FOCAL':state,'COEFFICIENT':f.params[state],'STANDARD_ERROR':f.bse[state],'P_VALUE':f.pvalues[state],'CI_LOWER_95':f.conf_int().loc[state,0],'CI_UPPER_95':f.conf_int().loc[state,1],'N_STATE_MONTHS':int(z[state].eq(1).sum())})
 return pd.DataFrame(rows)
def event_profiles(series):
 rows=[]
 for key,g in series.groupby(['CHANNEL','CHANNEL_ROLE','SORT_METHOD','SAMPLE_DESIGN']):
  ch,role,sort,design=key; g=g.sort_values('DATE').reset_index(drop=True); per=g.DATE.dt.year*12+g.DATE.dt.month
  for event in ['CPU_AND_VIX_STRESS','EXTREME_CPU_REGIME']:
   events=np.where(g[event].eq(1))[0]
   for h in [0,1,3,6,12]:
    diffs=[]
    for i in events:
     j=i+h
     if j>=len(g) or per.iloc[j]-per.iloc[i]!=h: continue
     rh=np.prod(1+g.loc[i:j,'HIGH_RETURN'])-1; rl=np.prod(1+g.loc[i:j,'LOW_RETURN'])-1; diffs.append(rh-rl)
    rows.append({'CHANNEL':ch,'CHANNEL_ROLE':role,'SORT_METHOD':sort,'SAMPLE_DESIGN':design,'EVENT':event,'HORIZON_MONTHS':h,'N_EVENTS':len(diffs),'MEAN_CUMULATIVE_HML':np.mean(diffs) if diffs else np.nan,'MEDIAN_CUMULATIVE_HML':np.median(diffs) if diffs else np.nan,'MIN_CUMULATIVE_HML':np.min(diffs) if diffs else np.nan,'MAX_CUMULATIVE_HML':np.max(diffs) if diffs else np.nan})
 return pd.DataFrame(rows)
def risk_stats(series):
 rows=[]
 for key,g in series.groupby(['CHANNEL','CHANNEL_ROLE','SORT_METHOD','SAMPLE_DESIGN']):
  ch,role,sort,design=key; x=g.HML_RETURN.dropna(); q=x.quantile(.05); cvar=x[x<=q].mean(); downside=np.sqrt(np.mean(np.minimum(x,0)**2)); wealth=(1+x).cumprod(); dd=wealth/wealth.cummax()-1
  rows.append({'CHANNEL':ch,'CHANNEL_ROLE':role,'SORT_METHOD':sort,'SAMPLE_DESIGN':design,'N_MONTHS':len(x),'MEAN_MONTHLY_HML':x.mean(),'ANNUALIZED_HML_MEAN':12*x.mean(),'ANNUALIZED_HML_VOLATILITY':np.sqrt(12)*x.std(),'SHARPE_NO_RF':np.sqrt(12)*x.mean()/x.std() if x.std()>0 else np.nan,'VAR_5':q,'CVAR_5':cvar,'DOWNSIDE_DEVIATION':downside,'MAX_DRAWDOWN':dd.min()})
 return pd.DataFrame(rows)
def multiplicity(res):
 x=res.copy(); x['HOLM_P_PRIMARY_FAMILY']=np.nan; x['BH_Q_PRIMARY_FAMILY']=np.nan
 m=(x.CHANNEL_ROLE=='PRIMARY')&(x.SORT_METHOD=='MEDIAN_SPLIT')&(x.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')
 for test,g in x[m].groupby('TEST'):
  p=g.P_VALUE.to_numpy(); idx=g.index; x.loc[idx,'HOLM_P_PRIMARY_FAMILY']=multipletests(p,method='holm')[1]; x.loc[idx,'BH_Q_PRIMARY_FAMILY']=multipletests(p,method='fdr_bh')[1]
 return x
def figures(series,events):
 import matplotlib.pyplot as plt
 labels={'INTERNAL_FINANCIAL_CAPACITY':'Internal financial capacity','EXTERNAL_FINANCING_DEPENDENCE':'External financing dependence','GROWTH_DURATION_EXPOSURE_FINAL':'Growth-duration exposure','PORTFOLIO_CONCENTRATION_FINAL':'Portfolio concentration'}
 for ch,title in labels.items():
  g=series[(series.CHANNEL==ch)&(series.SORT_METHOD=='MEDIAN_SPLIT')&(series.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')].sort_values('DATE')
  if g.empty: continue
  wh=(1+g.HIGH_RETURN).cumprod(); wl=(1+g.LOW_RETURN).cumprod()
  fig=plt.figure(figsize=(8.2,4.6)); ax=fig.add_subplot(111); ax.plot(g.DATE,wh,label='High architecture'); ax.plot(g.DATE,wl,label='Low architecture'); ax.set_title(f'Architecture-sorted wealth: {title}'); ax.set_ylabel('Growth of one unit'); ax.set_xlabel('Month'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/f'31_wealth_{ch.lower()}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'31_wealth_{ch.lower()}.pdf',bbox_inches='tight'); plt.close(fig)
  e=events[(events.CHANNEL==ch)&(events.SORT_METHOD=='MEDIAN_SPLIT')&(events.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')&(events.EVENT=='CPU_AND_VIX_STRESS')].sort_values('HORIZON_MONTHS')
  fig=plt.figure(figsize=(7.2,4.5)); ax=fig.add_subplot(111); ax.plot(e.HORIZON_MONTHS,100*e.MEAN_CUMULATIVE_HML,marker='o'); ax.axhline(0,linewidth=1); ax.set_title(f'High-minus-low profile after joint stress: {title}'); ax.set_xlabel('Months from event'); ax.set_ylabel('Cumulative high-minus-low return (percentage points)'); ax.set_xticks([0,1,3,6,12]); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/f'31_joint_event_{ch.lower()}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'31_joint_event_{ch.lower()}.pdf',bbox_inches='tight'); plt.close(fig)
def main():
 df=pd.read_csv(SOURCE,usecols=lambda c:c in set(req()),parse_dates=['DATE'],low_memory=False)
 for c in req():
  if c in df and c not in ['DATE','ETF_ID','ETF_TICKER','ETF_NAME_RETURN']: df[c]=pd.to_numeric(df[c],errors='coerce')
 members=[]; series=[]
 for ch in CHANNELS:
  for sort in ['MEDIAN_SPLIT','TERCILE']:
   mem=membership(df,ch,sort); members.append(mem)
   for design in ['AVAILABLE_CROSS_SECTION','BALANCED_COMMON_WINDOW']: series.append(port_series(df,mem,ch,sort,design))
 members=pd.concat(members,ignore_index=True); members.to_csv(OUT/'31_portfolio_sort_membership.csv',index=False)
 series=pd.concat(series,ignore_index=True); series.to_csv(OUT/'31_architecture_sorted_portfolio_returns.csv',index=False)
 res=multiplicity(evaluate(series)); res.to_csv(OUT/'31_sorted_portfolio_regression_results.csv',index=False)
 primary=res[(res.CHANNEL_ROLE=='PRIMARY')&(res.SORT_METHOD=='MEDIAN_SPLIT')&(res.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')]; primary.to_csv(OUT/'31_primary_sorted_portfolio_results.csv',index=False)
 events=event_profiles(series); events.to_csv(OUT/'31_sorted_portfolio_event_profiles.csv',index=False)
 risk=risk_stats(series); risk.to_csv(OUT/'31_sorted_portfolio_risk_statistics.csv',index=False)
 figures(series,events)
 meta={'source':str(SOURCE),'sorting':'fixed 2025Q4 architecture assignment','primary_sort':'median split','robustness_sort':'top-minus-bottom tercile','weights':'equal weighted','sample_designs':['available cross-section','balanced common window'],'inference':'Newey-West HAC, 3 lags'}
 (OUT/'31_portfolio_sort_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print(primary[['CHANNEL','TEST','COEFFICIENT','P_VALUE','HOLM_P_PRIMARY_FAMILY','BH_Q_PRIMARY_FAMILY','STATE_MEAN' if 'STATE_MEAN' in primary else 'N_MONTHS','N_MONTHS']].to_string(index=False))
 print('\nEVENT PROFILES\n',events[(events.CHANNEL_ROLE=='PRIMARY')&(events.SORT_METHOD=='MEDIAN_SPLIT')&(events.SAMPLE_DESIGN=='AVAILABLE_CROSS_SECTION')].to_string(index=False))
if __name__=='__main__': main()
