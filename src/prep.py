import pandas as pd, numpy as np, re
L=pd.read_csv('../data/curves_long.csv')
M=pd.read_csv('../data/curve_metadata.csv')
M=M[M.use_for_training.isin(['yes','yes_with_caution','maybe'])]
D=L.merge(M[['sub_csv_path','route_curve','regimen','tau_h','data_origin','use_for_training','unit_fix']],on='sub_csv_path')
D=D[(D.Species=='human')&(D.route_curve=='PO')]
D=D[D.source_pmid!=32553717].copy() if False else D
# unit & time
tu=D['Time Unit'].astype(str).str.lower()
D['t']=np.where(tu.isin(['days','d']),D.time*24,D.time)
mult=D['Analyte Unit'].map({'ng/mL':1,'µg/mL':1000,'ug/mL':1000,'mg/L':1000}).fillna(np.nan)
D.loc[D.source_pmid==11557462,'Analyte Unit']='ng/mL'; mult=D['Analyte Unit'].map({'ng/mL':1,'µg/mL':1000,'ug/mL':1000,'mg/L':1000})
D['c']=D.conc*mult
D=D[D.c.notna()&D.t.notna()&(D.t>=0)]
D['cid']=D.sub_csv_path
D['dose']=pd.to_numeric(D['Administration Value'].astype(str).str.split('/').str[0],errors='coerce')
D['analyte']=D['Analyte Name'].str.lower().replace({'tenofovir alafenamide':'taf','pmpa':'tenofovir'})
# dose schedule: list of (time, dose); ss tau
def sched(r):
    reg=r.regimen; d=r.dose
    if r.source_pmid==31675437: return ('multi',[(0,d),(24,d/2),(48,d/2)],None)
    if r.source_pmid==37787861: return ('ss',[(0,d)],24.0)
    if 'steady' in reg and pd.notna(r.tau_h) and str(r.tau_h)!='': return ('ss',[(0,d)],float(r.tau_h))
    if r.source_pmid==32553717:
        return ('prior',[(0,d)],{'Perfectadherence':24.0,'Moderateadherence':42.0,'Lowadherence':84.0}[r.curve_label])
    if 'single' in reg or reg=='': return ('single',[(0,d)],None)
    return ('skip',[(0,d)],None)
S=D.drop_duplicates('cid').set_index('cid')
S['kind']=[sched(r)[0] for r in S.itertuples()]
S['doses']=[sched(r)[1] for r in S.itertuples()]
S['tau']=[sched(r)[2] for r in S.itertuples()]
# covariates
def cov(r):
    k=(str(r.regimen)+' '+str(r.curve_label)).lower()
    preg=int(r.source_pmid in (33252495,33318014,23392467,25959631) or 'pregn' in k)
    renal=int(r.source_pmid in (25921689,) or 'severerenal' in k)
    ddi=int(any(x in k for x in ['probenecid','rifampin','fam','cobicistat','plus_']))
    return preg,renal,ddi
S[['preg','renal','ddi']]=[cov(r) for r in S.itertuples()]
S=S[S.kind!='skip']
D=D[D.cid.isin(S.index)]
D=D.join(S[['kind','tau','preg','renal','ddi']],on='cid')
D=D[~(D.kind.isin(['single','multi'])&(D.t==0))]
D['paper']=D.source_pmid
D.to_pickle('D.pkl'); S.to_pickle('S.pkl')
print(len(S),S.source_pmid.nunique()); print(S.groupby(['analyte','kind']).size())
print(S.groupby('source_pmid').size().to_string())
