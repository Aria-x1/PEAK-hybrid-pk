import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import model as mdl
D=pd.read_pickle('D.pkl'); S=pd.read_pickle('S.pkl'); F=pd.read_pickle('F.pkl')
M=pd.read_csv('../data/curve_metadata.csv').set_index('sub_csv_path')
S['akey']=['tfv_after_TAF' if (r.source_pmid==27216057 and r.analyte=='tenofovir') else r.analyte for r in S.itertuples()]
F['akey']=S.loc[F.index,'akey']; F['use']=M.loc[F.index,'use_for_training']
D=D[D.cid.isin(F.index)&(D.c>0)].copy()
D['akey']=S.loc[D.cid,'akey'].values
D['ldose']=np.log10(D.dose); D['tauf']=D.tau.fillna(0).astype(float)
D['kindc']=D.kind.map({'single':0,'multi':1,'ss':2,'prior':3}); D['ly']=np.log10(D.c)
keys=sorted(D.akey.unique()); P=['logV','logka','logke']
def feats(df):
    X=df[['t','ldose','tauf','kindc','preg','renal','ddi']].copy()
    for k in keys: X['a_'+k]=(df.akey==k).astype(int)
    return X
def ccov(df):
    X=df[['ldose','preg','renal','ddi']].copy()
    for k in keys: X['a_'+k]=(df.akey==k).astype(int)
    return X
F['ldose']=np.log10(S.loc[F.index,'dose'])
def pm(cid,par):
    r=S.loc[cid]; g=D[D.cid==cid]
    ke=np.exp(np.clip(par['logke'],np.log(0.003),np.log(1.5)))
    ka=ke+0.02+max(np.exp(par['logka'])-np.exp(par['logke'])-0.02,0.01)
    with np.errstate(all='ignore'): c=mdl.conc(g.t.values,r.kind,r.doses,r.tau,ka,ke,np.exp(par['logV']))
    return np.log10(np.clip(c,1e-3,None))
def fold(train_c,test_c,name,rows,drop_cov=()):
    tr=D[D.cid.isin(train_c)]; Ftr=F.loc[list(train_c)]
    gbm=GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(feats(tr),tr.ly)
    med_a=Ftr.groupby('akey')[P].median(); med_g=Ftr[P].median()
    Xc=ccov(Ftr).drop(columns=list(drop_cov)); mu=Xc.mean(); sd=Xc.std().replace(0,1)
    rid=Ridge(alpha=3.0).fit((Xc-mu)/sd,Ftr[P])
    for cid in test_c:
        g=D[D.cid==cid]
        if len(g)==0: continue
        y=g.ly.values; ak=F.loc[cid,'akey']
        par=med_a.loc[ak] if ak in med_a.index else med_g
        xc=(ccov(F.loc[[cid]]).drop(columns=list(drop_cov))-mu)/sd
        ph=pd.Series(rid.predict(xc)[0],index=P)
        rows.append(dict(exp=name,cid=cid,paper=F.loc[cid,'source_pmid'],label=F.loc[cid,'curve_label'],dose=S.loc[cid,'dose'],akey=ak,
          GBM=np.abs(y-gbm.predict(feats(g))).mean(),mech_pop=np.abs(y-pm(cid,par)).mean(),hybrid=np.abs(y-pm(cid,ph)).mean()))
rows=[]; allc=list(F.index)
# 1 leave-one-dose-out, 11557462 (train: everything else incl. other doses of same paper)
for cid in F.index[F.source_pmid==11557462]:
    fold([c for c in allc if c!=cid],[cid],'dose_11557462',rows)
# 2 leave-population-out
pop={'pregnancy':F.index[F.source_pmid.isin([33252495,33318014,23392467])],
     'renal':F.index[(F.source_pmid==25921689)|((F.source_pmid==27216057)&F.curve_label.str.contains('Severe'))],
     'DDI':F.index[(F.source_pmid.isin([15673751,31675437,21770762]))&(S.loc[F.index,'ddi']==1)]}
for k,idx in pop.items():
    covdrop={'pregnancy':['preg'],'renal':['renal'],'DDI':['ddi']}[k]
    fold([c for c in allc if c not in set(idx)],list(idx),'pop_'+k,rows,drop_cov=[]) 
R=pd.DataFrame(rows); R.to_csv('../results/extrap_per_curve.csv',index=False)
pd.set_option('display.width',200)
print(R[R.exp=='dose_11557462'][['label','dose','GBM','mech_pop','hybrid']].round(3).to_string(index=False))
print(R.groupby('exp')[['GBM','mech_pop','hybrid']].agg(['mean']).round(3)); print(R.groupby('exp').size())
print(R[R.exp!='dose_11557462'].groupby(['exp','paper'])[['GBM','mech_pop','hybrid']].mean().round(2))
