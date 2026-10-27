import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import model as mdl
D=pd.read_pickle('D.pkl'); S=pd.read_pickle('S.pkl'); F=pd.read_pickle('F.pkl')
M=pd.read_csv('../data/curve_metadata.csv').set_index('sub_csv_path')
def akey(r):
    return 'tfv_after_TAF' if (r.source_pmid==27216057 and r.analyte=='tenofovir') else r.analyte
S['akey']=[akey(r) for r in S.itertuples()]
F['akey']=S.loc[F.index,'akey']; F['use']=M.loc[F.index,'use_for_training']
D=D[D.cid.isin(F.index)&(D.c>0)].copy()
D['akey']=S.loc[D.cid,'akey'].values; D['use']=M.loc[D.cid,'use_for_training'].values
D['ldose']=np.log10(D.dose); D['tauf']=D.tau.fillna(0).astype(float)
D['kindc']=D.kind.map({'single':0,'multi':1,'ss':2,'prior':3})
D['ly']=np.log10(D.c)
keys=sorted(D.akey.unique())
def feats(df):
    X=df[['t','ldose','tauf','kindc','preg','renal','ddi']].copy()
    for k in keys: X['a_'+k]=(df.akey==k).astype(int)
    return X
def curve_cov(df):
    X=df[['ldose','preg','renal','ddi']].copy()
    for k in keys: X['a_'+k]=(df.akey==k).astype(int)
    return X
F['ldose']=np.log10(S.loc[F.index,'dose']); 
P=['logV','logka','logke']
def predict_mech(cid,par):
    r=S.loc[cid]; g=D[D.cid==cid]
    ke=np.exp(np.clip(par['logke'],np.log(0.003),np.log(1.5))); ka=ke+0.02+np.exp(np.clip(np.log(max(np.exp(par['logka'])-np.exp(par['logke'])-0.02,0.01)),-5,3))
    V=np.exp(par['logV'])
    with np.errstate(all='ignore'): c=mdl.conc(g.t.values,r.kind,r.doses,r.tau,ka,ke,V)
    return np.log10(np.clip(c,1e-3,None))
rows=[]
papers=sorted(D.paper.unique())
for p in papers:
    tr=D[D.paper!=p]; te=D[D.paper==p]; Ftr=F[F.source_pmid!=p]; Fte=F[F.source_pmid==p]
    # M0, M1
    m0=tr.ly.mean(); m1=tr.groupby('akey').ly.mean()
    gbm=GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(feats(tr),tr.ly)
    # mech pop
    med_a=Ftr.groupby('akey')[P].median(); med_g=Ftr[P].median()
    # hybrid ridge on log params
    Xc=curve_cov(Ftr); mu=Xc.mean(); sd=Xc.std().replace(0,1)
    rid=Ridge(alpha=3.0).fit((Xc-mu)/sd,Ftr[P])
    for cid in Fte.index:
        g=te[te.cid==cid]
        if len(g)==0: continue
        y=g.ly.values
        e={}
        e['mean']=np.abs(y-m0).mean()
        e['analyte_mean']=np.abs(y-m1.get(Fte.loc[cid,'akey'],m0)).mean()
        e['GBM']=np.abs(y-gbm.predict(feats(g))).mean()
        ak=Fte.loc[cid,'akey']; par=(med_a.loc[ak] if ak in med_a.index else med_g)
        e['mech_pop']=np.abs(y-predict_mech(cid,par)).mean()
        xc=(curve_cov(Fte.loc[[cid]])-mu)/sd
        ph=pd.Series(rid.predict(xc)[0],index=P)
        e['hybrid']=np.abs(y-predict_mech(cid,ph)).mean()
        e['seen_analyte']=ak in med_a.index
        rows.append(dict(paper=p,cid=cid,akey=ak,use=Fte.loc[cid,'use'],n=len(g),**e))
R=pd.DataFrame(rows); R.to_pickle('R.pkl')
def summarize(R,name):
    mods=['mean','analyte_mean','GBM','mech_pop','hybrid']
    pm=R.groupby('paper')[mods].mean()
    rng=np.random.default_rng(0); bs=[]
    for _ in range(2000):
        ix=rng.choice(len(pm),len(pm)); bs.append(pm.iloc[ix].mean().values)
    bs=np.array(bs); lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    out=pd.DataFrame({'curve_MAE':R[mods].mean(),'paper_MAE':pm.mean(),'lo':lo,'hi':hi}).round(3)
    d=(pm['hybrid']-pm['GBM']); dd=np.array([d.iloc[rng.choice(len(d),len(d))].mean() for _ in range(2000)])
    print(f'--- {name}: {len(R)} curves, {R.paper.nunique()} papers'); print(out.to_string())
    print('hybrid-GBM paper-level diff %.3f  95%%CI [%.3f, %.3f]  hybrid better in %d/%d papers'%(d.mean(),*np.percentile(dd,[2.5,97.5]),(d<0).sum(),len(d)))
summarize(R,'ALL human PO (yes+caution+maybe)')
summarize(R[R.use.isin(['yes','yes_with_caution'])],'yes + yes_with_caution only')
summarize(R[R.akey=='tenofovir'],'tenofovir analyte only (TDF-derived)')
summarize(R[R.seen_analyte],'analyte seen in training')
print(R[['paper','akey','use','n','GBM','mech_pop','hybrid']].round(2).groupby('paper').mean(numeric_only=True).round(2))
