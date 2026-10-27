import numpy as np, pandas as pd, warnings
from scipy.optimize import least_squares
warnings.filterwarnings('ignore')
D=pd.read_pickle('D.pkl'); S=pd.read_pickle('S.pkl')
def bateman(t,d,ka,ke,V): return d/V*ka/(ka-ke)*(np.exp(-ke*t)-np.exp(-ka*t))
def conc(t,kind,doses,tau,ka,ke,V):
    t=np.asarray(t,float)
    if kind=='single': return bateman(t,doses[0][1],ka,ke,V)
    if kind=='multi':
        out=np.zeros_like(t)
        for tk,d in doses:
            s=t-tk; m=s>=0
            out[m]+=bateman(s[m],d,ka,ke,V)
        return out
    te=np.where(t>tau,np.mod(t,tau),t) if kind=='ss' else t
    d=doses[0][1]; A=d/V*ka/(ka-ke)
    return A*(np.exp(-ke*te)/(1-np.exp(-ke*tau))-np.exp(-ka*te)/(1-np.exp(-ka*tau)))
def unpack(p):
    ke=np.exp(p[1]); ka=ke+0.02+np.exp(p[2]); return ka,ke,np.exp(p[0])
def fit_curve(cid):
    g=D[(D.cid==cid)&(D.c>0)]; r=S.loc[cid]
    t=g.t.values; y=np.log10(g.c.values)
    if len(g)<4: return None
    best=None
    for ke0 in [0.03,0.1,0.3]:
        for ka0 in [0.3,1.5,5]:
            V0=r.doses[0][1]/max(g.c.max(),1e-3)*1.0
            p0=[np.log(V0),np.log(ke0),np.log(max(ka0-ke0-0.02,0.05))]
            def res(p):
                ka,ke,V=unpack(p)
                with np.errstate(all='ignore'):
                    c=conc(t,r.kind,r.doses,r.tau,ka,ke,V)
                return np.log10(np.clip(c,1e-6,None))-y
            try:
                o=least_squares(res,p0,bounds=([-10,np.log(0.003),np.log(0.01)],[25,np.log(1.5),np.log(15)]))
            except Exception: continue
            if best is None or o.cost<best.cost: best=o
    if best is None: return None
    ka,ke,V=unpack(best.x)
    return dict(cid=cid,logV=np.log(V),logke=np.log(ke),logka=np.log(ka),rmse=np.sqrt(2*best.cost/len(g)),n=len(g))
if __name__=='__main__':
    F=pd.DataFrame([x for x in (fit_curve(c) for c in S.index) if x]).set_index('cid')
    F=F.join(S[['source_pmid','curve_label','analyte','kind','dose','preg','renal','ddi']])
    F['t_half']=np.log(2)/np.exp(F.logke)
    F.to_pickle('F.pkl'); pd.set_option('display.width',220)
    print(F[['source_pmid','curve_label','analyte','kind','n','rmse','t_half']].round(2).to_string())
    print('median in-sample rmse (log10)',F.rmse.median())
