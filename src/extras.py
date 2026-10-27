"""Three follow-up experiments (run after prep.py and model.py):
  A. dose-normalised GBM control for the dose-extrapolation test
  B. leave-one-paper-out with / without PMID 33252495
  C. learning curve: LOPO MAE vs number of training papers
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import model as mdl

D = pd.read_pickle('D.pkl'); S = pd.read_pickle('S.pkl'); F = pd.read_pickle('F.pkl')
S['akey'] = ['tfv_after_TAF' if (r.source_pmid == 27216057 and r.analyte == 'tenofovir') else r.analyte for r in S.itertuples()]
F['akey'] = S.loc[F.index, 'akey']
D = D[D.cid.isin(F.index) & (D.c > 0)].copy()
D['akey'] = S.loc[D.cid, 'akey'].values
D['ldose'] = np.log10(D.dose); D['tauf'] = D.tau.fillna(0).astype(float)
D['kindc'] = D.kind.map({'single': 0, 'multi': 1, 'ss': 2, 'prior': 3}); D['ly'] = np.log10(D.c)
keys = sorted(D.akey.unique()); P = ['logV', 'logka', 'logke']

def feats(df, drop_dose=False):
    X = df[['t', 'ldose', 'tauf', 'kindc', 'preg', 'renal', 'ddi']].copy()
    for k in keys: X['a_' + k] = (df.akey == k).astype(int)
    return X.drop(columns=['ldose']) if drop_dose else X

def ccov(df):
    X = df[['preg', 'renal', 'ddi']].copy(); X.insert(0, 'ldose', np.log10(S.loc[df.index, 'dose']))
    for k in keys: X['a_' + k] = (df.akey == k).astype(int)
    return X

def pm(cid, par):
    r = S.loc[cid]; g = D[D.cid == cid]
    ke = np.exp(np.clip(par['logke'], np.log(0.003), np.log(1.5)))
    ka = ke + 0.02 + max(np.exp(par['logka']) - np.exp(par['logke']) - 0.02, 0.01)
    with np.errstate(all='ignore'):
        c = mdl.conc(g.t.values, r.kind, r.doses, r.tau, ka, ke, np.exp(par['logV']))
    return np.log10(np.clip(c, 1e-3, None))

def gbm(): return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)

def fold(train_c, test_c, gbm_variants=True):
    """Train on curve ids train_c, return per-curve MAE dict list for test_c."""
    tr = D[D.cid.isin(train_c)]; Ftr = F.loc[list(train_c)]
    m1 = tr.groupby('akey').ly.mean(); m0 = tr.ly.mean()
    g1 = gbm().fit(feats(tr), tr.ly)
    if gbm_variants:
        # dose-normalised target: log10(C/D); dose scaling is built in, as in the mechanistic model
        g2 = gbm().fit(feats(tr), tr.ly - tr.ldose)
        g3 = gbm().fit(feats(tr, True), tr.ly - tr.ldose)   # ... and without dose as a feature
    med_a = Ftr.groupby('akey')[P].median(); med_g = Ftr[P].median()
    Xc = ccov(Ftr); mu = Xc.mean(); sd = Xc.std().replace(0, 1)
    rid = Ridge(alpha=3.0).fit((Xc - mu) / sd, Ftr[P])
    out = []
    for cid in test_c:
        g = D[D.cid == cid]
        if len(g) == 0: continue
        y = g.ly.values; ak = F.loc[cid, 'akey']
        par = med_a.loc[ak] if ak in med_a.index else med_g
        ph = pd.Series(rid.predict((ccov(F.loc[[cid]]) - mu) / sd)[0], index=P)
        e = dict(cid=cid, paper=F.loc[cid, 'source_pmid'], label=F.loc[cid, 'curve_label'], dose=S.loc[cid, 'dose'], n=len(g),
                 analyte_mean=np.abs(y - m1.get(ak, m0)).mean(),
                 GBM=np.abs(y - g1.predict(feats(g))).mean(),
                 mech_pop=np.abs(y - pm(cid, par)).mean(), hybrid=np.abs(y - pm(cid, ph)).mean())
        if gbm_variants:
            e['GBM_norm'] = np.abs(y - (g2.predict(feats(g)) + g.ldose.values)).mean()
            e['GBM_norm_nodose'] = np.abs(y - (g3.predict(feats(g, True)) + g.ldose.values)).mean()
        out.append(e)
    return out

def paper_boot(pm_df, a, b, n=2000, seed=0):
    d = (pm_df[a] - pm_df[b]); rng = np.random.default_rng(seed)
    bs = [d.iloc[rng.choice(len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d < 0).sum()), len(d)

allc = list(F.index); papers = sorted(D.paper.unique())
which = sys.argv[1:] or ['A', 'B', 'C']

# ---------------------------------------------------------------- A
if 'A' in which:
    print('=== A. dose extrapolation, PMID 11557462 (leave one dose out) ===')
    rows = []
    for cid in F.index[F.source_pmid == 11557462]:
        rows += fold([c for c in allc if c != cid], [cid])
    A = pd.DataFrame(rows).sort_values('dose')
    cols = ['GBM', 'GBM_norm', 'GBM_norm_nodose', 'mech_pop', 'hybrid']
    print(A[['dose'] + cols].round(3).to_string(index=False)); print('mean'); print(A[cols].mean().round(3).to_string())
    A.to_csv('../results/dose_control_per_curve.csv', index=False)
    # same test with ALL doses of that paper removed (paper-shift + dose-shift), for reference
    rows = []
    ids = list(F.index[F.source_pmid == 11557462])
    rows += fold([c for c in allc if c not in ids], ids)
    B = pd.DataFrame(rows).sort_values('dose')
    print('\nSame paper, ALL its doses held out (no same-study doses in training):')
    print(B[['dose'] + cols].round(3).to_string(index=False)); print('mean'); print(B[cols].mean().round(3).to_string())

# ---------------------------------------------------------------- B
if 'B' in which:
    print('\n=== B. LOPO with / without PMID 33252495 ===')
    npts = D.groupby(['paper', 'cid']).size().groupby('paper').agg(['min', 'median', 'max', 'count'])
    print('points per curve by paper (min / median / max / n curves):'); print(npts.to_string())
    cols = ['analyte_mean', 'GBM', 'GBM_norm', 'mech_pop', 'hybrid']
    def lopo(train_exclude=()):
        rows = []
        for p in papers:
            if p in train_exclude: continue
            tr = [c for c in allc if F.loc[c, 'source_pmid'] != p and F.loc[c, 'source_pmid'] not in train_exclude]
            rows += fold(tr, list(F.index[F.source_pmid == p]))
        return pd.DataFrame(rows)
    full = lopo(); full.to_csv('../results/lopo_full_extras.csv', index=False)
    noX = lopo(train_exclude=(33252495,)); noX.to_csv('../results/lopo_without_33252495_extras.csv', index=False)
    for name, R in [('(1) all 12 papers, 33252495 in train and test', full),
                    ('(2) 33252495 kept in TRAINING but dropped from the test average', full[full.paper != 33252495]),
                    ('(3) 33252495 removed from training AND test (11 papers)', noX)]:
        pmx = R.groupby('paper')[cols].mean()
        print(f'\n--- {name}: {len(R)} curves, {len(pmx)} papers'); print(pmx.mean().round(3).to_string())
        for a, b in [('hybrid', 'GBM'), ('mech_pop', 'GBM'), ('hybrid', 'GBM_norm')]:
            m, lo, hi, nb, nt = paper_boot(pmx, a, b)
            print(f'  {a} - {b}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {a} better in {nb}/{nt} papers')

# ---------------------------------------------------------------- C
if 'C' in which:
    print('\n=== C. learning curve (train on k random other papers, test on the held-out paper) ===')
    ks = [2, 4, 6, 8, 11]; REPS = 10
    cols = ['analyte_mean', 'GBM', 'GBM_norm', 'mech_pop', 'hybrid']
    rng = np.random.default_rng(123); rows = []
    for p in papers:
        others = [q for q in papers if q != p]
        te = list(F.index[F.source_pmid == p])
        for k in ks:
            reps = 1 if k >= len(others) else REPS
            for r in range(reps):
                sub = list(rng.choice(others, k, replace=False)) if k < len(others) else others
                tr = [c for c in allc if F.loc[c, 'source_pmid'] in sub]
                for e in fold(tr, te): e.update(k=k, rep=r, test_paper=p); rows.append(e)
        print('  done', p, flush=True)
    L = pd.DataFrame(rows); L.to_csv('../results/learning_curve_per_curve.csv', index=False)
    # per (k, rep): mean over test papers of the paper-mean MAE; then mean/percentiles over reps
    pp = L.groupby(['k', 'rep', 'test_paper'])[cols].mean().reset_index()
    per_rep = pp.groupby(['k', 'rep'])[cols].mean().reset_index()
    summ = per_rep.groupby('k')[cols].agg(['mean', lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
    print(per_rep.groupby('k')[cols].mean().round(3).to_string())
    per_rep.to_csv('../results/learning_curve_summary.csv', index=False)
    # hybrid - GBM per k with paper-level bootstrap (reps averaged within paper first)
    print('\nhybrid - GBM and mech_pop - GBM by k (paper bootstrap over the 12 test papers):')
    for k in ks:
        pmx = pp[pp.k == k].groupby('test_paper')[cols].mean()
        for a in ['hybrid', 'mech_pop']:
            m, lo, hi, nb, nt = paper_boot(pmx, a, 'GBM')
            print(f'  k={k:2d} {a:9s} - GBM: {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  better in {nb}/{nt} papers')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    sty = {'GBM': ('#14213D', 'GBM (black box)'), 'mech_pop': ('#1B6B73', 'Mechanistic (drug median)'),
           'hybrid': ('#B4540F', 'Hybrid (ridge on V, ka, ke)'), 'GBM_norm': ('#6B4FA0', 'GBM on log10(C/dose) (dose-scaled black box)'),
           'analyte_mean': ('#9AA3AE', 'Drug mean baseline')}
    for m, (c, lab) in sty.items():
        g = per_rep.groupby('k')[m]; mu = g.mean(); lo = g.quantile(.1); hi = g.quantile(.9)
        ax.plot(mu.index, mu.values, '--s' if m == 'GBM_norm' else '-o', color=c, label=lab, lw=2)
        ax.fill_between(mu.index, lo.values, hi.values, color=c, alpha=.12)
    ax.set_xlabel('Number of training papers (k)'); ax.set_ylabel('Paper-level MAE of log10 concentration')
    ax.set_xticks(ks); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9)
    ax.set_title('Learning curve, leave-one-paper-out (band: 10th-90th pct over random training subsets)', fontsize=9)
    fig.tight_layout(); fig.savefig('../results/learning_curve.png', dpi=200)
