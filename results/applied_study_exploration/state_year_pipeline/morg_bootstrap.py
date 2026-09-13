import pandas as pd, numpy as np
rng=np.random.default_rng(0)
OUT="/tmp/opencode/wcf_pipeline"
df=pd.read_stata(f"{OUT}/morg19.dta",convert_categoricals=False)
base=(df.age>=16)&(df.age<=64)&(df.earnwke>0)&(df.earnwt>0)
d=df[base]
grid=(np.arange(25)+0.5)/25
def wq(v,w,qs):
    o=np.argsort(v); v,w=v[o],w[o]; cw=np.cumsum(w); cw/=cw[-1]
    return np.interp(qs,cw,v)
print("MORG 2019 bootstrap (200 reps, resample persons, earnwt):")
for st,nm in [(6,'CA'),(36,'NY'),(48,'TX'),(23,'ME')]:
    g=d[d.stfips==st]
    v=g.earnwke.values.astype(float); w=g.earnwt.values.astype(float)
    q0=wq(v,w,grid)
    B=200; Q=np.empty((B,25))
    for b in range(B):
        idx=rng.integers(0,len(v),len(v))
        Q[b]=wq(v[idx],w[idx],grid)
    se=Q.std(0,ddof=1)
    print(f"  {nm}: N={len(v)}, u=.02 CV={se[0]/q0[0]:.3f}, u=.10 CV={se[2]/q0[2]:.3f}, u=.50 CV={se[12]/q0[12]:.3f}, u=.90 CV={se[22]/q0[22]:.3f}, u=.98 CV={se[24]/q0[24]:.3f}")
