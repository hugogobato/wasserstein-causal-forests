import pandas as pd, numpy as np, os
OUT="/tmp/opencode/wcf_pipeline"
K=25; grid=(np.arange(K,dtype=float)+0.5)/K
def wq(v,w,qs):
    v=np.asarray(v,float); w=np.asarray(w,float)
    m=np.isfinite(v)&np.isfinite(w)&(w>0); v,w=v[m],w[m]
    o=np.argsort(v); v,w=v[o],w[o]; cw=np.cumsum(w); cw/=cw[-1]
    return np.interp(qs,cw,v)
rows=[]
for yr in [10,15,19]:
    df=pd.read_stata(f"{OUT}/morg{yr}.dta",convert_categoricals=False)
    y=int(df["year"].iloc[0])
    base=(df["age"]>=16)&(df["age"]<=64)&(df["earnwke"]>0)&(df["earnwt"]>0)
    d=df[base]
    for st,g in d.groupby("stfips"):
        q=wq(g["earnwke"],g["earnwt"],grid)
        rows+=[dict(year=y,state_fips=int(st),k=k+1,u_k=round(grid[k],4),earnwke_q=float(q[k])) for k in range(K)]
    ns=d.groupby("stfips").size()
    print(f"{y}: N={len(d):,}, states={len(ns)}, per-state n: min={int(ns.min())} p25={int(ns.quantile(.25))} median={int(ns.median())} max={int(ns.max())}")
out=pd.DataFrame(rows)
out.to_csv(f"{OUT}/morg_state_weekly_earnings_quantiles_2010_2015_2019_long.csv",index=False)
out.pivot_table(index=["year","state_fips"],columns="k",values="earnwke_q").add_prefix("q_").reset_index().to_csv(
    f"{OUT}/morg_state_weekly_earnings_quantiles_2010_2015_2019_wide.csv",index=False)
print("saved MORG panel files")
