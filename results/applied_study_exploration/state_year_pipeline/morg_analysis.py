import pandas as pd, numpy as np
f = "/tmp/opencode/wcf_pipeline/morg19.dta"
df = pd.read_stata(f, convert_categoricals=False)
print("rows:", len(df))
print("year values:", df["year"].unique(), "ym range:", df["ym"].min(), df["ym"].max())
print("stfips unique:", df["stfips"].nunique(), "min/max:", df["stfips"].min(), df["stfips"].max())
print("weight col stats: earnwt>0:", (df["earnwt"]>0).mean(), " weight>0:", (df["weight"]>0).mean())
print("age 16-64 share:", ((df["age"]>=16)&(df["age"]<=64)).mean())
print("earnwke>0:", (df["earnwke"]>0).sum(), "earnhre>0:", (df["earnhre"]>0).sum(), "paidhre:", df["paidhre"].value_counts().to_dict(), "hourslw:", df["hourslw"].value_counts().to_dict())

def wq(v, w, qs):
    v = np.asarray(v, float); w = np.asarray(w, float)
    m = np.isfinite(v) & np.isfinite(w) & (w>0)
    v, w = v[m], w[m]
    o = np.argsort(v); v, w = v[o], w[o]
    cw = np.cumsum(w); cw /= cw[-1]
    return np.interp(qs, cw, v)

mask = (df["age"]>=16)&(df["age"]<=64)&(df["earnwke"]>0)&(df["earnwt"]>0)
d = df[mask]
qs = np.array([0.10,0.25,0.5,0.75,0.90])
print("\nMORG 2019, age 16-64, earnwke>0, earnwt weight; N =", len(d))
print("US weighted quantiles of earnwke (weekly $):", np.round(wq(d["earnwke"], d["earnwt"], qs),2))
print("BLS checkpoint ~ median usual weekly earnings 2019: 917 (full-time wage&salary)")

# hourly, paid hourly workers
m2 = mask & (df["paidhre"]==1) & (df["earnhre"]>0)
d2 = df[m2]
print("\nPaid-hourly only; N =", len(d2), " USA weighted quantiles earnhre ($/h):", np.round(wq(d2["earnhre"], d2["earnwt"], qs),2))

# per-state counts
cnt = d.groupby("stfips").size()
print("\nPer-state ORG sample sizes (earnwke>0, 16-64), 2019:")
print("n states:", len(cnt), " min:", int(cnt.min()), " p25:", int(cnt.quantile(.25)), " median:", int(cnt.median()), " p75:", int(cnt.quantile(.75)), " max:", int(cnt.max()))
print("smallest 10 states:", cnt.nsmallest(10).to_dict())
# hourly per-state counts
cnt2 = d2.groupby("stfips").size()
print("paid-hourly per-state: median:", int(cnt2.median()), "min:", int(cnt2.min()), "max:", int(cnt2.max()))
# state-level median weekly earnings for CA/NY/TX
for st in [6,36,48]:
    dd = d[d["stfips"]==st]
    print(f"state {st}: N={len(dd)}, med earnwke={wq(dd['earnwke'],dd['earnwt'],[0.5])[0]:.0f}")
