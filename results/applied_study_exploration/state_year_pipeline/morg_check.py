import pandas as pd, numpy as np
df = pd.read_stata("/tmp/opencode/wcf_pipeline/morg19.dta", convert_categoricals=False)
def wq(v,w,qs):
    v=np.asarray(v,float); w=np.asarray(w,float)
    m=np.isfinite(v)&np.isfinite(w)&(w>0); v,w=v[m],w[m]
    o=np.argsort(v); v,w=v[o],w[o]; cw=np.cumsum(w); cw/=cw[-1]
    return np.interp(qs,cw,v)
qs=np.array([.1,.25,.5,.75,.9])
base=(df.age>=16)&(df.earnwke>0)&(df.earnwt>0)
ft=base&(df.hourslw>=35)
print("Full-time (hourslw>=35), all ages: N=",ft.sum(), " US earnwke quantiles:", np.round(wq(df.earnwke[ft],df.earnwt[ft],qs),0))
print("  -> BLS 2019 published median usual weekly earnings, FT wage & salary = $917 (check)")
for st,nm in [(6,'CA'),(36,'NY'),(48,'TX')]:
    m=ft&(df.stfips==st)
    print(f"  {nm} FT median weekly = {wq(df.earnwke[m],df.earnwt[m],[.5])[0]:.0f}, N={m.sum()}")
# paid hourly full-time
ph=base&(df.paidhre==1)&(df.earnhre>0)
print("\nAll paid-hourly (any hours): US earnhre(/100) quantiles:", np.round(wq(df.earnhre[ph]/100,df.earnwt[ph],qs),2))
phft=ph&(df.hourslw>=35)
print("FT paid-hourly: US earnhre(/100) quantiles:", np.round(wq(df.earnhre[phft]/100,df.earnwt[phft],qs),2), " N=",phft.sum())
print("  -> BLS 2019 published median hourly earnings, paid hourly rates = ?")
