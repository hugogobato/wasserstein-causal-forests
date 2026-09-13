import pandas as pd, numpy as np
OUT="/tmp/opencode/wcf_pipeline"
need=set(["AGEP","WAGP","PWGTP","WKHP","WKWN","ADJINC"])
def wq(v,w,qs):
    v=np.asarray(v,float); w=np.asarray(w,float); o=np.argsort(v); v,w=v[o],w[o]
    cw=np.cumsum(w); cw/=cw[-1]; return np.interp(qs,cw,v)
qs=np.array([.5])
for part in ["psam_pusa.csv","psam_pusb.csv"]:
    df=pd.read_csv(f"{OUT}/{part}",usecols=lambda c:c in need,low_memory=False)
    df["wagp_adj"]=df.WAGP*(df.ADJINC/1e6)
    d=df[(df.AGEP>=16)&(df.AGEP<=64)&(df.WAGP>0)&(df.PWGTP>0)]
    ftyr=d[(d.WKHP>=35)&(d.WKWN>=50)]
    print(part,"US all wage earners 16-64 median:",round(wq(d.wagp_adj,d.PWGTP,qs)[0],0),
          "| FTYR median:",round(wq(ftyr.wagp_adj,ftyr.PWGTP,qs)[0],0),
          "| n:",len(d),"ftyr:",len(ftyr))
    if part=="psam_pusa.csv":
        store=[d[["wagp_adj","PWGTP","AGEP","WKHP","WKWN"]], ftyr[["wagp_adj","PWGTP"]]]
    else:
        d2=pd.concat([store[0],d[["wagp_adj","PWGTP","AGEP","WKHP","WKWN"]]])
        f2=pd.concat([store[1],ftyr[["wagp_adj","PWGTP"]]])
        print("US TOTAL all wage earners 16-64 median:",round(wq(d2.wagp_adj,d2.PWGTP,qs)[0],0),
              "| FTYR median:",round(wq(f2.wagp_adj,f2.PWGTP,qs)[0],0))
