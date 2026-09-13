import pandas as pd, numpy as np, os

OUT = "/tmp/opencode/wcf_pipeline"
files = {"06": ("psam_p06.csv", "California"), "36": ("psam_p36.csv", "New York"), "48": ("psam_p48.csv", "Texas")}
K = 25
grid = (np.arange(K, dtype=float) + 0.5) / K
rep_cols = [f"PWGTP{i}" for i in range(1, 81)]
need = set(["ST","PUMA","AGEP","WAGP","PWGTP","WKHP","WKWN","SCHL","SEX","ESR","ADJINC"] + rep_cols)

def sorted_quantiles(v_sorted, w_sorted, qs):
    cw = np.cumsum(w_sorted); cw /= cw[-1]
    return np.interp(qs, cw, v_sorted)

rows, summaries = [], []
for fips, (fn, name) in files.items():
    df = pd.read_csv(os.path.join(OUT, fn), usecols=lambda c: c in need, low_memory=False)
    n_all = len(df)
    popw = float(df["PWGTP"].sum())
    adj = float(df["ADJINC"].median()) / 1e6
    d = df[(df["AGEP"] >= 16) & (df["AGEP"] <= 64) & (df["WAGP"] > 0) & (df["PWGTP"] > 0)].copy()
    d["wagp_adj"] = d["WAGP"] * (d["ADJINC"] / 1e6)
    o = np.argsort(d["wagp_adj"].values)
    vs = d["wagp_adj"].values[o]
    w0 = d["PWGTP"].values[o].astype(float)
    q0 = sorted_quantiles(vs, w0, grid)
    Wrep = d[rep_cols].values[o].astype(float)
    Qrep = np.empty((80, K))
    for r in range(80):
        Qrep[r] = sorted_quantiles(vs, Wrep[:, r], grid)
    se = np.sqrt(4.0 / 80.0 * ((Qrep - q0[None, :]) ** 2).sum(axis=0))
    ftyr = d[(d["WKHP"] >= 35) & (d["WKWN"] >= 50)]
    fv = np.sort(ftyr["wagp_adj"].values)
    fw = ftyr["PWGTP"].values[np.argsort(ftyr["wagp_adj"].values)].astype(float)
    med_fty = sorted_quantiles(fv, fw, np.array([0.5]))[0]
    hr = (ftyr["wagp_adj"] / (ftyr["WKHP"].astype(float) * ftyr["WKWN"].astype(float)).clip(1)).values
    oo = np.argsort(hr)
    med_hr = sorted_quantiles(hr[oo], ftyr["PWGTP"].values[oo].astype(float), np.array([0.5]))[0]
    cv = se / np.maximum(q0, 1.0)
    rows += [dict(state_fips=fips, state=name, year=2019, k=k + 1, u_k=round(grid[k], 4),
                  wage_q=q0[k], wage_q_se=se[k], cv=cv[k]) for k in range(K)]
    summaries.append(dict(state_fips=fips, state=name, year=2019, adjinc=adj,
                          n_person_records=n_all, sum_PWGTP=popw,
                          n_wage_earners_16_64=len(d), sum_PWGTP_wage=float(d["PWGTP"].sum()),
                          n_ftyr=int(len(ftyr)),
                          median_wage_all_wage_earners_16_64=float(q0[12]),
                          median_wage_ftyr=float(med_fty), median_implied_hourly_ftyr=float(med_hr),
                          se_median=q0[12] and float(se[12]), cv_median=float(cv[12]),
                          cv_p90=float(cv[22]), cv_p10=float(cv[2])))
    print(f"{name}: records={n_all}, sum PWGTP={popw:,.0f}, wage earners 16-64 WAGP>0={len(d):,}, FTYR={len(ftyr):,}")
    print(f"   grid p10={q0[2]:,.0f} p25={q0[6]:,.0f} p50={q0[12]:,.0f} (SE {se[12]:,.0f}) p75={q0[18]:,.0f} p90={q0[22]:,.0f}; CV p10={cv[2]:.3f} p50={cv[12]:.3f} p90={cv[22]:.3f}")
    print(f"   FTYR median WAGP={med_fty:,.0f}; implied hourly median={med_hr:.2f}; ADJINC={adj}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "acs_pums_state_wage_quantiles.csv"), index=False)
pd.DataFrame(summaries).to_csv(os.path.join(OUT, "acs_pums_state_summary.csv"), index=False)
print("\nSaved acs_pums_state_wage_quantiles.csv and acs_pums_state_summary.csv")
