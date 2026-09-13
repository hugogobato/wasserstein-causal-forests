import pandas as pd, numpy as np, os, json, time

OUT = "/tmp/opencode/wcf_pipeline"
K = 25
grid = (np.arange(K, dtype=float) + 0.5) / K
rep_cols = [f"PWGTP{i}" for i in range(1, 81)]
need = set(["ST","AGEP","WAGP","PWGTP","WKHP","WKWN","ADJINC","SCHL","SEX"] + rep_cols)

def sq(v_sorted, w_sorted, qs):
    cw = np.cumsum(w_sorted); cw /= cw[-1]
    return np.interp(qs, cw, v_sorted)

rows, sums = [], []
t0 = time.time()
for part in ["psam_pusa.csv", "psam_pusb.csv"]:
    df = pd.read_csv(os.path.join(OUT, part), usecols=lambda c: c in need, low_memory=False)
    df["wagp_adj"] = df["WAGP"] * (df["ADJINC"] / 1e6)
    d = df[(df["AGEP"] >= 16) & (df["AGEP"] <= 64) & (df["WAGP"] > 0) & (df["PWGTP"] > 0)]
    # US-level
    for st, g in d.groupby("ST"):
        o = np.argsort(g["wagp_adj"].values)
        vs = g["wagp_adj"].values[o]
        w0 = g["PWGTP"].values[o].astype(float)
        q0 = sq(vs, w0, grid)
        W = g[rep_cols].values[o].astype(float)
        Qr = np.empty((80, K))
        for r in range(80):
            Qr[r] = sq(vs, W[:, r], grid)
        se = np.sqrt(4.0 / 80.0 * ((Qr - q0[None, :]) ** 2).sum(axis=0))
        ftyr = g[(g["WKHP"] >= 35) & (g["WKWN"] >= 50)]
        med_ftyr = float(np.interp(0.5, np.cumsum(ftyr["PWGTP"].values[np.argsort(ftyr["wagp_adj"].values)]) /
                                   ftyr["PWGTP"].sum(), np.sort(ftyr["wagp_adj"].values))) if len(ftyr) else np.nan
        rows += [dict(state_fips=int(st), k=k + 1, u_k=round(grid[k], 4),
                      wage_q=float(q0[k]), wage_q_se=float(se[k])) for k in range(K)]
        sums.append(dict(state_fips=int(st), n_records=int(len(g)), n_wage_earners_16_64=int((g["WAGP"] > 0).sum()),
                         sum_PWGTP=float(g["PWGTP"].sum()), n_ftyr=int(len(ftyr)),
                         median_wage_all_16_64=float(q0[12]), median_wage_ftyr=med_ftyr,
                         se_median=float(se[12]), cv_median=float(se[12] / max(q0[12], 1)),
                         cv_p90=float(se[22] / max(q0[22], 1))))
    print(f"{part} processed, elapsed {time.time()-t0:.0f}s, states so far {len(sums)}", flush=True)

qdf = pd.DataFrame(rows); sdf = pd.DataFrame(sums)
qdf.to_csv(os.path.join(OUT, "acs_pums_all_states_2019_quantiles_long.csv"), index=False)
sdf.to_csv(os.path.join(OUT, "acs_pums_all_states_2019_summary.csv"), index=False)
wide = qdf.pivot(index="state_fips", columns="k", values="wage_q").add_prefix("q_")
wide.to_csv(os.path.join(OUT, "acs_pums_all_states_2019_quantiles_wide.csv"))
print("\nstates:", len(sdf), " total wage earners 16-64:", int(sdf.n_wage_earners_16_64.sum()))
print(sdf[["n_wage_earners_16_64","median_wage_all_16_64","cv_median","cv_p90"]].describe().round(3).to_string())
print("\nSample of summary:")
print(sdf.sort_values("state_fips").head(8).to_string(index=False))
print("elapsed", round(time.time()-t0), "s")
