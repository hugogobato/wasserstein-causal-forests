import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/tmp/opencode/wcf_pipeline"
q=pd.read_csv(f"{OUT}/acs_pums_all_states_2019_quantiles_long.csv")
summ=pd.read_csv(f"{OUT}/acs_pums_all_states_2019_summary.csv")
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
ax=axes[0]
for st,g in q.groupby("state_fips"):
    g=g.sort_values("u_k")
    ax.plot(g.u_k, g.wage_q/1000, color="0.8", lw=0.6)
for st,c in [(6,"tab:blue"),(36,"tab:red"),(48,"tab:green")]:
    g=q[q.state_fips==st].sort_values("u_k")
    ax.plot(g.u_k, g.wage_q/1000, color=c, lw=2, label={6:"CA",36:"NY",48:"TX"}[st])
ax.set_xlabel("quantile level u (K=25 midpoint grid)"); ax.set_ylabel("annual wage income ($1000s)")
ax.set_title("ACS PUMS 2019: state wage-income quantile curves (51 states)"); ax.legend(); ax.grid(alpha=.25)
ax=axes[1]
cv=q.assign(cv=q.wage_q_se/q.wage_q).groupby("k").agg(med=("cv","median"), zero=("wage_q_se", lambda s:(s==0).mean()), k=("u_k","first"))
ax.plot(cv.k, 100*cv.med, "o-", color="tab:purple", label="median CV across states (nonzero SE)")
ax.plot(cv.k, 100*cv["zero"], "s--", color="0.4", label="share with zero replicate SE")
ax.set_xlabel("quantile level u"); ax.set_ylabel("percent")
ax.set_title("Sampling error across the grid (ACS 2019, SDR replicate weights)"); ax.legend(); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(f"{OUT}/prototype_figures.png", dpi=140)
print("saved", f"{OUT}/prototype_figures.png")
print(cv.to_string(index=False))
