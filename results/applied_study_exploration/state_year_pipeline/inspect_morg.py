import pandas as pd, numpy as np
f = "/tmp/opencode/wcf_pipeline/morg19.dta"
r = pd.read_stata(f, convert_categoricals=False)
print("shape:", r.shape)
print("columns:", list(r.columns))
