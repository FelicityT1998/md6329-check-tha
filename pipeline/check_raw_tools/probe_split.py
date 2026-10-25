import sys, pandas as pd
t0 = pd.read_csv(sys.argv[1], sep="\t", low_memory=False).set_index("nsin")
h = pd.read_csv(sys.argv[2])
h["sex"] = h.nsin.map(t0.T0_SEXE).map({1: "male", 2: "female"})
h["edu_high"] = h.nsin.map(t0.T0_a08).ge(6).map({True: "edu>=HBO", False: "edu<HBO"})
h["care"] = h.nsin.map(t0.T0_f01_geen).map({1: "no care", 0: "any care"})
h["pain"] = [t0.loc[n, f"T0_JVHPIJN{s}"] for n, s in zip(h.nsin, h.side)]
for c in ["sex", "edu_high", "care", "pain", "group", "img_fmt"]:
    g = h.groupby(c, dropna=False).agg(hips=("event", "size"), events=("event", "sum"), subjects=("nsin", "nunique"))
    print(c, g.to_dict("index"))
