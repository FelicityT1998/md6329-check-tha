import sys, pandas as pd, numpy as np
t0 = pd.read_csv(sys.argv[1], sep="\t", low_memory=False).set_index("nsin")
img = pd.read_csv(sys.argv[2]); dev = pd.read_csv(sys.argv[3], dtype=str).fillna("")
dev["nsin"] = dev.nsin.astype(int)
img = img.drop_duplicates("nsin")
tif = img[img.fmt == "tif"].copy()
tif["sig"] = "TIF|" + tif.BitsStored.astype(str) + "bit|" + tif.Software.fillna("none").str[:12] + "|" + np.where(tif.Rows > 3500, "large", "small")
d = dev.drop_duplicates("nsin").copy()
fam = lambda s: "A19" if s == "A19" else "VIPS" if s.startswith("VIPS") else "sol" if s.startswith("sol") else "Siemens-V" if s[:2] in ("VA","VB") else "4.x/3.x" if s[:1] in "34" else "Version1.3" if s.startswith("Version") else "none"
d["sig"] = "DCM|" + d.sw.map(fam) + "|" + d.rows + "x" + d.cols + "|" + d.bits + "bit"
sig = pd.concat([tif.set_index("nsin").sig, d.set_index("nsin").sig])
t0["sig"] = sig
t0["rom_missing"] = t0.T0_HEXOL.isna()
t0["pain_dir_missing"] = t0.T0_HEXOPIJNL.isna()
g = t0.groupby("sig").agg(n=("rom_missing", "size"), rom_miss=("rom_missing", "mean"), exo_pain_miss=("pain_dir_missing", "mean"),
                          interval_grp=("T0_JVCON", lambda x: (x == 1).mean()), female=("T0_SEXE", lambda x: (x == 2).mean()),
                          age=("T0_Lft_T0", "mean"), no_care=("T0_f01_geen", "mean"))
g = g[g.n >= 8].sort_values("n", ascending=False).round(2)
print(g.to_string()); print("subjects in clusters n>=8:", int(g.n.sum()), "/", int(t0.sig.notna().sum()))
# purity score: share of subjects in clusters whose ROM-missing rate is <10% or >90%
pure = g[(g.rom_miss < .1) | (g.rom_miss > .9)].n.sum() / g.n.sum()
print("ROM-protocol purity (clusters with <10% or >90% missing):", round(pure, 2))
