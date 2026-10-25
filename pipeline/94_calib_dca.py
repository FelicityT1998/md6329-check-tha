"""Calibration curves and decision curves for the main models (repeat-0 out-of-fold predictions, Platt-calibrated).

usage: python -I 94_calib_dca.py <project dir>
writes: data/derived/figs/res_calibration_dca.png, data/derived/results/dca.csv
Calibration: deciles of predicted risk (mean predicted vs observed, Wilson 95% CI) + identity line.
Decision curve (Vickers 2006): net benefit = TP/n - FP/n * pt/(1-pt) over threshold pt in [1%, 40%], vs treat-all / treat-none.
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = pathlib.Path(sys.argv[1]); O = P / "data/derived/oof"; FG = P / "data/derived/figs"; R = P / "data/derived/results"
MODELS = [("m0_DESCLB_mice", "M0 elastic net (clinical)", "#2a78d6"), ("m2ens_DESCLB_lam1_dtinit_es_s3", "M2 FT-Transformer (clinical)", "#1baf7a"),
          ("m5_pix_logit_DESCLB", "M0 + pixel image score", "#eb6834")]
INK2, MUTED, GRID, SURF = "#52514e", "#898781", "#e4e3dd", "#fcfcfb"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "legend.frameon": False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
a1.plot([0, 0.6], [0, 0.6], color=MUTED, lw=1, ls=(0, (3, 3)))
pts = np.linspace(0.01, 0.40, 79); rows = []
for stem, lab, col in MODELS:
    d = pd.read_parquet(O / f"{stem}.parquet"); d = d[(d.repeat == 0) & d.y.notna()]
    y, p = d.y.values, d.p_cal.values
    q = pd.qcut(p, 10, labels=False, duplicates="drop"); g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
    mp, ob, n = g.p.mean(), g.y.mean(), g.y.size()
    z = 1.96; lo = (ob + z * z / (2 * n) - z * np.sqrt(ob * (1 - ob) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    hi = (ob + z * z / (2 * n) + z * np.sqrt(ob * (1 - ob) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    a1.errorbar(mp, ob, yerr=[ob - lo, hi - ob], fmt="o-", color=col, ms=5, lw=1.5, capsize=0, label=lab, mec=SURF, mew=1)
    nb = [((p >= t) & (y == 1)).mean() - ((p >= t) & (y == 0)).mean() * t / (1 - t) for t in pts]
    a2.plot(pts, nb, color=col, lw=2, label=lab)
    rows += [{"model": stem, "threshold": t, "net_benefit": v} for t, v in zip(pts, nb)]
prev = y.mean()
a2.plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], color=MUTED, lw=1.2, ls=(0, (3, 3)), label="treat all")
a2.axhline(0, color=INK2, lw=1, label="treat none")
a1.set_xlabel("predicted 10-year THA risk (decile mean)"); a1.set_ylabel("observed proportion (95% CI)"); a1.set_xlim(0, 0.6); a1.set_ylim(0, 0.75)
a1.set_title("Calibration (repeat 0, out-of-fold)", loc="left", fontsize=10.5); a1.legend(loc="upper left", fontsize=8)
a2.set_xlabel("threshold probability"); a2.set_ylabel("net benefit"); a2.set_ylim(-0.01, 0.06)
a2.set_title("Decision curve", loc="left", fontsize=10.5); a2.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(FG / "res_calibration_dca.png", dpi=200)
pd.DataFrame(rows).to_csv(R / "dca.csv", index=False)
D = pd.DataFrame(rows).pivot(index="threshold", columns="model", values="net_benefit")
print(D.loc[D.index[[4, 9, 19, 29]]].round(4).to_string())
