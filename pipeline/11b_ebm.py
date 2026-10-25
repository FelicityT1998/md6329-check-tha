"""M1b Explainable Boosting Machine (GA2M) - the generalised-additive family Jang et al. 2023 used for 10-year hip
replacement in OAI. Same nested repeated grouped CV as M0/M1.

usage: python -I 11b_ebm.py <groups=DESCLB> [n_jobs=7]
writes: data/derived/oof/m1b_<groups>.parquet + _selection.csv + _importance.csv + .json

Small inner grid (EBMs are slow-ish and robust to tuning): interactions {0, 5} x learning_rate {0.005, 0.02};
max_bins 64, min_samples_leaf 10, outer_bags 8; scored by inner Platt-recalibrated log-loss. NaN kept (EBM bins missing).
"""
import sys, json, pathlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from interpret.glassbox import ExplainableBoostingClassifier
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, fold_rows, D
from lib_eval import Platt, summarise, cluster_boot, select_score

warnings.filterwarnings("ignore")
GROUPS = sys.argv[1] if len(sys.argv) > 1 else "DESCLB"
NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 7
H, FD, F = load(); spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec); y = H.y10.values
GRID = [(i, lr) for i in (0, 5) for lr in (0.005, 0.02)]
OUT = D / "oof"

def model(inter, lr):
    return ExplainableBoostingClassifier(interactions=inter, learning_rate=lr, max_bins=64, min_samples_leaf=10, outer_bags=8,
                                         n_jobs=1, random_state=0)

def run(rep, k):
    tr, te = fold_rows(H, F, rep, k)
    inner = []
    for j in range(3):
        itr, iva = fold_rows(H, F, rep, k, j)
        pp = TabPrep(spec, "none", scale=False); inner.append((pp.fit_transform(Xraw.iloc[itr]), y[itr], pp.transform(Xraw.iloc[iva]), y[iva]))
    scores, oof_in = [], []
    for g in GRID:
        ps, ys = [], []
        for Xa, ya, Xb, yb in inner: ps.append(model(*g).fit(Xa, ya).predict_proba(Xb)[:, 1]); ys.append(yb)
        p, yy = np.concatenate(ps), np.concatenate(ys); scores.append(select_score(p, yy)); oof_in.append((p, yy))
    best = int(np.argmin(scores))
    pp = TabPrep(spec, "none", scale=False); Xa = pp.fit_transform(Xraw.iloc[tr]); Xb = pp.transform(Xraw.iloc[te])
    m = model(*GRID[best]).fit(Xa, y[tr]); p = m.predict_proba(Xb)[:, 1]; cal = Platt().fit(*oof_in[best])
    imp = pd.Series(m.term_importances(), index=m.term_names_, name=f"{rep}_{k}")
    return (pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": y[te],
                          "p_raw": p, "p_cal": cal(p), "cfg": best}),
            {"repeat": rep, "outer": k, "interactions": GRID[best][0], "lr": GRID[best][1], "inner_score": scores[best]}, imp)

res = Parallel(n_jobs=NJ, verbose=5)(delayed(run)(r, k) for r in range(5) for k in range(5))
oof = pd.concat([r[0] for r in res]); sel = pd.DataFrame([r[1] for r in res]); imp = pd.concat([r[2] for r in res], axis=1)
tag = f"m1b_{GROUPS}"
oof.to_parquet(OUT / f"{tag}.parquet", index=False); sel.to_csv(OUT / f"{tag}_selection.csv", index=False); imp.to_csv(OUT / f"{tag}_importance.csv")
summ = {}
for col in ("p_raw", "p_cal"):
    R, agg = summarise(oof.rename(columns={col: "p"}))
    ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
    summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                 "ci95_repeat0": ci.round(4).to_dict()}
summ["selection"] = {str(k): int(v) for k, v in sel.groupby(["interactions", "lr"]).size().items()}
(OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal")}, indent=1))
