"""M0 elastic-net logistic regression with nested repeated grouped CV (plan 7.3 / 7.9).

usage: python -I 10_m0.py <groups=DESCLB> [impute=mice] [n_jobs=7]
writes: data/derived/oof/m0_<groups>_<impute>.parquet (repeat, nsin, side, y, p_raw, p_cal, cfg) + .json summary

Inner loop: the 40 pre-sampled candidates (hpo/m0_elasticnet.csv) are scored by mean inner Platt-recalibrated log-loss (AUPRC has ~25
events per inner fold and is too noisy to select on); the winner is refitted on the whole outer training set.
Platt recalibration is fitted on the winner's inner out-of-fold predictions and applied to the outer test fold.
"""
import sys, json, pathlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, fold_rows, D
from lib_eval import Platt, summarise, cluster_boot, select_score

warnings.filterwarnings("ignore")
GROUPS = sys.argv[1] if len(sys.argv) > 1 else "DESCLB"
IMPUTE = sys.argv[2] if len(sys.argv) > 2 else "mice"
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 7
H, FD, F = load(); CFG = pd.read_csv(D / "hpo/m0_elasticnet.csv")
spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec); y = H.y10.values
OUT = D / "oof"; OUT.mkdir(exist_ok=True)

def model(c):
    return LogisticRegression(solver="saga", C=c.C, l1_ratio=c.l1_ratio, max_iter=4000, tol=1e-4,
                              class_weight=None if c.class_weight == "none" else "balanced")

def run(rep, k):
    tr, te = fold_rows(H, F, rep, k)
    inner = []
    for j in range(3):
        itr, iva = fold_rows(H, F, rep, k, j)
        pp = TabPrep(spec, IMPUTE); Xa = pp.fit_transform(Xraw.iloc[itr]); Xb = pp.transform(Xraw.iloc[iva])
        inner.append((Xa.values, y[itr], Xb.values, y[iva]))
    scores = np.zeros(len(CFG)); oof_in = {}
    for ci, c in enumerate(CFG.itertuples()):
        ps, ys = [], []
        for Xa, ya, Xb, yb in inner:
            ps.append(model(c).fit(Xa, ya).predict_proba(Xb)[:, 1]); ys.append(yb)
        p, yy = np.concatenate(ps), np.concatenate(ys)
        scores[ci] = select_score(p, yy); oof_in[ci] = (p, yy)
    best = int(np.argmin(scores)); c = CFG.iloc[best]
    pp = TabPrep(spec, IMPUTE); Xa = pp.fit_transform(Xraw.iloc[tr]); Xb = pp.transform(Xraw.iloc[te])
    m = model(c).fit(Xa.values, y[tr]); p = m.predict_proba(Xb.values)[:, 1]
    cal = Platt().fit(*oof_in[best])
    coef = pd.Series(m.coef_[0], index=Xa.columns)
    return (pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": y[te],
                          "p_raw": p, "p_cal": cal(p), "cfg": best}),
            {"repeat": rep, "outer": k, "best": best, "inner_logloss": scores[best], "n_feat": Xa.shape[1], "n_nonzero": int((coef != 0).sum())},
            coef.rename(f"{rep}_{k}"))

res = Parallel(n_jobs=NJ, verbose=5)(delayed(run)(r, k) for r in range(5) for k in range(5))
oof = pd.concat([r[0] for r in res]); sel = pd.DataFrame([r[1] for r in res]); coefs = pd.concat([r[2] for r in res], axis=1)
tag = f"m0_{GROUPS}_{IMPUTE}"
oof.to_parquet(OUT / f"{tag}.parquet", index=False); sel.to_csv(OUT / f"{tag}_selection.csv", index=False); coefs.to_csv(OUT / f"{tag}_coefs.csv")
summ = {}
for col in ("p_raw", "p_cal"):
    R, agg = summarise(oof.rename(columns={col: "p"}))
    ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
    summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                 "ci95_repeat0": ci.round(4).to_dict()}
summ["features"] = len(spec); summ["selected_cfg_counts"] = sel.best.value_counts().to_dict()
(OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal")}, indent=1))
