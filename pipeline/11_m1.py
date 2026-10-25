"""M1 LightGBM with nested repeated grouped CV (plan 7.9). NaN kept (native missing handling), no scaling.

usage: python -I 11_m1.py <groups=DESCLB> [n_jobs=7]
writes: data/derived/oof/m1_<groups>.parquet + _selection.csv + _importance.csv + .json

Inner loop: 60 pre-sampled candidates (hpo/m1_lightgbm.csv); each inner fit early-stops on the inner validation fold
(binary log-loss, patience 100); candidate score = mean best inner log-loss; refit on the outer training set with
n_estimators = median of the three inner best iterations. Platt recalibration from the winner's inner OOF predictions.
"""
import sys, json, pathlib, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import log_loss
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, fold_rows, D
from lib_eval import Platt, summarise, cluster_boot, select_score

warnings.filterwarnings("ignore")
GROUPS = sys.argv[1] if len(sys.argv) > 1 else "DESCLB"
NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 7
H, FD, F = load(); CFG = pd.read_csv(D / "hpo/m1_lightgbm.csv")
spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec); y = H.y10.values
OUT = D / "oof"; OUT.mkdir(exist_ok=True)

def params(c, n=None):
    return dict(objective="binary", num_leaves=int(c.num_leaves), max_depth=int(c.max_depth), learning_rate=float(c.learning_rate),
                min_child_samples=int(c.min_child_samples), subsample=float(c.subsample), subsample_freq=1,
                colsample_bytree=float(c.colsample_bytree), reg_lambda=float(c.reg_lambda), reg_alpha=float(c.reg_alpha),
                scale_pos_weight=float(c.scale_pos_weight), n_estimators=n or int(c.n_estimators_max), n_jobs=1, verbose=-1, random_state=0)

def run(rep, k):
    tr, te = fold_rows(H, F, rep, k)
    inner = []
    for j in range(3):
        itr, iva = fold_rows(H, F, rep, k, j)
        pp = TabPrep(spec, "none", scale=False); Xa = pp.fit_transform(Xraw.iloc[itr]); Xb = pp.transform(Xraw.iloc[iva])
        inner.append((Xa, y[itr], Xb, y[iva]))
    scores, iters, oof_in = np.zeros(len(CFG)), {}, {}
    for ci, c in enumerate(CFG.itertuples()):
        ps, ys, its = [], [], []
        for Xa, ya, Xb, yb in inner:
            m = lgb.LGBMClassifier(**params(c)).fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="binary_logloss",
                                                    callbacks=[lgb.early_stopping(100, verbose=False)])
            its.append(m.best_iteration_ or m.n_estimators); ps.append(m.predict_proba(Xb, num_iteration=its[-1])[:, 1]); ys.append(yb)
        p, yy = np.concatenate(ps), np.concatenate(ys)
        scores[ci] = select_score(p, yy); iters[ci] = int(np.median(its)); oof_in[ci] = (p, yy)
    best = int(np.argmin(scores)); c = CFG.iloc[best]
    pp = TabPrep(spec, "none", scale=False); Xa = pp.fit_transform(Xraw.iloc[tr]); Xb = pp.transform(Xraw.iloc[te])
    m = lgb.LGBMClassifier(**params(c, max(iters[best], 10))).fit(Xa, y[tr])
    p = m.predict_proba(Xb)[:, 1]; cal = Platt().fit(*oof_in[best])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=Xa.columns, name=f"{rep}_{k}")
    return (pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": y[te],
                          "p_raw": p, "p_cal": cal(p), "cfg": best}),
            {"repeat": rep, "outer": k, "best": best, "inner_logloss": scores[best], "n_trees": iters[best]}, imp)

res = Parallel(n_jobs=NJ, verbose=5)(delayed(run)(r, k) for r in range(5) for k in range(5))
oof = pd.concat([r[0] for r in res]); sel = pd.DataFrame([r[1] for r in res]); imp = pd.concat([r[2] for r in res], axis=1)
tag = f"m1_{GROUPS}"
oof.to_parquet(OUT / f"{tag}.parquet", index=False); sel.to_csv(OUT / f"{tag}_selection.csv", index=False); imp.to_csv(OUT / f"{tag}_importance.csv")
summ = {}
for col in ("p_raw", "p_cal"):
    R, agg = summarise(oof.rename(columns={col: "p"}))
    ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
    summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                 "ci95_repeat0": ci.round(4).to_dict()}
summ["selected_cfg_counts"] = sel.best.value_counts().to_dict(); summ["median_trees"] = float(sel.n_trees.median())
(OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal")}, indent=1))
