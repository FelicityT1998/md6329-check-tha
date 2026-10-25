"""M3 image model (frozen ResNet-18 embedding of the hip crop) and its linear early fusion with tabular groups (plan 7.9 B4).

usage: python -I 13_m3.py <emb_tag=imagenet> [groups=-] [n_jobs=7]
       groups '-' = image only (M3); e.g. 'D' = D+I (experiment A2), 'DESCLB' = all clinical + I (M4-linear)
reads : data/derived/emb/emb_<tag>.npy + _index.csv (06_embed.py), hips.parquet, folds.csv, hpo/m0_elasticnet.csv
writes: data/derived/oof/m3_<tag>_<groups>.parquet + .json (same layout as M0)

Image block, fitted on the training rows of each split only: PCA to 32 components of the 512-d embedding (hips with a crop),
z-scored; hips without a crop get 0 + an 'img_missing' indicator. Classifier: the M0 elastic-net candidate list, selected by
inner Platt-recalibrated log-loss, refitted on the outer training set; Platt from the winner's inner OOF predictions.
"""
import sys, json, pathlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, fold_rows, D
from lib_eval import Platt, summarise, cluster_boot, select_score

warnings.filterwarnings("ignore")
TAG = sys.argv[1] if len(sys.argv) > 1 else "imagenet"
GROUPS = sys.argv[2] if len(sys.argv) > 2 else "-"
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 7
NPC = 32
H, FD, F = load(); CFG = pd.read_csv(D / "hpo/m0_elasticnet.csv"); y = H.y10.values
OUT = D / "oof"


from lib_img import image_matrix, ImgPrep
IMG = image_matrix(H, TAG, D)
spec = feature_spec(FD, GROUPS) if GROUPS != "-" else None
Xraw = raw_frame(H, spec) if spec else None


def build(itr, ite):
    ip = ImgPrep(NPC).fit(IMG[itr]); A, B = [ip.transform(IMG[itr])], [ip.transform(IMG[ite])]
    if spec:
        pp = TabPrep(spec, "mice"); A.append(pp.fit_transform(Xraw.iloc[itr]).values); B.append(pp.transform(Xraw.iloc[ite]).values)
    return np.column_stack(A), np.column_stack(B)


def model(c):
    return LogisticRegression(solver="saga", C=c.C, l1_ratio=c.l1_ratio, max_iter=4000, tol=1e-4,
                              class_weight=None if c.class_weight == "none" else "balanced")


def run(rep, k):
    tr, te = fold_rows(H, F, rep, k)
    inner = []
    for j in range(3):
        itr, iva = fold_rows(H, F, rep, k, j); Xa, Xb = build(itr, iva); inner.append((Xa, y[itr], Xb, y[iva]))
    scores = np.zeros(len(CFG)); oof_in = {}
    for ci, c in enumerate(CFG.itertuples()):
        ps, ys = [], []
        for Xa, ya, Xb, yb in inner: ps.append(model(c).fit(Xa, ya).predict_proba(Xb)[:, 1]); ys.append(yb)
        p, yy = np.concatenate(ps), np.concatenate(ys); scores[ci] = select_score(p, yy); oof_in[ci] = (p, yy)
    best = int(np.argmin(scores)); c = CFG.iloc[best]
    Xa, Xb = build(tr, te); m = model(c).fit(Xa, y[tr]); p = m.predict_proba(Xb)[:, 1]; cal = Platt().fit(*oof_in[best])
    return (pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": y[te],
                          "p_raw": p, "p_cal": cal(p), "cfg": best}),
            {"repeat": rep, "outer": k, "best": best, "inner_logloss": scores[best], "n_feat": Xa.shape[1]})


if __name__ == "__main__":
    res = Parallel(n_jobs=NJ, verbose=5)(delayed(run)(r, k) for r in range(5) for k in range(5))
    oof = pd.concat([r[0] for r in res]); sel = pd.DataFrame([r[1] for r in res])
    tag = f"m3_{TAG}_{GROUPS}"
    oof.to_parquet(OUT / f"{tag}.parquet", index=False); sel.to_csv(OUT / f"{tag}_selection.csv", index=False)
    summ = {}
    for col in ("p_raw", "p_cal"):
        R, agg = summarise(oof.rename(columns={col: "p"}))
        ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
        summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                     "ci95_repeat0": ci.round(4).to_dict()}
    summ["img_coverage_known"] = float((~np.isnan(IMG[:, 0]))[H.y10.notna().values].mean())
    summ["selected_cfg_counts"] = sel.best.value_counts().to_dict()
    (OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal")}, indent=1))
