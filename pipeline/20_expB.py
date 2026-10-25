"""Experiment B: pseudo-external validation inside CHECK (plan 6.3 B / 7.3).

usage: python -I 20_expB.py <model=m0|m1> [groups=DESCLB] [emb_tag=-] [n_jobs=7]
reads : splits_B.csv (domain labels per hip), oof/<main CV run>.parquet (internal reference), selection files
writes: data/derived/results/expB_<model>_<groups>[_I<tag>].csv  one row per (split, direction)
        data/derived/results/expB_<...>_preds.parquet               transported predictions

Protocol per split (source -> target):
  - hyper-parameters fixed to the configuration chosen most often in the main nested CV of the same model (no re-tuning:
    the question is how a model developed in one setting transports, not how well it could be re-tuned);
  - preprocessing + model fitted on all source hips with known 10-y status; Platt map from 3-fold subject-grouped OOF
    predictions inside the source; applied unchanged to the target (= what an external user would receive);
  - internal reference = the main 5x5 nested-CV OOF predictions of the same model restricted to the target hips
    (target hips seen 'in distribution'); delta = transported - internal, paired subject-clustered bootstrap (500, repeat 0);
  - fixed threshold = source OOF probability giving 80% sensitivity in the source; report target sensitivity / specificity.
Leave-one-centre-out: each of C01-C10 is the target once (all other hips incl. 'train_only' are the source); predictions
are pooled across the ten centres and also reported per centre.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, D
from lib_eval import Platt, metrics

warnings.filterwarnings("ignore")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "m0"
GROUPS = sys.argv[2] if len(sys.argv) > 2 else "DESCLB"
ETAG = sys.argv[3] if len(sys.argv) > 3 else "-"
NJ = int(sys.argv[4]) if len(sys.argv) > 4 else 7
H, FD, F = load(); spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec); y = H.y10.values
S = pd.read_csv(D / "splits_B.csv"); assert (S.nsin.values == H.nsin.values).all() and (S.side.values == H.side.values).all()
O = D / "oof"; R = D / "results"; R.mkdir(exist_ok=True)
if ETAG != "-":
    from lib_img import image_matrix, ImgPrep
    IMG = image_matrix(H, ETAG, D)
main_stem = {"m0": f"m0_{GROUPS}_mice", "m1": f"m1_{GROUPS}"}[MODEL] if ETAG == "-" else f"m3_{ETAG}_{GROUPS}"
SEL = pd.read_csv(O / f"{main_stem}_selection.csv")
if MODEL == "m0" or ETAG != "-":
    C = pd.read_csv(D / "hpo/m0_elasticnet.csv").iloc[int(SEL.best.mode()[0])]
else:
    C = pd.read_csv(D / "hpo/m1_lightgbm.csv").iloc[int(SEL.best.mode()[0])]; NTREE = int(SEL[SEL.best == SEL.best.mode()[0]].n_trees.median())
REF = pd.read_parquet(O / f"{main_stem}.parquet"); REF = REF[REF.y.notna()]


def features(itr, ite):
    if MODEL == "m1" and ETAG == "-":
        pp = TabPrep(spec, "none", scale=False); return pp.fit_transform(Xraw.iloc[itr]), pp.transform(Xraw.iloc[ite])
    pp = TabPrep(spec, "mice"); A, B = [pp.fit_transform(Xraw.iloc[itr]).values], [pp.transform(Xraw.iloc[ite]).values]
    if ETAG != "-":
        ip = ImgPrep(32).fit(IMG[itr]); A.append(ip.transform(IMG[itr])); B.append(ip.transform(IMG[ite]))
    return np.column_stack(A), np.column_stack(B)


def fit_predict(itr, ite):
    Xa, Xb = features(itr, ite)
    if MODEL == "m1" and ETAG == "-":
        m = lgb.LGBMClassifier(objective="binary", num_leaves=int(C.num_leaves), max_depth=int(C.max_depth), learning_rate=float(C.learning_rate),
                               min_child_samples=int(C.min_child_samples), subsample=float(C.subsample), subsample_freq=1,
                               colsample_bytree=float(C.colsample_bytree), reg_lambda=float(C.reg_lambda), reg_alpha=float(C.reg_alpha),
                               scale_pos_weight=float(C.scale_pos_weight), n_estimators=max(NTREE, 10), n_jobs=1, verbose=-1, random_state=0)
    else:
        m = LogisticRegression(solver="saga", C=C.C, l1_ratio=C.l1_ratio, max_iter=4000, tol=1e-4,
                               class_weight=None if C.class_weight == "none" else "balanced")
    return m.fit(Xa, y[itr]).predict_proba(Xb)[:, 1]


def transport(src_mask, tgt_mask):
    known = ~np.isnan(y)
    # subject-disjoint: hip-level domains (pain, format) can put a subject's two hips on both sides -> drop from source
    src_mask = src_mask & ~np.isin(H.nsin.values, H.nsin.values[tgt_mask])
    src = np.where(src_mask & known)[0]; tgt = np.where(tgt_mask & known)[0]
    # source-internal OOF for Platt and the fixed threshold
    oof = np.zeros(len(src))
    for a, b in GroupKFold(3).split(src, groups=H.nsin.values[src]):
        oof[b] = fit_predict(src[a], src[b])
    cal = Platt().fit(oof, y[src]); oc = cal(oof)
    thr = np.quantile(oc[y[src] == 1], 0.2)            # 80% source sensitivity
    p = cal(fit_predict(src, tgt))
    return tgt, p, thr


def boot_delta(d, n=500, seed=0):
    """d: target hips with y, p_ext, p_int (repeat-0 internal OOF). paired subject bootstrap of dAUPRC, dAUROC."""
    g = d.groupby("nsin").indices; ids = np.array(list(g)); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        ix = np.concatenate([g[i] for i in rng.choice(ids, len(ids))]); yy = d.y.values[ix]
        if yy.sum() < 2 or yy.sum() == len(yy): continue
        out.append((average_precision_score(yy, d.p_ext.values[ix]) - average_precision_score(yy, d.p_int.values[ix]),
                    roc_auc_score(yy, d.p_ext.values[ix]) - roc_auc_score(yy, d.p_int.values[ix]),
                    average_precision_score(yy, d.p_ext.values[ix])))
    return np.quantile(np.array(out), [0.025, 0.975], axis=0)


def summarise_split(name, src_lab, tgt_lab, tgt, p, thr):
    d = pd.DataFrame({"nsin": H.nsin.values[tgt], "side": H.side.values[tgt], "y": y[tgt], "p_ext": p})
    ref = REF[REF.repeat == 0][["nsin", "side", "p_cal"]].rename(columns={"p_cal": "p_int"})
    d = d.merge(ref, on=["nsin", "side"])
    me, mi = metrics(d.y, d.p_ext), metrics(d.y, d.p_int)
    ci = boot_delta(d) if d.y.sum() >= 3 else np.full((2, 3), np.nan)
    pos = d.y == 1
    return {"split": name, "source": src_lab, "target": tgt_lab, "n_target": len(d), "events_target": int(d.y.sum()),
            "prev_target": d.y.mean(), "auprc_ext": me["auprc"], "auprc_ext_lo": ci[0, 2], "auprc_ext_hi": ci[1, 2],
            "auprc_int": mi["auprc"], "d_auprc": me["auprc"] - mi["auprc"], "d_auprc_lo": ci[0, 0], "d_auprc_hi": ci[1, 0],
            "auroc_ext": me["auroc"], "auroc_int": mi["auroc"], "d_auroc": me["auroc"] - mi["auroc"], "d_auroc_lo": ci[0, 1], "d_auroc_hi": ci[1, 1],
            "cal_int_ext": me["cal_intercept"], "cal_slope_ext": me["cal_slope"], "cal_int_int": mi["cal_intercept"], "cal_slope_int": mi["cal_slope"],
            "brier_ext": me["brier"], "sens_at_src80": float((d.p_ext[pos] >= thr).mean()) if pos.any() else np.nan,
            "spec_at_src80": float((d.p_ext[~pos] < thr).mean())}, d.assign(split=name, target=tgt_lab)


def job(name, col, a, b):
    sm, tm = (S[col] == a).values, (S[col] == b).values
    tgt, p, thr = transport(sm, tm)
    return summarise_split(name, a, b, tgt, p, thr)


if __name__ == "__main__":
    jobs = [("format", "dom_fmt", "tif", "dcm"), ("format", "dom_fmt", "dcm", "tif"),
            ("followup_group", "dom_group", "annual", "interval"), ("pain", "dom_pain", "painful", "painless"),
            ("sex", "dom_sex", "female", "male"), ("education", "dom_edu", "high", "low"), ("education", "dom_edu", "low", "high"),
            ("prior_care", "dom_care", "care", "no_care"), ("prior_care", "dom_care", "no_care", "care")]
    res = Parallel(n_jobs=NJ)(delayed(job)(*j) for j in jobs)
    # leave-one-centre-out
    centres = sorted(c for c in S.dom_centre.unique() if c.startswith("C"))
    def loco(c):
        tgt, p, thr = transport((S.dom_centre != c).values, (S.dom_centre == c).values)
        return tgt, p, thr, c
    lo = Parallel(n_jobs=NJ)(delayed(loco)(c) for c in centres)
    rows = [r[0] for r in res]; preds = [r[1] for r in res]
    for tgt, p, thr, c in lo:
        r, d = summarise_split("loco_centre", "other centres", c, tgt, p, thr); rows.append(r); preds.append(d)
    tg = np.concatenate([x[0] for x in lo]); pp = np.concatenate([x[1] for x in lo])
    thr_mean = float(np.mean([x[2] for x in lo]))
    r, d = summarise_split("loco_pooled", "9 other centres", "all C01-C10", tg, pp, thr_mean); rows.append(r); preds.append(d)
    T = pd.DataFrame(rows); tag = f"expB_{MODEL}_{GROUPS}" + ("" if ETAG == "-" else f"_I{ETAG}")
    T.to_csv(R / f"{tag}.csv", index=False); pd.concat(preds).to_parquet(R / f"{tag}_preds.parquet", index=False)
    pd.set_option("display.width", 250)
    print(T[["split", "source", "target", "n_target", "events_target", "auprc_ext", "auprc_int", "d_auprc", "d_auprc_lo", "d_auprc_hi",
             "auroc_ext", "auroc_int", "cal_int_ext", "cal_slope_ext", "sens_at_src80", "spec_at_src80"]].round(3).to_string())
