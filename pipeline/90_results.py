"""Collect every out-of-fold prediction file into one results table (+ paired deltas vs reference models).

usage: python -I 90_results.py <project dir>
reads : data/derived/oof/*.parquet (rows = repeat x hip, columns y, p_raw, p_cal)
writes: data/derived/results/results_all.csv     one row per model: per-repeat mean (sd) of AUPRC/AUROC/Brier/cal, CI95 (repeat 0)
        data/derived/results/deltas.csv          paired subject-clustered bootstrap (repeat 0, 1,000 draws) of dAUPRC / dAUROC
                                                 for each model vs. its reference
Metrics are computed on Platt-recalibrated probabilities (p_cal) unless noted; AUPRC/AUROC are rank-based and equal
before/after a monotone Platt map within a fold, but pooling across folds makes them differ slightly.
"""
import sys, pathlib, json, re
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_eval import metrics, summarise
from sklearn.metrics import average_precision_score, roc_auc_score

P = pathlib.Path(sys.argv[1]); O = P / "data/derived/oof"; R = P / "data/derived/results"; R.mkdir(exist_ok=True)
SKIP = re.compile(r"(_all|_importance|_selection|_coefs|_curves)$")


def load_oof(stem):
    d = pd.read_parquet(O / f"{stem}.parquet")
    return d[d.y.notna()]


def row(stem, col="p_cal"):
    d = load_oof(stem)
    if col not in d: col = "p_raw"
    Rr, agg = summarise(d.rename(columns={col: "p"}))
    out = {"model": stem, "prob": col, "n_hips": int(agg.loc["mean", "n"]), "events": int(agg.loc["mean", "events"])}
    for m in ("auprc", "auroc", "brier", "cal_intercept", "cal_slope"):
        out[m] = agg.loc["mean", m]; out[m + "_sd"] = agg.loc["std", m]
    js = O / f"{stem}.json"
    if js.exists():
        j = json.loads(js.read_text()).get(col, {}).get("ci95_repeat0", {})
        for m in ("auprc", "auroc"):
            if m in j: out[m + "_lo"], out[m + "_hi"] = j[m]["0.025"], j[m]["0.975"]
    return out


def paired_delta(a, b, n=1000, seed=0, col="p_cal"):
    """model a minus model b on the hips both predict, repeat 0, resampling subjects."""
    A, B = load_oof(a), load_oof(b)
    A, B = A[A.repeat == 0], B[B.repeat == 0]
    m = A.merge(B, on=["nsin", "side"], suffixes=("_a", "_b"))
    ca = col if f"{col}_a" in m else "p_raw"; cb = col if f"{col}_b" in m else "p_raw"
    y, pa, pb = m.y_a.values, m[f"{ca}_a"].values, m[f"{cb}_b"].values
    groups = m.groupby("nsin").indices; ids = np.array(list(groups)); rng = np.random.default_rng(seed); D = []
    for _ in range(n):
        ix = np.concatenate([groups[i] for i in rng.choice(ids, len(ids))])
        if y[ix].sum() < 2: continue
        D.append((average_precision_score(y[ix], pa[ix]) - average_precision_score(y[ix], pb[ix]),
                  roc_auc_score(y[ix], pa[ix]) - roc_auc_score(y[ix], pb[ix])))
    D = np.array(D)
    pt = (average_precision_score(y, pa) - average_precision_score(y, pb), roc_auc_score(y, pa) - roc_auc_score(y, pb))
    return {"model": a, "reference": b, "n": len(m), "d_auprc": pt[0], "d_auprc_lo": np.quantile(D[:, 0], .025), "d_auprc_hi": np.quantile(D[:, 0], .975),
            "p_auprc_le0": float((D[:, 0] <= 0).mean()), "d_auroc": pt[1], "d_auroc_lo": np.quantile(D[:, 1], .025),
            "d_auroc_hi": np.quantile(D[:, 1], .975)}


if __name__ == "__main__":
    stems = sorted(f.stem for f in O.glob("*.parquet") if not SKIP.search(f.stem))
    rows = [row(s) for s in stems]
    for s in stems:   # discrete-time head of the FT-Transformers: 10-y cumulative incidence as a second prediction
        if "ci10" in pd.read_parquet(O / f"{s}.parquet", columns=None).columns:
            r = row(s, "ci10"); r["model"] = s + " [10y CI head]"; rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(R / "results_all.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["model", "n_hips", "events", "auprc", "auprc_sd", "auprc_lo", "auprc_hi", "auroc", "brier", "cal_intercept", "cal_slope"]].round(3).to_string())
    # paired comparisons: each family vs its D-only baseline, and every main model vs M0 on the same groups
    pairs = []
    for s in stems:
        mm = re.match(r"(m0|m1|m1b)_(\w+?)(_mice)?$", s)
        if mm and mm.group(2) != "D":
            base = f"{mm.group(1)}_D{mm.group(3) or ''}"
            if base in stems: pairs.append((s, base))
        if s.startswith(("m2", "m3", "m4", "m1_", "m1b_")) and "DESCLB" in s and "m0_DESCLB_mice" in stems and s != "m0_DESCLB_mice":
            pairs.append((s, "m0_DESCLB_mice"))
        if s.startswith("m3_") and s.endswith("_D") and "m0_D_mice" in stems: pairs.append((s, "m0_D_mice"))
        if s.startswith("m3_") and s.endswith("_D") and "m0_DR_mice" in stems: pairs.append((s, "m0_DR_mice"))
    for extra in [("m4_imagenet_DESCLB", "m2ens_DESCLB_lam1"), ("m4_simclr_DESCLB", "m2ens_DESCLB_lam1"),
                  ("m4_imagenet_DESCLB_film", "m2ens_DESCLB_lam1"), ("m4_imagenet_DESCLB_film", "m4_imagenet_DESCLB"),
                  ("m3_imagenet_DESCLB", "m3_imagenet_-"), ("m3_simclr_-", "m3_imagenet_-"),
                  ("m3_imagenet_ms_-", "m3_imagenet_-"), ("m3_densenet121_-", "m3_imagenet_-"),
                  ("m3_densenet121_DESCLB", "m3_imagenet_DESCLB"), ("m3_densenet121_D", "m3_imagenet_D"),
                  ("m3_densenet121_DESCLB", "m1_DESCLB"), ("m3_densenet121_DESCLB", "m0_DES_mice"),
                  ("m2ens_DESCLB_lam1_dtinit", "m2ens_DESCLB_lam1"),
                  ("m2ens_DESCLB_lam1_dtinit_es", "m2ens_DESCLB_lam1"), ("m2ens_DESCLB_lam1_dtinit_es_s3", "m2ens_DESCLB_lam1"),
                  ("m2ens_DESCLB_lam1_dtinit_es_s3", "m2ens_DESCLB_lam1_dtinit_es"),
                  ("m4_densenet121_DESCLB_film_dtinit_es", "m2ens_DESCLB_lam1_dtinit_es"),
                  # 10-23 trainable image head (repeat 0 only; paired on repeat 0)
                  ("m3_head_res_-", "m3_densenet121_-"), ("m3_head_res_-", "m3_head_linear_-"), ("m3_head_res_-", "m3_head_res_noaux_-"),
                  ("m3_head_linear_-", "m3_densenet121_-"), ("m3_densenet121_adabn_-", "m3_densenet121_-"),
                  ("m5_pix_logit_DESCLB", "m0_DESCLB_mice"), ("m5_pix_logit_DESCLB", "m0_DESCLBR_mice"),
                  ("m5_pix_logit_DESCLB", "m3_densenet121_DESCLB"), ("m5_pix_logit_items_DESCLB", "m0_DESCLB_mice"),
                  ("m5_pix_items_DESCLB", "m0_DESCLB_mice"), ("m5_pix_logit_items_only", "m0_DR_mice"),
                  ("m5_pix_logit_DESCLB", "m5_pix_items_DESCLB"),
                  ("m2ens_DESCLB_v2_ln", "m2ens_DESCLB_lam1_dtinit_es"), ("m2ens_DESCLB_v2_bn", "m2ens_DESCLB_v2_ln"), ("m4_imagenet_DESCLB_dtinit", "m2ens_DESCLB_lam1_dtinit"),
                  ("m3_imagenet_DESCLB", "m0_DESCLB_mice"), ("m3_simclr_DESCLB", "m0_DESCLB_mice")]:
        if all(e in stems for e in extra): pairs.append(extra)
    pairs = list(dict.fromkeys(pairs))
    Dl = pd.DataFrame([paired_delta(a, b) for a, b in pairs])
    Dl.to_csv(R / "deltas.csv", index=False)
    print(Dl.round(3).to_string())
