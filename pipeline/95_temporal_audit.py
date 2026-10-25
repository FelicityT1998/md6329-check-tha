"""Temporal-admissibility audit of the inputs and of the 'too good' image head (2026-10-23 night).

usage: python -I 95_temporal_audit.py <project dir>
reads : data/derived/hips.parquet, images.csv, emb/emb_densenet121.npy, oof/*.parquet
writes: data/derived/results/temporal_window.csv   per model: AUROC by event-time window (events in years 1-3 / 4-7 / 8-10
                                                     vs all event-free hips) and lagged AUROC/AUPRC (events in the first
                                                     2 or 3 years removed) - mean over the available repeats
        data/derived/results/temporal_lag_deltas.csv paired subject-clustered bootstrap (repeat 0) of the lag-2 deltas
        data/derived/results/temporal_checks.json    DICOM acquisition years, cross-subject duplicate films, does the image
                                                     score predict censoring, orthopaedist visits by event window
Why: a label-permutation test only shows that the pipeline does not couple test labels to predictions; a within-KL
analysis is not a clean control because the baseline KL in the public CHECK file was scored at the end of follow-up,
paired and in known sequence with up to 10 years of later films (Macri 2022). Whether the image signal is 'future
information' or disease stage is decided by (i) when every input was acquired and (ii) how the signal behaves across
the time-to-event axis, which this script tabulates.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; O = D / "oof"; R = D / "results"; R.mkdir(exist_ok=True)
H = pd.read_parquet(D / "hips.parquet")
MODELS = {"m0_DES_mice": "M0 D+E+S", "m0_DESCLB_mice": "M0 DESCLB (clinical)", "m2ens_DESCLB_lam1_dtinit_es_s3": "M2 FT-Transformer",
          "m3_head_res_-": "Image head (res + aux)", "m3_head_res_noaux_-": "Image head (res, no aux)",
          "m5_pix_logit_DESCLB": "M0 + pixel logit", "m5noaux_pix_logit_DESCLB": "M0 + pixel logit (no-aux head)",
          "m0_DR_mice": "D + R reads (inadmissible)", "m0_DESCLBR_mice": "M0 + R reads (inadmissible)"}
WIN = {"1-3": (1, 3), "4-7": (4, 7), "8-10": (8, 10)}
ev = H.set_index(["nsin", "side"])[["t_event", "r_kl"]]


def scores(d, col):
    d = d.join(ev, on=["nsin", "side"]); out = {"n_events": int(d.y.sum()), "auroc": roc_auc_score(d.y, d[col]), "auprc": average_precision_score(d.y, d[col])}
    for w, (a, b) in WIN.items():
        s = d[(d.y == 0) | d.t_event.between(a, b)]; out[f"auroc_yr{w}"] = roc_auc_score(s.y, s[col]); out[f"ev_yr{w}"] = int(s.y.sum())
    for lag in (2, 3):
        s = d[~((d.y == 1) & (d.t_event <= lag))]
        out[f"lag{lag}_auroc"] = roc_auc_score(s.y, s[col]); out[f"lag{lag}_auprc"] = average_precision_score(s.y, s[col]); out[f"lag{lag}_events"] = int(s.y.sum())
    return out


rows = []
for stem, name in MODELS.items():
    f = O / f"{stem}.parquet"
    if not f.exists(): continue
    d = pd.read_parquet(f); d = d[d.y.notna()]; col = "p_cal" if "p_cal" in d and d.p_cal.notna().all() else "p_raw"
    per = pd.DataFrame([scores(g, col) for _, g in d.groupby("repeat")])
    r = {"model": name, "stem": stem, "repeats": int(d.repeat.nunique())} | per.mean().to_dict()
    r["lag2_auprc_sd"] = per.lag2_auprc.std() if len(per) > 1 else np.nan
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(R / "temporal_window.csv", index=False)
pd.set_option("display.width", 250)
print(T[["model", "repeats", "auroc", "auroc_yr1-3", "auroc_yr4-7", "auroc_yr8-10", "lag2_auroc", "lag2_auprc", "lag3_auroc", "lag3_auprc"]].round(3).to_string())


def lag_delta(a, b, lag=2, n=1000, seed=0):
    A, B = [pd.read_parquet(O / f"{s}.parquet").query("repeat == 0") for s in (a, b)]
    m = A.merge(B, on=["nsin", "side"], suffixes=("_a", "_b")); m = m[m.y_a.notna()].join(ev, on=["nsin", "side"])
    m = m[~((m.y_a == 1) & (m.t_event <= lag))].reset_index(drop=True)
    ca = "p_cal_a" if "p_cal_a" in m and m.p_cal_a.notna().all() else "p_raw_a"; cb = "p_cal_b" if "p_cal_b" in m and m.p_cal_b.notna().all() else "p_raw_b"
    y, pa, pb = m.y_a.values, m[ca].values, m[cb].values
    g = m.groupby("nsin").indices; ids = np.array(list(g)); rng = np.random.default_rng(seed); Dd = []
    for _ in range(n):
        ix = np.concatenate([g[i] for i in rng.choice(ids, len(ids))])
        if y[ix].sum() < 2: continue
        Dd.append((average_precision_score(y[ix], pa[ix]) - average_precision_score(y[ix], pb[ix]), roc_auc_score(y[ix], pa[ix]) - roc_auc_score(y[ix], pb[ix])))
    Dd = np.array(Dd)
    return {"model": a, "reference": b, "lag": lag, "events": int(y.sum()),
            "d_auprc": average_precision_score(y, pa) - average_precision_score(y, pb), "d_auprc_lo": np.quantile(Dd[:, 0], .025), "d_auprc_hi": np.quantile(Dd[:, 0], .975),
            "d_auroc": roc_auc_score(y, pa) - roc_auc_score(y, pb), "d_auroc_lo": np.quantile(Dd[:, 1], .025), "d_auroc_hi": np.quantile(Dd[:, 1], .975)}


pairs = [("m5_pix_logit_DESCLB", "m0_DESCLB_mice"), ("m3_head_res_-", "m0_DESCLB_mice"), ("m5noaux_pix_logit_DESCLB", "m0_DESCLB_mice"),
         ("m3_head_res_noaux_-", "m3_head_res_-")]
Dl = pd.DataFrame([lag_delta(a, b) for a, b in pairs if (O / f"{a}.parquet").exists() and (O / f"{b}.parquet").exists()])
Dl.to_csv(R / "temporal_lag_deltas.csv", index=False); print(Dl.round(3).to_string())

# ---------------------------------------------------------------- acquisition-time and data-integrity checks
chk = {}
I = pd.read_csv(D / "images.csv")
yr = I.loc[I.fmt == "dcm", "StudyDate"].astype(str).str[:4]
chk["dicom_study_year_counts"] = yr.value_counts().sort_index().to_dict()
chk["tiff_dateable"] = "no (digitised film, IrfanView; no acquisition date) - located in the T00 folder of the baseline release"
E = np.load(D / "emb/emb_densenet121.npy"); ix = pd.read_csv(D / "emb/emb_densenet121_index.csv")
X = E / np.linalg.norm(E, axis=1, keepdims=True); S = X @ X.T
S[ix.nsin.values[:, None] == ix.nsin.values[None, :]] = -1
a, b = np.where(np.triu(S, 1) > 0.99)
chk["cross_subject_duplicate_crops_cos>0.99"] = [{"nsin_a": int(ix.nsin[i]), "side_a": ix.side[i], "nsin_b": int(ix.nsin[j]), "side_b": ix.side[j],
                                                   "cos": round(float(S[i, j]), 4)} for i, j in zip(a, b)]
pr = pd.read_parquet(D / "imghead/pred_res.parquet").query("inner == -1").set_index("row").img_logit
c = H.assign(img=pr.reindex(H.index).values); c = c[c.img.notna() & (c.event == 0)]
chk["image_score_predicts_censoring_auroc"] = round(roc_auc_score(c.status10 == "censored<10y", c.img), 3)
H["win"] = pd.cut(H.t_event, [0, 3, 7, 10], labels=list(WIN))
chk["orthopaedist_last_year_by_event_window"] = {str(k): v for k, v in H.groupby("win", observed=True).care_orthopaedist.agg(["sum", "count"]).T.to_dict().items()}
chk["orthopaedist_rate_event_free"] = round(float(H.loc[H.y10 == 0, "care_orthopaedist"].mean()), 3)
chk["baseline_kl_by_event_window"] = pd.crosstab(H.win, H.r_kl).to_dict()
(R / "temporal_checks.json").write_text(json.dumps(chk, indent=1, default=str)); print(json.dumps(chk, indent=1, default=str)[:1500])
