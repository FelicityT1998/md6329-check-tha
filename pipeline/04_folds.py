"""Fixed resampling plan + search-space files shared by every model (plan 7.3).

usage: python -I 04_folds.py <project dir>
reads : data/derived/hips.parquet, subjects.parquet
writes: data/derived/folds.csv      nsin, repeat, outer_fold, inner_fold (-1 = in the outer test fold), centre, fmt, strata
        data/derived/splits_B.csv   hip-level domain labels for the pseudo-external experiments (experiment B)
        data/derived/hpo/           M0/M1 random-search candidates, FT-Transformer NAS space + candidates (pre-sampled, seeded)

Design
- unit of splitting = subject (nsin), so both hips are always on the same side of every split.
- outer: 5 folds x 5 repeats; inner: 3 folds inside each outer training set (tuning / early stopping only).
- strata = subject THA count (0/1/2) x baseline film format; sparse strata (<10 subjects) fall back to the event count.
- every candidate configuration is drawn ONCE here with a fixed seed, so all outer folds evaluate the identical candidate
  list (random search becomes a paired comparison across folds and the per-fold selected config is reproducible).
"""
import sys, json, math, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; HP = D / "hpo"; HP.mkdir(exist_ok=True)
SEED, N_REP, K_OUT, K_IN = 20261020, 5, 5, 3
H = pd.read_parquet(D / "hips.parquet"); S = pd.read_parquet(D / "subjects.parquet").set_index("nsin")

# ---------------------------------------------------------------- folds
subj = S[["n_events", "fmt_subject", "centre"]].copy()
subj["n_events"] = subj.n_events.clip(upper=2).astype(int)
subj["strata"] = subj.n_events.astype(str) + "|" + subj.fmt_subject
small = subj.strata.map(subj.strata.value_counts()) < 10
subj.loc[small, "strata"] = subj.loc[small, "n_events"].astype(str) + "|any"
rows = []
for r in range(N_REP):
    outer = StratifiedKFold(K_OUT, shuffle=True, random_state=SEED + r)
    for k, (tr, te) in enumerate(outer.split(subj, subj.strata)):
        inner_lab = pd.Series(-1, index=subj.index)
        trs = subj.iloc[tr]
        inner = StratifiedKFold(K_IN, shuffle=True, random_state=SEED + 100 * r + k)
        for j, (_, va) in enumerate(inner.split(trs, trs.strata)): inner_lab.iloc[tr[va]] = j
        rows.append(pd.DataFrame({"nsin": subj.index, "repeat": r, "outer_fold": k, "is_test": np.isin(np.arange(len(subj)), te),
                                  "inner_fold": inner_lab.values}))
F = pd.concat(rows)
# one row per (subject, repeat): outer_fold = the fold in which the subject is TESTED; inner_fold is defined per (repeat, outer_fold)
test = F[F.is_test][["nsin", "repeat", "outer_fold"]]
inner = F[~F.is_test][["nsin", "repeat", "outer_fold", "inner_fold"]].rename(columns={"outer_fold": "for_outer"})
inner_w = inner.pivot_table(index=["nsin", "repeat"], columns="for_outer", values="inner_fold").add_prefix("inner_for_o").reset_index()
folds = test.merge(inner_w, on=["nsin", "repeat"]).merge(subj[["centre", "fmt_subject", "strata"]], left_on="nsin", right_index=True)
for c in [c for c in folds if c.startswith("inner_for_o")]: folds[c] = folds[c].fillna(-1).astype(int)
folds.to_csv(D / "folds.csv", index=False)

# balance report on 10-y known hips
k = H[H.y10.notna()].merge(folds[["nsin", "repeat", "outer_fold"]], on="nsin")
bal = k.groupby(["repeat", "outer_fold"]).agg(hips=("y10", "size"), events=("y10", "sum")).reset_index()
print("outer test folds: events per fold min/median/max =", int(bal.events.min()), int(bal.events.median()), int(bal.events.max()),
      "| hips", int(bal.hips.min()), "-", int(bal.hips.max()))
ik = k[k.repeat == 0].merge(folds[folds.repeat == 0][["nsin", "inner_for_o0"]], on="nsin")
ik = ik[ik.inner_for_o0 >= 0].groupby("inner_for_o0").y10.agg(["size", "sum"])
print("inner folds (repeat 0, outer 0) hips/events:", ik.to_dict("index"))

# ---------------------------------------------------------------- experiment B domain labels (hip level)
B = H[["nsin", "side", "y10", "event", "centre", "img_fmt", "group", "hip_pain", "female", "education", "care_no_care"]].copy()
B["dom_centre"] = np.where(B.centre.str.match(r"C\d\d"), B.centre, "train_only")
B["dom_fmt"] = B.img_fmt.fillna("none")
B["dom_group"] = B.group
B["dom_pain"] = B.hip_pain.map({1.0: "painful", 0.0: "painless"})
B["dom_sex"] = B.female.map({1.0: "female", 0.0: "male"})
B["dom_edu"] = np.where(B.education >= 6, "high", np.where(B.education.notna(), "low", None))   # >= HAVO/VWO
B["dom_care"] = B.care_no_care.map({1.0: "no_care", 0.0: "care"})
B = B.drop(columns=["centre", "img_fmt", "group", "hip_pain", "female", "education", "care_no_care"])
B.to_csv(D / "splits_B.csv", index=False)
kb = B[B.y10.notna()]
for c in [c for c in B if c.startswith("dom_")]:
    print(c, kb.groupby(c, dropna=False).y10.agg(["size", "sum"]).astype(int).to_dict("index"))

# ---------------------------------------------------------------- HPO candidates (random search, non-NN)
rng = np.random.default_rng(SEED)
loguni = lambda lo, hi, n: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
m0 = pd.DataFrame({"C": loguni(1e-3, 10, 40), "l1_ratio": rng.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 40),
                   "class_weight": rng.choice(["none", "balanced"], 40)})
m0.index.name = "cfg"; m0.to_csv(HP / "m0_elasticnet.csv")
n1 = 60
m1 = pd.DataFrame({"num_leaves": rng.choice([4, 7, 15, 31], n1), "max_depth": rng.choice([2, 3, 4, 6], n1),
                   "learning_rate": loguni(0.01, 0.1, n1), "min_child_samples": rng.choice([10, 20, 40, 80], n1),
                   "subsample": rng.uniform(0.6, 1.0, n1).round(2), "colsample_bytree": rng.uniform(0.4, 1.0, n1).round(2),
                   "reg_lambda": loguni(1e-2, 30, n1), "reg_alpha": loguni(1e-3, 3, n1),
                   "scale_pos_weight": rng.choice([1, 5, 10, 20], n1), "n_estimators_max": 1500})
m1.index.name = "cfg"; m1.to_csv(HP / "m1_lightgbm.csv")

# ---------------------------------------------------------------- NAS: FT-Transformer block search space
NAS = {
    "tokenizer": {"numeric": ["linear", "periodic", "piecewise_linear"], "missing": ["learned_MISS_token", "indicator_token"],
                  "feature_id_embedding": ["learned", "lm_frozen_desc"]},  # lm_frozen_desc = text embedding of the variable description
    "d_token": [16, 32, 48, 64],
    "n_blocks": [1, 2, 3],
    "n_heads": [2, 4, 8],
    "ffn_mult": [1.0, 4 / 3, 2.0],
    "attn_dropout": [0.0, 0.1, 0.2, 0.3],
    "ffn_dropout": [0.0, 0.1, 0.2, 0.3],
    "residual_dropout": [0.0, 0.1],
    "norm": ["pre_ln"],
    "readout": ["cls", "mean"],
    "group_tokens": [False, True],            # one extra learnable token per feature group (D/E/S/C/L/B/R/I) -> group attention maps
    "image_fusion": ["none", "token", "film"],  # how the image embedding enters (M4b): extra token or FiLM modulation of [CLS]
    "loss": ["bce_weighted", "focal_g1", "focal_g2"],
    "lr": [1e-4, 3e-4, 1e-3], "weight_decay": [1e-5, 1e-4, 1e-3], "batch_size": [128, 256],
}
(HP / "nas_space.json").write_text(json.dumps(NAS, indent=1), encoding="utf-8")

N_FEAT = 60  # main tabular tokens (ipsi + contra + subject); exact count is set by the feature set in each model script
def ftt_params(d, L, ffm, n_feat=N_FEAT, tok="linear", group_tokens=False):
    tok_p = n_feat * (2 * d if tok == "linear" else 2 * d + 16 * d if tok == "periodic" else 2 * d + 8 * d)
    hid = int(d * ffm)
    blk = 4 * (d * d + d) + 2 * (d * hid) + hid + d + 4 * d          # attention + GEGLU-free FFN + 2 LayerNorms
    return tok_p + L * blk + d * 2 + (8 * d if group_tokens else 0) + d + 1
cands = []
r2 = np.random.default_rng(SEED + 7)
while len(cands) < 48:
    c = {k: (v[r2.integers(len(v))] if isinstance(v, list) else None) for k, v in NAS.items() if isinstance(v, list)}
    c["numeric_tok"] = NAS["tokenizer"]["numeric"][r2.integers(3)]
    if c["d_token"] % c["n_heads"] or c["d_token"] // c["n_heads"] < 8: continue   # head dim >= 8
    c["params"] = ftt_params(c["d_token"], c["n_blocks"], c["ffn_mult"], tok=c["numeric_tok"], group_tokens=c["group_tokens"])
    if c["params"] > 60_000: continue                                               # cap: ~93 events -> keep the model small
    cands.append(c)
nas = pd.DataFrame(cands); nas.index.name = "arch"
nas.to_csv(HP / "nas_ftt_candidates.csv")
print(f"HPO: M0 {len(m0)} cfgs, M1 {len(m1)} cfgs | NAS: {len(nas)} FT-Transformer candidates, params {nas.params.min():,}-{nas.params.max():,}")
