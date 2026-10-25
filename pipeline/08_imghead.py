"""Trainable image head on frozen DenseNet-121 block-3 maps, with radiographic-reading auxiliary supervision (2026-10-23).

usage:
  python -I 08_imghead.py <project dir> probe            what do the frozen global embeddings encode? (reading items)
  python -I 08_imghead.py <project dir> train [n_jobs]   variants linear / res_noaux / res, repeat 0, 5 outer folds
  python -I 08_imghead.py <project dir> fuse  [n_jobs]   M0 elastic net on DESCLB + pixel-derived features (cross-fitted)

Why (analysis in 实验结果汇总.md §0/§2): frozen ImageNet embeddings give AUROC 0.61-0.69 for THA while the radiologist's
reading (KL / OARSI) lifts AUPRC 0.37 -> 0.58. OA pathology on an AP radiograph = joint-space narrowing (cartilage loss,
superior / medial), osteophytes (acetabular / femoral, superior / inferior), subchondral sclerosis, cysts, femoral-head
flattening and buttressing; global average pooling of ImageNet features discards exactly this small, localised geometry.
So: keep the 14x14 spatial map, learn a small residual conv head with BatchNorm, attention pooling, and teach it WHAT to
look at through the 11 baseline OARSI items as auxiliary regression targets (cf. Tiulpin 2019 KL auxiliary, von Schacky
2020 OARSI multi-task). Only pixels are used at prediction time; reading scores are training targets only.

Head variants (all: AdamW, linear warm-up 3 epochs + cosine, 30 epochs fixed - no early stopping, batch 64,
spatial jitter +-1 cell and Dropout2d 0.1 on the maps, pos_weight 5 for THA):
  linear    : GAP(1024) -> BatchNorm1d -> dropout 0.3 -> linear heads (THA + 11 items)       [no residual, no spatial]
  res_noaux : 1x1 conv 1024->64 + BN + ReLU -> 2 residual blocks (3x3 conv-BN-ReLU-3x3 conv-BN + identity skip)
              -> [attention pooling ; GAP] -> dropout 0.3 -> THA head only
  res       : same as res_noaux + the 11 reading items (masked MSE on train-standardised targets, weight 1)
Fold protocol: repeat 0 outer folds; inside each outer training set the 'res' head is cross-fitted on the 3 inner folds of
folds.csv (so every training hip gets an out-of-fold pixel score for the fusion model) and refitted on the whole outer
training set for the test fold. Platt for the image-only THA score is fitted on the cross-fitted inner OOF.
"""
import sys, pathlib, json, warnings, math, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); MODE = sys.argv[2]; NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 6
# 10-23 night: repeats 1-4 (defaults reproduce the original repeat-0 run and file names)
#   IMG_REPEAT   which repeat of folds.csv to use (outputs of repeat r > 0 get the suffix _r<r>)
#   IMG_KINDS    head variants to train;  IMG_CROSSFIT  variants that are also cross-fitted on the 3 inner folds (fusion input)
#   FUSE_KIND    which cross-fitted head feeds 'fuse' (res -> m5_pix_*, res_noaux -> m5noaux_pix_*)
REP = int(os.environ.get("IMG_REPEAT", 0)); SUF = "" if REP == 0 else f"_r{REP}"
KINDS = os.environ.get("IMG_KINDS", "linear,res_noaux,res").split(",")
CROSSFIT = os.environ.get("IMG_CROSSFIT", "res").split(",")
FUSE_KIND = os.environ.get("FUSE_KIND", "res")
sys.path.insert(0, str(P))
from lib_tab import load as load_tab, feature_spec, raw_frame, TabPrep, fold_rows, D
from lib_eval import Platt, metrics, cluster_boot, select_score
FM = pathlib.Path(r"D:\datasets\CHECK\fmaps"); OUT = D / "imghead"; OUT.mkdir(exist_ok=True)
ITEMS = ["r_kl", "r_jsn_sup", "r_jsn_med", "r_ost_sup_acet", "r_ost_inf_acet", "r_ost_sup_fem", "r_ost_inf_fem",
         "r_sclerosis", "r_cysts", "r_flattening", "r_buttressing"]
H, FD, F = load_tab(); y10 = H.y10.values
idx = pd.read_csv(FM / "densenet_b3_index.csv")
pos = pd.Series(np.arange(len(idx)), index=pd.MultiIndex.from_frame(idx[["nsin", "side"]]))
MAPROW = pos.reindex(pd.MultiIndex.from_frame(H[["nsin", "side"]])).values          # NaN = no crop
HAS = ~np.isnan(MAPROW); MAPROW = np.where(HAS, MAPROW, 0).astype(int)
R = H[ITEMS].values.astype(np.float32)


# ------------------------------------------------------------------------------------------------ model
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1, self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
    def forward(self, x):
        return Fn.relu(x + self.b2(self.c2(Fn.relu(self.b1(self.c1(x))))))


# res_roi (10-25): attention and average pooling restricted to a disc around the femoral-head centre. The crop is 5 head
# radii wide and centred on the head, so on the 14x14 map r = 2.8 cells; disc radius 1.6 r keeps the head, the joint space
# and the acetabular rim and drops the medial pelvic wall and the shaft.
_yy, _xx = np.mgrid[0:14, 0:14]
ROI = torch.tensor((np.hypot(_yy - 6.5, _xx - 6.5) <= 1.6 * 2.8).reshape(-1))


class Head(nn.Module):
    def __init__(self, kind, n_aux=len(ITEMS), c=64):
        super().__init__()
        self.kind = kind; self.drop2d = nn.Dropout2d(0.1); self.drop = nn.Dropout(0.3)
        if kind == "linear":
            self.bn = nn.BatchNorm1d(1024); self.out = nn.Linear(1024, 1 + n_aux)
        else:
            self.reduce = nn.Sequential(nn.Conv2d(1024, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
            self.blocks = nn.Sequential(ResBlock(c), ResBlock(c))
            self.att = nn.Conv2d(c, 1, 1)
            self.out = nn.Linear(2 * c, 1 + n_aux)
    def forward(self, x, return_att=False):
        x = self.drop2d(x)
        if self.kind == "linear":
            return self.out(self.drop(self.bn(x.mean((2, 3)))))
        z = self.blocks(self.reduce(x)); s = self.att(z).flatten(1)
        if self.kind == "res_roi":
            s = s.masked_fill(~ROI, float("-inf")); a = torch.softmax(s, 1)
            f = torch.cat([(z.flatten(2) * a[:, None]).sum(2), z.flatten(2)[:, :, ROI].mean(2)], 1)
        else:
            a = torch.softmax(s, 1)                                                                # (B, 196) attention over the joint
            f = torch.cat([(z.flatten(2) * a[:, None]).sum(2), z.mean((2, 3))], 1)
        o = self.out(self.drop(f))
        return (o, a) if return_att else o


def train_head(kind, tr, epochs=30, seed=0, aux_w=1.0):
    """tr: H-row indices (training subjects, all hips with a crop; THA loss only where y10 known)."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    M = np.load(FM / "densenet_b3.npy", mmap_mode="r")
    tr = tr[HAS[tr]]
    mu, sd = np.nanmean(R[tr], 0), np.nanstd(R[tr], 0) + 1e-6
    net = Head(kind); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    steps = epochs * math.ceil(len(tr) / 64); warm = 3 * math.ceil(len(tr) / 64)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    use_aux = kind in ("linear", "res", "res_roi") and aux_w > 0
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr)
        for b in range(0, len(perm), 64):
            ix = perm[b:b + 64]
            if len(ix) < 4: continue
            x = torch.from_numpy(M[np.sort(MAPROW[ix])].astype(np.float32))
            order = np.argsort(np.argsort(MAPROW[ix])); x = x[order]                               # back to ix order
            dx, dy = np.random.randint(-1, 2, 2); x = torch.roll(x, (int(dy), int(dx)), (2, 3))  # spatial jitter
            o = net(x); yb = y10[ix]; kb = ~np.isnan(yb)
            loss = torch.tensor(0.0)
            if kb.any():
                loss = loss + Fn.binary_cross_entropy_with_logits(o[kb, 0], torch.tensor(yb[kb], dtype=torch.float32), pos_weight=torch.tensor(5.0))
            if use_aux:
                t = torch.tensor((R[ix] - mu) / sd); m = ~torch.isnan(t)
                loss = loss + aux_w * ((o[:, 1:] - torch.nan_to_num(t)) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval(); return net, mu, sd


def predict_head(net, mu, sd, rows):
    M = np.load(FM / "densenet_b3.npy", mmap_mode="r"); out = np.full((len(rows), 1 + len(ITEMS)), np.nan, np.float32)
    ok = HAS[rows]; r = rows[ok]
    with torch.no_grad():
        o = torch.cat([net(torch.from_numpy(M[MAPROW[r[b:b + 128]]].astype(np.float32))) for b in range(0, len(r), 128)]).numpy()
    o[:, 1:] = o[:, 1:] * sd + mu
    out[ok] = o; return out


def subj(rep, k, j=None):
    f = F[F.repeat == rep].set_index("nsin"); te = H.nsin.isin(f.index[f.outer_fold == k]).values
    va = H.nsin.isin(f.index[f[f"inner_for_o{k}"] == j]).values if j is not None else np.zeros(len(H), bool)
    return te, va


def job(kind, k, j):
    """j = None -> fit on the whole outer-training set, predict outer test; j in 0..2 -> cross-fit, predict inner fold j."""
    te, va = subj(REP, k, j)
    tr = np.where(~te & ~va)[0]; ev = np.where(te if j is None else va)[0]
    net, mu, sd = train_head(kind, tr, aux_w=0.0 if kind == "res_noaux" else 1.0)
    o = predict_head(net, mu, sd, ev)
    d = pd.DataFrame(o, columns=["img_logit"] + [f"hat_{c}" for c in ITEMS]); d["row"] = ev; d["outer"] = k; d["inner"] = -1 if j is None else j
    if j is None and kind == "res" and REP == 0: torch.save(net.state_dict(), OUT / f"head_res_o{k}.pt")
    return kind, d


if MODE == "export":   # merge repeat 0 (oof/<tag>) with repeats 1-4 (imghead/...) into oof/<tag> with a repeat column
    from lib_eval import summarise
    O = D / "oof"; (O / "rep0_backup").mkdir(exist_ok=True)
    fams = {"m3_head_res_-": "oof_m3_head_res", "m3_head_res_noaux_-": "oof_m3_head_res_noaux", "m3_head_linear_-": "oof_m3_head_linear",
            "m5_pix_logit_DESCLB": "m5_pix_logit_DESCLB", "m5_pix_items_DESCLB": "m5_pix_items_DESCLB",
            "m5_pix_logit_items_DESCLB": "m5_pix_logit_items_DESCLB", "m5_pix_logit_items_only": "m5_pix_logit_items_only",
            "m5noaux_pix_logit_DESCLB": "m5noaux_pix_logit_DESCLB"}
    for tag, stem in fams.items():
        bk = O / "rep0_backup" / f"{tag}.parquet"
        if not bk.exists() and (O / f"{tag}.parquet").exists(): (O / f"{tag}.parquet").replace(bk)
        if (O / f"{tag}.json").exists() and not (O / "rep0_backup" / f"{tag}.json").exists():
            (O / "rep0_backup" / f"{tag}.json").write_text((O / f"{tag}.json").read_text())
        # repeat 0: a fresh cross-fitted file in imghead/ (has p_cal) wins over the original oof/ file
        parts = [pd.read_parquet(OUT / f"{stem}.parquet")] if stem.startswith("oof_") and (OUT / f"{stem}.parquet").exists() else \
                [pd.read_parquet(bk)] if bk.exists() else []
        parts += [pd.read_parquet(f) for r in range(1, 5) for f in [OUT / f"{stem}_r{r}.parquet"] if f.exists()]
        if not parts: print(tag, "nothing to merge"); continue
        A = pd.concat(parts, ignore_index=True); A.to_parquet(O / f"{tag}.parquet", index=False)
        col = "p_cal" if "p_cal" in A and A.p_cal.notna().all() else "p_raw"
        R_, agg = summarise(A[A.y.notna()].rename(columns={col: "p"}))
        j = json.loads((O / "rep0_backup" / f"{tag}.json").read_text()) if (O / "rep0_backup" / f"{tag}.json").exists() else {col: {}}
        j.setdefault(col, {})["per_repeat_mean"] = agg.loc["mean"].to_dict(); j[col]["per_repeat_sd"] = agg.loc["std"].to_dict()
        j[col]["per_repeat"] = R_[["repeat", "auprc", "auroc", "cal_slope"]].to_dict("records")
        j["note"] = f"repeats {sorted(A.repeat.unique().tolist())}; CI from repeat 0"
        (O / f"{tag}.json").write_text(json.dumps(j, indent=1, default=float))
        print(tag, "repeats", sorted(A.repeat.unique()), "AUPRC", round(agg.loc["mean", "auprc"], 3), "+-", round(agg.loc["std", "auprc"], 3),
              "AUROC", round(agg.loc["mean", "auroc"], 3), "+-", round(agg.loc["std", "auroc"], 3))

elif MODE == "transport":   # pseudo-external validation of the image head (res + aux): leave-one-centre-out and TIFF <-> DICOM
    from sklearn.metrics import roc_auc_score, average_precision_score
    S = pd.read_csv(D / "splits_B.csv")
    def tjob(name, src, tgt):
        net, mu, sd = train_head("res", np.where(src)[0]); ev = np.where(tgt)[0]
        o = predict_head(net, mu, sd, ev); return name, ev, o[:, 0]
    jobs = [(f"loco_{c}", (S.dom_centre != c).values, (S.dom_centre == c).values) for c in sorted(S.dom_centre.unique()) if c.startswith("C")]
    jobs += [("tif->dcm", (S.dom_fmt == "tif").values, (S.dom_fmt == "dcm").values), ("dcm->tif", (S.dom_fmt == "dcm").values, (S.dom_fmt == "tif").values)]
    res = Parallel(n_jobs=NJ)(delayed(tjob)(*j) for j in jobs)
    internal = pd.read_parquet(D / "oof/m3_head_res_-.parquet").set_index(["nsin", "side"]).p_raw
    rows = []
    def score(name, ev, z):
        d = pd.DataFrame({"nsin": H.nsin.values[ev], "side": H.side.values[ev], "y": y10[ev], "ext": z}).dropna()
        d["int"] = internal.reindex(pd.MultiIndex.from_frame(d[["nsin", "side"]])).values; d = d.dropna()
        return {"split": name, "n": len(d), "events": int(d.y.sum()), "auroc_ext": roc_auc_score(d.y, d.ext), "auroc_int": roc_auc_score(d.y, d["int"]),
                "auprc_ext": average_precision_score(d.y, d.ext), "auprc_int": average_precision_score(d.y, d["int"])}
    for name, ev, z in res:
        if np.nansum(y10[ev]) >= 3: rows.append(score(name, ev, z))
    lo = [r for r in res if r[0].startswith("loco")]
    rows.append(score("loco_pooled", np.concatenate([r[1] for r in lo]), np.concatenate([r[2] for r in lo])))
    T = pd.DataFrame(rows); T.to_csv(OUT / "transport_res.csv", index=False); print(T.round(3).to_string())

elif MODE == "perm":   # leakage sanity check: labels permuted across SUBJECTS, same CV; AUROC vs the permuted labels must be ~0.5
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7); subs = H.nsin.unique(); perm = dict(zip(subs, rng.permutation(subs)))
    lab = H.set_index(["nsin", "side"]).y10
    y_perm = np.array([lab.get((perm[n], s), np.nan) for n, s in zip(H.nsin, H.side)])
    y_true = y10.copy(); y10[:] = y_perm
    res = Parallel(n_jobs=NJ)(delayed(job)("res_noaux", k, None) for k in range(5))
    t = pd.concat([r[1] for r in res]); t = t[t.img_logit.notna()]
    ok = ~np.isnan(y_perm[t.row]); print("permuted-label AUROC", round(roc_auc_score(y_perm[t.row][ok], t.img_logit[ok]), 3),
                                         "| vs true labels", round(roc_auc_score(np.nan_to_num(y_true[t.row][~np.isnan(y_true[t.row])]), t.img_logit[~np.isnan(y_true[t.row])]), 3))

elif MODE == "smoke":   # timing / shape check: 2 epochs of the residual head on outer fold 0
    import time
    te, _ = subj(0, 0); t = time.perf_counter(); net, mu, sd = train_head("res", np.where(~te)[0], epochs=2)
    o = predict_head(net, mu, sd, np.where(te)[0][:50]); print("2 epochs", round(time.perf_counter() - t, 1), "s; out", o.shape, np.nanmean(o, 0)[:3])

elif MODE == "probe":
    # frozen global embeddings -> each reading item: ridge on PCA-64, 5-fold by subject (repeat-0 outer folds), Spearman rho
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    from lib_img import image_matrix
    fold = H.nsin.map(F[F.repeat == 0].set_index("nsin").outer_fold).values
    rows = []
    for tag in ("imagenet", "imagenet_ms", "densenet121", "densenet121_adabn", "simclr"):
        if not (D / f"emb/emb_{tag}.npy").exists(): continue
        X = image_matrix(H, tag, D); ok0 = ~np.isnan(X[:, 0])
        for it in ITEMS:
            ok = ok0 & ~np.isnan(R[:, ITEMS.index(it)]); pr = np.full(len(H), np.nan)
            for k in range(5):
                a, b = np.where(ok & (fold != k))[0], np.where(ok & (fold == k))[0]
                p = PCA(64, random_state=0).fit(X[a]); m = Ridge(10.0).fit(p.transform(X[a]), R[a, ITEMS.index(it)])
                pr[b] = m.predict(p.transform(X[b]))
            rows.append({"embedding": tag, "item": it, "spearman": spearmanr(pr[ok], R[ok, ITEMS.index(it)])[0],
                         "prevalence_ge1": float((R[ok, ITEMS.index(it)] >= 1).mean())})
    T = pd.DataFrame(rows); T.to_csv(OUT / "probe_items_frozen.csv", index=False)
    print(T.pivot(index="item", columns="embedding", values="spearman").round(3).to_string())

elif MODE == "train":
    jobs = [(kind, k, j) for kind in KINDS for k in range(5) for j in ((None, 0, 1, 2) if kind in CROSSFIT else (None,))]
    res = Parallel(n_jobs=NJ, verbose=5)(delayed(job)(*a) for a in jobs)
    for kind in KINDS:
        d = pd.concat([r[1] for r in res if r[0] == kind]); d.to_parquet(OUT / f"pred_{kind}{SUF}.parquet", index=False)
    # evaluation: image-only THA (Platt: res -> cross-fitted inner OOF; others -> raw logit ranks + Platt on test-free inner? none,
    # so AUPRC/AUROC are reported on the raw score, calibration only for res) + reading-item agreement on the test folds
    from scipy.stats import spearmanr
    summ = {}
    for kind in KINDS:
        d = pd.read_parquet(OUT / f"pred_{kind}{SUF}.parquet"); t = d[d.inner == -1].copy()
        t["y"] = y10[t.row]; t["nsin"] = H.nsin.values[t.row]; t = t[t.y.notna() & t.img_logit.notna()]
        p = 1 / (1 + np.exp(-t.img_logit.values))
        if kind in CROSSFIT:
            cal = []
            for k in range(5):
                ci = d[(d.outer == k) & (d.inner >= 0)]; ci = ci[~np.isnan(y10[ci.row]) & ci.img_logit.notna()]
                pl = Platt().fit(1 / (1 + np.exp(-ci.img_logit.values)), y10[ci.row]); m = (t.outer == k).values
                p[m] = pl(p[m])
        m = metrics(t.y.values, p); tt = t.assign(p=p).reset_index(drop=True)
        ci95 = cluster_boot(tt[["nsin", "y", "p"]], "p", n=500)
        rho = {}
        if kind != "res_noaux":
            for it in ITEMS:
                v = d[d.inner == -1]; ok = ~np.isnan(R[v.row, ITEMS.index(it)]) & v[f"hat_{it}"].notna().values
                rho[it] = spearmanr(v[f"hat_{it}"].values[ok], R[v.row.values[ok], ITEMS.index(it)])[0]
        summ[kind] = {"auprc": m["auprc"], "auroc": m["auroc"], "cal_slope": m["cal_slope"], "ci95": ci95.round(4).to_dict(),
                      "n": int(len(t)), "events": int(t.y.sum()), "item_spearman": rho}
        print(kind, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ[kind].items() if k != "ci95"}))
        # OOF file in the common format (90_results.py); Platt only where a cross-fitted inner OOF exists
        o = pd.DataFrame({"repeat": REP, "outer": t.outer.values, "nsin": t.nsin.values, "side": H.side.values[t.row],
                          "y": t.y.values, "p_raw": 1 / (1 + np.exp(-t.img_logit.values))})
        if kind in CROSSFIT: o["p_cal"] = p
        o.to_parquet(OUT / f"oof_m3_head_{kind}{SUF}.parquet", index=False)
    sf = OUT / f"train_summary{SUF}.json"; old = json.loads(sf.read_text()) if sf.exists() else {}
    sf.write_text(json.dumps(old | summ, indent=1, default=float))      # keep variants trained in earlier calls

elif MODE == "fuse":
    # M0 (elastic net, same 40 candidates + inner selection) on DESCLB + pixel-derived features. Training rows carry their
    # cross-fitted pixel features (inner fold j predicted by the head trained without fold j), test rows the full-fit head.
    from sklearn.linear_model import LogisticRegression
    CFG = pd.read_csv(D / "hpo/m0_elasticnet.csv"); spec = feature_spec(FD, "DESCLB"); Xraw = raw_frame(H, spec)
    d = pd.read_parquet(OUT / f"pred_{FUSE_KIND}{SUF}.parquet")
    SETS = {"logit": ["img_logit"], "items": [f"hat_{c}" for c in ITEMS], "logit+items": ["img_logit"] + [f"hat_{c}" for c in ITEMS]}

    def pix(k, rows, cols, test):
        sub = d[(d.outer == k) & ((d.inner == -1) if test else (d.inner >= 0))].set_index("row")
        v = sub.reindex(rows)[cols].values.astype(float)
        miss = np.isnan(v[:, 0]); v = np.nan_to_num(v); return v, miss

    def model(c):
        return LogisticRegression(solver="saga", C=c.C, l1_ratio=c.l1_ratio, max_iter=4000, tol=1e-4,
                                  class_weight=None if c.class_weight == "none" else "balanced")

    def run(k, setname, with_tab=True):
        cols = SETS[setname]
        tr, te = fold_rows(H, F, REP, k)

        def build(a, b, b_is_test):
            va, ma = pix(k, a, cols, False); vb, mb = pix(k, b, cols, b_is_test)
            mu, sdv = va[~ma].mean(0), va[~ma].std(0) + 1e-6
            A = [np.column_stack([(va - mu) / sdv * ~ma[:, None], ma])]; B = [np.column_stack([(vb - mu) / sdv * ~mb[:, None], mb])]
            if with_tab:
                pp = TabPrep(spec, "mice"); A.append(pp.fit_transform(Xraw.iloc[a]).values); B.append(pp.transform(Xraw.iloc[b]).values)
            return np.column_stack(A), np.column_stack(B)
        inner = []
        for j in range(3):
            itr, iva = fold_rows(H, F, REP, k, j); Xa, Xb = build(itr, iva, False); inner.append((Xa, y10[itr], Xb, y10[iva]))
        sc = np.zeros(len(CFG)); oo = {}
        for ci, c in enumerate(CFG.itertuples()):
            ps, ys = [], []
            for Xa, ya, Xb, yb in inner: ps.append(model(c).fit(Xa, ya).predict_proba(Xb)[:, 1]); ys.append(yb)
            p, yy = np.concatenate(ps), np.concatenate(ys); sc[ci] = select_score(p, yy); oo[ci] = (p, yy)
        best = int(np.argmin(sc)); Xa, Xb = build(tr, te, True)
        p = model(CFG.iloc[best]).fit(Xa, y10[tr]).predict_proba(Xb)[:, 1]
        return pd.DataFrame({"repeat": REP, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": y10[te],
                             "p_raw": p, "p_cal": Platt().fit(*oo[best])(p), "cfg": best})

    # repeat 0 writes straight into oof/ (original behaviour); repeats > 0 go to imghead/ and are merged by 'export'
    O = D / "oof" if REP == 0 else OUT
    fam = {"res": "m5_pix", "res_noaux": "m5noaux_pix"}.get(FUSE_KIND, f"m5{FUSE_KIND}_pix")
    todo = [(s, True) for s in SETS] + [("logit+items", False)]
    if FUSE_KIND != "res": todo = [("logit", True)]
    res = Parallel(n_jobs=NJ, verbose=5)(delayed(run)(k, s, t) for s, t in todo for k in range(5))
    for i, (s, t) in enumerate(todo):
        oof = pd.concat(res[i * 5:(i + 1) * 5]); tag = f"{fam}_{s.replace('+', '_')}" + ("_DESCLB" if t else "_only") + SUF
        oof.to_parquet(O / f"{tag}.parquet", index=False)
        m = metrics(oof.y.values[oof.y.notna()], oof.p_cal.values[oof.y.notna()])
        ci = cluster_boot(oof[oof.y.notna()].reset_index(drop=True).rename(columns={"p_cal": "p"}), "p", n=500)
        (O / f"{tag}.json").write_text(json.dumps({"p_cal": {"per_repeat_mean": m, "per_repeat_sd": {}, "ci95_repeat0": ci.round(4).to_dict()},
                                                   "note": "repeat 0 only (image head trained on repeat-0 folds)"}, indent=1, default=float))
        print(tag, {k: round(v, 3) for k, v in m.items() if k in ("auprc", "auroc", "cal_slope")})
