"""Small, self-contained assets for the Lectrace lecture (lectrace/01_seeing_surgery_coming.py), 2026-10-24.

usage: python -I 98_lectrace_assets.py <project dir>
writes: lectrace/assets/{demo.json, fmap.npy, head_res_o0.pt, crop.png, oof_rep0.csv}

One demo hip is taken from the outer test fold 0 of repeat 0: a THA hip that the clinical model ranks modestly and the
fusion model ranks high. For that split, M0 and the fusion model are refitted exactly as in 10_m0.py / 08_imghead.py fuse
(same 40 candidates, same inner selection, same Platt), and their test predictions are checked against the saved OOF
files, so every number the lecture computes is the number in the report.
"""
import sys, json, shutil, pathlib, warnings
import numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

P = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(P)); warnings.filterwarnings("ignore")
from lib_tab import load, feature_spec, raw_frame, TabPrep, fold_rows, D
from lib_eval import Platt, select_score

OUT = P / "lectrace/assets"; OUT.mkdir(parents=True, exist_ok=True)
FM = pathlib.Path(r"D:\datasets\CHECK\fmaps")
REP, K = 0, 0
ITEMS = ["r_kl", "r_jsn_sup", "r_jsn_med", "r_ost_sup_acet", "r_ost_inf_acet", "r_ost_sup_fem", "r_ost_inf_fem",
         "r_sclerosis", "r_cysts", "r_flattening", "r_buttressing"]
H, FD, F = load(); y10 = H.y10.values
spec = feature_spec(FD, "DESCLB"); Xraw = raw_frame(H, spec)
CFG = pd.read_csv(D / "hpo/m0_elasticnet.csv")
pred = pd.read_parquet(D / "imghead/pred_res.parquet")
crops = pd.read_csv(D / "crops.csv")

# ------------------------------------------------------------------ choose the demo hip
m0 = pd.read_parquet(D / "oof/m0_DESCLB_mice.parquet").query("repeat == @REP")
fu = pd.read_parquet(D / "oof/m5_pix_logit_DESCLB.parquet").query("repeat == @REP")
oof = m0[["outer", "nsin", "side", "y", "p_cal", "cfg"]].rename(columns={"p_cal": "p_m0", "cfg": "cfg_m0"}).merge(
    fu[["nsin", "side", "p_cal", "cfg"]].rename(columns={"p_cal": "p_fused", "cfg": "cfg_fused"}), on=["nsin", "side"])
oof[["nsin", "y", "p_m0", "p_fused"]].to_csv(OUT / "oof_rep0.csv", index=False)
t = oof[oof.outer == K].copy()
t["rank_m0"], t["rank_fused"] = t.p_m0.rank(pct=True), t.p_fused.rank(pct=True)
ok_crop = crops[~crops.get("qc_exclude", pd.Series(False, index=crops.index)).fillna(False).astype(bool)][["nsin", "side"]]
cand = t[(t.y == 1)].merge(ok_crop, on=["nsin", "side"])
demo = cand.loc[(cand.rank_fused - cand.rank_m0).idxmax()]
row = int(np.where((H.nsin == demo.nsin) & (H.side == demo.side))[0][0])
print("demo hip", demo.nsin, demo.side, "| M0 p", round(demo.p_m0, 3), "rank", round(demo.rank_m0, 3),
      "| fused p", round(demo.p_fused, 3), "rank", round(demo.rank_fused, 3))


def model(c):
    return LogisticRegression(solver="saga", C=c.C, l1_ratio=c.l1_ratio, max_iter=4000, tol=1e-4,
                              class_weight=None if c.class_weight == "none" else "balanced")


def pix(rows, test):
    sub = pred[(pred.outer == K) & ((pred.inner == -1) if test else (pred.inner >= 0))].set_index("row")
    v = sub.reindex(rows)["img_logit"].values.astype(float); miss = np.isnan(v); return np.nan_to_num(v), miss


def refit(with_img, best):
    """Refit the selected configuration on outer-train fold K; Platt from that configuration's inner OOF predictions."""
    tr, te = fold_rows(H, F, REP, K)

    def build(a, b, b_test):
        pp = TabPrep(spec, "mice"); A = pp.fit_transform(Xraw.iloc[a]); B = pp.transform(Xraw.iloc[b]); names = list(A.columns)
        A, B = A.values, B.values; extra = {}
        if with_img:
            va, ma = pix(a, False); vb, mb = pix(b, b_test); mu, sd = va[~ma].mean(), va[~ma].std() + 1e-6
            A = np.column_stack([(va - mu) / sd * ~ma, ma, A]); B = np.column_stack([(vb - mu) / sd * ~mb, mb, B])
            names = ["img_logit_z", "img_missing"] + names; extra = {"img_mu": mu, "img_sd": sd}
        return A, B, names, pp, extra
    c = CFG.iloc[best]; ps, ys = [], []
    for j in range(3):
        itr, iva = fold_rows(H, F, REP, K, j); A, B, *_ = build(itr, iva, False)
        ps.append(model(c).fit(A, y10[itr]).predict_proba(B)[:, 1]); ys.append(y10[iva])
    cal = Platt().fit(np.concatenate(ps), np.concatenate(ys))
    A, B, names, pp, extra = build(tr, te, True)
    m = model(c).fit(A, y10[tr]); p = m.predict_proba(B)[:, 1]
    i = int(np.where(te == row)[0][0])
    return dict(names=names, coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), x=B[i].tolist(),
                platt_a=float(cal.m.coef_[0, 0]), platt_b=float(cal.m.intercept_[0]), p_raw=float(p[i]), p_cal=float(cal(p[i:i + 1])[0]),
                cfg=dict(C=float(c.C), l1_ratio=float(c.l1_ratio), class_weight=c.class_weight), **extra), pp


M0, pp = refit(False, int(demo.cfg_m0)); FU, _ = refit(True, int(demo.cfg_fused))
print("check M0  p_cal refit / saved:", round(M0["p_cal"], 4), round(float(demo.p_m0), 4))
print("check fus p_cal refit / saved:", round(FU["p_cal"], 4), round(float(demo.p_fused), 4))

# ------------------------------------------------------------------ preprocessing walk-through for a few features
walk = {}
for name in ["ir_rom", "ir_rom_asym", "ir_pain", "womac_pain", "hip_stiff", "sos_mean"]:
    if name not in pp.num: continue
    s = next(s for s in spec if s["name"] == name)
    w = {"raw": None if pd.isna(Xraw.iloc[row][name]) else float(Xraw.iloc[row][name]), "transform": s["transform"],
         "lo": float(pp.lo_[name]), "hi": float(pp.hi_[name]), "mu": float(pp.mu_[name]), "sd": float(pp.sd_[name])}
    if s["src_c"]: w.update(ipsi=float(H.iloc[row][s["src"]]), contra=float(H.iloc[row][s["src_c"]]))
    if s["transform"] == "yeojohnson": w["lambda"] = float(pp.pt_[name].lambdas_[0])
    walk[name] = w

# ------------------------------------------------------------------ image assets
idx = pd.read_csv(FM / "densenet_b3_index.csv"); j = int(idx.index[(idx.nsin == demo.nsin) & (idx.side == demo.side)][0])
np.save(OUT / "fmap.npy", np.load(FM / "densenet_b3.npy", mmap_mode="r")[j].astype(np.float16))
shutil.copy(D / "imghead/head_res_o0.pt", OUT / "head_res_o0.pt")
cp = crops[(crops.nsin == demo.nsin) & (crops.side == demo.side)].crop.iloc[0]
a = np.array(Image.open(cp)).astype(np.float32); Image.fromarray((a / a.max() * 255).astype(np.uint8)).save(OUT / "crop.png")
tr_all = np.where(~H.nsin.isin(F[(F.repeat == REP) & (F.outer_fold == K)].nsin))[0]
has = H.iloc[tr_all][["nsin", "side"]].merge(idx[["nsin", "side"]], how="left", indicator=True)._merge.eq("both").values
Rtr = H.iloc[tr_all[has]][ITEMS].values.astype(float)
head_logit = pred[(pred.outer == K) & (pred.inner == -1) & (pred.row == row)].img_logit.iloc[0]

json.dump(dict(nsin=int(demo.nsin), side=demo.side, y10=int(demo.y), t_event=float(H.iloc[row].t_event),
               event_rate=float(np.nanmean(y10)), m0=M0, fused=FU, walk=walk, items=ITEMS,
               item_mu=np.nanmean(Rtr, 0).tolist(), item_sd=(np.nanstd(Rtr, 0) + 1e-6).tolist(),
               item_reading=[None if pd.isna(v) else float(v) for v in H.iloc[row][ITEMS]],
               n_test_fold=int(len(t)), head_logit_saved=float(head_logit), rank_m0=float(demo.rank_m0), rank_fused=float(demo.rank_fused)),
          open(OUT / "demo.json", "w"), indent=1)
print("assets ->", OUT, "| head logit saved", round(float(head_logit), 3))
