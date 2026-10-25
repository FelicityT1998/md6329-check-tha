"""M4 multimodal FT-Transformer: tabular tokens + one image token [I] (plan 7.9 B5), FiLM as the fusion ablation.

usage: python -I 14_m4.py <emb_tag=imagenet> [fusion=token|film] [groups=DESCLB] [lam=1.0] [n_jobs=7]
reads : oof/m2_<groups>_selection.csv (architecture / lr / wd chosen per outer fold by the M2 nested search), hpo/nas_ftt_candidates.csv,
        emb/emb_<tag>.npy
writes: oof/m4_<tag>_<groups>[_film].parquet (+ _all.parquet with censored hips and 2/5/10-y cumulative incidence) + .json

Same CV-ensemble protocol as `12_m2.py ens` so M4 vs M2ens is a paired comparison on identical folds, architectures and seeds:
per outer fold the three early-stopped inner-fold models predict the test fold and are averaged; Platt is fitted on
their pooled inner OOF predictions. The image block (PCA-32 + z-score) is fitted on each inner training set only.
Hips without a crop get a zero image vector (= training mean).
"""
import sys, json, pathlib, warnings, time
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, D
from lib_eval import Platt, summarise, cluster_boot
from lib_ftt import FTT, fit, predict, ple_bins, ES_TOTAL
from lib_img import image_matrix, ImgPrep

warnings.filterwarnings("ignore")
TAG = sys.argv[1] if len(sys.argv) > 1 else "imagenet"
FUSION = sys.argv[2] if len(sys.argv) > 2 else "token"
GROUPS = sys.argv[3] if len(sys.argv) > 3 else "DESCLB"
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
NJ = int(sys.argv[5]) if len(sys.argv) > 5 else 7
SUFFIX = sys.argv[6] if len(sys.argv) > 6 else ""
NPC = 32
H, FD, F = load(); spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec)
GMAP = {g: i for i, g in enumerate("DESCLBRI")}
OUT = D / "oof"; HP = D / "hpo"
y_all = H.y10.fillna(0).values.astype(np.float32); known = H.y10.notna().values
t_obs = np.where(H.event == 1, H.t_event, H.fu_end).astype(np.int64); ev = H.event.values.astype(np.int64)
EPOCHS, PATIENCE, POS_W = 200, 20, 5.0
IMG = image_matrix(H, TAG, D)
SEL = pd.read_csv(OUT / f"m2_{GROUPS}_selection.csv"); ARCH = pd.read_csv(HP / "nas_ftt_candidates.csv").set_index("arch")


def subjects(rep, k, j=None):
    f = F[F.repeat == rep].set_index("nsin"); test = set(f.index[f.outer_fold == k])
    val = set(f.index[f[f"inner_for_o{k}"] == j]) if j is not None else set()
    return H.nsin.isin(test).values, H.nsin.isin(val).values


def make_data(train_idx):
    pp = TabPrep(spec, "none", scale=True).fit(Xraw.iloc[train_idx])
    xn, mask, xc = pp.ftt_arrays(Xraw)
    groups = [GMAP[s["group"]] for s in spec if s["kind"] != "nom"] + [GMAP[s["group"]] for s in spec if s["kind"] == "nom"]
    data = {"xn": torch.tensor(xn), "mask": torch.tensor(mask), "xc": torch.tensor(xc, dtype=torch.long),
            "y": torch.tensor(y_all), "ybin_mask": torch.tensor(known), "t_obs": torch.tensor(t_obs), "event": torch.tensor(ev)}
    img = torch.tensor(ImgPrep(NPC).fit(IMG[train_idx]).transform(IMG, indicator=False))
    return data, ple_bins(xn[train_idx], mask[train_idx]), [len(pp.cats_[c]) for c in pp.nom], groups, img


def build(a, bins, cards, groups, n_num):
    return FTT(n_num, cards, groups, d=int(a["d_token"]), n_blocks=int(a["n_blocks"]), heads=int(a["n_heads"]),
               ffn_mult=float(a["ffn_mult"]), attn_do=float(a["attn_dropout"]), ffn_do=float(a["ffn_dropout"]),
               res_do=float(a["residual_dropout"]), num_tok=a["numeric_tok"], bins=bins, group_tokens=bool(a["group_tokens"]),
               readout=a["readout"], img_dim=NPC, fusion=FUSION)


def run_ens(rep, k):
    torch.set_num_threads(2)
    s = SEL[(SEL.repeat == rep) & (SEL.outer == k)].iloc[0]; a = ARCH.loc[int(s.arch)]
    is_te, _ = subjects(rep, k); te = np.where(is_te)[0]
    P10, CI, po, yo = [], [], [], []
    for j in range(3):
        _, is_va = subjects(rep, k, j)
        tr = np.where(~is_te & ~is_va)[0]; va = np.where(is_va & known)[0]
        data, bins, cards, groups, img = make_data(tr)
        m = build(a, bins, cards, groups, data["xn"].shape[1])
        fit(m, data, tr, np.where(is_va)[0] if ES_TOTAL else va, EPOCHS, float(s.lr), float(s.wd), patience=PATIENCE, seed=j,
            loss_kw=dict(kind=a["loss"], pos_weight=POS_W, lam=LAM), img=img)
        p, _ = predict(m, data, va, img=img); po.append(p); yo.append(y_all[va])
        p, ci = predict(m, data, te, img=img); P10.append(p); CI.append(ci)
    cal = Platt().fit(np.concatenate(po), np.concatenate(yo))
    p, ci = np.mean(P10, 0), np.mean(CI, 0)
    return pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": H.y10.values[te],
                         "p_raw": p, "p_cal": cal(p), "ci2": ci[:, 1], "ci5": ci[:, 4], "ci10": ci[:, 9], "time": t_obs[te], "event": ev[te]})


if __name__ == "__main__":
    t = time.perf_counter()
    allp = pd.concat(Parallel(n_jobs=NJ, verbose=5)(delayed(run_ens)(r, k) for r in range(5) for k in range(5)))
    tag = f"m4_{TAG}_{GROUPS}" + ("" if FUSION == "token" else f"_{FUSION}") + SUFFIX
    allp.to_parquet(OUT / f"{tag}_all.parquet", index=False)
    oof = allp[allp.y.notna()].copy(); oof.to_parquet(OUT / f"{tag}.parquet", index=False)
    summ = {}
    for col in ("p_raw", "p_cal", "ci10"):
        R, agg = summarise(oof.rename(columns={col: "p"}))
        ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
        summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                     "ci95_repeat0": ci.round(4).to_dict()}
    summ.update(lam=LAM, fusion=FUSION, minutes=(time.perf_counter() - t) / 60)
    (OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal", "ci10")}, indent=1))
