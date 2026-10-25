"""M2 FT-Transformer: two-stage architecture search + nested repeated grouped CV (plan 7.9).

usage:
  python -I 12_m2.py nas  [groups=DESCLB] [n_jobs=7]   stage 1: 48 pre-sampled architectures (hpo/nas_ftt_candidates.csv)
                                                       on the 3 inner folds of repeat 0 / outer fold 0 -> hpo/nas_stage1_<g>.csv
  python -I 12_m2.py cv   [groups=DESCLB] [n_jobs=7] [lam=0.5] [tag]
                                                       stage 2: top-3 architectures x lr {3e-4,1e-3} x wd {1e-5,1e-4} selected
                                                       per outer fold by inner Platt-recalibrated log-loss; refit with the median
                                                       inner best epoch, 3 seeds averaged -> oof/m2_<g><tag>.parquet

Training rows = all hips of training subjects (censored hips feed only the discrete-time head); the binary head and every
metric use hips with known 10-year status. Preprocessing (Yeo-Johnson/log1p, winsorising, z-score, PLE bin edges) is fitted
on the training rows of each split.
"""
import sys, json, pathlib, warnings, time
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, D
from lib_eval import Platt, summarise, cluster_boot, select_score
from lib_ftt import FTT, fit, predict, ple_bins, ES_TOTAL
import os
N_SEEDS = int(os.environ.get("FTT_SEEDS", 1))

warnings.filterwarnings("ignore")
MODE = sys.argv[1]; GROUPS = sys.argv[2] if len(sys.argv) > 2 else "DESCLB"
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 7
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
TAG = sys.argv[5] if len(sys.argv) > 5 else ""
H, FD, F = load(); spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec)
GMAP = {g: i for i, g in enumerate("DESCLBRI")}
OUT = D / "oof"; OUT.mkdir(exist_ok=True); HP = D / "hpo"
y_all = H.y10.fillna(0).values.astype(np.float32); known = H.y10.notna().values
t_obs = np.where(H.event == 1, H.t_event, H.fu_end).astype(np.int64); ev = H.event.values.astype(np.int64)
EPOCHS, PATIENCE, POS_W = 200, 20, 5.0


def subjects(rep, k, j=None):
    f = F[F.repeat == rep].set_index("nsin"); test = set(f.index[f.outer_fold == k])
    val = set(f.index[f[f"inner_for_o{k}"] == j]) if j is not None else set()
    is_te, is_va = H.nsin.isin(test).values, H.nsin.isin(val).values
    return is_te, is_va


def make_data(train_idx):
    pp = TabPrep(spec, "none", scale=True).fit(Xraw.iloc[train_idx])
    xn, mask, xc = pp.ftt_arrays(Xraw)
    groups = [GMAP[s["group"]] for s in spec if s["kind"] != "nom"] + [GMAP[s["group"]] for s in spec if s["kind"] == "nom"]
    data = {"xn": torch.tensor(xn), "mask": torch.tensor(mask), "xc": torch.tensor(xc, dtype=torch.long),
            "y": torch.tensor(y_all), "ybin_mask": torch.tensor(known), "t_obs": torch.tensor(t_obs), "event": torch.tensor(ev)}
    bins = ple_bins(xn[train_idx], mask[train_idx])
    cards = [len(pp.cats_[c]) for c in pp.nom]
    return data, bins, cards, groups


def build(arch, bins, cards, groups, n_num):
    return FTT(n_num, cards, groups, d=int(arch["d_token"]), n_blocks=int(arch["n_blocks"]), heads=int(arch["n_heads"]),
               ffn_mult=float(arch["ffn_mult"]), attn_do=float(arch["attn_dropout"]), ffn_do=float(arch["ffn_dropout"]),
               res_do=float(arch["residual_dropout"]), num_tok=arch["numeric_tok"], bins=bins,
               group_tokens=bool(arch["group_tokens"]), readout=arch["readout"])


def inner_eval(rep, k, arch, lr, wd, seed=0):
    """pooled inner OOF predictions for one configuration on outer fold (rep, k)."""
    torch.set_num_threads(2)
    is_te, _ = subjects(rep, k)
    ps, ys, eps = [], [], []
    for j in range(3):
        _, is_va = subjects(rep, k, j)
        tr = np.where(~is_te & ~is_va)[0]; va = np.where(is_va & known)[0]
        data, bins, cards, groups = make_data(tr)
        m = build(arch, bins, cards, groups, data["xn"].shape[1])
        be, _ = fit(m, data, tr, va, EPOCHS, lr, wd, patience=PATIENCE, seed=seed,
                    loss_kw=dict(kind=arch["loss"], pos_weight=POS_W, lam=LAM))
        p, _ = predict(m, data, va); ps.append(p); ys.append(y_all[va]); eps.append(be)
    p, yy = np.concatenate(ps), np.concatenate(ys)
    return select_score(p, yy), p, yy, int(np.median(eps))


if MODE == "nas":
    A = pd.read_csv(HP / "nas_ftt_candidates.csv")
    t = time.perf_counter()
    res = Parallel(n_jobs=NJ, verbose=5)(delayed(inner_eval)(0, 0, a, float(a.lr), float(a.weight_decay)) for _, a in A.iterrows())
    A["score"] = [r[0] for r in res]; A["best_epoch"] = [r[3] for r in res]
    A["inner_auprc"] = [__import__("sklearn.metrics", fromlist=["x"]).average_precision_score(r[2], r[1]) for r in res]
    A.sort_values("score").to_csv(HP / f"nas_stage1_{GROUPS}.csv", index=False)
    print(f"stage 1 done in {time.perf_counter() - t:.0f}s"); print(A.sort_values("score").head(8).to_string())

elif MODE == "ens":
    # CV-ensemble: per outer fold, reuse the configuration selected by stage 2 (oof/m2_<g>_selection.csv); the three
    # early-stopped inner-fold models predict the outer test fold and are averaged; Platt is fitted on their own pooled
    # inner OOF predictions, so the calibration map belongs to exactly the models that make the test predictions.
    # (Fixes the refit/Platt mismatch of mode "cv": calibration intercept -1.07.)
    SEL = pd.read_csv(OUT / f"m2_{GROUPS}_selection.csv")
    ARCH = pd.read_csv(HP / "nas_ftt_candidates.csv").set_index("arch")

    def run_ens(rep, k):
        torch.set_num_threads(2)
        s = SEL[(SEL.repeat == rep) & (SEL.outer == k)].iloc[0]; a = ARCH.loc[int(s.arch)].copy(); a["arch"] = int(s.arch)
        is_te, _ = subjects(rep, k); te = np.where(is_te)[0]
        P10, CI, po, yo, EPLOG = [], [], [], [], []
        for j in range(3):
            _, is_va = subjects(rep, k, j)
            tr = np.where(~is_te & ~is_va)[0]; va = np.where(is_va & known)[0]
            va_fit = np.where(is_va)[0] if ES_TOTAL else va      # total-loss early stopping also sees censored val hips
            data, bins, cards, groups = make_data(tr)
            pv, pt, ct = [], [], []
            for sd in range(N_SEEDS):                              # FTT_SEEDS > 1: seed ensemble inside each inner fold
                m = build(a, bins, cards, groups, data["xn"].shape[1])
                be, hist = fit(m, data, tr, va_fit, EPOCHS, float(s.lr), float(s.wd), patience=PATIENCE, seed=j + 10 * sd,
                               loss_kw=dict(kind=a["loss"], pos_weight=POS_W, lam=LAM))
                EPLOG.append({"repeat": rep, "outer": k, "inner": j, "seed": sd, "best_epoch": be, "stopped_at": len(hist),
                              "arch": int(s.arch), "lr": float(s.lr)})
                pv.append(predict(m, data, va)[0]); p, ci = predict(m, data, te); pt.append(p); ct.append(ci)
            po.append(np.mean(pv, 0)); yo.append(y_all[va]); P10.append(np.mean(pt, 0)); CI.append(np.mean(ct, 0))
        cal = Platt().fit(np.concatenate(po), np.concatenate(yo))
        p, ci = np.mean(P10, 0), np.mean(CI, 0)
        # discrete-time risk is recalibrated on the same inner OOF via its own 10-y cumulative incidence
        return pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": H.y10.values[te],
                             "p_raw": p, "p_cal": cal(p), "ci2": ci[:, 1], "ci5": ci[:, 4], "ci10": ci[:, 9], "time": t_obs[te], "event": ev[te]}), EPLOG

    t = time.perf_counter()
    res = Parallel(n_jobs=NJ, verbose=5)(delayed(run_ens)(r, k) for r in range(5) for k in range(5))
    allp = pd.concat([r[0] for r in res])
    tag = f"m2ens_{GROUPS}{TAG}"
    pd.DataFrame([e for r in res for e in r[1]]).to_csv(OUT / f"{tag}_epochs.csv", index=False)   # best / stopping epoch per model
    allp.to_parquet(OUT / f"{tag}_all.parquet", index=False)
    oof = allp[allp.y.notna()].copy(); oof.to_parquet(OUT / f"{tag}.parquet", index=False)
    summ = {}
    for col in ("p_raw", "p_cal", "ci10"):
        R, agg = summarise(oof.rename(columns={col: "p"}))
        ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
        summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                     "ci95_repeat0": ci.round(4).to_dict()}
    summ["lam"] = LAM; summ["minutes"] = (time.perf_counter() - t) / 60
    (OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal", "ci10")}, indent=1))

elif MODE == "cv":
    S1 = pd.read_csv(HP / f"nas_stage1_{GROUPS}.csv").head(3)
    grid = [(a, lr, wd) for _, a in S1.iterrows() for lr in (3e-4, 1e-3) for wd in (1e-5, 1e-4)]

    def run(rep, k):
        torch.set_num_threads(2)
        scores = [inner_eval(rep, k, a, lr, wd) for a, lr, wd in grid]
        best = int(np.argmin([s[0] for s in scores])); a, lr, wd = grid[best]; n_ep = max(5, scores[best][3])
        is_te, _ = subjects(rep, k); tr = np.where(~is_te)[0]; te = np.where(is_te)[0]
        data, bins, cards, groups = make_data(tr)
        P10, CI, hists = [], [], []
        for seed in range(3):
            m = build(a, bins, cards, groups, data["xn"].shape[1])
            _, h = fit(m, data, tr, None, n_ep, lr, wd, seed=seed, loss_kw=dict(kind=a["loss"], pos_weight=POS_W, lam=LAM))
            p, ci = predict(m, data, te); P10.append(p); CI.append(ci); hists.append(h)
        p, ci = np.mean(P10, 0), np.mean(CI, 0)
        cal = Platt().fit(scores[best][1], scores[best][2])
        te_known = known[te]
        oof = pd.DataFrame({"repeat": rep, "outer": k, "nsin": H.nsin.values[te], "side": H.side.values[te], "y": H.y10.values[te],
                            "p_raw": p, "p_cal": cal(p), "ci2": ci[:, 1], "ci5": ci[:, 4], "ci10": ci[:, 9],
                            "time": t_obs[te], "event": ev[te]})
        sel = {"repeat": rep, "outer": k, "arch": int(a["arch"]), "lr": lr, "wd": wd, "epochs": n_ep, "inner_score": scores[best][0]}
        return oof, sel, {"repeat": rep, "outer": k, "hist": hists[0]}

    t = time.perf_counter()
    res = Parallel(n_jobs=NJ, verbose=5)(delayed(run)(r, k) for r in range(5) for k in range(5))
    allp = pd.concat([r[0] for r in res]); sel = pd.DataFrame([r[1] for r in res])
    tag = f"m2_{GROUPS}{TAG}"
    allp.to_parquet(OUT / f"{tag}_all.parquet", index=False)
    oof = allp[allp.y.notna()].copy(); oof.to_parquet(OUT / f"{tag}.parquet", index=False)
    sel.to_csv(OUT / f"{tag}_selection.csv", index=False)
    (OUT / f"{tag}_curves.json").write_text(json.dumps([r[2] for r in res]))
    summ = {}
    for col in ("p_raw", "p_cal", "ci10"):
        R, agg = summarise(oof.rename(columns={col: "p"}))
        ci = cluster_boot(oof[oof.repeat == 0].reset_index(drop=True).rename(columns={col: "p"}), "p", n=500)
        summ[col] = {"per_repeat_mean": agg.loc["mean"].round(4).to_dict(), "per_repeat_sd": agg.loc["std"].round(4).to_dict(),
                     "ci95_repeat0": ci.round(4).to_dict()}
    summ["selection"] = {str(k): int(v) for k, v in sel.groupby(["arch", "lr", "wd"]).size().items()}; summ["lam"] = LAM; summ["minutes"] = (time.perf_counter() - t) / 60
    (OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(tag, json.dumps({k: summ[k]["per_repeat_mean"] for k in ("p_raw", "p_cal", "ci10")}, indent=1))
