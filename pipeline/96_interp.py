"""Interpretability of the tabular models (Experiment C): FT-Transformer attention + LightGBM TreeSHAP, repeat 0.

usage: python -I 96_interp.py <project dir> [n_jobs=5]
reads : oof/m2_DESCLB_selection.csv (architecture / lr / wd chosen per outer fold), oof/m1_DESCLB_selection.csv, hpo/
writes: data/derived/results/interp_ftt_attention.csv   per feature: mean attention received, events vs non-events
        data/derived/results/interp_m1_shap.csv          per feature: mean |SHAP|
        data/derived/figs/res_interp_tabular.png

FT-Transformer: the main M2 recipe (lam = 1, discrete-time bias -5.3, early stopping on the total validation loss) with one
seed per inner fold. For every outer test hip of repeat 0 we take the self-attention maps of all blocks (heads averaged) and
record the attention each feature token *receives*, averaged over query tokens and blocks; this is defined for both readouts
the stage-2 search selects ([CLS] and mean pooling). The three inner-fold models are averaged.
LightGBM: the configuration chosen most often in each outer fold, refitted on the outer training set; exact TreeSHAP values
from LightGBM (pred_contrib) on the outer test hips. Both are descriptive: attention weights are not attributions.
"""
import os, sys, pathlib, warnings
os.environ.setdefault("FTT_DT_BIAS", "-5.3"); os.environ.setdefault("FTT_ES", "total")
import numpy as np, pandas as pd, torch, lightgbm as lgb
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib_tab import load, feature_spec, raw_frame, TabPrep, fold_rows
from lib_ftt import FTT, fit, ple_bins

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 5
D = P / "data/derived"; OUT, R, FG = D / "oof", D / "results", D / "figs"
H, FD, F = load(); GROUPS = "DESCLB"; spec = feature_spec(FD, GROUPS); Xraw = raw_frame(H, spec)
GROUP_OF = {s["name"]: s["group"] for s in spec}
GMAP = {g: i for i, g in enumerate("DESCLBRI")}
y_all = H.y10.fillna(0).values.astype(np.float32); known = H.y10.notna().values
t_obs = np.where(H.event == 1, H.t_event, H.fu_end).astype(np.int64); ev = H.event.values.astype(np.int64)
ARCH = pd.read_csv(D / "hpo/nas_ftt_candidates.csv").set_index("arch")


def subjects(rep, k, j=None):
    f = F[F.repeat == rep].set_index("nsin"); test = set(f.index[f.outer_fold == k])
    val = set(f.index[f[f"inner_for_o{k}"] == j]) if j is not None else set()
    return H.nsin.isin(test).values, H.nsin.isin(val).values


def ftt_fold(k):
    torch.set_num_threads(2)
    s = pd.read_csv(OUT / "m2_DESCLB_selection.csv"); s = s[(s.repeat == 0) & (s.outer == k)].iloc[0]; a = ARCH.loc[int(s.arch)]
    is_te, _ = subjects(0, k); te = np.where(is_te & known)[0]; A = []
    for j in range(3):
        _, is_va = subjects(0, k, j); tr = np.where(~is_te & ~is_va)[0]; va = np.where(is_va)[0]
        pp = TabPrep(spec, "none", scale=True).fit(Xraw.iloc[tr]); xn, mask, xc = pp.ftt_arrays(Xraw)
        names = pp.num + pp.nom
        data = {"xn": torch.tensor(xn), "mask": torch.tensor(mask), "xc": torch.tensor(xc, dtype=torch.long), "y": torch.tensor(y_all),
                "ybin_mask": torch.tensor(known), "t_obs": torch.tensor(t_obs), "event": torch.tensor(ev)}
        m = FTT(xn.shape[1], [len(pp.cats_[c]) for c in pp.nom], [GMAP[GROUP_OF[n]] for n in names], d=int(a.d_token),
                n_blocks=int(a.n_blocks), heads=int(a.n_heads), ffn_mult=float(a.ffn_mult), attn_do=float(a.attn_dropout),
                ffn_do=float(a.ffn_dropout), res_do=float(a.residual_dropout), num_tok=a.numeric_tok, bins=ple_bins(xn[tr], mask[tr]),
                group_tokens=bool(a.group_tokens), readout=a.readout)
        fit(m, data, tr, va, 200, float(s.lr), float(s.wd), patience=20, seed=j, loss_kw=dict(kind=a.loss, pos_weight=5.0, lam=1.0))
        m.eval()
        with torch.no_grad():
            m(data["xn"][te], data["mask"][te], data["xc"][te], keep_attn=True)
        att = torch.stack([b.last_attn for b in m.blocks]).mean(0)        # (n, T, T), heads and blocks averaged
        A.append(att.mean(1)[:, 1:].numpy())                               # attention received by each feature token
    return pd.DataFrame(np.mean(A, 0), columns=names).assign(y=H.y10.values[te], outer=k)


def shap_fold(k):
    CFG = pd.read_csv(D / "hpo/m1_lightgbm.csv"); sel = pd.read_csv(OUT / "m1_DESCLB_selection.csv")
    s = sel[(sel.repeat == 0) & (sel.outer == k)].iloc[0]; c = CFG.iloc[int(s.best)]
    tr, te = fold_rows(H, F, 0, k)
    pp = TabPrep(spec, "none", scale=False); Xa = pp.fit_transform(Xraw.iloc[tr]); Xb = pp.transform(Xraw.iloc[te])
    m = lgb.LGBMClassifier(objective="binary", num_leaves=int(c.num_leaves), max_depth=int(c.max_depth), learning_rate=float(c.learning_rate),
                           min_child_samples=int(c.min_child_samples), subsample=float(c.subsample), subsample_freq=1,
                           colsample_bytree=float(c.colsample_bytree), reg_lambda=float(c.reg_lambda), reg_alpha=float(c.reg_alpha),
                           scale_pos_weight=float(c.scale_pos_weight), n_estimators=max(int(s.n_trees), 10), n_jobs=1, verbose=-1,
                           random_state=0).fit(Xa, H.y10.values[tr])
    sv = m.predict(Xb, pred_contrib=True)[:, :-1]
    return pd.DataFrame(sv, columns=Xa.columns), Xb.reset_index(drop=True)


if __name__ == "__main__":
    att = pd.concat(Parallel(n_jobs=NJ)(delayed(ftt_fold)(k) for k in range(5)), ignore_index=True)
    feats = [c for c in att.columns if c not in ("y", "outer")]
    A = pd.DataFrame({"feature": feats, "group": [GROUP_OF.get(f, "?") for f in feats],
                      "att_all": att[feats].mean().values, "att_event": att.loc[att.y == 1, feats].mean().values,
                      "att_nonevent": att.loc[att.y == 0, feats].mean().values}).sort_values("att_all", ascending=False)
    A["att_uniform"] = 1 / (len(feats) + 1)
    A.to_csv(R / "interp_ftt_attention.csv", index=False)
    sh = Parallel(n_jobs=NJ)(delayed(shap_fold)(k) for k in range(5))
    SV = pd.concat([s[0] for s in sh], ignore_index=True); XV = pd.concat([s[1] for s in sh], ignore_index=True)
    base = lambda c: c.split("=")[0] if c not in GROUP_OF else c
    S = pd.DataFrame({"feature": SV.columns, "group": [GROUP_OF.get(base(c), "?") for c in SV.columns],
                      "mean_abs_shap": SV.abs().mean().values}).sort_values("mean_abs_shap", ascending=False)
    S.to_csv(R / "interp_m1_shap.csv", index=False)
    pd.set_option("display.width", 200)
    print(A.head(15).round(4).to_string()); print(S.head(15).round(4).to_string())

    # ------------------------------------------------------------- figure: (a) FTT attention, (b) SHAP beeswarm
    GC = {"D": "#2a78d6", "E": "#eb6834", "S": "#1baf7a", "C": "#eda100", "L": "#e87ba4", "B": "#008300"}   # categorical slots 1-6, fixed order
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3dd", "#fcfcfb"
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
    N = 15; fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8))
    top = A.head(N).iloc[::-1]
    for i, r in enumerate(top.itertuples()):
        a1.plot([r.att_nonevent, r.att_event], [i, i], color=GRID, lw=2, zorder=1)
        a1.scatter(r.att_nonevent, i, s=22, color="white", edgecolor=GC.get(r.group, MUTED), lw=1.3, zorder=2)
        a1.scatter(r.att_event, i, s=22, color=GC.get(r.group, MUTED), zorder=3)
    a1.axvline(top.att_uniform.iloc[0], color=MUTED, lw=1, ls=(0, (3, 3)))
    a1.set_yticks(range(N)); a1.set_yticklabels([f"{r.group} · {r.feature}" for r in top.itertuples()])
    a1.set_xlabel("attention received (heads and blocks averaged)")
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker="o", ls="", color=GC[g], label=l) for g, l in
                       [("D", "D demographic"), ("E", "E examination"), ("S", "S symptoms"), ("C", "C care / drugs"),
                        ("L", "L social"), ("B", "B biochemistry")] if g in set(top.group)], loc="lower right", fontsize=7.5)
    a1.set_title("(a) FT-Transformer: top-15 tokens\nfilled = THA hips, open = no THA; dashed = uniform", loc="left", fontsize=9, color=INK)
    topS = S.head(N).iloc[::-1]; rng = np.random.default_rng(0)
    for i, r in enumerate(topS.itertuples()):
        v = SV[r.feature].values; x = XV[r.feature].astype(float).values
        q = pd.Series(x).rank(pct=True).values; q = np.where(np.isnan(x), 0.5, q)
        a2.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), c=q, cmap="coolwarm", s=4, alpha=0.6, lw=0)
    a2.axvline(0, color=MUTED, lw=0.8)
    a2.set_yticks(range(N)); a2.set_yticklabels([f"{r.group} · {r.feature}" for r in topS.itertuples()])
    a2.set_xlabel("TreeSHAP value (log-odds of THA)")
    a2.set_title("(b) LightGBM: top-15 features by mean |SHAP|\ncolour = feature value percentile (blue low, red high)", loc="left", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(FG / "res_interp_tabular.png", dpi=200); plt.close(fig)
