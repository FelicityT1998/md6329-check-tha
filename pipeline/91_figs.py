"""Result figures for the report (static PNG, light surface; palette = dataviz reference instance, same as diagrams.py).

usage: python -I 91_figs.py <project dir>
reads : data/derived/results/{results_all,deltas,expB_m0_DESCLB,expB_m1_DESCLB,probe_imagenet}.csv
writes: data/derived/figs/res_A_increment.png, res_A_leave_one_out.png, res_models_DESCLB.png, res_B_transport.png, res_probe.png
"""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))

P = pathlib.Path(sys.argv[1]); R = P / "data/derived/results"; FG = P / "data/derived/figs"; FG.mkdir(exist_ok=True)
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"      # categorical slots in fixed order (4 = violet: bars only)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3dd", "#fcfcfb"
PREV = 93 / 1705
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
T = pd.read_csv(R / "results_all.csv").set_index("model")


def get(m):
    return T.loc[m] if m in T.index else None


def prevalence(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(PREV, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(PREV, 0.005, " event rate 5.5%", transform=ax.get_xaxis_transform(), fontsize=7.5, color=MUTED, va="bottom", rotation=90)


# ---------------------------------------------------------------- A: incremental feature groups (3 model families)
SETS = [("D", "D demographics"), ("DE", "+ E examination"), ("DES", "+ S symptoms"), ("DESC", "+ C care / drugs"),
        ("DESCL", "+ L social / work"), ("DESCLB", "+ B biochemistry  (all clinical)"), ("DESCLBR", "+ R radiograph reading"),
        ("DR", "D + R only")]
FAM = [("m0_{}_mice", "Elastic net (M0)", S1), ("m1_{}", "LightGBM (M1)", S2), ("m1b_{}", "EBM (M1b)", S3)]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for fi, (pat, lab, col) in enumerate(FAM):
    for si, (g, _) in enumerate(SETS):
        r = get(pat.format(g))
        if r is None: continue
        y = si + (fi - 1) * 0.22
        if not np.isnan(r.get("auprc_lo", np.nan)): ax.plot([r.auprc_lo, r.auprc_hi], [y, y], color=col, lw=2, solid_capstyle="round", alpha=0.55)
        ax.plot(r.auprc, y, "o", ms=6, color=col, mec=SURF, mew=1.5, label=lab if not any(t.get_label() == lab for t in ax.lines) else None)
ax.set_yticks(range(len(SETS))); ax.set_yticklabels([s[1] for s in SETS]); ax.invert_yaxis()
ax.axhspan(5.5, 7.5, color="#f3f2ee", zorder=0); ax.text(0.99, 6.5, "reading scores:\nB4 leakage risk", transform=ax.get_yaxis_transform(),
                                                         ha="right", va="center", fontsize=7.5, color=INK2)
prevalence(ax); ax.set_xlabel("AUPRC (mean of 5 repeats; bar = 95% CI, subject bootstrap)"); ax.set_xlim(0, 0.75); ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", fontsize=8); ax.set_title("Experiment A - adding feature groups to demographics", loc="left", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FG / "res_A_increment.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- A: leave-one-group-out from DESCLB (paired deltas, M0 + M1)
from importlib import import_module
pr = import_module("90_results")
LOO = [("ESCLB", "- D demographics"), ("DSCLB", "- E examination"), ("DECLB", "- S symptoms"), ("DESLB", "- C care / drugs"),
       ("DESCB", "- L social / work"), ("DESCL", "- B biochemistry")]
rows = []
for pat, lab, col in FAM[:2]:
    for g, gl in LOO:
        a, b = pat.format(g), pat.format("DESCLB")
        if a in T.index and b in T.index:
            pr.O = P / "data/derived/oof"; d = pr.paired_delta(a, b); rows.append({**d, "family": lab, "group": gl, "col": col})
L = pd.DataFrame(rows); L.to_csv(R / "leave_one_group_out.csv", index=False)
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for fi, (pat, lab, col) in enumerate(FAM[:2]):
    sub = L[L.family == lab]
    for i, gl in enumerate([x[1] for x in LOO]):
        r = sub[sub.group == gl]
        if r.empty: continue
        r = r.iloc[0]; y = i + (fi - 0.5) * 0.3
        ax.plot([r.d_auprc_lo, r.d_auprc_hi], [y, y], color=col, lw=2, alpha=0.55, solid_capstyle="round")
        ax.plot(r.d_auprc, y, "o", ms=6, color=col, mec=SURF, mew=1.5, label=lab if i == 0 else None)
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(len(LOO))); ax.set_yticklabels([x[1] for x in LOO]); ax.invert_yaxis()
ax.set_xlabel("change in AUPRC when the group is removed from all-clinical (DESCLB), paired bootstrap 95% CI"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", fontsize=8); ax.set_title("Experiment A - leave one group out", loc="left", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FG / "res_A_leave_one_out.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- main model comparison on the all-clinical set
MODELS = [("m0_DESCLB_mice", "M0 elastic net"), ("m1_DESCLB", "M1 LightGBM"), ("m1b_DESCLB", "M1b EBM"),
          ("m2ens_DESCLB_lam1_dtinit_es_s3", "M2 FT-Transformer"), ("m3_imagenet_-", "M3 image only (ResNet-18 frozen)"),
          ("m3_densenet121_-", "M3 image only (DenseNet-121 frozen)"), ("m3_simclr_-", "M3 image only (ResNet-18 + SimCLR)"),
          ("m3_imagenet_DESCLB", "M0 + ResNet-18 image PCA (linear fusion)"),
          ("m3_densenet121_DESCLB", "M0 + DenseNet-121 image PCA (linear fusion)"),
          ("m3_head_res_-", "M3 image only, residual head + OARSI aux"),
          ("m5_pix_logit_DESCLB", "M0 + pixel image score"), ("m4_imagenet_DESCLB", "M4 FT-T + image token"),
          ("m4_imagenet_DESCLB_film", "M4 FT-T + image FiLM"),
          ("m4_densenet121_DESCLB_film_dtinit_es", "M4 FT-T + DenseNet FiLM (fixed training)")]   # R reference kept out of the main figure (inadmissible, see temporal audit)
MODELS = [m for m in MODELS if m[0] in T.index]
fig, axs = plt.subplots(1, 2, figsize=(9.2, 0.42 * len(MODELS) + 1.2), sharey=True)
for ax, met, xl in ((axs[0], "auprc", "AUPRC"), (axs[1], "auroc", "AUROC")):
    for i, (m, lab) in enumerate(MODELS):
        r = T.loc[m]; col = S2 if "image" in lab or "M3" in lab or "M4" in lab else (MUTED if "R reading" in lab else S1)
        if not np.isnan(r.get(f"{met}_lo", np.nan)): ax.plot([r[f"{met}_lo"], r[f"{met}_hi"]], [i, i], color=col, lw=2, alpha=0.55, solid_capstyle="round")
        ax.plot(r[met], i, "o", ms=6.5, color=col, mec=SURF, mew=1.5)
        ax.text(r[met], i - 0.28, f"{r[met]:.3f}", ha="center", fontsize=7, color=INK2)
    ax.set_xlabel(f"{xl} (mean of 5 repeats; 95% CI)"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(MODELS))); axs[0].set_yticklabels([m[1] for m in MODELS]); axs[0].invert_yaxis()
prevalence(axs[0]); axs[1].axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
axs[0].set_title("Models on all clinical features (blue) and with pixels (orange)", loc="left", color=INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(FG / "res_models_DESCLB.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- B: transport (dumbbell) + calibration intercept
for mdl in ("m0", "m1"):
    f = R / f"expB_{mdl}_DESCLB.csv"
    if not f.exists(): continue
    B = pd.read_csv(f); B = B[B.split != "loco_centre"].copy()
    B["label"] = B.source + " -> " + B.target + "  (" + B.events_target.astype(str) + " ev)"
    B.loc[B.split == "loco_pooled", "label"] = "leave-one-centre-out, pooled (" + B.events_target.astype(str) + " ev)"
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 0.42 * len(B) + 1.3), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    for i, r in enumerate(B.itertuples()):
        axs[0].plot([r.auprc_int, r.auprc_ext], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        axs[0].plot(r.auprc_int, i, "o", ms=6.5, color=S1, mec=SURF, mew=1.5, zorder=2, label="internal (nested CV, same hips)" if i == 0 else None)
        axs[0].plot(r.auprc_ext, i, "o", ms=6.5, color=S2, mec=SURF, mew=1.5, zorder=3, label="transported (trained on source only)" if i == 0 else None)
        axs[0].text(max(r.auprc_int, r.auprc_ext) + 0.015, i, f"{r.d_auprc:+.3f} [{r.d_auprc_lo:+.2f}, {r.d_auprc_hi:+.2f}]", va="center", fontsize=7, color=INK2)
        axs[1].plot(r.cal_int_int, i, "o", ms=6.5, color=S1, mec=SURF, mew=1.5); axs[1].plot(r.cal_int_ext, i, "o", ms=6.5, color=S2, mec=SURF, mew=1.5)
    axs[0].set_yticks(range(len(B))); axs[0].set_yticklabels(B.label); axs[0].invert_yaxis(); axs[0].set_xlim(0, 0.85)
    axs[0].set_xlabel("AUPRC in the target domain (label: transported - internal, 95% CI)"); axs[0].legend(loc="upper right", fontsize=7.5)
    axs[1].axvline(0, color=INK2, lw=1); axs[1].set_xlabel("calibration-in-the-large (0 = calibrated)")
    for a in axs: a.grid(axis="y", visible=False)
    axs[0].set_title(f"Experiment B - pseudo-external validation ({'elastic net' if mdl == 'm0' else 'LightGBM'}, all clinical)",
                     loc="left", color=INK, fontsize=10.5)
    fig.tight_layout(); fig.savefig(FG / f"res_B_transport_{mdl}.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- shortcut probes
f = R / "probe_imagenet.csv"
if f.exists():
    Q = pd.read_csv(f)
    INP = [("whole_film_embedding", "whole film (ImageNet)", S4), ("hip_crop_embedding", "hip crop (ImageNet)", S2),
           ("hip_crop_embedding_residualised", "hip crop, centre x format means removed", S3), ("clinical_table_DESCLB", "clinical table (DESCLB)", S1)]
    TG = [("centre", "centre (10 classes, macro)"), ("format", "TIFF vs DICOM"), ("followup_group", "annual vs interval group"),
          ("outcome_THA10", "outcome: 10-y THA (reference)")]
    fig, ax = plt.subplots(figsize=(7.6, 4.0)); w = 0.19
    for ii, (k, lab, col) in enumerate(INP):
        for ti, (t, _) in enumerate(TG):
            r = Q[(Q.input == k) & (Q.target == t)]
            if r.empty: continue
            v = r.auroc.iloc[0]; x = ti + (ii - 1.5) * (w + 0.02)
            ax.bar(x, v - 0.5, w, bottom=0.5, color=col, label=lab if ti == 0 or (ti == 2 and ii == 2) else None)
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h)); ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc="upper right")
    ax.set_xticks(range(len(TG))); ax.set_xticklabels([t[1] for t in TG], fontsize=8); ax.set_ylim(0.5, 1.05); ax.set_ylabel("probe AUROC (5-fold, by subject)")
    ax.grid(axis="x", visible=False); ax.set_title("Shortcut probes - what else can the inputs predict?", loc="left", color=INK, fontsize=10.5)
    fig.tight_layout(); fig.savefig(FG / "res_probe.png", dpi=200); plt.close(fig)
print("figures ->", FG)
