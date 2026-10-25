"""Report figures that are drawn from data (not by the image model): literature forest plot (Table 1), study timeline
(Fig 1), clinical-feature correlation (Fig 3), pipeline and fold layout (Fig 4), hyper-parameter / NAS summary (Fig 7).

usage: python -I 97_report_figs.py <project dir> [out dir = report/figs]
reads : hips.parquet, features.csv (via lib_tab), hpo/nas_stage1_DESCLB.csv, oof/m2_DESCLB_selection.csv, oof/m2ens_*.parquet,
        results/results_all.csv, results/expB_m0_DESCLB.csv
Literature numbers are typed in from the abstracts (litreview/abstracts.md); rows marked * need a full-text check.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score

P = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(P))
OUTD = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else P / "report/figs"; OUTD.mkdir(parents=True, exist_ok=True)
D = P / "data/derived"
S1, S2, S3, S4, S5, S6, S7 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3dd", "#fcfcfb"
FILL = {S1: "#e3eefb", S2: "#fde6dc", S3: "#dcf3ea", S7: "#e6e3f8", MUTED: "#f1f0ec", S4: "#fcefd2"}
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": False, "legend.frameon": False,
                     "savefig.dpi": 300})
T = pd.read_csv(D / "results/results_all.csv").set_index("model")


def hanley_mcneil(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return max(0, a - 1.96 * se), min(1, a + 1.96 * se)


# =============================================================== Table 1: literature forest plot
# (panel, study, cohort, endpoint, hips, events, modality, model label, auroc, lo, hi, ci_kind, auprc, prevalence, validation)
# ci_kind: "rep" reported, "hm" Hanley-McNeil from n/events, "sd" mean +- SD across held-out cohorts, None not estimable
LIT = [
    ("hip", "Agricola 2013", "CHECK", "THA, 5 y", 1400, 35, "I", "SSM shape modes", 0.81, None, None, "hm", None, 0.025, "apparent (GEE)"),
    ("hip", "Hirvasniemi 2019", "CHECK", "rHOA/THA, 10 y", 987, 435, "C", "age, sex, BMI", 0.59, None, None, "hm", None, 0.44, "internal CV"),
    ("hip", "", "", "", 987, 435, "M", "+ bone texture", 0.68, None, None, "hm", None, 0.44, "internal CV"),
    ("hip", "Gielis 2020", "CHECK, painful", "rHOA/THA, 8 y", 1044, 185, "C", "clinical + reader scores", 0.795, 0.757, 0.834, "rep", None, 0.177, "internal"),
    ("hip", "", "", "", 1044, 185, "M", "+ Shape-Score", 0.864, 0.833, 0.895, "rep", None, 0.177, "internal"),
    ("hip", "Runhaar 2021", "CHECK", "clin. HOA, 5-10 y", 735, 162, "M", "history+exam+X-ray", 0.71, None, None, "hm", None, 0.22, "internal"),
    ("hip", "Saberi Hosnijeh 2018", "Rotterdam", "rHOA, 10 y", 2327, None, "C", "basic clinical", 0.67, None, None, None, None, None, "ext. to CHECK"),
    ("hip", "", "", "", 2327, None, "M", "+ imaging", 0.78, None, None, None, None, None, "ext. to CHECK"),
    ("hip", "Jang 2023", "OAI", "THA, 10 y", 9592, 230, "C", "110 clinical vars", 0.681, 0.609, 0.752, "rep", 0.08, 0.024, "held-out test"),
    ("hip", "", "", "", 9592, 230, "I", "26 DL measurements", 0.769, 0.693, 0.845, "rep", 0.22, 0.024, "held-out test"),
    ("hip", "", "", "", 9592, 230, "M", "combined (GAM)", 0.807, 0.741, 0.874, "rep", 0.28, 0.024, "held-out test"),
    ("hip", "van den Berg 2026", "World COACH", "rHOA, 4-8 y", 25090, 1280, "C", "demographics", 0.71, None, None, "hm", None, 0.051, "internal CV"),
    ("hip", "", "", "", 25090, 1280, "I", "DL features", 0.81, None, None, "hm", None, 0.051, "internal CV"),
    ("hip", "", "", "", 25090, 1280, "M", "hybrid, LOCO (mean ± SD)", 0.64, 0.55, 0.73, "sd", None, 0.051, "LOCO"),
    ("hip", "Hidaka 2024", "Japan, 1 centre", "KL>=3, 5 y", 703, 203, "M", "CNN + GBDT", 0.846, None, None, "hm", None, 203 / 703, "5-fold CV"),
    ("knee", "Tiulpin 2019", "OAI -> MOST", "KL progression", 3918, 1501, "C", "clinical, no KL (LR)", 0.68, 0.66, 0.69, "rep", 0.56, 1501 / 3918, "external"),
    ("knee", "", "", "", 3918, 1501, "I", "CNN, image only", 0.79, 0.77, 0.80, "rep", 0.68, 1501 / 3918, "external"),
    ("knee", "", "", "", 3918, 1501, "M", "CNN + clinical (GBM)", 0.79, 0.78, 0.81, "rep", 0.68, 1501 / 3918, "external"),
    ("knee", "Musbahi 2026", "MOST -> OAI", "TKA", None, None, "M", "DenseNet-201 + clinical", 0.85, None, None, None, 0.62, 0.10, "internal"),
    ("knee", "", "", "", None, None, "M", "same, external OAI", 0.79, None, None, None, None, None, "external"),
    ("knee", "Castagno 2025", "OAI (FNIH)", "rapid progr., 2 y", None, None, "C", "clinical-only autoML", None, None, None, None, 0.764, None, "external"),
]
loco = pd.read_csv(D / "results/expB_m0_DESCLB.csv"); loco = loco[loco.split == "loco_pooled"].iloc[0]
OURS = [("m0_DESCLB_mice", "C", "M0 clinical elastic net", "nested CV"),
        ("m3_densenet121_DESCLB", "M", "M0 + frozen DenseNet (v1)", "nested CV"),
        ("m3_head_res_-", "I", "residual image head only", "nested CV"),
        ("m5_pix_logit_DESCLB", "M", "M0 + pixel logit (final)", "nested CV")]
rows = [dict(zip(["panel", "study", "cohort", "endpoint", "n", "ev", "mod", "label", "auroc", "lo", "hi", "ci", "auprc", "prev", "val"], r)) for r in LIT]
for i, (m, mod, lab, val) in enumerate(OURS):
    r = T.loc[m]
    rows.append(dict(panel="ours", study="This study" if i == 0 else "", cohort="CHECK" if i == 0 else "", endpoint="THA, 10 y" if i == 0 else "",
                     n=1705, ev=93, mod=mod, label=lab, auroc=r.auroc, lo=r.auroc_lo, hi=r.auroc_hi, ci="rep", auprc=r.auprc, prev=93 / 1705, val=val))
rows.append(dict(panel="ours", study="", cohort="", endpoint="", n=1705, ev=92, mod="C", label="M0, leave-one-centre-out", auroc=loco.auroc_ext,
                 lo=None, hi=None, ci="hm", auprc=loco.auprc_ext, prev=93 / 1705, val="pseudo-external"))
F = pd.DataFrame(rows)
for i, r in F.iterrows():
    if r.ci == "hm" and pd.notna(r.ev) and pd.notna(r.n) and pd.notna(r.auroc):
        F.loc[i, ["lo", "hi"]] = hanley_mcneil(r.auroc, r.ev, r.n - r.ev)
MC = {"C": S1, "I": S2, "M": S3}
fig = plt.figure(figsize=(7.4, 6.6))
gs = fig.add_gridspec(1, 3, width_ratios=[4.3, 1.45, 0.95], wspace=0.05, left=0.01, right=0.985, top=0.95, bottom=0.115)
at, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
ypos, y, prev_panel = [], 0, None
heads = {"hip": "Hip OA prognosis", "knee": "Knee OA (reference)", "ours": "This study (CHECK, 1,705 hips, 93 THA, 5.5%)"}
for i, r in F.iterrows():
    if r.panel != prev_panel:
        y += 0.6 if prev_panel else 0; at.text(0.0, y, heads[r.panel], fontsize=7.8, fontweight="bold", color=INK, va="center")
        y += 1; prev_panel = r.panel
    ypos.append(y); y += 1
F["y"] = ypos; Y = y
cols = [(0.0, "Study"), (1.02, "Cohort · endpoint"), (2.42, "Hips (events; %)"), (3.38, "Model input")]
for x, h in cols: at.text(x, -1.2, h, fontsize=7, color=INK2, fontweight="bold", va="center")
for r in F.itertuples():
    at.text(0.0, r.y, r.study, fontsize=6.6, color=INK, va="center")
    ce = (r.cohort + (" · " + r.endpoint if r.endpoint else "")) if r.cohort else ""
    at.text(1.02, r.y, ce, fontsize=6.2, color=INK2, va="center")
    ns = "" if pd.isna(r.n) else (f"{int(r.n):,}" + ("" if pd.isna(r.ev) else f" ({int(r.ev)}; {100 * r.ev / r.n:.1f}%)"))
    if r.study or r.panel == "ours" and r.label.startswith("M0 clinical"): at.text(2.42, r.y, ns, fontsize=6.2, color=INK2, va="center")
    at.text(3.38, r.y, r.label, fontsize=6.2, color=INK2, va="center")
at.set_xlim(0, 4.75); at.set_ylim(Y, -2); at.axis("off")
for ax in (a1, a2):
    ax.set_ylim(Y, -2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    for r in F.itertuples():
        if r.panel == "ours": ax.axhspan(r.y - 0.5, r.y + 0.5, color="#f4f3ef", zorder=0, lw=0)
for r in F.itertuples():
    if pd.isna(r.auroc): continue
    c = MC[r.mod]
    if pd.notna(r.lo):
        a1.plot([r.lo, r.hi], [r.y, r.y], color=c, lw=1.2, ls="-" if r.ci == "rep" else (0, (2, 1.2)), zorder=2)
    sz = 14 if pd.isna(r.ev) else 10 + 22 * np.log10(r.ev)
    a1.scatter(r.auroc, r.y, s=sz, marker="s" if r.panel == "ours" else "o", color=c, edgecolor="white", lw=0.6, zorder=3)
a1.axvline(0.5, color=MUTED, lw=0.8); a1.set_xlim(0.5, 1.0); a1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
a1.set_xlabel("AUROC (95% CI)"); a1.set_title("Discrimination", fontsize=7.8, color=INK, loc="left")
for v in (0.6, 0.7, 0.8, 0.9): a1.axvline(v, color=GRID, lw=0.5, zorder=0)
for r in F.itertuples():
    if pd.notna(r.auprc) and pd.notna(r.prev):
        lift = r.auprc / r.prev
        a2.scatter(lift, r.y, s=16, marker="s" if r.panel == "ours" else "o", color=MC[r.mod], edgecolor="white", lw=0.6, zorder=3)
        a2.text(lift + 0.5, r.y, f"{r.auprc:.2f}", fontsize=5.8, color=INK2, va="center")
    elif pd.notna(r.auprc):
        a2.text(0.6, r.y, f"AUPRC {r.auprc:.2f}; rate n.r.", fontsize=5.6, color=MUTED, va="center")
a2.axvline(1, color=MUTED, lw=0.8); a2.set_xlim(0, 15); a2.set_xticks([1, 5, 10, 15])
a2.set_xlabel("AUPRC lift (÷ event rate)"); a2.set_title("Precision", fontsize=7.8, color=INK, loc="left")
for v in (5, 10): a2.axvline(v, color=GRID, lw=0.5, zorder=0)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=S1, label="clinical / demographic"),
                    Line2D([], [], marker="o", ls="", color=S2, label="imaging only"),
                    Line2D([], [], marker="o", ls="", color=S3, label="combined"),
                    Line2D([], [], color=INK2, lw=1.2, label="reported CI"),
                    Line2D([], [], color=INK2, lw=1.2, ls=(0, (2, 1.2)), label="Hanley-McNeil CI or ±SD")],
           loc="lower left", ncol=5, fontsize=6.3, bbox_to_anchor=(0.01, -0.005))
fig.savefig(OUTD / "tab1_forest.png"); plt.close(fig)
F.drop(columns="y").to_csv(OUTD / "tab1_forest_data.csv", index=False)

# =============================================================== Fig 1: study timeline
H = pd.read_parquet(D / "hips.parquet")
fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.set_xlim(-1.6, 11.6); ax.set_ylim(-0.5, 7.2); ax.axis("off")
lanes = {"inputs": 6.4, "quest": 5.5, "xray": 4.6, "reads": 3.7, "events": 1.6}
for x in range(0, 11):
    ax.plot([x, x], [0.2, 6.9], color=GRID, lw=0.5, zorder=0); ax.text(x, 0.0, f"{x}", ha="center", fontsize=7, color=INK2)
ax.text(5, -0.45, "years after baseline (t₀ = first presentation, enrolment Oct 2002 – Sep 2005)", ha="center", fontsize=7, color=INK2)
ax.add_patch(Rectangle((0.6, 0.2), 1.9, 6.7, color="#f4f3ef", zorder=0, lw=0))
ax.text(1.55, 6.95, "lag-2 analysis drops\nTHA in years 1-2", ha="center", fontsize=6.2, color=MUTED, va="bottom")
lab = lambda y, s: ax.text(-1.55, y, s, fontsize=7, color=INK, va="center", fontweight="bold")
lab(lanes["inputs"], "Predictors"); lab(lanes["quest"], "Questionnaire"); lab(lanes["xray"], "AP pelvis film"); lab(lanes["reads"], "KL / OARSI read"); lab(lanes["events"] + 0.8, "Hips per year")
ax.scatter(0, lanes["inputs"], s=60, marker="D", color=S1, zorder=3)
ax.text(0.45, lanes["inputs"], "D · E · S · C · L · B  (76 features)  +  baseline radiograph pixels  - all fixed at t₀", fontsize=7, color=INK, va="center")
for x in range(0, 11):
    ax.scatter(x, lanes["quest"], s=16, color=S1, zorder=3)
for x in (0, 2, 5, 8, 10):
    ax.scatter(x, lanes["quest"] - 0.22, s=16, facecolor="white", edgecolor=S1, lw=1, zorder=3)
ax.text(10.35, lanes["quest"], "annual group (86%)\n○ interval group (14%)", fontsize=6.2, color=INK2, va="center")
for x in (0, 2, 5, 8, 10):
    ax.scatter(x, lanes["xray"], s=26, marker="s", color=S2 if x == 0 else "white", edgecolor=S2, lw=1, zorder=3)
ax.text(10.35, lanes["xray"], "■ input (t₀ only)\n□ outcome check only", fontsize=6.2, color=INK2, va="center")
ax.annotate("", xy=(0.2, lanes["reads"]), xytext=(10.2, lanes["reads"]), arrowprops=dict(arrowstyle="->", color=S7, lw=1.2))
ax.text(5, lanes["reads"] + 0.25, "all films read together after year 10, paired, in known order (Macri 2022)  → excluded as predictors;"
        " used only as training targets of the image head", fontsize=6.3, color=S7, ha="center")
ev = H[H.event == 1].t_event.value_counts().sort_index(); ce = H[H.y10.isna()].fu_end.value_counts().sort_index()
for t, n in ev.items(): ax.bar(t, n / 40 * 1.6, bottom=lanes["events"], width=0.32, color=S3, align="center")
for t, n in ce.items(): ax.bar(t + 0.34, n / 40 * 1.6, bottom=lanes["events"], width=0.32, color=MUTED, align="center")
ax.plot([-0.3, 10.6], [lanes["events"]] * 2, color=INK2, lw=0.6)
ax.scatter([10.45, 10.45], [lanes["events"] + 1.25, lanes["events"] + 0.85], marker="s", s=14, color=[S3, MUTED])
ax.text(10.6, lanes["events"] + 1.25, "THA (93 hips, 75 people)", fontsize=6.2, color=INK2, va="center"); ax.text(10.6, lanes["events"] + 0.85, "censored < 10 y (299 hips)", fontsize=6.2, color=INK2, va="center")
ax.text(-1.55, lanes["events"] - 0.55, "2,004 hips · 1,705 with known 10-y status · 850 / 1,002 participants completed year 10\n"
        "THA = prosthesis flag, first post-operative visit, or prosthesis seen on a follow-up film (checked against the films)",
        fontsize=6.2, color=INK2, va="center")
fig.savefig(OUTD / "fig2_timeline.png", bbox_inches="tight"); plt.close(fig)

# =============================================================== Fig 3: correlation of the 76 main clinical features
from lib_tab import load, feature_spec, raw_frame
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
H2, FD, Fo = load(); spec = feature_spec(FD, "DESCLB"); X = raw_frame(H2, spec)
num = [s["name"] for s in spec if s["kind"] != "nom"]; grp = {s["name"]: s["group"] for s in spec}
C = X[num].astype(float).corr(method="spearman").fillna(0)
order = leaves_list(linkage(squareform(1 - np.abs(C.values), checks=False), "average")); C = C.iloc[order, order]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b62f30"])
fig = plt.figure(figsize=(7.4, 6.9)); ax = fig.add_axes([0.2, 0.2, 0.68, 0.73]); sx = fig.add_axes([0.2, 0.935, 0.68, 0.012])
im = ax.imshow(C.values, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(C))); ax.set_yticks(range(len(C)))
ax.set_xticklabels(C.columns, rotation=90, fontsize=4.6); ax.set_yticklabels([f"{grp[c]} · {c}" for c in C.columns], fontsize=4.6)
for s in ax.spines.values(): s.set_visible(False)
GC = {"D": S1, "E": S2, "S": S3, "C": S4, "L": S5, "B": S6}
sx.imshow([[i for i in range(len(C))]], aspect="auto", cmap=matplotlib.colors.ListedColormap([GC[grp[c]] for c in C.columns])); sx.axis("off")
hi = [(a, b, C.loc[a, b]) for i, a in enumerate(C.columns) for b in C.columns[i + 1:] if abs(C.loc[a, b]) >= 0.7]
for a, b, v in hi:
    i, j = list(C.columns).index(a), list(C.columns).index(b)
    for (u, w) in ((i, j), (j, i)): ax.add_patch(Rectangle((w - 0.5, u - 0.5), 1, 1, fill=False, ec=INK, lw=0.6))
cb = fig.add_axes([0.9, 0.45, 0.015, 0.3]); fig.colorbar(im, cax=cb); cb.tick_params(labelsize=6); cb.set_title("ρ", fontsize=7)
fig.legend(handles=[Rectangle((0, 0), 1, 1, color=GC[g], label=l) for g, l in
                    [("D", "D demographic"), ("E", "E examination"), ("S", "S symptoms"), ("C", "C care / drugs"), ("L", "L social / work"), ("B", "B biochemistry")]],
           loc="lower left", ncol=6, fontsize=6.3, bbox_to_anchor=(0.2, 0.0))
fig.savefig(OUTD / "fig3_correlation.png"); plt.close(fig)
pd.DataFrame(hi, columns=["a", "b", "rho"]).to_csv(OUTD / "fig3_high_pairs.csv", index=False)
print("Fig 3: numeric features", len(num), "| |rho|>=0.7 pairs:", [(a, b, round(v, 2)) for a, b, v in hi])


# =============================================================== Fig 4: pipeline and fold layout
def box(ax, x, y, w, h, title, body, col, fs=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012", fc=FILL.get(col, "#f1f0ec"), ec=col, lw=1.1))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=fs + 0.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.06, body, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.35)


def arrow(ax, a, b, col=MUTED, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, ls=ls, mutation_scale=8))


fig, ax = plt.subplots(figsize=(7.4, 4.8)); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0, 0.985, "Inputs fixed at t₀ · follow-up used only for labels · every fitted step inside the training fold", fontsize=7.6, fontweight="bold", color=INK, va="top")
# tabular row
box(ax, 0.00, 0.66, 0.18, 0.25, "CHECK T0 tables", "371 variables\n1,002 people → 2,004 hips\n1,705 known 10-y status\n93 THA (5.5%)", S1)
box(ax, 0.21, 0.66, 0.19, 0.25, "Admissibility gate", "t₀ values only\nR readings out (future info)\ncentre-protocol items out\n→ 76 features, 6 groups", S7)
box(ax, 0.43, 0.66, 0.20, 0.25, "In-fold tabular prep", "asymmetry terms (row-wise)\nlog1p / Yeo-Johnson, winsorise\nmissing flags + MICE\nz-score · one-hot", S1)
box(ax, 0.66, 0.66, 0.15, 0.25, "Tabular models", "M0 elastic net\nM1 LightGBM\nM1b EBM\nM2 FT-Transformer", S1)
# image row
box(ax, 0.00, 0.33, 0.18, 0.25, "Baseline AP pelvis", "1,123 films\nTIFF 637 · DICOM 487\n10 recovered centres", S2)
box(ax, 0.21, 0.33, 0.19, 0.25, "Localise & crop", "rotation fix (~110 films)\nheat-map locator, 2 peaks\n2.5 × head diameter, 256²\n1,963 hips with a crop", S2)
box(ax, 0.43, 0.33, 0.20, 0.25, "Image head (M3)", "frozen DenseNet-121 block 3\nresidual head, attention pool\n11 OARSI aux targets (train)\n→ cross-fitted THA logit", S2)
box(ax, 0.66, 0.33, 0.15, 0.25, "Fusion", "M0 + pixel logit\n(main model)\ncontrols: PCA fusion,\n[I] token, FiLM (M4)", S3)
for y0 in (0.785, 0.455):
    for x0, x1 in ((0.18, 0.21), (0.40, 0.43), (0.63, 0.66)): arrow(ax, (x0, y0), (x1, y0))
arrow(ax, (0.735, 0.66), (0.735, 0.58), S1)
# evaluation column
box(ax, 0.84, 0.50, 0.16, 0.41, "Evaluation", "5 × 5 outer folds\ngrouped by participant\nstratified THA × format\n3 inner folds: tuning,\nearly stopping, Platt\n\nAUPRC (primary)\nAUROC, calibration,\nBrier, decision curve", S4)
box(ax, 0.84, 0.03, 0.16, 0.43, "Auxiliary analyses", "temporal audit, lag-2/3\nleave-one-centre-out\nformat / pain / care /\nsex / education transport\nshortcut probes\nattention · SHAP\nfeature-group ablation", MUTED)
arrow(ax, (0.81, 0.785), (0.84, 0.75)); arrow(ax, (0.81, 0.455), (0.84, 0.6))
# fold schematic
ax.text(0.0, 0.265, "One outer split (repeat r, fold k)", fontsize=7, fontweight="bold", color=INK)
seg = [(0.0, 0.64, "outer training participants (80%)", S1), (0.64, 0.16, "outer test (20%)", S4)]
for x, w, t, c in seg:
    ax.add_patch(Rectangle((x, 0.19), w - 0.004, 0.05, fc=FILL.get(c), ec=c, lw=0.9)); ax.text(x + w / 2, 0.215, t, ha="center", va="center", fontsize=6.4, color=INK)
for j in range(3):
    x = 0.0 + j * 0.2133
    ax.add_patch(Rectangle((x, 0.115), 0.2093, 0.045, fc="white", ec=S1, lw=0.7, ls=(0, (2, 1))))
    ax.text(x + 0.104, 0.1375, f"inner fold {j + 1}", ha="center", va="center", fontsize=5.6, color=INK2)
ax.text(0.0, 0.055, "Inner folds: hyper-parameters (Platt-recalibrated log-loss), early stopping, the Platt map, and the\n"
        "cross-fitted image logit of every training hip (head trained on the other two inner folds).\n"
        "Refit on the whole outer training set → predict the outer test fold once.",
        fontsize=6.2, color=INK2, va="center")
fig.savefig(OUTD / "fig4_pipeline.png", bbox_inches="tight"); plt.close(fig)

# =============================================================== Fig 7: HPO / NAS summary
N1 = pd.read_csv(D / "hpo/nas_stage1_DESCLB.csv"); SEL = pd.read_csv(D / "oof/m2_DESCLB_selection.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.15, 0.75, 1.5], wspace=0.45))
a = axs[0]; top3 = N1.head(3)
for tok, mk in (("linear", "o"), ("periodic", "^"), ("piecewise_linear", "s")):
    d = N1[N1.numeric_tok == tok]; a.scatter(d.params / 1e3, d.score, s=14, marker=mk, color=S1, alpha=0.55, lw=0, label=tok.replace("_", "-"))
a.scatter(top3.params / 1e3, top3.score, s=40, facecolor="none", edgecolor=S2, lw=1.3, label="top 3 → stage 2")
for r in top3.itertuples(): a.annotate(f"#{r.arch}", (r.params / 1e3, r.score), xytext=(4, -6), textcoords="offset points", fontsize=6, color=INK2)
a.set_xlabel("trainable parameters (×1,000)"); a.set_ylabel("inner log-loss (Platt-recalibrated)"); a.legend(fontsize=5.6, loc="lower left", ncol=2, borderaxespad=0.3, handletextpad=0.2, columnspacing=0.6)
a.set_ylim(N1.score.min() - 0.012, N1.score.max() + 0.002)   # free band under the points for the legend
a.set_title("(a) NAS stage 1: 48 architectures", loc="left", fontsize=7.5, color=INK); a.grid(True, color=GRID, lw=0.5)
b = axs[1]; cnt = SEL.groupby(["arch", "lr"]).size().reset_index(name="n")
labs = [f"#{r.arch}\nlr {r.lr:g}" for r in cnt.itertuples()]
b.barh(range(len(cnt)), cnt.n, color=[S2 if r.lr == 1e-3 else "#f3b498" for r in cnt.itertuples()], height=0.7)
b.set_yticks(range(len(cnt))); b.set_yticklabels(labs, fontsize=5.8); b.invert_yaxis(); b.set_xlabel("outer folds (of 25)")
b.set_title("(b) Stage 2 choices", loc="left", fontsize=7.5, color=INK)
c = axs[2]
VERS = [("m2ens_DESCLB_lam1", "v1, DT head\nuntrained"), ("m2ens_DESCLB_lam1_dtinit", "+ DT bias\ninit"),
        ("m2ens_DESCLB_lam1_dtinit_es", "+ total-loss\nstopping"), ("m2ens_DESCLB_lam1_dtinit_es_s3", "+ 3 seeds\n(final)"),
        ("m2ens_DESCLB_v2_bn_s3", "BatchNorm\nablation")]
SDS = []
for i, (stem, lab) in enumerate(VERS):
    d = pd.read_parquet(D / f"oof/{stem}.parquet"); d = d[d.y.notna()]
    v = [average_precision_score(g.y, g.p_cal) for _, g in d.groupby("repeat")]
    c.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=10, color=S1 if i != 3 else S2, alpha=0.8, lw=0)
    c.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=INK, lw=1.2)
    SDS.append(np.std(v, ddof=1))
c.axhline(T.loc["m0_DESCLB_mice"].auprc, color=MUTED, lw=0.8, ls=(0, (3, 3))); c.text(4.4, T.loc["m0_DESCLB_mice"].auprc + 0.004, "M0", fontsize=6, color=MUTED, ha="right")
c.set_xticks(range(len(VERS))); c.set_xticklabels([f"{v[1]}\nSD {s:.3f}" for v, s in zip(VERS, SDS)], fontsize=5.2); c.set_ylabel("AUPRC per repeat")
c.set_title("(c) M2 training recipe, 5 repeats", loc="left", fontsize=7.5, color=INK); c.grid(True, axis="y", color=GRID, lw=0.5)
fig.savefig(OUTD / "fig8_hpo_nas.png", bbox_inches="tight"); plt.close(fig)
print("done ->", OUTD)
