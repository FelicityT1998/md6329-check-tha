"""Exploratory data analysis E1-E10 + collinearity block (plan 7.2).

usage: python -I 03_eda.py <project dir> [--no-montage]
reads : data/derived/hips.parquet, subjects.parquet, features.csv, images.csv (01_cohort.py / eda.py)
writes: data/derived/eda/  (E*_*.png figures, E*_*.csv tables, summary.txt)

Figures are static PNGs for the NeurIPS report (light surface). Colour roles: categorical slots in fixed order,
sequential blue for magnitude (missingness), diverging blue<->red with grey midpoint for correlations.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; OUT = D / "eda"; OUT.mkdir(parents=True, exist_ok=True)
MONTAGE = "--no-montage" not in sys.argv
sys.path.insert(0, str(P))

H = pd.read_parquet(D / "hips.parquet"); S = pd.read_parquet(D / "subjects.parquet").set_index("nsin")
FD = pd.read_csv(D / "features.csv"); IMG = pd.read_csv(D / "images.csv")
LOG = open(OUT / "summary.txt", "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n")

# ---------------------------------------------------------------- style (reference palette, light)
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#b8302f"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "font.family": ["Segoe UI", "DejaVu Sans"]})
def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight"); plt.close(fig)

def km(df):
    """Hip-level KM cumulative incidence at years 0..10 (annual grid) + at-risk counts."""
    s, cum, risk = 1.0, [0.0], [len(df)]
    for t in range(1, 11):
        at = (df.time >= t).sum(); d = ((df.event == 1) & (df.t_event == t)).sum()
        if at: s *= 1 - d / at
        cum.append(1 - s); risk.append(at)
    return np.array(cum), np.array(risk)

known = H[H.y10.notna()]
CENTRES = [f"C{i:02d}" for i in range(1, 11)]

# ================================================================ E1 flow
n_img = int(H.has_img.sum()); n_known = len(known); n_known_img = int(known.has_img.sum())
fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.axis("off"); ax.grid(False)
boxes = [(0.5, 0.92, f"CHECK participants\nn = {len(S):,}"),
         (0.5, 0.72, f"Hips (2 per participant)\nn = {len(H):,}  |  THA events = {int(H.event.sum())}"),
         (0.2, 0.47, f"Censored < 10 y, no event\nn = {int((H.status10 == 'censored<10y').sum())} hips\n(IPCW / survival only)"),
         (0.75, 0.47, f"10-y status known (main)\nn = {n_known:,}  |  events = {int(known.y10.sum())}"),
         (0.55, 0.17, f"+ baseline AP pixels\nn = {n_known_img:,}  |  events = {int(known[known.has_img].y10.sum())}"),
         (0.92, 0.17, f"centre C01-C10\nn = {int(known.centre.isin(CENTRES).sum()):,}")]
for x, y, t in boxes:
    ax.text(x, y, t, ha="center", va="center", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec=AXIS, lw=0.8))
for (x0, y0), (x1, y1) in [((0.5, 0.86), (0.5, 0.78)), ((0.5, 0.66), (0.22, 0.54)), ((0.5, 0.66), (0.73, 0.54)),
                           ((0.75, 0.41), (0.57, 0.23)), ((0.75, 0.41), (0.9, 0.23))]:
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
save(fig, "E1_flow.png")

# ================================================================ E2 Table 1
def fmt_cont(x): return f"{x.mean():.1f} ± {x.std():.1f}" if x.notna().any() else ""
def fmt_bin(x): return f"{x.mean() * 100:.0f}%" if x.notna().any() else ""
rows = []
spec = [("age", "cont", "Age, y"), ("female", "bin", "Female"), ("bmi", "cont", "BMI"), ("education", "cont", "Education (2-8)"),
        ("not_working", "bin", "Not in paid work"), ("comor_n", "cont", "Comorbidities (0-4)"), ("hip_pain", "bin", "Ipsilateral hip pain"),
        ("hip_pain_c", "bin", "Contralateral hip pain"), ("knee_pain", "bin", "Ipsilateral knee pain"), ("hip_stiff", "bin", "Hip morning stiffness"),
        ("womac_pain", "cont", "WOMAC pain (0-20)"), ("womac_func", "cont", "WOMAC function (0-68)"), ("nrs_week", "cont", "NRS past week"),
        ("ir_rom", "cont", "Internal rotation, °"), ("ir_le25", "bin", "IR ≤ 25°"), ("flex_rom", "cont", "Flexion, °"),
        ("ir_pain", "cont", "Pain on IR (1-4)"), ("care_no_care", "bin", "No care in prior year"), ("care_gp", "bin", "GP visit prior year"),
        ("med_nsaid", "bin", "NSAID use"), ("r_kl", "cont", "Baseline KL (0-4)")]
groups = {"All hips": H, "THA ≤10 y": H[H.event == 1], "No THA (10-y known)": known[known.y10 == 0], "Censored <10 y": H[H.status10 == "censored<10y"]}
for col, kind, lab in spec:
    r = {"Variable": lab}
    for g, df in groups.items(): r[g] = (fmt_cont if kind == "cont" else fmt_bin)(df[col])
    r["missing"] = f"{H[col].isna().mean() * 100:.1f}%"; rows.append(r)
for g in ["annual", "interval"]:
    rows.append({"Variable": f"Visit group = {g}", **{k: fmt_bin(df.group.eq(g).astype(float)) for k, df in groups.items()}, "missing": ""})
for k in ["tif", "dcm"]:
    rows.append({"Variable": f"Film format = {k}", **{g: fmt_bin(df.img_fmt.eq(k).astype(float)) for g, df in groups.items()}, "missing": ""})
T1 = pd.DataFrame(rows); T1.loc[-1] = ["n hips", *[str(len(df)) for df in groups.values()], ""]; T1 = T1.sort_index()
T1.to_csv(OUT / "E2_table1.csv", index=False, encoding="utf-8-sig")
log("== E2 Table 1\n" + T1.to_string(index=False))

# ================================================================ E3 outcome: KM panels
def km_panel(ax, strata, title, colors=None):
    for i, (lab, df) in enumerate(strata):
        c, r = km(df)
        col = (colors or SER)[i]
        ax.step(range(11), c * 100, where="post", color=col, lw=2, label=f"{lab} (n={len(df)}, ev={int(df.event.sum())})")
        ax.text(10.15, c[-1] * 100, f"{c[-1] * 100:.1f}%", color=INK2, fontsize=7.5, va="center")
    ax.set_title(title, loc="left"); ax.set_xlim(0, 11); ax.set_xticks([0, 2, 5, 8, 10])
    ax.set_xlabel("Years from baseline"); ax.set_ylabel("Cumulative THA, %"); ax.legend(fontsize=7, loc="upper left")
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
km_panel(axs[0, 0], [("All hips", H)], "Overall")
km_panel(axs[0, 1], [(f"KL {k:.0f}", H[H.r_kl == k]) for k in [0, 1, 2, 3]], "By baseline KL (reader-scored, B4)")
km_panel(axs[0, 2], [("No pain", H[H.hip_pain == 0]), ("Ipsilateral pain", H[H.hip_pain == 1])], "By ipsilateral hip pain")
km_panel(axs[1, 0], [("Annual", H[H.group == "annual"]), ("Interval", H[H.group == "interval"])], "By visiting group (B3)")
km_panel(axs[1, 1], [("TIFF", H[H.img_fmt == "tif"]), ("DICOM", H[H.img_fmt == "dcm"])], "By film format (B5)")
km_panel(axs[1, 2], [("IR > 25°", H[H.ir_le25 == 0]), ("IR ≤ 25°", H[H.ir_le25 == 1])], "By internal rotation ≤ 25°")
fig.tight_layout(); save(fig, "E3_km_strata.png")

# centre: small multiples, each centre against the pooled curve
fig, axs = plt.subplots(2, 5, figsize=(13, 5), sharex=True, sharey=True)
pooled, _ = km(H[H.centre.isin(CENTRES)])
for ax, c in zip(axs.ravel(), CENTRES):
    df = H[H.centre == c]; cc, _ = km(df)
    ax.step(range(11), pooled * 100, where="post", color=AXIS, lw=1.5)
    ax.step(range(11), cc * 100, where="post", color=SER[0], lw=2)
    ax.set_title(f"{c}  n={df.nsin.nunique()}  ev={int(df.event.sum())}", loc="left", fontsize=8.5)
    ax.text(10, cc[-1] * 100, f"{cc[-1] * 100:.1f}%", color=INK2, fontsize=7.5, ha="right", va="bottom")
    ax.set_xticks([0, 5, 10])
for ax in axs[:, 0]: ax.set_ylabel("Cum. THA, %")
fig.suptitle("Cumulative THA by recovered centre (blue) vs pooled (grey)", x=0.01, ha="left", color=INK)
fig.tight_layout(); save(fig, "E3_km_centre.png")

fig, axs = plt.subplots(1, 2, figsize=(9, 3))
ev = H[H.event == 1].t_event.value_counts().reindex(range(1, 11), fill_value=0)
axs[0].bar(ev.index, ev.values, color=SER[0], width=0.7); axs[0].set_title("Year of THA (n hips)", loc="left"); axs[0].set_xticks(range(1, 11))
cen = H[H.status10 == "censored<10y"].fu_end.value_counts().reindex(range(0, 10), fill_value=0)
axs[1].bar(cen.index, cen.values, color=MUTED, width=0.7); axs[1].set_title("Last follow-up year of censored hips", loc="left"); axs[1].set_xticks(range(0, 10))
for a in axs: a.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "E3_event_censor_years.png")

# ================================================================ E4 loss to follow-up (subject level, SMD)
S["completed10"] = S.fu_end >= 10
sub = H.drop_duplicates("nsin").set_index("nsin")
comp = []
for col, lab in [("age", "Age"), ("female", "Female"), ("bmi", "BMI"), ("education", "Education"), ("not_working", "Not working"),
                 ("comor_n", "Comorbidities"), ("womac_pain", "WOMAC pain"), ("womac_func", "WOMAC function"), ("sf36_mh", "SF-36 mental"),
                 ("care_no_care", "No care prior year"), ("smoking", None)]:
    if lab is None: continue
    a, b = sub.loc[S.completed10[sub.index], col], sub.loc[~S.completed10[sub.index], col]
    smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    comp.append({"Variable": lab, "Completed 10 y": f"{a.mean():.2f}", "Lost < 10 y": f"{b.mean():.2f}", "SMD": round(smd, 3)})
comp.append({"Variable": "Interval group", "Completed 10 y": f"{(S[S.completed10].group == 'interval').mean():.2f}",
             "Lost < 10 y": f"{(S[~S.completed10].group == 'interval').mean():.2f}", "SMD": np.nan})
E4 = pd.DataFrame(comp); E4.to_csv(OUT / "E4_dropout.csv", index=False, encoding="utf-8-sig")
log(f"\n== E4 completed 10 y: {int(S.completed10.sum())} / {len(S)} subjects\n" + E4.to_string(index=False))

# ================================================================ E5 missingness heatmap (feature x centre)
main_cols = FD[(FD.laterality != "contra")].col.tolist()
cols_ord = FD[FD.laterality != "contra"].sort_values(["group", "col"]).col.tolist()
Mc = H.groupby("centre")[cols_ord].apply(lambda d: d.isna().mean()).reindex(CENTRES + ["small", "unassigned"]).T
Mc["TIFF"] = H[H.img_fmt == "tif"][cols_ord].isna().mean(); Mc["DICOM"] = H[H.img_fmt == "dcm"][cols_ord].isna().mean()
keep = Mc.index[Mc.max(axis=1) > 0.02]          # only show variables with some missingness
M = Mc.loc[keep]
fig, ax = plt.subplots(figsize=(7.5, 0.16 * len(M) + 1.5))
im = ax.imshow(M.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
ax.set_yticks(range(len(M))); ax.set_yticklabels([f"{FD.set_index('col').group[c]} · {c}" for c in M.index], fontsize=7)
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=60, fontsize=7.5)
ax.axvline(11.5, color=SURF, lw=3)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Missing fraction"); cb.outline.set_visible(False)
ax.set_title("Missingness by centre / film format (variables with >2% missing anywhere)", loc="left")
save(fig, "E5_missing_centre.png"); Mc.round(3).to_csv(OUT / "E5_missing_centre.csv", encoding="utf-8-sig")
spread = (M[CENTRES].max(axis=1) - M[CENTRES].min(axis=1)).sort_values(ascending=False)
log("\n== E5 variables whose missingness differs most between centres (max-min)\n" + spread.head(15).round(2).to_string())

# ================================================================ E6 centre profile
prof = []
for c in CENTRES + ["small", "unassigned"]:
    d = H[H.centre == c]; s = d.drop_duplicates("nsin"); k = d[d.y10.notna()]
    prof.append({"centre": c, "subjects": len(s), "TIFF/DICOM": f"{(s.fmt_subject == 'tif').sum()}/{(s.fmt_subject == 'dcm').sum()}",
                 "THA hips": int(d.event.sum()), "KM10 %": round(km(d)[0][-1] * 100, 1), "10y-known event %": round(k.y10.mean() * 100, 1) if len(k) else np.nan,
                 "female %": round(s.female.mean() * 100), "age": round(s.age.mean(), 1), "interval %": round((s.group == "interval").mean() * 100),
                 "no care %": round(s.care_no_care.mean() * 100), "hip pain %": round(d.hip_pain.mean() * 100),
                 "KL≥2 %": round((d.r_kl >= 2).mean() * 100), "ER ROM missing %": round(d.er_rom.isna().mean() * 100),
                 "censored<10y %": round((d.status10 == "censored<10y").mean() * 100)})
E6 = pd.DataFrame(prof); E6.to_csv(OUT / "E6_centre_profile.csv", index=False, encoding="utf-8-sig")
log("\n== E6 centre profile\n" + E6.to_string(index=False))
# centre-outcome association: GEE logistic of y10 on centre dummies (cluster = subject), Wald test
kc = known[known.centre.isin(CENTRES)]
Xc = pd.get_dummies(kc.centre, drop_first=True, dtype=float); Xc.insert(0, "const", 1.0)
g = sm.GEE(kc.y10.values, Xc.values, groups=kc.nsin.values, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
R_ = np.eye(Xc.shape[1])[1:]; w = g.wald_test(R_, scalar=True)
log(f"centre -> y10 GEE Wald chi2={float(w.statistic):.2f} df={R_.shape[0]} p={float(w.pvalue):.3f} (cluster-robust)")

# ================================================================ E7 image QC
IMG["sig"] = np.where(IMG.fmt == "dcm", "DICOM " + IMG.BitsStored.astype(str) + "b", "TIFF " + IMG.BitsStored.astype(str) + "b " + IMG.Software.fillna("?").astype(str))
IMG["mpx"] = IMG.Rows * IMG.Columns / 1e6
IMG["centre"] = IMG.nsin.map(S.centre)
def spacing(r):
    for k in ("PixelSpacing", "ImagerPixelSpacing"):
        v = r.get(k)
        if isinstance(v, str) and v.strip(): return float(v.strip("[]").split(",")[0])
    return np.nan
IMG["spacing_mm"] = IMG.apply(spacing, axis=1)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for i, (f, d) in enumerate(IMG.groupby("fmt")):
    axs[0].hist(d.mpx, bins=40, color=SER[i], alpha=0.85, label=f"{f.upper()} (n={len(d)})")
axs[0].set_title("Image size (megapixels)", loc="left"); axs[0].legend(fontsize=7.5)
sig = IMG.sig.value_counts().head(8)[::-1]
axs[1].barh(range(len(sig)), sig.values, color=SER[0], height=0.65); axs[1].set_yticks(range(len(sig))); axs[1].set_yticklabels(sig.index, fontsize=7)
axs[1].set_title("Acquisition signature (top 8)", loc="left"); axs[1].grid(axis="y", visible=False)
d = IMG[IMG.fmt == "dcm"].spacing_mm.dropna()
axs[2].hist(d, bins=30, color=SER[0]); axs[2].set_title(f"DICOM pixel spacing, mm (n={len(d)}; TIFF has none)", loc="left")
fig.tight_layout(); save(fig, "E7_image_qc.png")
ct = pd.crosstab(IMG.centre.fillna("?"), IMG.sig); ct.to_csv(OUT / "E7_signature_by_centre.csv", encoding="utf-8-sig")
log("\n== E7 image signatures by centre\n" + ct.loc[:, ct.sum() > 5].to_string())
if MONTAGE:
    from render_lib import load
    picks = [IMG[(IMG.centre == c) & (IMG.view == "O")].sample(1, random_state=0) for c in CENTRES if ((IMG.centre == c) & (IMG.view == "O")).any()]
    picks = pd.concat(picks)
    fig, axs = plt.subplots(2, 5, figsize=(14, 6.4))
    for ax in axs.ravel(): ax.axis("off")
    for ax, r in zip(axs.ravel(), picks.itertuples()):
        a, _ = load(r.path); st = max(1, max(a.shape) // 700); a = a[::st, ::st]
        lo, hi = np.percentile(a, [1, 99]); ax.imshow(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
        ax.set_title(f"{r.centre} · {r.sig[:22]}", fontsize=8, color=INK)
    fig.suptitle("One baseline AP pelvis per centre (raw, 1-99% window): burned-in text, masks, calibration ball, field of view", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(); save(fig, "E7_montage_centre.png")

# ================================================================ E8 univariable associations (GEE, cluster = subject)
cand = FD[(FD.laterality != "contra") & FD.kind.isin(["cont", "ord", "bin"]) & FD.main].col.tolist()
cand += ["hip_pain_c", "r_kl_c", "ir_rom_c"]
res = []
for c in cand:
    d = known[["nsin", "y10", c]].dropna()
    if d[c].nunique() < 2 or d.y10.sum() < 5: continue
    if FD.set_index("col").kind[c] == "bin" and min(d[d[c] == 1].y10.sum(), d[d[c] == 0].y10.sum()) < 3: continue  # sparse cell
    x = d[c].astype(float); kind = FD.set_index("col").kind[c]
    if kind != "bin": x = (x - x.mean()) / x.std()
    try:
        m = sm.GEE(d.y10.values, sm.add_constant(x.values), groups=d.nsin.values, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()
        b, se = m.params[1], m.bse[1]
        res.append({"col": c, "group": FD.set_index("col").group[c], "kind": kind, "n": len(d), "events": int(d.y10.sum()),
                    "OR": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se), "p": m.pvalues[1],
                    "nl_lit": isinstance(FD.set_index("col").nl_lit[c], str)})
    except Exception as e:
        log("GEE failed", c, repr(e))
U = pd.DataFrame(res).sort_values(["group", "OR"]); U.to_csv(OUT / "E8_univariable.csv", index=False, encoding="utf-8-sig")
U = U[(U.hi / U.lo) < 60]  # drop degenerate (sparse) estimates from the plot only
fig, ax = plt.subplots(figsize=(6.5, 0.17 * len(U) + 1))
y = np.arange(len(U))
for i, r in enumerate(U.itertuples()):
    col = SER[0] if r.nl_lit else MUTED
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.4); ax.plot(r.OR, i, "o", ms=4.5, color=col, mec=SURF, mew=1)
ax.axvline(1, color=AXIS, lw=1); ax.set_xscale("log")
ax.set_yticks(y); ax.set_yticklabels([f"{r.group} · {r.col}{' (per SD)' if r.kind != 'bin' else ''}" for r in U.itertuples()], fontsize=6.5)
ax.set_xlabel("Odds ratio for THA ≤ 10 y (95% CI, GEE cluster-robust)"); ax.grid(axis="y", visible=False)
ax.plot([], [], "o", color=SER[0], label="reported in Dutch / CHECK literature"); ax.plot([], [], "o", color=MUTED, label="other candidate")
ax.legend(fontsize=7, loc="lower right"); ax.set_title("E8 Univariable associations (10-y known hips)", loc="left")
save(fig, "E8_forest.png")
log("\n== E8 top associations\n" + pd.DataFrame(res).sort_values("p").head(20)[["col", "OR", "lo", "hi", "p"]].round(3).to_string(index=False))

# ================================================================ E9 left-right dependence
both = known.pivot_table(index="nsin", columns="side", values="y10").dropna()
tab = pd.crosstab(both.L, both.R); log("\n== E9 left x right 10-y THA (subjects with both hips known)\n" + tab.to_string())
po = (both.L == both.R).mean(); pe = both.L.mean() * both.R.mean() + (1 - both.L.mean()) * (1 - both.R.mean())
g0 = sm.GEE(known.y10.values, np.ones((len(known), 1)), groups=known.nsin.values, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
log(f"kappa={(po - pe) / (1 - pe):.3f}  | P(THA other side | THA one side)={both[(both.L == 1) | (both.R == 1)].min(axis=1).mean():.2f}"
    f"  | GEE exchangeable within-subject corr={g0.cov_struct.dep_params:.3f}")
q = known[["hip_pain", "hip_pain_q"]].dropna()
log(f"agreement exam hip_pain vs questionnaire hip_pain_q: {(q.hip_pain == q.hip_pain_q).mean():.3f}")

# ================================================================ E10 leakage audit
E10 = FD[["col", "group", "kind", "laterality", "main", "note"]].copy()
E10["source"] = np.where(E10.group == "R", "Rontgen_opT10.dta (T0 scores, read paired with T2-T10 films)", "CHECK_T0_DANS_ENG_20161128.tab")
E10["wave"] = "T0"; E10["leakage_risk"] = np.where(E10.group == "R", "HIGH (B4)", np.where(~E10.main, "centre proxy / sensitivity only", "low"))
extra = pd.DataFrame([
    {"col": "T0_JVCON (visit group)", "group": "-", "source": "T0 tab", "wave": "T0", "leakage_risk": "severity proxy assigned by investigators (B3) - stratifier only"},
    {"col": "centre", "group": "-", "source": "02_centres.py (T0-T10 DICOM headers)", "wave": "T0-T10", "leakage_risk": "uses follow-up headers - split variable only"},
    {"col": "img_fmt", "group": "-", "source": "images.csv", "wave": "T0", "leakage_risk": "acquisition proxy - split variable only"},
    {"col": "T*_f01_orth (follow-up orthopaedic visits)", "group": "-", "source": "T1-T10 dta", "wave": "T1-T10", "leakage_risk": "mediator on the path to surgery - excluded"},
    {"col": "T0_dyspl", "group": "-", "source": "T0 tab", "wave": "T0", "leakage_risk": "constant (exclusion criterion) - excluded"},
    {"col": "T0_RAS", "group": "-", "source": "T0 tab", "wave": "T0", "leakage_risk": "97% white, descriptive only"}])
pd.concat([E10, extra]).to_csv(OUT / "E10_leakage_audit.csv", index=False, encoding="utf-8-sig")

# ================================================================ COLLINEARITY  (C1 exam, C2 Dutch-literature set, C3 all clinical tabular)
def spearman(cols, df=H):
    return df[cols].astype(float).corr(method="spearman", min_periods=100)
def cluster_order(Cm):
    Dm = 1 - Cm.abs().fillna(0).values; np.fill_diagonal(Dm, 0); Dm = (Dm + Dm.T) / 2
    return Cm.index[leaves_list(linkage(squareform(Dm, checks=False), "average"))]
def heat(Cm, name, title, annot=False, size=None):
    o = cluster_order(Cm); Cm = Cm.loc[o, o]; n = len(Cm)
    fig, ax = plt.subplots(figsize=size or (0.24 * n + 2.5, 0.24 * n + 1.8))
    im = ax.imshow(Cm.values, cmap=DIV, vmin=-1, vmax=1); ax.grid(False)
    lab = [f"{FD.set_index('col').group.get(c, '?')} · {c}" for c in Cm.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=90, fontsize=6.5); ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=6.5)
    if annot:
        for i in range(n):
            for j in range(n):
                v = Cm.values[i, j]
                if i != j and abs(v) >= 0.3: ax.text(j, i, f"{v:.1f}".replace("0.", "."), ha="center", va="center", fontsize=5.5,
                                                     color="#ffffff" if abs(v) > 0.65 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Spearman ρ (pairwise complete)"); cb.outline.set_visible(False)
    ax.set_title(title, loc="left"); save(fig, name)
def vif_table(cols, df=H):
    X = df[cols].astype(float).dropna(); X = X.loc[:, X.std() > 0]; X = (X - X.mean()) / X.std()
    Xc = sm.add_constant(X.values)
    v = pd.DataFrame({"col": X.columns, "VIF": [variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])]})
    cond = np.linalg.cond(X.values)
    return v.sort_values("VIF", ascending=False), len(X), cond
def high_pairs(Cm, thr):
    out = []
    for i, a in enumerate(Cm.index):
        for b in Cm.index[i + 1:]:
            if abs(Cm.loc[a, b]) >= thr: out.append((a, b, round(Cm.loc[a, b], 3)))
    return pd.DataFrame(out, columns=["a", "b", "rho"]).sort_values("rho", key=abs, ascending=False)

# C1 all physical-examination variables (ipsilateral + contralateral, main + sensitivity)
c1 = FD[FD.group == "E"].col.tolist()
C1 = spearman(c1); heat(C1, "C1_exam_corr.png", "C1 Physical examination: all items, ipsi + contra (clustered)")
v1, n1, k1 = vif_table(FD[(FD.group == "E") & FD.main].col.tolist())
v1.to_csv(OUT / "C1_exam_vif.csv", index=False)
log(f"\n== C1 exam (main set) VIF on {n1} complete hips, condition number {k1:.1f}\n" + v1.head(12).round(2).to_string(index=False))
log("C1 |rho|>=0.6 pairs\n" + high_pairs(C1, 0.6).head(25).to_string(index=False))

# C2 factors reported in Dutch-system / CHECK literature (ipsilateral) -> heatmap with values + VIF
c2 = FD[FD.nl_lit.notna() & (FD.laterality != "contra") & (FD.kind != "nom")].col.tolist()
FD[FD.col.isin(c2)][["col", "group", "nl_lit"]].to_csv(OUT / "C2_dutch_lit_features.csv", index=False, encoding="utf-8-sig")
C2 = spearman(c2); heat(C2, "C2_dutch_lit_corr.png", "C2 Predictors reported in Dutch / CHECK literature", annot=True)
v2, n2, k2 = vif_table([c for c in c2 if c != "r_kl"]); v2r, _, k2r = vif_table(c2)
pd.concat([v2.assign(set="without R"), v2r.assign(set="with r_kl")]).to_csv(OUT / "C2_dutch_lit_vif.csv", index=False)
log(f"\n== C2 Dutch-literature set: {len(c2)} vars; VIF (no KL) on {n2} complete hips, cond {k2:.1f} | with KL cond {k2r:.1f}\n" + v2.round(2).to_string(index=False))

# C3 all main clinical tabular features (ipsi + subject-level), every group
c3 = FD[FD.main & FD.kind.isin(["cont", "ord", "bin"]) & (FD.laterality != "contra")].col.tolist()
C3 = spearman(c3); heat(C3, "C3_all_clinical_corr.png", "C3 All main tabular features (ipsilateral + subject level)")
hp = high_pairs(C3, 0.7); hp.to_csv(OUT / "C3_high_pairs.csv", index=False)
v3, n3, k3 = vif_table([c for c in c3 if FD.set_index("col").group[c] != "R" and H[c].isna().mean() < 0.3])
v3.to_csv(OUT / "C3_vif.csv", index=False)
log(f"\n== C3 {len(c3)} vars; |rho|>=0.7 pairs: {len(hp)}\n" + hp.head(30).to_string(index=False))
log(f"C3 VIF (non-R, <30% missing) on {n3} complete hips, condition number {k3:.1f}\n" + v3.head(20).round(2).to_string(index=False))
# ================================================================ E5b missingness mechanism audit (raw T0, subject level)
T0 = pd.read_csv(P / "data/check_raw/T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False).set_index("nsin")
gates = {"no hip pain": T0.T0_JVHPIJN == 2, "no knee pain": T0.T0_JVKPIJN == 2, "male": T0.T0_SEXE == 1,
         "not in paid work": T0.T0_ZW01 != 1, "no alcohol": T0.T0_J02a == 2, "diet unchanged": T0.T0_J03 == 2}
cen_s = S.centre.reindex(T0.index)
aud = []
for c in T0.columns:
    m = T0[c].isna()
    if not (0.02 <= m.mean() <= 0.995): continue
    r = {"var": c, "missing": m.mean()}
    for g, s in gates.items(): r[f"Δ {g}"] = m[s].mean() - m[~s & T0.index.isin(s.index)].mean()
    cm = m.groupby(cen_s).mean().reindex(CENTRES); r["centre range"] = cm.max() - cm.min()
    gmax = max(abs(r[f"Δ {g}"]) for g in gates)
    r["type"] = "centre protocol" if r["centre range"] > 0.5 else "structural (gated)" if gmax > 0.3 else "random (MCAR/MAR)"
    aud.append(r)
AUD = pd.DataFrame(aud).sort_values(["type", "missing"], ascending=[True, False]); AUD.round(3).to_csv(OUT / "E5b_missing_mechanism.csv", index=False)
log("\n== E5b missingness mechanism: " + AUD.type.value_counts().to_dict().__repr__())
# figure: one row per variable block (collapse item batteries to their first item) x gate contrasts
show = AUD[~AUD["var"].str.match(r"T0_(zw05[b-p]|J04[b-m]|I09[b-i]|I13[b-j]|wm\w+s$|PCI(?!pijn)|Sos[fik]|soscrec|sP|sC(?!OMP)|sOC|sHA|uNTX|CTXI(?!I)|Lept|Adipo|Resis|HEB|BOU|r[a-z]+$|mobility|selfcare|daily|pain$|mood|BEN|MVH)")]
cols = [f"Δ {g}" for g in gates] + ["centre range"]
fig, ax = plt.subplots(figsize=(7.8, 0.17 * len(show) + 1.6))
im = ax.imshow(show[cols].values, cmap=DIV, vmin=-1, vmax=1, aspect="auto"); ax.grid(False)
ax.set_yticks(range(len(show))); ax.set_yticklabels([f"{r['var']}  ({r['missing']:.0%})" for _, r in show.iterrows()], fontsize=6.5)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=7.5)
for i, t in enumerate(show.type):
    ax.text(len(cols) - 0.4, i, t, fontsize=6.5, va="center", color=INK2)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.25); cb.set_label("Missing-rate difference (gate − rest) / range across centres"); cb.outline.set_visible(False)
ax.set_title("E5b Missingness mechanism: gated by complaint/status, by centre, or random", loc="left")
save(fig, "E5b_missing_mechanism.png")

# ================================================================ E11 distributions, skewness and the transform rule
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
tcols = FD[(FD.laterality != "contra") & (FD.kind == "cont") & (FD.group != "R")]
rows = []
for r in tcols.itertuples():
    v = H[r.col].dropna().astype(float)
    if len(v) < 50: continue
    if r.transform == "log1p": w = np.log1p(v.clip(lower=0))
    elif r.transform == "yeojohnson": w = pd.Series(PowerTransformer("yeo-johnson", standardize=False).fit_transform(v.values[:, None])[:, 0])
    else: w = v
    rows.append({"col": r.col, "group": r.group, "transform": r.transform, "skew_before": skew(v), "skew_after": skew(w), "main": r.main})
SK = pd.DataFrame(rows).sort_values("skew_before", key=abs); SK.round(3).to_csv(OUT / "E11_skewness.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(12, 0.2 * len(SK) + 1.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]
for i, r in enumerate(SK.itertuples()):
    ax.plot([r.skew_before, r.skew_after], [i, i], color=GRID, lw=1.5, zorder=1)
    ax.plot(r.skew_before, i, "o", color=MUTED, ms=5, zorder=2)
    ax.plot(r.skew_after, i, "o", color=SER[0] if r.transform != "none" else MUTED, ms=5, mec=SURF, zorder=3)
ax.axvline(0, color=AXIS, lw=1); ax.axvspan(-1, 1, color=GRID, alpha=0.35, lw=0)
ax.set_yticks(range(len(SK))); ax.set_yticklabels([f"{r.group} · {r.col}  [{r.transform}]" for r in SK.itertuples()], fontsize=6.5)
ax.set_xlabel("Skewness (grey = raw, blue = after fold-fitted transform)"); ax.grid(axis="y", visible=False)
ax.set_title("E11 Skewness before / after transform (|skew| ≤ 1 band shaded)", loc="left")
top = SK[SK["transform"] != "none"].sort_values("skew_before", key=abs, ascending=False).head(4)
axs[1].axis("off")
for i, r in enumerate(top.itertuples()):
    v = H[r.col].dropna().astype(float)
    w = np.log1p(v.clip(lower=0)) if r.transform == "log1p" else PowerTransformer("yeo-johnson", standardize=False).fit_transform(v.values[:, None])[:, 0]
    for j, (vals, lab, col) in enumerate([(v, "raw", MUTED), (w, r.transform, SER[0])]):
        a = axs[1].inset_axes([0.05 + 0.5 * j, 1 - (i + 1) * 0.245, 0.42, 0.19])
        a.hist(vals, bins=40, color=col); a.set_title(f"{r.col} · {lab}", fontsize=7, loc="left"); a.tick_params(labelsize=6); a.grid(False)
save(fig, "E11_skewness.png")
log("\n== E11 skewness (|skew|>1 raw): " + str(int((SK.skew_before.abs() > 1).sum())) + " -> after: " + str(int((SK.skew_after.abs() > 1).sum())))
LOG.close()
