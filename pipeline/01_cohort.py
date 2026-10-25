"""Build the analysis cohort: one row per subject and one row per hip, plus a feature dictionary.

usage: python -I 01_cohort.py <check_raw dir> <derived dir>
reads : T0T5/CHECK_T0_DANS_ENG_20161128.tab, sav/T1..T10.dta, sav/Rontgen_opT10.dta,
        <derived>/centres.csv (02_centres.py), <derived>/images.csv (eda.py)
writes: <derived>/subjects.parquet, hips.parquet, features.csv

Rules (see plan 6.6 / 7.3 / 7.4):
- outcome = T10 THA flag, or T10 _VISIT, or KL=5 scored before definite dropout (same rule as eda.py);
  event year = _VISIT, else first flag, else first KL=5 wave.
- y10 = 1 event within 10 y; 0 event-free and followed to year 10; NaN censored < 10 y (excluded from the main analysis).
- inputs are T0 only; JVCON (visiting group), centre and image format are kept as metadata, never as features.
- each hip gets ipsilateral (no suffix) and contralateral (_c) versions of side-specific variables.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd, pyreadstat

warnings.filterwarnings("ignore")
raw, out = map(pathlib.Path, sys.argv[1:3])
SIDES = {"L": "li", "R": "re"}
OTHER = {"L": "R", "R": "L"}
VISITS_INTERVAL = [0, 2, 5, 8, 10]

t0 = pd.read_csv(raw / "T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False).set_index("nsin")
rs = pyreadstat.read_dta(raw / "sav/Rontgen_opT10.dta")[0].set_index("nsin")
W = {k: pyreadstat.read_dta(raw / f"sav/T{k}.dta")[0].set_index("nsin") for k in range(1, 11)}
for df in [rs, *W.values()]: df.index = df.index.astype(int)
centres = pd.read_csv(out / "centres.csv").set_index("nsin")
img = pd.read_csv(out / "images.csv")
INTERVAL_CODE = 1  # T0_JVCON: 1 = variable (interval) group, 2 = annual group

# ---------------------------------------------------------------- outcome
def fu_end(n):
    dd = W[10].loc[n, "T10_dropout_def_visit"]
    last = 10 if pd.isna(dd) else int(dd) - 1
    if last == 10 and W[10].loc[n, "T10_dropout_temp"] == 1: last = 9
    if t0.loc[n, "T0_JVCON"] == INTERVAL_CODE: last = max(v for v in VISITS_INTERVAL if v <= last)
    return last

rows = []
for n in t0.index:
    end = fu_end(n)
    for S, s in SIDES.items():
        flags = [k for k in range(1, 11) if W[k].loc[n, f"T{k}_THA{S}"] == 1]
        visit = W[10].loc[n, f"T10_THA{S}_VISIT"]
        flag10 = W[10].loc[n, f"T10_THA{S}"] == 1
        kl5 = [t for t in ["T0", "T2", "T5", "T8", "T10"] if rs.loc[n, f"{t}_H_KL_{s}_def"] == 5 and int(t[1:]) <= end]
        ev = flag10 or pd.notna(visit) or bool(kl5)
        t_ev = (visit if pd.notna(visit) else min(flags) if flag10 else int(kl5[0][1:])) if ev else np.nan
        rows.append({"nsin": n, "side": S, "event": int(ev), "t_event": t_ev, "fu_end": end})
H = pd.DataFrame(rows)
H["time"] = np.where(H.event == 1, H.t_event, H.fu_end)
H["status10"] = np.select([H.event == 1, H.fu_end >= 10], ["event", "event-free@10y"], "censored<10y")
H["y10"] = np.where(H.status10 == "censored<10y", np.nan, H.event)

# ---------------------------------------------------------------- metadata (never model inputs)
H["group"] = H.nsin.map(lambda n: "interval" if t0.loc[n, "T0_JVCON"] == INTERVAL_CODE else "annual")
H["centre"] = H.nsin.map(centres.centre).fillna("unassigned")
H.loc[H.centre.str.startswith("small"), "centre"] = "small"
views = img.groupby("nsin").view.agg(lambda x: set(x))
fmts = img.groupby(["nsin", "view"]).fmt.first()
def hip_img(r):
    v = views.get(r.nsin, set())
    for view in ("O", r.side):  # whole-pelvis film preferred, else the single-hip film
        if view in v: return pd.Series({"has_img": True, "img_view": view, "img_fmt": fmts[(r.nsin, view)]})
    return pd.Series({"has_img": False, "img_view": None, "img_fmt": None})
H = pd.concat([H, H.apply(hip_img, axis=1)], axis=1)
H["fmt_subject"] = H.groupby("nsin").img_fmt.transform(lambda x: x.dropna().iloc[0] if x.notna().any() else "none")

# ---------------------------------------------------------------- feature construction
yes12 = lambda s: s.map({1: 1.0, 2: 0.0})           # 1 = yes / present, 2 = no / absent
def side_col(base, S):                              # exam columns end with L/R
    return t0[f"T0_{base}{S}"]

F, META = {}, []   # F[name] -> (L-series, R-series) or subject-level series
def add(name, group, kind, L, R=None, main=True, nl_lit="", note=""):
    """Register a feature. If R is given the feature is side-specific -> ipsilateral + contralateral copies."""
    F[name] = (L, R)
    META.append(dict(feature=name, group=group, kind=kind, side_specific=R is not None, main=main, nl_lit=nl_lit, note=note))

# D  demographics / biology
add("age", "D", "cont", t0.T0_Lft_T0, nl_lit="Zarringam 2020; Gademan 2018")
add("female", "D", "bin", (t0.T0_SEXE == 2).astype(float), nl_lit="Zarringam 2020 (male HR 2.1)")
add("bmi", "D", "cont", t0.T0_BMI, nl_lit="Zarringam 2020 (n.s. hip); Teirlinck 2019 (no assoc)")
add("comor_n", "D", "ord", t0.T0_comor, nl_lit="Gademan 2018")
add("comor_back", "D", "bin", t0.T0_V18O, note="back disorder - competing source of hip-region pain")
add("comor_inflam", "D", "bin", t0.T0_V18Q)
add("comor_diabetes", "D", "bin", t0.T0_V18M)
add("comor_cardiac", "D", "bin", t0.T0_V18C, note="surgical fitness proxy (ASA not available)")
menop = pd.Series(np.where(t0.T0_SEXE == 1, "male", t0.T0_Menopauze_01.map({0: "pre", 1: "post"})), index=t0.index).replace("nan", np.nan)
add("menopause", "D", "nom", menop, note="structural: male -> own category (7.4)")

# E  physical examination (side-specific unless noted)
for base, nm in [("HENDO", "ir"), ("HFLEX", "flex"), ("HEXO", "er"), ("HADD", "add"), ("HABD", "abd")]:
    main = nm in ("ir", "flex")  # er/add/abd ROM missingness == centre protocol (7.1) -> sensitivity only
    add(f"{nm}_rom", "E", "cont", side_col(base, "L"), side_col(base, "R"), main=main,
        nl_lit="Agricola 2013/2024; Bastick 2017" if nm == "ir" else "", note="" if main else "missing = centre (7.1)")
    add(f"{nm}_pain", "E", "ord", side_col(base + "PIJN", "L"), side_col(base + "PIJN", "R"), main=main,
        nl_lit="Bastick 2017 (IR pain)" if nm == "ir" else "", note="" if main else "missing = centre (audit: 0-100% by centre)")
    add(f"{nm}_start", "E", "cont", side_col(base + "STRT", "L"), side_col(base + "STRT", "R"), main=False,
        note="starting position of the goniometer, mostly 0")
add("ir_le25", "E", "bin", (side_col("HENDO", "L") <= 25).astype(float).where(side_col("HENDO", "L").notna()),
    (side_col("HENDO", "R") <= 25).astype(float).where(side_col("HENDO", "R").notna()), nl_lit="Agricola 2024 FAIS cut-off")
add("flex_le115", "E", "bin", (side_col("HFLEX", "L") <= 115).astype(float).where(side_col("HFLEX", "L").notna()),
    (side_col("HFLEX", "R") <= 115).astype(float).where(side_col("HFLEX", "R").notna()), note="ACR clinical criterion")
add("contracture", "E", "bin", yes12(side_col("HCONTRAC", "L")), yes12(side_col("HCONTRAC", "R")))
for base, nm in [("TROCH", "troch_bursitis"), ("ADDUC", "adductor_tend"), ("ILIO", "iliopect_bursitis")]:
    absent = t0[f"T0_{base}"] == 2  # global 'present?' = no -> side = 0 (structural skip)
    add(nm, "E", "bin", side_col(base, "L").mask(absent, 0.0), side_col(base, "R").mask(absent, 0.0), main=False,
        note="differential diagnosis; missingness ~centre (E5) -> sensitivity only")
add("heberden_n", "E", "cont", t0[[f"T0_HEB{d}{s}" for d in "2345" for s in "LR"]].sum(axis=1, min_count=1), note="generalised OA")
add("bouchard_n", "E", "cont", t0[[f"T0_BOU{d}{s}" for d in "2345" for s in "LR"]].sum(axis=1, min_count=1))

# S  symptoms / patient-reported
add("hip_pain", "S", "bin", t0.T0_JVHPIJNL.astype(float), t0.T0_JVHPIJNR.astype(float), nl_lit="Jang 2023; van Berkel 2021")
add("hip_pain_q", "S", "bin", t0.T0_D01a, t0.T0_d01b, note="questionnaire version of hip pain")
add("knee_pain", "S", "bin", t0.T0_JVKPIJNL.astype(float), t0.T0_JVKPIJNR.astype(float), note="same-side knee")
add("n_pain_sites", "S", "ord", t0[["T0_D01a", "T0_d01b", "T0_d01c", "T0_d01d", "T0_d01e"]].sum(axis=1, min_count=1),
    nl_lit="Gademan 2018 (number of painful joints)")
# asked only of participants with hip complaints (missing 9% without hip pain vs 0% with) -> structural: no hip pain = no stiffness
add("hip_stiff", "S", "bin", yes12(t0.T0_JVHSTIJF).mask(t0.T0_JVHSTIJF.isna() & (t0.T0_JVHPIJN == 2), 0.0),
    note="ACR morning stiffness; structural skip: no hip pain -> 0")
add("nrs_now", "S", "ord", t0.T0_D02)
add("nrs_week", "S", "ord", t0.T0_D03)
add("womac_pain", "S", "cont", t0.T0_wmpyn, nl_lit="Zarringam 2020; Runhaar 2021; Gademan 2018")
add("womac_stiff", "S", "cont", t0.T0_wmstf)
add("womac_func", "S", "cont", t0.T0_wmfun, nl_lit="Gademan 2016 indication criteria (function)")
add("womac_total", "S", "cont", t0.T0_wmtot, main=False, note="= pain + stiff + func (exact linear combination)")
for c, nm in [("T0_rfys", "sf36_pf"), ("T0_rpijn", "sf36_bp"), ("T0_rment", "sf36_mh"), ("T0_rvit", "sf36_vit"),
              ("T0_rrolfy", "sf36_rp"), ("T0_rsoc", "sf36_sf")]:
    add(nm, "S", "cont", t0[c])
add("vas_health", "S", "cont", t0.T0_b20)
for c, nm in [("T0_PCIpijn", "pci_transform"), ("T0_PCIaflei", "pci_distract"), ("T0_PCIeisen", "pci_reduce_demand"),
              ("T0_PCIterug", "pci_retreat"), ("T0_PCIpiekr", "pci_worry"), ("T0_PCIrust", "pci_rest")]:
    add(nm, "S", "cont", t0[c], note="pain coping inventory")
add("sos_mean", "S", "cont", t0.T0_Sosgem, note="SOS (activity avoidance) mean")

# C  care use in the year before baseline + medication (Dutch GP gatekeeping)
for c, nm in [("T0_F01_ha", "gp"), ("T0_f01_fys", "physio"), ("T0_f01_oeft", "exercise_ther"), ("T0_f01_orth", "orthopaedist"),
              ("T0_f01_reu", "rheumatologist"), ("T0_f01_rev", "rehab_doctor"), ("T0_f01_geen", "no_care")]:
    add(f"care_{nm}", "C", "bin", t0[c], nl_lit="Hofstede 2015; Oomen 2024 (GP-led OA care)" if nm in ("gp", "physio") else "")
for c, nm in [("T0_parac", "paracetamol"), ("T0_ibu", "nsaid"), ("T0_asp", "aspirin"), ("T0_anders", "other_drug"), ("T0_geen", "no_drug")]:
    add(f"med_{nm}", "C", "bin", t0[c], nl_lit="van Brug 2023 (pre-op analgesics); Jang 2023" if nm in ("nsaid", "other_drug") else "")
add("glucosamine", "C", "bin", t0.T0_J06D)

# L  social / occupational / lifestyle
add("education", "L", "ord", t0.T0_a08, nl_lit="Bonsel 2024; Hoelen 2024 (SES)")
add("married", "L", "bin", (t0.T0_A03 == 2).astype(float).where(t0.T0_A03.notna()))
emp = t0.T0_ZW01.map({1: "paid", 2: "unemployed", 3: "disabled", 4: "other", 5: "retired", 6: "homemaker", 7: "other"})
add("employment", "L", "nom", emp)
working = t0.T0_ZW01 == 1
# ZW02/ZW03 codes contradict their value labels (94% of paid workers coded "2 = sick leave due to hip/knee";
# employed coded ZW02 "1 = NO"), found by the C3 collinearity check (rho -0.96 with not_working) -> excluded from main
add("sick_leave_hk", "L", "bin", (t0.T0_ZW03 == 2).astype(float).where(t0.T0_ZW03.notna()).mask(~working & t0.T0_ZW01.notna(), 0.0),
    main=False, note="EXCLUDED: ZW03 coding contradicts labels (03_eda C3)")
heavy = t0[["T0_zw05f", "T0_zw05g", "T0_zw05h", "T0_zw05i"]].mean(axis=1)  # >5 kg, >25 kg, max exertion, heavy work
add("phys_load", "L", "cont", heavy.where(working), note="structural: non-workers NaN + not_working flag (7.4)")
add("not_working", "L", "bin", (~working).astype(float).where(t0.T0_ZW01.notna()))
smk = t0.T0_J01.map({1: "current", 2: "current", 3: "former", 4: "former", 5: "never"})
add("smoking", "L", "nom", smk, nl_lit="Peters 2020; Hoelen 2024 (case-mix)")
add("alcohol", "L", "bin", yes12(t0.T0_J02a))
add("activity_freq", "L", "ord", t0.T0_I14, nl_lit="Latijnhouwers 2024 (activity recommendation)")
add("heavy_garden", "L", "ord", t0.T0_I13h)

# B  biochemistry
for c, nm in [("T0_BSE", "esr"), ("T0_hsCRP", "hscrp"), ("T0_CTXIIngmmol", "uctx2"), ("T0_sCOMPUl", "scomp"),
              ("T0_sC12C", "sc12c"), ("T0_sCS846", "scs846")]:
    add(nm, "B", "cont", t0[c], note="ESR is an ACR criterion" if nm == "esr" else "")

# R  radiographic scores (leakage risk B4) - ipsilateral + contralateral
R = lambda c: rs[c].where(rs[c] < 9) if c in rs else pd.Series(np.nan, index=rs.index)
for tmpl, nm in [("T0_H_KL_{s}_def", "kl"), ("T0_H_AP_JSN_{s}_med", "jsn_med"), ("T0_H_AP_JSN_{s}_sup", "jsn_sup"),
                 ("T0_H_FP_JSN_{s}_pos", "fp_jsn_pos"), ("T0_H_FP_JSN_{s}_sup", "fp_jsn_sup"),
                 ("T0_H_OSTI_{s}_acet", "ost_inf_acet"), ("T0_H_OSTI_{s}_fem", "ost_inf_fem"),
                 ("T0_H_OSTS_{s}_acet", "ost_sup_acet"), ("T0_H_OSTS_{s}_fem", "ost_sup_fem"),
                 ("T0_H_{s}_SCL", "sclerosis"), ("T0_H_{s}_CYS", "cysts"), ("T0_H_{s}_FLA", "flattening"), ("T0_H_{s}_BUT", "buttressing")]:
    add(f"r_{nm}", "R", "ord", R(tmpl.format(s="li")).reindex(t0.index), R(tmpl.format(s="re")).reindex(t0.index),
        nl_lit="Gademan 2018; Reijman 2005; Macri 2022" if nm == "kl" else "", note="paired-read with follow-up (B4)")
add("r_knee_kl_max", "R", "ord", pd.concat([R("T0_K_KL_li_def"), R("T0_K_KL_re_def")], axis=1).max(axis=1).reindex(t0.index),
    nl_lit="Latijnhouwers 2024 (multiple joint arthroplasty)")

# ---------------------------------------------------------------- assemble hip table
feat = {}
for name, (L, R_) in F.items():
    if R_ is None:
        feat[name] = H.nsin.map(L).values
    else:
        ipsi = np.where(H.side == "L", H.nsin.map(L), H.nsin.map(R_))
        contra = np.where(H.side == "L", H.nsin.map(R_), H.nsin.map(L))
        feat[name], feat[name + "_c"] = ipsi, contra
X = pd.DataFrame(feat, index=H.index)
X["side_right"] = (H.side == "R").astype(float)
nominal = {m["feature"] for m in META if m["kind"] == "nom"}
for c in X.columns:
    X[c] = X[c].astype("string") if c.removesuffix("_c") in nominal else pd.to_numeric(X[c], errors="coerce").astype(float)
hips = pd.concat([H, X], axis=1)

meta = []
for m in META:
    meta.append(m | {"col": m["feature"], "laterality": "ipsi" if m["side_specific"] else "subject"})
    if m["side_specific"]: meta.append(m | {"col": m["feature"] + "_c", "laterality": "contra"})
meta.append(dict(feature="side_right", col="side_right", group="D", kind="bin", side_specific=False, main=True, nl_lit="", note="", laterality="hip"))
fd = pd.DataFrame(meta)[["col", "feature", "group", "kind", "laterality", "main", "nl_lit", "note"]]
fd["missing_rate"] = fd.col.map(hips[fd.col].isna().mean()).round(3)

# missingness type (audit in 03_eda E5b): structural gates are resolved above (assigned / own category), centre-protocol
# items are excluded from the main set, the rest is treated as random (MCAR/MAR) -> fold-wise MICE + indicator
centre_cols = {"er", "add", "abd"}
def miss_type(r):
    if r.missing_rate == 0: return "none"
    base = r.feature
    if (base.split("_")[0] in centre_cols and base.endswith(("_rom", "_pain", "_start"))) or base.endswith("_start") \
            or base in ("troch_bursitis", "adductor_tend", "iliopect_bursitis"): return "centre"
    if base in ("menopause", "phys_load", "not_working", "sick_leave_hk"): return "structural"
    return "random"
fd["missing_type"] = fd.apply(miss_type, axis=1)

# distribution: right-skewed biomarkers -> log1p, other skewed continuous -> Yeo-Johnson (lambda fitted in the training fold)
from scipy.stats import skew
def transform(r):
    if r.kind != "cont" or r.group == "R": return "none"
    sk = skew(hips[r.col].dropna()) if hips[r.col].notna().sum() > 20 else 0
    if abs(sk) <= 1: return "none"
    return "log1p" if r.group == "B" else "yeojohnson"
fd["transform"] = fd.apply(transform, axis=1)

subj = t0.copy()
subj["group"] = np.where(subj.T0_JVCON == INTERVAL_CODE, "interval", "annual")
subj["centre"] = hips.groupby("nsin").centre.first()
subj["fmt_subject"] = hips.groupby("nsin").fmt_subject.first()
subj["n_events"] = hips.groupby("nsin").event.sum()
subj["fu_end"] = hips.groupby("nsin").fu_end.first()
subj.reset_index().to_parquet(out / "subjects.parquet", index=False)
hips.to_parquet(out / "hips.parquet", index=False)
fd.to_csv(out / "features.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- report
print("subjects", len(subj), "| hips", len(hips), "| events", int(hips.event.sum()), "in", hips[hips.event == 1].nsin.nunique(), "subjects")
print("status10", hips.status10.value_counts().to_dict(), "| y10 known", int(hips.y10.notna().sum()), "events", int(hips.y10.sum()))
print("features", len(fd), "| by group", fd.groupby("group").size().to_dict(), "| main", int(fd.main.sum()))
print("centre", hips.drop_duplicates("nsin").centre.value_counts().to_dict())
print("img", hips.img_fmt.value_counts(dropna=False).to_dict())
print(fd.sort_values("missing_rate", ascending=False).head(15)[["col", "group", "missing_rate", "main"]].to_string(index=False))
