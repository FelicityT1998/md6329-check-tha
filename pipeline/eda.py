"""Preliminary data analysis for the CHECK hip-replacement project.

usage: python -I eda.py <check_raw dir> <T00_AP image dir> <out dir>
Writes hips.csv (one row per hip), images.csv, dicom_headers.csv, missing_by_group.csv and prints the summary.
"""
import sys, pathlib, re, warnings
import numpy as np, pandas as pd, pyreadstat, pydicom, tifffile

warnings.filterwarnings("ignore")
raw, imgdir, out = map(pathlib.Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
SIDES = {"L": "li", "R": "re"}
VISITS_INTERVAL = [0, 2, 5, 8, 10]

# ---------- tables ----------
t0 = pd.read_csv(raw / "T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False).set_index("nsin")
rs = pyreadstat.read_dta(raw / "sav/Rontgen_opT10.dta")[0].set_index("nsin")
W = {k: pyreadstat.read_dta(raw / f"sav/T{k}.dta")[0].set_index("nsin") for k in range(1, 11)}
for df in [rs, *W.values()]: df.index = df.index.astype(int)  # .dta stores nsin as string
annual = t0["T0_JVCON"]  # coding checked below
print("JVCON values:", t0["T0_JVCON"].value_counts().to_dict())

# ---------- images ----------
rows = []
for p in sorted(imgdir.iterdir()):
    m = re.match(r"(\d+)_T00_AP([OLR])\.(tif|dcm)$", p.name)
    if m: rows.append({"nsin": int(m[1]), "view": m[2], "fmt": m[3], "path": str(p), "mb": p.stat().st_size / 1e6})
img = pd.DataFrame(rows)
hdr = []
for r in img.itertuples():
    h = {"path": r.path}
    try:
        if r.fmt == "dcm":
            ds = pydicom.dcmread(r.path, stop_before_pixels=True, force=True)
            for k in ["InstitutionName", "Manufacturer", "ManufacturerModelName", "Modality", "StudyDate", "Rows", "Columns",
                      "BitsStored", "PixelSpacing", "ImagerPixelSpacing", "PhotometricInterpretation", "StationName", "BodyPartExamined"]:
                v = getattr(ds, k, None); h[k] = str(v) if v is not None else None
        else:
            with tifffile.TiffFile(r.path) as tf:
                pg = tf.pages[0]
                h.update(Rows=pg.shape[0], Columns=pg.shape[1], BitsStored=pg.bitspersample, Photometric=str(pg.photometric),
                         XRes=str(pg.tags.get("XResolution").value) if pg.tags.get("XResolution") else None,
                         Software=pg.tags.get("Software").value if pg.tags.get("Software") else None)
    except Exception as e:
        h["error"] = repr(e)
    hdr.append(h)
hdr = pd.DataFrame(hdr)
img = img.merge(hdr, on="path")
img.to_csv(out / "images.csv", index=False)

# ---------- hip-level outcome ----------
def fu_end(nsin):
    """Last year the subject was seen (0-10) given definite-dropout visit and visiting group."""
    dd = W[10].loc[nsin, "T10_dropout_def_visit"]
    temp10 = W[10].loc[nsin, "T10_dropout_temp"] == 1
    last = 10 if pd.isna(dd) else int(dd) - 1
    if last == 10 and temp10: last = 9
    if t0.loc[nsin, "T0_JVCON"] == INTERVAL_CODE:
        last = max(v for v in VISITS_INTERVAL if v <= last)
    return last

# visiting-group code: the smaller group is the interval group
INTERVAL_CODE = t0["T0_JVCON"].value_counts().idxmin()

hips = []
for nsin in t0.index:
    end = fu_end(nsin)
    for S, s in SIDES.items():
        flags = [k for k in range(1, 11) if W[k].loc[nsin, f"T{k}_THA{S}"] == 1]
        visit = W[10].loc[nsin, f"T10_THA{S}_VISIT"]
        kl = {t: rs.loc[nsin, f"{t}_H_KL_{s}_def"] for t in ["T0", "T2", "T5", "T8", "T10"]}
        kl5 = [t for t, v in kl.items() if v == 5]
        # Outcome rule (checked against follow-up films, see 6.6 of the plan):
        # T10 carries the corrected cumulative THA status; flags that appear only in earlier waves were side-entry errors
        # (5880459, 6224931). KL=5 without any flag = prosthesis on film (8 hips) -> event, unless scored after definite dropout.
        kl5_obs = [t for t in kl5 if int(t[1:]) <= end]
        ev = W[10].loc[nsin, f"T10_THA{S}"] == 1 or pd.notna(visit) or bool(kl5_obs)
        t_ev = (visit if pd.notna(visit) else min(flags) if W[10].loc[nsin, f"T10_THA{S}"] == 1 else int(kl5_obs[0][1:])) if ev else np.nan
        hips.append({"nsin": nsin, "side": S, "event": int(ev), "t_event": t_ev, "first_flag": min(flags) if flags else np.nan,
                     "flag_T10": int(W[10].loc[nsin, f"T10_THA{S}"] == 1), "fu_end": end,
                     "kl0": kl["T0"], "kl10": kl["T10"], "kl5_any": int(bool(kl5)), "kl5_first": kl5[0] if kl5 else None,
                     "group": "interval" if t0.loc[nsin, "T0_JVCON"] == INTERVAL_CODE else "annual"})
H = pd.DataFrame(hips)
H["time"] = np.where(H.event == 1, H.t_event, H.fu_end)
H["status10"] = np.select([H.event == 1, H.fu_end >= 10], ["event", "event-free@10y"], "censored<10y")

# image availability per hip
v = img.groupby("nsin").agg(views=("view", lambda x: "".join(sorted(set(x)))), fmts=("fmt", lambda x: "/".join(sorted(set(x)))), nfile=("view", "size"))
H = H.merge(v, left_on="nsin", right_index=True, how="left")
H["has_img"] = [isinstance(vw, str) and ("O" in vw or S in vw) for vw, S in zip(H.views, H.side)]
def hip_fmt(r):
    if not r.has_img: return None
    sub = img[(img.nsin == r.nsin) & (img.view.isin(["O", r.side]))]
    return "/".join(sorted(set(sub.fmt)))
H["img_fmt"] = H.apply(hip_fmt, axis=1)

# ---------- feature groups (hip-level: ipsilateral columns) ----------
def ip(base, S):  # ipsilateral column name for side-specific exam variables
    return f"T0_{base}{S}"
def side_cols(S):
    o = "R" if S == "L" else "L"
    E = [ip(b, S) for b in ["HENDO", "HEXO", "HFLEX", "HADD", "HABD", "HENDOPIJN", "HEXOPIJN", "HFLEXPIJN", "HADDPIJN", "HABDPIJN", "HCONTRAC"]]
    return {
        "D 人口学": ["T0_Lft_T0", "T0_SEXE", "T0_BMI", "T0_comor"],
        "D' 绝经(仅女性)": ["T0_Menopauze_01"],
        "E 查体(同侧)": E,
        "S 症状": [f"T0_JVHPIJN{S}", "T0_JVHSTIJF", "T0_D02", "T0_D03", "T0_wmpyn", "T0_wmstf", "T0_wmfun",
                  "T0_rfys", "T0_rpijn", "T0_PCIpijn", "T0_Sosgem"],
        "C 就医用药": ["T0_F01_ha", "T0_f01_fys", "T0_f01_orth", "T0_f01_reu", "T0_f01_geen", "T0_parac", "T0_ibu", "T0_asp"],
        "L 社会生活": ["T0_a08", "T0_ZW01", "T0_J01", "T0_I14"],
        "L' 体力负荷(仅在职)": [f"T0_zw05{c}" for c in "abcdefghijklmnop"],
        "B 生化": ["T0_CTXIIngmmol", "T0_sCOMPUl", "T0_hsCRP"],
    }
R_COLS = lambda s: [c for c in rs.columns if c.startswith("T0_H_") and f"_{s}_" in c]

miss = []
for r in H.itertuples():
    rec = {"nsin": r.nsin, "side": r.side}
    for g, cols in side_cols(r.side).items():
        cols = [c for c in cols if c in t0.columns]
        vals = t0.loc[r.nsin, cols]
        rec[g] = vals.isna().mean()
    rc = sorted(set(R_COLS(SIDES[r.side])))
    rv = rs.loc[r.nsin, rc]
    rec["R 读片评分"] = (rv.isna() | (rv == 9)).mean()
    rec["I 骨盆正位像素"] = 0.0 if r.has_img else 1.0
    miss.append(rec)
M = pd.DataFrame(miss)
H = H.merge(M, on=["nsin", "side"])
H.to_csv(out / "hips.csv", index=False)

# ---------- report ----------
P = lambda x, n: f"{x} ({x / n:.1%})"
print("\n===== 受试者 =====")
N = len(t0); print("T0 受试者", N, "| 随访分组", H.drop_duplicates("nsin").group.value_counts().to_dict())
print("有任何 T00_AP 图像的受试者", img.nsin.nunique(), "| 文件数", len(img), img.groupby(["view", "fmt"]).size().to_dict())
print("每人文件数分布", v.nfile.value_counts().sort_index().to_dict(), "| 视图组合", v.views.value_counts().to_dict(), "| 同一人混合格式", int((v.fmts == "dcm/tif").sum()))
dd = W[10]["T10_dropout_def_visit"]
print("10 年内彻底失访", int(dd.notna().sum()), "按失访访视", dd.value_counts().sort_index().to_dict(), "| T10 临时缺席", int((W[10].T10_dropout_temp == 1).sum()))

print("\n===== 髋 =====")
nh = len(H); print("髋总数", nh, "| 基线 KL 已评", int(H.kl0.between(0, 4).sum()), "基线KL分布", H.kl0.value_counts(dropna=False).sort_index().to_dict())
print("基线已置换(KL0=5)", int((H.kl0 == 5).sum()))
print("有基线骨盆正位像素的髋", P(int(H.has_img.sum()), nh), "格式", H.img_fmt.value_counts().to_dict())
full = H.has_img & H.kl0.between(0, 4) & (H["E 查体(同侧)"] < 1)
print("像素+KL+查体齐全的髋", P(int(full.sum()), nh))

print("\n===== 结局：10 年内同侧髋置换 =====")
e = H[H.event == 1]
print("置换髋", int(H.event.sum()), "| 左", int(e.side.eq("L").sum()), "右", int(e.side.eq("R").sum()), "| 涉及受试者", e.nsin.nunique(), "双侧", int((e.groupby("nsin").size() == 2).sum()))
print("T10 标记有、但只用 T10 会漏掉的", int(((H.event == 1) & (H.flag_T10 == 0)).sum()))
print("置换年份(VISIT/首次标记)", e.t_event.value_counts().sort_index().to_dict())
print("10y 状态", H.status10.value_counts().to_dict())
print("粗率 事件/全部髋", f"{H.event.mean():.2%}")
known = H.status10 != "censored<10y"
print("10 年状态已知的髋", int(known.sum()), "其中事件率", f"{H[known].event.mean():.2%}")
# Kaplan-Meier at hip level
def km(df):
    s, curve = 1.0, {}
    for t in range(1, 11):
        at_risk = ((df.time >= t)).sum()
        d = ((df.event == 1) & (df.t_event == t)).sum()
        if at_risk: s *= 1 - d / at_risk
        curve[t] = 1 - s
    return curve
c = km(H); print("KM 累计置换率 2/5/8/10 年", {t: f"{c[t]:.2%}" for t in [2, 5, 8, 10]})
for g, sub in H.groupby("group"): print("  KM 10 年", g, f"{km(sub)[10]:.2%}", "n", len(sub), "事件", int(sub.event.sum()))
for k, sub in H.groupby(H.kl0.fillna(-1)): print("  基线KL", k, "n", len(sub), "事件", int(sub.event.sum()), f"KM10 {km(sub)[10]:.1%}")
hp = H.apply(lambda r: t0.loc[r.nsin, f"T0_JVHPIJN{r.side}"], axis=1)
for k, sub in H.groupby(hp): print("  同侧基线髋痛", k, "n", len(sub), "事件", int(sub.event.sum()), f"KM10 {km(sub)[10]:.1%}")
for k, sub in H.groupby(H.has_img): print("  有像素", k, "n", len(sub), "事件", int(sub.event.sum()))
for k, sub in H[H.has_img].groupby("img_fmt"): print("  格式", k, "n", len(sub), "事件", int(sub.event.sum()), f"KM10 {km(sub)[10]:.1%}")

print("\n===== 置换标记 vs KL=5 =====")
print(pd.crosstab(H.event, H.kl5_any, margins=True))
print("事件但 KL 从未 =5：其 T10 KL", H[(H.event == 1) & (H.kl5_any == 0)].kl10.value_counts(dropna=False).to_dict())
print("KL=5 但无置换标记：", H[(H.event == 0) & (H.kl5_any == 1)][["nsin", "side", "kl5_first", "fu_end"]].to_dict("records"))

print("\n===== 缺失（髋级，变量缺失比例的均值 / 整组全缺的髋比例） =====")
grp = [c for c in M.columns if c not in ("nsin", "side")]
tab = pd.DataFrame({"变量平均缺失": H[grp].mean(), "整组全缺的髋": (H[grp] == 1).mean(),
                    "事件髋中平均缺失": H.loc[H.event == 1, grp].mean()}).round(3)
print(tab.to_string()); tab.to_csv(out / "missing_by_group.csv")
sexe = t0["T0_SEXE"]; print("性别编码", sexe.value_counts().to_dict())
emp = t0["T0_zw05a"].isna()
print("体力负荷缺失 × 就业(ZW01)", pd.crosstab(t0.T0_ZW01, emp).to_dict())

print("\n===== 影像头文件 =====")
dc = img[img.fmt == "dcm"]
for k in ["InstitutionName", "Manufacturer", "ManufacturerModelName", "Modality", "StationName", "PhotometricInterpretation", "BitsStored"]:
    print(k, dc[k].value_counts(dropna=False).head(15).to_dict())
print("DICOM StudyDate 年份", dc.StudyDate.str[:4].value_counts(dropna=False).sort_index().to_dict())
print("DICOM 尺寸", (dc.Rows.astype(str) + "x" + dc.Columns.astype(str)).value_counts().head(8).to_dict())
tf_ = img[img.fmt == "tif"]
print("TIFF 尺寸", (tf_.Rows.astype(str) + "x" + tf_.Columns.astype(str)).value_counts().head(8).to_dict(), "bits", tf_.BitsStored.value_counts().to_dict(),
      "photometric", tf_.Photometric.value_counts().to_dict(), "software", tf_.Software.value_counts(dropna=False).head(5).to_dict())
print("文件大小 MB 中位数", img.groupby("fmt").mb.median().round(1).to_dict())
print("读取错误", int(img.get("error", pd.Series(dtype=str)).notna().sum()))
