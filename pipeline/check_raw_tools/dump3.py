import sys, pathlib, pandas as pd, pyreadstat
d = pathlib.Path(sys.argv[1]); ids = [713242, 26001, 2626426]
t0 = pd.read_csv(d/"T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False).set_index("nsin")
rs = pyreadstat.read_dta(d/"sav/Rontgen_opT10.dta")[0]; rs["nsin"] = rs.nsin.astype(int); rs = rs.set_index("nsin")
cols = ["T0_JVCON","T0_SEXE","T0_Lft_T0","T0_BMI","T0_a08","T0_ZW01","T0_comor","T0_Menopauze_01",
        "T0_JVHPIJNL","T0_JVHPIJNR","T0_JVKPIJN","T0_D02","T0_wmpyn","T0_wmfun","T0_JVHSTIJF",
        "T0_HENDOL","T0_HENDOR","T0_HFLEXL","T0_HFLEXR","T0_HEXOL","T0_HEXOR","T0_HADDL","T0_HABDL",
        "T0_HENDOPIJNL","T0_HENDOPIJNR","T0_HCONTRACL","T0_HCONTRACR",
        "T0_F01_ha","T0_f01_fys","T0_f01_orth","T0_f01_geen","T0_parac","T0_ibu","T0_J01","T0_zw05a","T0_CTXIIngmmol","T0_hsCRP"]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("=== T0 表（原始编码，共 %d 列，这里摘 %d 列）" % (t0.shape[1], len(cols)))
print(t0.loc[ids, cols].T.to_string())
rc = [c for c in rs.columns if "KL" in c] + [c for c in rs.columns if c.startswith("T0_H_") and "KL" not in c][:8]
print("\n=== 读片评分表（%d 列，摘录）" % rs.shape[1]); print(rs.loc[ids, rc].T.to_string())
print("\n=== T1–T10 置换标记 / 失访")
for k in range(1, 11):
    w = pyreadstat.read_dta(d/f"sav/T{k}.dta", usecols=["nsin", f"T{k}_THAL", f"T{k}_THAR", f"T{k}_THAL_VISIT", f"T{k}_THAR_VISIT", f"T{k}_dropout_temp", f"T{k}_dropout_def_visit"])[0]
    w["nsin"] = w.nsin.astype(int); w = w.set_index("nsin").loc[ids]
    print(f"T{k}", {n: w.loc[n].dropna().to_dict() for n in ids})
