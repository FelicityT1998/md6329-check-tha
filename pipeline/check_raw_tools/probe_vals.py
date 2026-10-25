import sys, pathlib, pandas as pd, re
d = pathlib.Path(sys.argv[1])
t0 = pd.read_csv(d/"T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False)
cols = list(t0.columns)
for pat in [r"^T0_r", r"PCI", r"Sos", r"^T0_zw", r"^T0_ZW", r"^T0_J0", r"^T0_I1", r"CTX|COMP|CRP|hsc", r"asp|anders|geen", r"^T0_V18", r"RAS", r"dysp", r"^T0_wm"]:
    print(pat, [c for c in cols if re.search(pat, c)][:40])
for c in ["T0_HENDOL","T0_HENDOPIJNL","T0_JVHPIJNL","T0_BMI","T0_a08","T0_zw05a","T0_D02","T0_wmtot","T0_HCONTRACL","T0_Menopauze_01","T0_comor","T0_parac","T0_F01_ha"]:
    v = t0[c]; print(c, v.dtype, "nan", int(v.isna().sum()), "top", v.value_counts(dropna=False).head(8).to_dict(), "max", v.max() if v.dtype != object else "")
