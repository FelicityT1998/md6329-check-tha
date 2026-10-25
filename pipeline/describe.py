import sys, pathlib, pyreadstat, pandas as pd
d = pathlib.Path(sys.argv[1])
t0 = pd.read_csv(d.parent/"T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False)
_, m0 = pyreadstat.read_por(d/"T0.por", metadataonly=True)
L = m0.variable_value_labels
rs, _ = pyreadstat.read_dta(d/"Rontgen_opT10.dta")
t10, _ = pyreadstat.read_dta(d/"T10.dta")
def vc(col, lab=None):
    s = t0[col]; v = s.value_counts(dropna=False).sort_index()
    lab = L.get(lab or col.upper(), {})
    return ", ".join(f"{lab.get(k,k)}={n}({n/len(s):.0%})" for k, n in v.items())
print("N", len(t0))
print("age", t0["T0_Lft_T0"].describe()[["mean","std","min","max"]].round(1).to_dict())
print("sex", vc("T0_SEXE")); print("BMI", t0["T0_BMI"].describe()[["mean","std"]].round(1).to_dict())
print("race", vc("T0_RAS")); print("education", vc("T0_a08","T0_A08")); print("employment", vc("T0_ZW01"))
print("visit group", vc("T0_JVCON"))
for c,n in [("T0_zw05g","T0_ZW05G"),("T0_zw05i","T0_ZW05I")]: print(n, vc(c, n))
for c,n in [("T0_F01_ha","GP"),("T0_f01_fys","physio"),("T0_f01_orth","orthopedist"),("T0_f01_reu","rheumatologist"),("T0_f01_geen","none")]:
    print("baseline care in past yr:", n, f"{(t0[c]==1).mean():.0%}")
print("hip pain", vc("T0_JVHPIJN","T0_JVHPI")); print("knee pain", vc("T0_JVKPIJN","T0_JVKPI"))
print("dysplasia flag", vc("T0_dyspl","T0_DYSPL")); print("ROM HENDOL missing", f"{t0['T0_HENDOL'].isna().mean():.0%}")
for side, sfx in [("li","L"),("re","R")]:
    flag = t10[f"T10_THA{sfx}"]; visit = t10[f"T10_THA{sfx}_VISIT"]
    print(f"THA {side}: yes={int((flag==1).sum())} visit:", visit[flag==1].value_counts().sort_index().to_dict())
    print(pd.crosstab(rs[f"T10_H_KL_{side}_def"].fillna(-1), flag.fillna(-1)).to_string())
for t in ["T0","T2","T5","T8","T10"]:
    k = rs[f"{t}_H_KL_li_def"]
    print(t, "KL-left 0-5:", int(k.between(0,5).sum()), "code9:", int((k==9).sum()), "NaN:", int(k.isna().sum()))
