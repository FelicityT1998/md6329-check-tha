import sys, pathlib, pandas as pd, pyreadstat
d = pathlib.Path(sys.argv[1])
t0 = pd.read_csv(d/"T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False)
print("T0 rows", len(t0), "nsin dtype", t0.nsin.dtype, t0.nsin.head(3).tolist(), "unique", t0.nsin.nunique())
W = {}
for k in range(1, 11):
    df, _ = pyreadstat.read_dta(d/f"sav/T{k}.dta", usecols=["nsin", f"T{k}_THAL", f"T{k}_THAR", f"T{k}_THAL_VISIT", f"T{k}_THAR_VISIT", f"T{k}_dropout_def_visit", f"T{k}_dropout_temp"])
    W[k] = df.set_index("nsin")
    print(k, len(df), "THAL", df[f"T{k}_THAL"].value_counts(dropna=False).to_dict(), "temp", df[f"T{k}_dropout_temp"].value_counts(dropna=False).to_dict(),
          "def", df[f"T{k}_dropout_def_visit"].value_counts(dropna=False).sort_index().to_dict())
m = pd.concat([W[k][[f"T{k}_THAL"]] for k in range(1, 11)], axis=1)
# monotonic?
arr = m.fillna(-1).values
nonmono = sum(any(arr[i, j] == 1 and arr[i, j+1] == 0 for j in range(9)) for i in range(len(arr)))
print("left THA flag goes 1->0 later:", nonmono)
print("T10 THAL_VISIT for flagged:", W[10].loc[W[10].T10_THAL == 1, "T10_THAL_VISIT"].value_counts().sort_index().to_dict())
first = m.eq(1).idxmax(axis=1).where(m.eq(1).any(axis=1))
cmp = pd.DataFrame({"first": first.str.extract(r"T(\d+)_")[0].astype(float), "visit": W[10].T10_THAL_VISIT})
print("first-flag wave vs T10 VISIT (left):"); print(pd.crosstab(cmp["first"].fillna(-1), cmp["visit"].fillna(-1)))
rs, _ = pyreadstat.read_dta(d/"sav/Rontgen_opT10.dta")
print("Rontgen rows", len(rs), "T0 KL left", rs.T0_H_KL_li_def.value_counts(dropna=False).sort_index().to_dict(), "right", rs.T0_H_KL_re_def.value_counts(dropna=False).sort_index().to_dict())
