import sys, pathlib, csv, collections
b = pathlib.Path(sys.argv[1])
with open(b/"T0T5/CHECK_Radiographic_Scoring_OA_T0T2T5T8T10_DANS_ENG_20170726.tab", encoding="utf-8") as f:
    r = list(csv.reader(f, delimiter="\t")); h, rows = r[0], r[1:]
col = {c: i for i, c in enumerate(h)}
for t in ["T0","T2","T5","T8","T10"]:
    c = collections.Counter()
    for side in ["li","re"]:
        for row in rows: c[row[col[f"RontgT10_{t}_H_KL_{side}_def"]]] += 1
    print(t, dict(sorted(c.items())))
inc = collections.Counter()
for row in rows:
    for side in ["li","re"]:
        k0, k10 = row[col[f"RontgT10_T0_H_KL_{side}_def"]], row[col[f"RontgT10_T10_H_KL_{side}_def"]]
        if k0 in ("0","1"):
            inc["base<2"] += 1
            if k10 == "": inc["T10 missing"] += 1
            elif k10.isdigit() and int(k10) >= 2: inc["T10 KL>=2 (raw)"] += 1
            else: inc[f"T10={k10}"] += 1
print(dict(inc))
