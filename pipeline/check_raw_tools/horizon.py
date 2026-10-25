import sys, csv, pathlib
b = pathlib.Path(sys.argv[1])
with open(b/"T0T5/CHECK_Radiographic_Scoring_OA_T0T2T5T8T10_DANS_ENG_20170726.tab", encoding="utf-8") as f:
    r = list(csv.reader(f, delimiter="\t")); h, rows = r[0], r[1:]
c = {k: i for i, k in enumerate(h)}
def kl(row, t, s):
    v = row[c[f"RontgT10_{t}_H_KL_{s}_def"]].strip()
    return int(v) if v.isdigit() else None
print("horizon | at-risk hips with data | incident KL>=2 or code5 | rate | of which code5")
for t in ["T2","T5","T8","T10"]:
    n = ev = e5 = 0
    for row in rows:
        for s in ["li","re"]:
            k0 = kl(row,"T0",s)
            if k0 is None or k0 >= 2: continue
            prior5 = any(kl(row,tt,s) == 5 for tt in ["T2","T5","T8","T10"][:["T2","T5","T8","T10"].index(t)])
            kt = kl(row,t,s)
            if kt is None and not prior5: continue
            n += 1
            if prior5 or kt >= 2: ev += 1
            if prior5 or kt == 5: e5 += 1
    print(f"{t:>4} | {n} | {ev} | {ev/n:.1%} | {e5}")
end = 0
for row in rows:
    for s in ["li","re"]:
        k0 = kl(row,"T0",s); k10 = kl(row,"T10",s)
        if k0 is not None and k0 < 2 and k10 is not None and k10 in (3,4,5): end += 1
print("end-stage at T10 (KL>=3 or code5) among base<2 hips:", end)
