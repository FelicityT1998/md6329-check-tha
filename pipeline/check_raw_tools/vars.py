import sys, re, csv, pathlib, fitz
b = pathlib.Path(sys.argv[1])
t = "".join(p.get_text() for p in fitz.open(b/"T0T5/eng var quest corr 6.pdf"))
for kw in [r"occupation|profession|job\b|paid work|work(ing)? hours|physical(ly)? (demanding|work)|heavy (work|lifting)", r"sport|physical activity|SQUASH", r"smok", r"paired|known (time )?sequence|chronolog|blinded|re-?scor|Rontgen_?opT10|scored at T10"]:
    hits = [m for m in re.finditer(kw, t, re.I)][:4]
    print(f"== /{kw}/ hits={len(list(re.finditer(kw, t, re.I)))}")
    for m in hits: print("   ...", t[max(0,m.start()-120):m.end()+160].replace("\n"," "))
with open(b/"T0T5/CHECK_T0_DANS_ENG_20161128.tab", encoding="utf-8") as f: h = next(csv.reader(f, delimiter="\t"))
print("T0 cols re work/sport/smoke/KL:", [c for c in h if re.search(r"(?i)werk|work|beroep|job|sport|squash|rook|smok|KL|kell|alpha|lcea|ce_", c)])
