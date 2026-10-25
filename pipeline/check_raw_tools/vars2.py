import sys, re, csv, pathlib
b = pathlib.Path(sys.argv[1])
with open(b/"T0T5/CHECK_T0_DANS_ENG_20161128.tab", encoding="utf-8") as f: h = next(csv.reader(f, delimiter="\t"))
print([c for c in h if re.search(r"(?i)_ZW|_I1[0-9]|_J0|womac|_[abc]0?[0-9]+$|stijf|stiff|morn|THA|THR|TKA|HP\b|_E0", c)][:150])
print(len(h)); print(h[:370])
