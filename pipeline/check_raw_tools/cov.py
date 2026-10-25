import sys, pathlib, re, csv, collections
b = pathlib.Path(sys.argv[1])
ap = []
for f in sorted((b/"zip_index").glob("T0*.txt")):
    for line in f.read_text(encoding="utf-8").splitlines():
        n, s = line.split("\t")
        if "/T00_AP/" in n: ap.append((n, int(s)))
ids = collections.Counter(re.search(r"/(\d+)_", n).group(1) for n, _ in ap)
ext = collections.Counter(n.rsplit(".",1)[-1].lower() for n,_ in ap)
print("AP files", len(ap), "unique nsin", len(ids), "ext", dict(ext), "multi-file ids", sum(v>1 for v in ids.values()))
print("example names:", [n for n,_ in ap[:5]])
def load(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        r = list(csv.reader(f, delimiter="\t")); return r[0], r[1:]
h0, r0 = load(b/"T0T5/CHECK_T0_DANS_ENG_20161128.tab"); h1, r1 = load(b/"T0T5/CHECK_Radiographic_Scoring_OA_T0T2T5T8T10_DANS_ENG_20170726.tab")
nsin = {row[0] for row in r0}
print("clinical rows", len(r0), "AP ids in clinical:", len(set(ids) & nsin))
hipcols = [c for c in h1 if "_H_" in c]
print("hip scoring cols:", len(hipcols)); print(hipcols[:80])
print("KL-like:", [c for c in h1 if re.search(r"(?i)kl|kell|tka|tha|thp|prot|repl", c)])
