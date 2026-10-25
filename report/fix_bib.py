"""Complete bib entries from Europe PMC records (litreview/fulltext/records.json): authors, journal, volume, number,
pages, year, doi. Titles are kept. The '% verify' line after a completed entry is removed.

usage: python -I fix_bib.py <report dir> <records.json>
"""
import re, sys, json, pathlib

R = pathlib.Path(sys.argv[1]); recs = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf8"))
bib = (R / "refs.bib").read_text(encoding="utf8")
LATEX = {"é": "{\\'e}", "è": "{\\`e}", "ë": '{\\"e}', "ö": '{\\"o}', "ü": '{\\"u}', "ä": '{\\"a}', "á": "{\\'a}", "ó": "{\\'o}",
         "í": "{\\'i}", "ñ": "{\\~n}", "ç": "{\\c c}", "ø": "{\\o}", "Ö": '{\\"O}', "Ü": '{\\"U}', "É": "{\\'E}", "ž": "{\\v z}",
         "š": "{\\v s}", "ć": "{\\'c}", "–": "--", "’": "'"}


def tex(s):
    return "".join(LATEX.get(c, c) for c in s)


def authors(s):
    out = []
    for a in s.rstrip(".").split(", "):
        a = a.strip()
        if a.lower().startswith("check") or " " not in a: out.append("{" + tex(a) + "}"); continue
        last, ini = a.rsplit(" ", 1)
        out.append(f"{tex(last)}, {' '.join(c + '.' for c in ini if c.isupper())}")
    if len(out) > 10: out = out[:10] + ["others"]
    return " and ".join(out)


changed = []
for key, r in recs.items():
    m = re.search(r"@(\w+)\{" + re.escape(key) + r",(.*?)\n\}\n(% verify[^\n]*\n)?", bib, flags=re.S)
    if not m: continue
    title = re.search(r"title\s*=\s*(\{.*?\}),\n", m.group(2), flags=re.S)
    f = [f"  author  = {{{authors(r['authors'])}}}", f"  title   = {title.group(1) if title else '{' + r['title'].rstrip('.') + '}'}",
         f"  journal = {{{tex(r['journal'])}}}"]
    for k, v in (("volume", r.get("volume")), ("number", r.get("issue")), ("pages", (r.get("pages") or "").replace("-", "--"))):
        if v: f.append(f"  {k:7s} = {{{v}}}")
    f.append(f"  year    = {{{r['year']}}}")
    if r.get("doi"): f.append(f"  doi     = {{{r['doi']}}}")
    f.append(f"  note    = {{PMID {r['pmid']}}}")
    new = "@article{" + key + ",\n" + ",\n".join(f) + "\n}\n"
    bib = bib[:m.start()] + new + bib[m.end():]; changed.append(key)
(R / "refs.bib").write_text(bib, encoding="utf8")
print("completed:", changed)
print("still marked verify:", re.findall(r"@\w+\{(\w+),[^@]*?\n\}\n% verify", bib))
