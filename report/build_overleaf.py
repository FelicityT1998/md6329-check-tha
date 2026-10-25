"""Build a single-file Overleaf project (NeurIPS 2026 template) from report/main.tex + sections/.

usage: python -I build_overleaf.py <report dir>
writes: <report dir>/overleaf/{main.tex, neurips_2026.sty, refs.bib, figs/*.png}  and  <report dir>/overleaf.zip
Every \\input{...} is inlined, so the upload cannot break on missing sub-files; only figures that are referenced are copied.
"""
import re, sys, shutil, pathlib, zipfile

R = pathlib.Path(sys.argv[1]); OUT = R / "overleaf"
if OUT.exists(): shutil.rmtree(OUT)
(OUT / "figs").mkdir(parents=True)


def inline(path):
    txt = path.read_text(encoding="utf8")
    def rep(m):
        p = R / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return f"% ---- begin {p.name} ----\n" + inline(p) + f"\n% ---- end {p.name} ----\n"
    return re.sub(r"^\\input\{([^}]+)\}", rep, txt, flags=re.M)


tex = inline(R / "main.tex")
tex = re.sub(r"^%[^\n]*[^\x00-\x7f][^\n]*$", "%", tex, flags=re.M)          # drop non-ASCII comment lines
(OUT / "main.tex").write_text(tex, encoding="utf8")
for f in ("neurips_2026.sty", "refs.bib"): shutil.copy(R / f, OUT / f)
figs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))
missing = []
for f in sorted(figs):
    src = R / "figs" / f
    if src.exists():
        # pdfLaTeX re-encodes PNGs with an alpha channel slowly (free Overleaf compile limit): flatten to RGB on white and
        # cap the width at 2400 px (= 300 dpi at the full text width); the originals in figs/ stay untouched
        from PIL import Image
        im = Image.open(src)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA"); bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[-1]); im = bg
        elif im.mode != "RGB": im = im.convert("RGB")
        if im.width > 2400: im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
        im.save(OUT / "figs" / f, optimize=True)
    else: missing.append(f)
with zipfile.ZipFile(R / "overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p, p.relative_to(OUT).as_posix())
print("figures copied:", len(figs) - len(missing), "| missing (guarded by \\IfFileExists):", missing)
print("non-ASCII left in main.tex:", [l for l in tex.splitlines() if re.search(r"[^\x00-\x7f]", l)][:5])
print("->", R / "overleaf.zip")
