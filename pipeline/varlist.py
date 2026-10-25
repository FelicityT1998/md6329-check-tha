import sys, pathlib, pyreadstat
d = pathlib.Path(sys.argv[1]); out = []
files = ["T0.por"] + [f"T{i}.dta" for i in range(1,11)] + ["Rontgen_opT10.dta"]
for f in files:
    rd = pyreadstat.read_por if f.endswith(".por") else pyreadstat.read_dta
    try:
        df, meta = rd(d/f, metadataonly=True)
    except Exception as e:
        df, meta = rd(d/f, metadataonly=True, encoding="latin1")
    out.append(f"===== {f}: {meta.number_rows} rows x {meta.number_columns} cols")
    for n, l in zip(meta.column_names, meta.column_labels):
        vl = meta.variable_value_labels.get(n)
        out.append(f"{f}\t{n}\t{l or ''}\t{dict(list(vl.items())[:8]) if vl else ''}")
(d/"varlist.tsv").write_text("\n".join(out), encoding="utf-8")
print("\n".join(l for l in out if l.startswith("=====")))
