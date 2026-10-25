"""Recover CHECK study centres from X-ray device identity (plan section 7.1).

Bipartite graph subject-device over all pelvis AP DICOM films T0-T10; two devices are linked when >= MIN_SHARED subjects
were imaged on both; connected components of the device graph = pseudo-centres; each subject takes the component holding
most of its films. Assumes subjects stay at one research centre for 10 years (supported by identical serials at T5/T10).

usage: python -I 02_centres.py <project dir>   ->  data/derived/centres.csv, prints validation tables
"""
import sys, re, pathlib, collections
import numpy as np, pandas as pd

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"


def family(sw):
    sw = str(sw)
    if "PMS" in sw: return "Philips"
    for pat, name in [(r"^VIPS", "VIPS"), (r"^sol_", "sol"), (r"^V[A-F]\d", "SiemensV"), (r"^V\d", "Vnum"),
                      (r"^Version", "Version"), (r"^A\d", "A"), (r"^\d", "num")]:
        if re.match(pat, sw): return name
    return "none" if sw in ("", "nan") else "other"


def device(r):
    fam = family(r.sw)
    if str(r.serial) not in ("", "nan"): return f"{fam}|SN{r.serial}"
    return f"SIG|{fam}|{r.rows}x{r.cols}|{r.bits}b|{r.spacing}"   # fallback signature when the serial was stripped; never used to link devices


t0 = pd.read_csv(D / "dicom_t0_device.csv", dtype=str).fillna("").assign(wave="T00")
fu = pd.read_csv(D / "followup_headers.csv", dtype=str).fillna("")
fu = fu[fu.error == ""]
films = pd.concat([t0[["nsin", "wave", "serial", "sw", "rows", "cols", "bits", "spacing"]],
                   fu[["nsin", "wave", "serial", "sw", "rows", "cols", "bits", "spacing"]]], ignore_index=True)
films["nsin"] = films.nsin.astype(int)
films["spacing"] = films.spacing.str.replace(" ", "")
films["device"] = films.apply(device, axis=1)
print("films", len(films), "| subjects with >=1 DICOM film", films.nsin.nunique(), "| devices", films.device.nunique())
print("films per wave", films.wave.value_counts().sort_index().to_dict())

sub_dev = films.groupby(["nsin", "device"]).size().rename("n").reset_index()


def components(min_shared):
    parent = {d: d for d in sub_dev.device.unique()}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    pairs = collections.Counter()
    for _, g in sub_dev.groupby("nsin"):
        ds = sorted(g.device)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)): pairs[(ds[i], ds[j])] += 1
    sig = lambda d: d.startswith("SIG|")
    for (a, b), w in pairs.items():
        # signature-only "devices" collide across centres (same vendor model), so only serial-numbered devices may bridge
        if w >= min_shared and not sig(a) and not sig(b): parent[find(a)] = find(b)
    # stage 2: a signature device may attach to (never bridge) the one serial component it shares most subjects with
    attach = {}
    for d in [d for d in parent if sig(d)]:
        votes = collections.Counter()
        for (a, b), w in pairs.items():
            if d in (a, b):
                o = b if a == d else a
                if not sig(o): votes[find(o)] += w
        if votes and votes.most_common(1)[0][1] >= MIN_ATTACH: attach[d] = votes.most_common(1)[0][0]
    return {d: attach.get(d, find(d)) for d in parent}


MIN_ATTACH = 3
t0rom = pd.read_csv(P / "data/check_raw/T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False).set_index("nsin").T0_HEXOL.isna()
def assign(k):
    s = sub_dev.assign(c=sub_dev.device.map(components(k))).groupby(["nsin", "c"]).n.sum().reset_index()
    return s.sort_values("n", ascending=False).drop_duplicates("nsin").set_index("nsin")
for k in [1, 2, 3, 5, 8]:
    best = assign(k); sizes = best.c.value_counts(); big = sizes[sizes >= 10].index
    rm = t0rom.reindex(best.index).groupby(best.c).mean()
    pure = sizes[big][(rm[big] < .1) | (rm[big] > .9)].sum() / sizes[big].sum()
    print(f"min_shared={k}: components>=10: {len(big)}, sizes {sizes.head(14).tolist()}, ROM-protocol purity {pure:.2f}")

MIN_SHARED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
comp = components(MIN_SHARED)
s = sub_dev.assign(c=sub_dev.device.map(comp)).groupby(["nsin", "c"]).n.sum().reset_index()
tot = s.groupby("nsin").n.sum()
best = s.sort_values("n", ascending=False).drop_duplicates("nsin").set_index("nsin")
best["purity"] = best.n / tot.loc[best.index]          # share of the subject's films inside its assigned component
sizes = best.c.value_counts()
label = {c: f"C{i + 1:02d}" for i, c in enumerate(sizes.index)}
best["centre"] = best.c.map(lambda c: label[c] if sizes[c] >= 10 else "small")

t0tab = pd.read_csv(P / "data/check_raw/T0T5/CHECK_T0_DANS_ENG_20161128.tab", sep="\t", low_memory=False).set_index("nsin")
img = pd.read_csv(D / "images.csv").drop_duplicates("nsin").set_index("nsin")
hips = pd.read_csv(D / "hips.csv")
out = pd.DataFrame(index=t0tab.index)
out["centre"] = best.centre.reindex(out.index).fillna("unassigned")
out["purity"] = best.purity.reindex(out.index)
out["n_dicom_films"] = tot.reindex(out.index).fillna(0).astype(int)
out["fmt_T0"] = img.fmt.reindex(out.index)
out.reset_index().to_csv(D / "centres.csv", index=False)

ev = hips.groupby("nsin").agg(hip_events=("event", "sum"))
v = out.join(ev).join(t0tab[["T0_HEXOL", "T0_JVCON", "T0_f01_geen", "T0_SEXE"]])
tab = v.groupby("centre").agg(subjects=("purity", "size"), tif=("fmt_T0", lambda x: (x == "tif").sum()), dcm=("fmt_T0", lambda x: (x == "dcm").sum()),
                              hip_events=("hip_events", "sum"), rom_exo_missing=("T0_HEXOL", lambda x: x.isna().mean()),
                              interval_grp=("T0_JVCON", lambda x: (x == 1).mean()), no_care=("T0_f01_geen", "mean"),
                              female=("T0_SEXE", lambda x: (x == 2).mean()), purity=("purity", "mean")).round(2)
print(tab.to_string())
print("subjects assigned to a centre >=10:", int((~out.centre.isin(["unassigned", "small"])).sum()), "/", len(out))
print("TIFF-at-baseline subjects assigned:", int(((out.fmt_T0 == "tif") & ~out.centre.isin(["unassigned", "small"])).sum()), "/", int((out.fmt_T0 == "tif").sum()))
print("unassigned: subjects", int((out.centre == "unassigned").sum()), "| hip events", int(v[v.centre == "unassigned"].hip_events.sum()))


