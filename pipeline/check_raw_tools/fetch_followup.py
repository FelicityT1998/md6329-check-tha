"""Index follow-up image zips on DANS via HTTP range requests and pull only selected subjects' hip AP films.
usage: python -I fetch_followup.py <out dir> <nsin,nsin,...>"""
import sys, os, io, json, zipfile, urllib.request, shutil, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zipindex_http import HttpFile
out, want = sys.argv[1], {int(x) for x in sys.argv[2].split(",")}
os.makedirs(out, exist_ok=True)
base = "https://lifesciences.datastations.nl/api"
zips = []
for pid in ["doi:10.17026/DANS-ZC8-G4CW", "doi:10.17026/DANS-XU3-GCAT", "doi:10.17026/DANS-XEX-HZWW"]:
    d = json.load(urllib.request.urlopen(f"{base}/datasets/:persistentId/?persistentId={pid}", timeout=60))["data"]["latestVersion"]
    for f in d["files"]:
        n = f["dataFile"]["filename"]
        if re.match(r"T(5|8|10)(-\d+)?\.zip$", n): zips.append((n, f["dataFile"]["id"]))
idx = {}
for n, fid in sorted(zips):
    z = zipfile.ZipFile(HttpFile(f"{base}/access/datafile/{fid}"))
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    idx[n] = names
    folders = sorted({x.split("/")[1] for x in names if x.count("/") >= 2})
    print(n, len(names), folders, flush=True)
    for i in z.infolist():
        m = re.search(r"/(\d+)_T\d+_AP[OLR]?\.(tif|dcm)$", i.filename)
        if m and int(m[1]) in want:
            dst = os.path.join(out, os.path.basename(i.filename))
            if not os.path.exists(dst):
                with z.open(i) as r, open(dst, "wb") as w: shutil.copyfileobj(r, w, 16 << 20)
            print("  got", i.filename, round(i.file_size / 1e6, 1), "MB", flush=True)
json.dump(idx, open(os.path.join(out, "_followup_zip_index.json"), "w"))
