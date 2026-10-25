"""Read only the DICOM headers of every follow-up pelvis AP film (T2/T5/T8/T10) on DANS via HTTP range requests.
usage: python -I followup_headers.py <out.csv>
Resumable: rows already in <out.csv> are skipped."""
import sys, os, io, re, json, csv, zipfile, urllib.request
import pydicom

BASE = "https://lifesciences.datastations.nl/api"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file with a 256 KB read-ahead cache (one request per header in practice)."""
    def __init__(self, url, block=256 << 10):
        self.url, self.pos, self.block, self.cache = url, 0, block, (0, b"")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120) as r:
            self.url = r.geturl(); self.size = int(r.headers["Content-Range"].split("/")[-1])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def _fetch(self, start, n):
        end = min(start + n, self.size) - 1
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"}), timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == 4: raise
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        c0, cb = self.cache
        if not (c0 <= self.pos and self.pos + n <= c0 + len(cb)):
            self.cache = (self.pos, self._fetch(self.pos, max(n, self.block))); c0, cb = self.cache
        out = cb[self.pos - c0: self.pos - c0 + n]; self.pos += len(out); return out


FIELDS = ["wave", "zip", "member", "nsin", "view", "serial", "sw", "manufacturer", "model", "rows", "cols", "bits", "photo", "spacing", "year", "plate", "error"]
out = sys.argv[1]
done = set()
if os.path.exists(out):
    with open(out, encoding="utf-8") as f: done = {r["member"] for r in csv.DictReader(f)}
fh = open(out, "a", newline="", encoding="utf-8"); w = csv.DictWriter(fh, FIELDS)
if not done: w.writeheader()

zips = []
for pid in ["doi:10.17026/DANS-XEX-HZWW", "doi:10.17026/DANS-ZC8-G4CW", "doi:10.17026/DANS-XU3-GCAT"]:
    d = json.load(urllib.request.urlopen(f"{BASE}/datasets/:persistentId/?persistentId={pid}", timeout=60))["data"]["latestVersion"]
    zips += [(f["dataFile"]["filename"], f["dataFile"]["id"]) for f in d["files"] if re.match(r"T(2|5|8|10)(-\d+)?\.zip$", f["dataFile"]["filename"])]
for zname, fid in sorted(zips):
    z = zipfile.ZipFile(HttpFile(f"{BASE}/access/datafile/{fid}"))
    members = [i for i in z.infolist() if re.search(r"/T\d+_AP/(\d+)_T\d+_AP[OLR]?\.dcm$", i.filename) and i.filename not in done]
    print(zname, "todo", len(members), flush=True)
    for i in members:
        m = re.search(r"/(\d+)_(T\d+)_AP([OLR]?)\.dcm$", i.filename)
        row = dict(wave=m[2], zip=zname, member=i.filename, nsin=int(m[1]), view=m[3])
        try:
            with z.open(i) as r:
                ds = pydicom.dcmread(r, stop_before_pixels=True, force=True)
            g = lambda k: str(getattr(ds, k, "") or "")
            row.update(serial=g("DeviceSerialNumber"), sw=g("SoftwareVersions"), manufacturer=g("Manufacturer"), model=g("ManufacturerModelName"),
                       rows=g("Rows"), cols=g("Columns"), bits=g("BitsStored"), photo=g("PhotometricInterpretation"),
                       spacing=(g("PixelSpacing") or g("ImagerPixelSpacing")).replace(" ", ""), year=g("StudyDate")[:4], plate=g("PlateType"))
        except Exception as e:
            row["error"] = repr(e)[:200]
        w.writerow(row); fh.flush()
print("ALL DONE", flush=True)
