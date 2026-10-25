import sys, io, json, zipfile, urllib.request, collections, re
class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl(); self.size = int(r.headers["Content-Range"].split("/")[-1])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            if r.status != 206: raise RuntimeError(f"no range support: {r.status}")
            data = r.read()
        self.pos += len(data); return data
ids = json.load(open(sys.argv[2]))
summary = {}
for label, fid in ids.items():
    f = HttpFile(f"https://lifesciences.datastations.nl/api/access/datafile/{fid}")
    z = zipfile.ZipFile(f)
    rows = [(i.filename, i.file_size) for i in z.infolist() if not i.is_dir()]
    folders = collections.Counter(n.split("/")[1] for n, _ in rows if n.count("/") >= 2)
    sizes = collections.Counter()
    for n, s in rows: sizes[n.split("/")[1]] += s
    summary[label] = {k: [folders[k], round(sizes[k] / 1e9, 2)] for k in folders}
    with open(f"{sys.argv[1]}/{label}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"{n}\t{s}" for n, s in rows))
    print(label, summary[label], flush=True)
json.dump(summary, open(f"{sys.argv[1]}/summary.json", "w"), indent=1)



