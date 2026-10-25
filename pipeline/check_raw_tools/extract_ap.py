import sys, os, zipfile, glob, shutil, zlib, datetime, json
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
log = open(os.path.join(dst, "_extract.log"), "a", encoding="utf-8", buffering=1)
def L(m): log.write(f"{datetime.datetime.now():%H:%M:%S} {m}\n")
bad, n, nbytes = [], 0, 0
for zp in sorted(glob.glob(os.path.join(src, "*.zip"))):
    with zipfile.ZipFile(zp) as z:
        for i in z.infolist():
            if i.is_dir() or "/T00_AP/" not in i.filename: continue
            out = os.path.join(dst, os.path.basename(i.filename))
            if os.path.exists(out) and os.path.getsize(out) == i.file_size and os.path.exists(out + ".ok"):
                n += 1; nbytes += i.file_size; continue
            try:
                with z.open(i) as r, open(out, "wb") as w:   # zipfile raises BadZipFile on CRC mismatch at EOF
                    shutil.copyfileobj(r, w, 16 << 20)
                open(out + ".ok", "w").close()
                n += 1; nbytes += i.file_size
            except Exception as e:
                bad.append((os.path.basename(zp), i.filename, repr(e))); L(f"BAD {zp} {i.filename} {e!r}")
    L(f"{os.path.basename(zp)} done; total files {n}, {nbytes/1e9:.1f} GB, bad {len(bad)}")
for f in glob.glob(os.path.join(dst, "*.ok")): os.remove(f)
json.dump({"files": n, "gb": round(nbytes/1e9, 2), "bad": bad}, open(os.path.join(dst, "_extract_summary.json"), "w"), indent=1)
L("ALL DONE")
