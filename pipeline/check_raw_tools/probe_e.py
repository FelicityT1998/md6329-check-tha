import sys, pandas as pd
t0 = pd.read_csv(sys.argv[1], sep="\t", low_memory=False)
for b in ["HENDO","HEXO","HFLEX","HADD","HABD","HENDOPIJN","HEXOPIJN","HFLEXPIJN","HADDPIJN","HABDPIJN","HCONTRAC"]:
    print(b, {S: (f"T0_{b}{S}" in t0.columns) and round(t0[f"T0_{b}{S}"].isna().mean(), 3) for S in "LR"})
print([c for c in t0.columns if "HFLEX" in c or "HEXO" in c])
img = pd.read_csv(sys.argv[2])
d = img[img.fmt == "dcm"]
print("PixelSpacing", d.PixelSpacing.notna().sum(), d.PixelSpacing.value_counts().head(6).to_dict())
print("ImagerPixelSpacing", d.ImagerPixelSpacing.notna().sum(), d.ImagerPixelSpacing.value_counts().head(6).to_dict())
d = d.assign(sig=d.Rows.astype(str)+"x"+d.Columns.astype(str)+"|"+d.BitsStored.astype(str)+"|"+d.PhotometricInterpretation)
print("acq signature clusters", d.sig.value_counts().head(12).to_dict())
t = img[img.fmt == "tif"]; print("TIFF XRes", t.XRes.value_counts().head(6).to_dict())
