import sys, glob, os, pydicom, pandas as pd
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.dcm"))):
    ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
    g = lambda k: str(getattr(ds, k, "") or "")
    rows.append(dict(nsin=int(os.path.basename(p)[:7]), file=os.path.basename(p), serial=g("DeviceSerialNumber"), sw=g("SoftwareVersions"),
                     rows=g("Rows"), cols=g("Columns"), bits=g("BitsStored"), photo=g("PhotometricInterpretation"),
                     spacing=g("PixelSpacing") or g("ImagerPixelSpacing"), year=g("StudyDate")[:4], ts=ds.file_meta.TransferSyntaxUID.name if hasattr(ds, "file_meta") and "TransferSyntaxUID" in ds.file_meta else "",
                     plate=g("PlateType"), proc=g("AcquisitionDeviceProcessingDescription"), body=g("BodyPartExamined"), kvp=g("KVP"), dist=g("DistanceSourceToDetector")))
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
print("files", len(d), "serial non-empty", (d.serial != "").sum(), "sw non-empty", (d.sw != "").sum())
print("serial", d.serial.value_counts().head(15).to_dict()); print("sw", d.sw.value_counts().head(15).to_dict())
print("proc", d.proc.value_counts().head(10).to_dict()); print("plate", d.plate.value_counts().head(6).to_dict()); print("kvp non-empty", (d.kvp != "").sum(), "dist", d.dist.value_counts().head(5).to_dict())
d["sig"] = d.rows + "x" + d.cols + "|" + d.bits + "|" + d.photo + "|" + d.spacing.str.replace(" ", "")
print(pd.crosstab(d.sig, d.sw.replace("", "<none>")).loc[lambda x: x.sum(1) > 4].to_string())
