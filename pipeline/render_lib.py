"""Shared radiograph loader: returns float array with bone bright (MONOCHROME1 inverted) and a one-line description."""
import pathlib
import numpy as np, pydicom, tifffile


def load(path):
    p = pathlib.Path(path)
    if p.suffix == ".dcm":
        ds = pydicom.dcmread(p)
        a = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1": a = a.max() - a
        sp = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        info = f"DICOM {a.shape[1]}x{a.shape[0]} {ds.BitsStored}bit {ds.PhotometricInterpretation} spacing={sp} date={getattr(ds, 'StudyDate', '')}"
    else:
        a = tifffile.imread(p).astype(np.float32)
        if a.ndim == 3: a = a[..., :3].mean(-1)
        info = f"TIFF {a.shape[1]}x{a.shape[0]}"
    return a, info
