"""Render radiographs (DICOM/TIFF) to downsampled PNG previews.
usage: python -I render.py <out dir> <image> [<image> ...]
MONOCHROME1 is inverted so bone is always bright; intensities windowed at the 1st/99th percentile."""
import sys, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from render_lib import load

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for path in sys.argv[2:]:
    a, info = load(path)
    step = max(1, max(a.shape) // 1400)
    a = a[::step, ::step]
    lo, hi = np.percentile(a, [1, 99])
    fig = plt.figure(figsize=(a.shape[1] / 150, a.shape[0] / 150 + 0.3), dpi=150)
    plt.imshow(np.clip((a - lo) / (hi - lo), 0, 1), cmap="gray"); plt.axis("off")
    plt.title(f"{pathlib.Path(path).name} | {info}", fontsize=6)
    fig.savefig(out / (pathlib.Path(path).stem + ".png"), bbox_inches="tight"); plt.close(fig)
    print(pathlib.Path(path).name, info)
