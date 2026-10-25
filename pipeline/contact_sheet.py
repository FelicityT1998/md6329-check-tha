"""Contact sheet of radiographs with captions. usage: python -I contact_sheet.py <out.png> <ncols> <image>[::caption] ..."""
import sys, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from render_lib import load

out, ncol, items = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
nrow = -(-len(items) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 3.6), dpi=110)
for ax in np.ravel(axes): ax.axis("off")
for ax, it in zip(np.ravel(axes), items):
    path, _, cap = it.partition("::")
    a, info = load(path)
    a = a[:: max(1, max(a.shape) // 900), :: max(1, max(a.shape) // 900)]
    lo, hi = np.percentile(a, [1, 99])
    ax.imshow(np.clip((a - lo) / (hi - lo), 0, 1), cmap="gray")
    ax.set_title(f"{pathlib.Path(path).stem}\n{cap}", fontsize=8)
fig.tight_layout(); fig.savefig(out)
