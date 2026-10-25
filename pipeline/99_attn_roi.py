"""Where does the image head look, and how good is the femoral-head locator as a region? (2026-10-25)

usage: python -I 99_attn_roi.py <project dir>
reads : landmarks/locator_cv2.csv (3-fold subject CV of the locator), imghead/head_res_o{k}.pt (repeat-0 full-fit heads),
        D:/datasets/CHECK/fmaps/densenet_b3.npy, oof/m3_head_res_-.parquet
writes: results/locator_overlap.csv, results/attn_roi.csv

1. Locator as a region: the locator predicts a circle (centre + radius) per femoral head. Against the annotated circle, the
   overlap of two circles gives IoU and Dice (closed form from centre distance and the two radii).
2. Attention mass inside the joint disc: share of attention-pooling weight inside 1.6 head radii of the crop centre (the
   same disc as the res_roi head in 08_imghead.py), by outcome group on repeat-0 test hips.
"""
import sys, pathlib
import numpy as np, pandas as pd, torch
P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; sys.path.insert(0, str(P / "lectrace"))
from _helpers import Head

# ---------------------------------------------------------------- 1. locator overlap
def circle_overlap(d, r1, r2):
    if d >= r1 + r2: return 0.0
    if d <= abs(r1 - r2): return np.pi * min(r1, r2) ** 2
    a = r1 ** 2 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1)) + r2 ** 2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    return a - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

E = pd.read_csv(D / "landmarks/locator_cv2.csv"); E = E[E.present == 1]
out = []
for dec, g in E.groupby("decode"):
    inter = np.array([circle_overlap(d, 1.0, 1.0 + rr) for d, rr in zip(g.err_r, g.rad_rel_err)])
    a1, a2 = np.pi, np.pi * (1 + g.rad_rel_err.values) ** 2
    iou = inter / (a1 + a2 - inter); dice = 2 * inter / (a1 + a2)
    out.append(dict(decode=dec, heads=len(g), iou_median=np.median(iou), iou_mean=iou.mean(), dice_median=np.median(dice),
                    dice_mean=dice.mean(), share_dice_ge_0_8=(dice >= 0.8).mean(), centre_err_median_r=g.err_r.median()))
L = pd.DataFrame(out); L.to_csv(D / "results/locator_overlap.csv", index=False); print(L.round(3).to_string(index=False))

# ---------------------------------------------------------------- 2. attention mass in the joint disc
FM = pathlib.Path(r"D:\datasets\CHECK\fmaps")
idx = pd.read_csv(FM / "densenet_b3_index.csv"); M = np.load(FM / "densenet_b3.npy", mmap_mode="r")
pos = {(n, s): i for i, (n, s) in enumerate(zip(idx.nsin, idx.side))}
yy, xx = np.mgrid[0:14, 0:14]; ROI = (np.hypot(yy - 6.5, xx - 6.5) <= 1.6 * 2.8).reshape(-1)
o = pd.read_parquet(D / "oof/m3_head_res_-.parquet").query("repeat == 0")
o["top10"] = o.p_cal >= o.p_cal.quantile(0.90)
rows = []
for k in range(5):
    net = Head(); net.load_state_dict(torch.load(D / f"imghead/head_res_o{k}.pt", map_location="cpu")); net.eval()
    t = o[o.outer == k]
    X = torch.from_numpy(np.stack([M[pos[(n, s)]] for n, s in zip(t.nsin, t.side)]).astype(np.float32))
    with torch.no_grad():
        z = net.blocks(net.reduce(X)); a = torch.softmax(net.att(z).flatten(1), 1).numpy()
    rows.append(t.assign(roi_mass=a[:, ROI].sum(1), peak_in_roi=ROI[a.argmax(1)]))
A = pd.concat(rows)
A["group"] = np.select([(A.y == 1) & A.top10, (A.y == 1) & ~A.top10, (A.y == 0) & A.top10], ["THA, ranked top 10%", "THA, missed", "no THA, flagged"], "no THA, not flagged")
S = A.groupby("group").agg(hips=("y", "size"), roi_mass_mean=("roi_mass", "mean"), roi_mass_median=("roi_mass", "median"), peak_in_roi=("peak_in_roi", "mean"))
S.loc["uniform attention (reference)"] = [np.nan, ROI.mean(), ROI.mean(), ROI.mean()]
S.to_csv(D / "results/attn_roi.csv"); print(S.round(3).to_string())
