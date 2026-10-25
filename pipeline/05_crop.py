"""Hip crops from baseline AP films using the locator output (plan 7.4 steps 3-9).

usage: python -I 05_crop.py <project dir> [crop_dir=D:\\datasets\\CHECK\\crops]
reads : data/derived/landmarks/pred.csv (05b_locator predict), landmarks.csv (manual labels override predictions),
        data/derived/images.csv, hips.parquet
writes: <crop_dir>/{nsin}_{side}.png (256x256, 16-bit), data/derived/crops.csv, data/derived/landmarks/qc_*.png

Side mapping: the locator already outputs patient sides (R_*, L_*) after turning each film upright (05r_orient.py,
radiological convention: patient RIGHT = screen-LEFT once upright). Manual labels are mapped the same way. The annotator
never used the 'flipped' flag, so no film is treated as mirrored.
Film choice per hip: whole-pelvis film (APO) first, else the single-hip film of that side; if a subject has several
films of the same view, the one with the highest locator confidence for that side.
Crop: square of side 2.5 x femoral-head diameter (= 5 r) centred on the head, area-resampled to 256x256; intensities
clipped to the 1st/99th percentile inside the crop and scaled to [0, 65535]. Patient-LEFT hips are mirrored so every
crop has the lateral side on the left (right-hip orientation).
"""
import sys, pathlib
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from PIL import Image

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; L = D / "landmarks"
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(r"D:\datasets\CHECK\crops"); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P))
from render_lib import load
CONF_MIN, SIZE, BOX = 0.3, 256, 5.0

pred = pd.read_csv(L / "pred.csv")
man = pd.read_csv(L / "landmarks.csv").drop_duplicates("path", keep="last") if (L / "landmarks.csv").exists() else pd.DataFrame(columns=["path"])
img = pd.read_csv(D / "images.csv")
H = pd.read_parquet(D / "hips.parquet")[["nsin", "side", "centre"]]

def upright_x(x, y, k):
    """ordering key = x-position after k counter-clockwise turns (only the order matters)."""
    return [x, y, -x, -y][k % 4]

# manual labels override predictions (conf = 1); screen order -> patient side in the upright frame
pred["source"] = "model"
for r in man.itertuples():
    i = pred.index[pred.path == r.path]
    if len(i) == 0: continue
    k = int(pred.loc[i[0], "k_upright"])
    hips = [(getattr(r, f"{s}_x"), getattr(r, f"{s}_y"), getattr(r, f"{s}_r")) for s in ("sl", "sr") if pd.notna(getattr(r, f"{s}_x"))]
    if r.view == "O" and len(hips) == 2:
        hips.sort(key=lambda h: upright_x(h[0], h[1], k)); lab = {"R": hips[0], "L": hips[1]}
    elif r.view == "O":
        lab = {}   # single usable hip on a pelvis film: keep the model's prediction for both sides
    else:
        lab = {r.view: hips[0]} if hips else {}
    for side, h in lab.items():
        pred.loc[i, [f"{side}_x", f"{side}_y", f"{side}_r", f"{side}_conf"]] = [*h, 1.0]
    if lab: pred.loc[i, "source"] = "manual"
pred = pred.merge(H.drop_duplicates("nsin")[["nsin", "centre"]], on="nsin", how="left")

# candidate (film, patient side) for each hip; single-hip films only offer their own side
cand = []
for r in pred.itertuples():
    for side in ("R", "L") if r.view == "O" else (r.view,):
        x = getattr(r, f"{side}_x")
        if pd.isna(x): continue
        cand.append({"nsin": r.nsin, "side": side, "view": r.view, "path": r.path, "fmt": r.fmt, "x": x, "y": getattr(r, f"{side}_y"),
                     "r": getattr(r, f"{side}_r"), "conf": getattr(r, f"{side}_conf"), "source": r.source, "k_upright": r.k_upright})
C = pd.DataFrame(cand)
C["view_rank"] = (C.view != "O").astype(int)
C = C.sort_values(["nsin", "side", "view_rank", "conf"], ascending=[True, True, True, False]).drop_duplicates(["nsin", "side"])

rows, cache = [], {}
for k, r in enumerate(C.itertuples()):
    if r.path not in cache: cache.clear(); cache[r.path] = load(r.path)[0]
    a = cache[r.path]; half = BOX * r.r / 2
    x0, y0, x1, y1 = int(round(r.x - half)), int(round(r.y - half)), int(round(r.x + half)), int(round(r.y + half))
    pad = max(0, -x0, -y0, x1 - a.shape[1], y1 - a.shape[0])
    a_p = np.pad(a, pad, mode="constant", constant_values=float(np.median(a))) if pad else a
    crop = np.ascontiguousarray(np.rot90(a_p[y0 + pad:y1 + pad, x0 + pad:x1 + pad], int(r.k_upright))).astype(np.float32)
    t = Fn.interpolate(torch.from_numpy(crop)[None, None], size=(SIZE, SIZE), mode="area")[0, 0].numpy()
    lo, hi = np.percentile(t, [1, 99]); t = np.clip((t - lo) / (hi - lo + 1e-6), 0, 1)
    if r.side == "L": t = t[:, ::-1]
    f = OUT / f"{r.nsin}_{r.side}.png"
    Image.fromarray((t * 65535).astype(np.uint16)).save(f)
    rows.append({"nsin": r.nsin, "side": r.side, "crop": str(f), "src_path": r.path, "view": r.view, "fmt": r.fmt, "cx": r.x, "cy": r.y,
                 "head_r_px": r.r, "conf": r.conf, "source": r.source, "k_upright": r.k_upright, "padded": pad > 0,
                 "qc_flag": (r.conf < CONF_MIN) or pad > 0 or not (0.03 < r.r / max(a.shape) < 0.11)})   # annotated: 0.054-0.088
    if k % 200 == 0: print(k, "/", len(C))
CR = pd.DataFrame(rows); CR.to_csv(D / "crops.csv", index=False)
missing = H.merge(CR[["nsin", "side"]], how="left", indicator=True).query("_merge == 'left_only'")
print(f"crops: {len(CR)} | flagged for QC: {int(CR.qc_flag.sum())} | hips without crop: {len(missing)}")

# QC contact sheets: flagged first, then a random sample per centre
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CR = CR.merge(H.drop_duplicates("nsin")[["nsin", "centre"]], on="nsin", how="left")
sheets = [("qc_flagged", CR[CR.qc_flag])] + [(f"qc_{c}", g.sample(min(30, len(g)), random_state=0)) for c, g in CR.groupby("centre")]
for name, g in sheets:
    if g.empty: continue
    n = len(g); nc = 10; nr = -(-n // nc)
    for page in range(0, nr, 6):
        sub = g.iloc[page * nc:(page + 6) * nc]; rr = -(-len(sub) // nc)
        fig, axs = plt.subplots(rr, nc, figsize=(nc * 1.6, rr * 1.75))
        for ax in np.ravel(axs): ax.axis("off")
        for ax, r in zip(np.ravel(axs), sub.itertuples()):
            ax.imshow(np.asarray(Image.open(r.crop)), cmap="gray")
            ax.set_title(f"{r.nsin}{r.side} {r.conf:.2f}", fontsize=6, color="#d03b3b" if r.qc_flag else "#52514e")
        fig.tight_layout(); fig.savefig(L / f"{name}_{page // 6}.png", dpi=110); plt.close(fig)
print("QC sheets ->", L)
