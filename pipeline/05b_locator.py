"""Femoral-head locator trained on the click annotations (05a_annotate.py), then applied to every baseline AP film.

usage:
  python -I 05b_locator.py <project dir> train      -> data/derived/landmarks/locator.pt + cv report (3-fold by subject)
  python -I 05b_locator.py <project dir> predict    -> data/derived/landmarks/pred.csv (one row per film: sl/sr centre, radius, confidence)

Model: ResNet-18 (ImageNet) trunk on the whole film letterboxed to 384x384, U-shaped upsampling to stride 4,
three output maps: heatmap of the PATIENT-RIGHT head centre, heatmap of the PATIENT-LEFT head centre, and a radius map
(regressed at the true centre only). Single-hip films (APL/APR) and skipped hips simply have no target for that side;
a "presence" logit per side is read from the heatmap maximum.
Orientation (2026-10-22): every film is first turned upright with the k from 05r_orient.py (orient.csv); in an upright
film the patient-right head is the screen-left one (radiological convention), so annotated pelvis films get their channel
by x-position after rotation (this also repairs the 4 films where the screen-right hip was clicked first); single-hip films
get the channel of their view. Predictions are rotated back to original pixel coordinates.
Augmentation: rotation +-8 deg, scale 0.85-1.15, translation, gamma / contrast, gaussian noise; NO horizontal flip
(left / right identity must be preserved).
"""
import sys, pathlib, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; L = D / "landmarks"; MODE = sys.argv[2]
sys.path.insert(0, str(P))
from render_lib import load
from lib_img import letterbox, rot_pt, unrot_pt, S
torch.set_num_threads(12); torch.manual_seed(0); np.random.seed(0)
STRIDE, SIG = 4, 2.0            # output stride, heatmap sigma in output pixels
ORI = pd.read_csv(L / "orient.csv").set_index("path").k_upright


def cached(path):
    f = L / "cache" / (pathlib.Path(path).stem + ".pt")
    if f.exists(): return torch.load(f)
    out = letterbox(load(path)[0]); torch.save(out, f); return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool); self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        up = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.u3, self.u2, self.u1 = up(512 + 256, 128), up(128 + 128, 64), up(64 + 64, 64)
        self.head = nn.Conv2d(64, 3, 1)       # heat_sl, heat_sr, radius
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1); x = (x - 0.449) / 0.226
        x0 = self.stem(x); x1 = self.l1(x0); x2 = self.l2(x1); x3 = self.l3(x2); x4 = self.l4(x3)
        y = self.u3(torch.cat([Fn.interpolate(x4, scale_factor=2), x3], 1))
        y = self.u2(torch.cat([Fn.interpolate(y, scale_factor=2), x2], 1))
        y = self.u1(torch.cat([Fn.interpolate(y, scale_factor=2), x1], 1))
        return self.head(y)                    # (B,3,S/4,S/4)

def augment(img, pts):
    """img (S,S) tensor, pts list of (x,y,r) or None in letterbox pixels -> random affine applied to both."""
    ang = math.radians(np.random.uniform(-8, 8)); sc = np.random.uniform(0.85, 1.15); tx, ty = np.random.uniform(-25, 25, 2)
    c, s_ = math.cos(ang) * sc, math.sin(ang) * sc
    A = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0]], dtype=torch.float32)  # in normalised coords
    A[:, 2] = torch.tensor([tx, ty]) / (S / 2)
    grid = Fn.affine_grid(torch.linalg.inv(torch.vstack([A, torch.tensor([0, 0, 1.0])]))[:2][None], (1, 1, S, S), align_corners=False)
    out = Fn.grid_sample(img[None, None], grid, align_corners=False)[0, 0]
    g = np.random.uniform(0.7, 1.4); out = out.clamp(0, 1) ** g
    out = (out - 0.5) * np.random.uniform(0.8, 1.25) + 0.5 + np.random.uniform(-0.08, 0.08)
    out = (out + torch.randn_like(out) * np.random.uniform(0, 0.03)).clamp(0, 1)
    new = []
    for p in pts:
        if p is None: new.append(None); continue
        x, y, r = p; u, v = (x - S / 2) / (S / 2), (y - S / 2) / (S / 2)
        u2, v2 = c * u - s_ * v + A[0, 2].item(), s_ * u + c * v + A[1, 2].item()
        new.append((u2 * S / 2 + S / 2, v2 * S / 2 + S / 2, r * sc))
    return out, new

def targets(pts):
    G = S // STRIDE; yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    T = torch.zeros(3, G, G); M = torch.zeros(3)  # M = which channels have supervision
    rad_mask = torch.zeros(G, G)
    for k, p in enumerate(pts):
        if p is None: continue
        x, y, r = p[0] / STRIDE, p[1] / STRIDE, p[2]
        T[k] = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIG ** 2)); M[k] = 1
        near = ((xx - x) ** 2 + (yy - y) ** 2) <= 4
        T[2][near] = r / S; rad_mask[near] = 1
    return T, M, rad_mask

def load_annot():
    A = pd.read_csv(L / "landmarks.csv").drop_duplicates("path", keep="last")
    A = A[A[["sl_x", "sr_x"]].notna().any(axis=1)]
    (L / "cache").mkdir(exist_ok=True); X, PTS = [], []
    for r in A.itertuples():
        img, s, ox, oy = cached(r.path); k = int(ORI[r.path])
        img = torch.rot90(img, k, (0, 1))
        pts = []
        for side in ("sl", "sr"):
            x, y, rr = getattr(r, f"{side}_x"), getattr(r, f"{side}_y"), getattr(r, f"{side}_r")
            pts.append(None if pd.isna(x) else (*rot_pt(x * s + ox, y * s + oy, k), rr * s))
        hips = [p for p in pts if p is not None]
        if r.view == "O":   # channel 0 = patient right = smaller x once upright
            hips.sort(key=lambda p: p[0]); pts = hips + [None] * (2 - len(hips)) if len(hips) == 2 else [None, None]
            if len(hips) == 1:   # one usable hip on a pelvis film: side from its position relative to the image centre
                pts = [hips[0], None] if hips[0][0] < S / 2 else [None, hips[0]]
        else:
            pts = [hips[0], None] if r.view == "R" else [None, hips[0]]
        X.append(img); PTS.append(pts)
    return A.reset_index(drop=True), X, PTS

def decode(out):
    """(3,G,G) logits -> per side (x, y, r, conf) in letterbox pixels."""
    res = []
    for k in range(2):
        h = torch.sigmoid(out[k]); idx = int(h.argmax()); G = h.shape[1]; y, x = divmod(idx, G)
        # sub-pixel refinement: soft-argmax in a 5x5 window
        y0, y1, x0, x1 = max(0, y - 2), min(G, y + 3), max(0, x - 2), min(G, x + 3)
        w = h[y0:y1, x0:x1]; yy, xx = torch.meshgrid(torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij")
        xs, ys = float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())
        res.append((xs * STRIDE + STRIDE / 2, ys * STRIDE + STRIDE / 2, float(out[2, y, x]) * S, float(h.max())))
    return res

def decode_pair(out, view):
    """geometry-based decoding (2026-10-22): the two side channels are merged into one femoral-head map, the top peak is
    taken, a disc of 3 head radii around it is suppressed, and the second peak is taken; on a pelvis film the two heads are
    assigned to patient right / left by their x-order in the upright frame, on a single-hip film the top peak goes to the
    film's side. Fixes the channel confusion of decode() (15% of hips > 1 radius off in CV, both channels on one head)."""
    h = torch.sigmoid(torch.maximum(out[0], out[1])).clone(); G = h.shape[1]
    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"); peaks = []
    for _ in range(2 if view == "O" else 1):
        idx = int(h.argmax()); y, x = divmod(idx, G); conf = float(h.max())
        y0, y1, x0, x1 = max(0, y - 2), min(G, y + 3), max(0, x - 2), min(G, x + 3)
        w = h[y0:y1, x0:x1]; sy, sx = torch.meshgrid(torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij")
        xs, ys = float((w * sx).sum() / w.sum()), float((w * sy).sum() / w.sum())
        r = float(out[2, y, x]) * S
        peaks.append((xs * STRIDE + STRIDE / 2, ys * STRIDE + STRIDE / 2, r, conf))
        h[((xx - x) ** 2 + (yy - y) ** 2) <= (3 * max(r, 8) / STRIDE) ** 2] = 0
    if view == "O":
        peaks.sort(key=lambda p: p[0]); return peaks
    nan = (np.nan, np.nan, np.nan, 0.0)
    return [peaks[0], nan] if view == "R" else [nan, peaks[0]]


def train(X, PTS, idx, epochs=60, log=None):
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * math.ceil(len(idx) / 8))
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx); tot = 0
        for b in range(0, len(perm), 8):
            batch = [augment(X[i], PTS[i]) for i in perm[b:b + 8]]
            xb = torch.stack([im for im, _ in batch])[:, None]
            tg = [targets(p) for _, p in batch]
            T = torch.stack([t[0] for t in tg]); M = torch.stack([t[1] for t in tg]); RM = torch.stack([t[2] for t in tg])
            out = net(xb)
            # focal-style heatmap loss on supervised channels; absent sides are trained towards an all-zero map
            hm = Fn.binary_cross_entropy_with_logits(out[:, :2], T[:, :2], reduction="none")
            wpos = 1 + 50 * T[:, :2]
            l_h = (hm * wpos).mean()
            l_r = ((out[:, 2] - T[:, 2]).abs() * RM).sum() / RM.sum().clamp(min=1)
            loss = l_h + 5 * l_r
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if log and (ep % 10 == 9 or ep == epochs - 1): log(f"  epoch {ep + 1}/{epochs} loss {tot / math.ceil(len(idx) / 8):.4f}")
    return net

if MODE in ("train", "cv"):   # cv = re-run only the 3-fold CV (both decoders), keep locator.pt
    A, X, PTS = load_annot(); n = len(A); print(f"{n} annotated films")
    # 3-fold CV by subject: centre error in units of the annotated head radius
    subj = A.nsin.unique(); np.random.shuffle(subj); fold = {s: i % 3 for i, s in enumerate(subj)}
    errs = []
    for k in range(3):
        te = [i for i in range(n) if fold[A.nsin[i]] == k]; tr = [i for i in range(n) if fold[A.nsin[i]] != k]
        t = time.perf_counter(); net = train(X, PTS, tr, epochs=25); net.eval()
        with torch.no_grad():
            for i in te:
                o = net(X[i][None, None])[0]
                for dec, pr in (("channel", decode(o)), ("pair", decode_pair(o, A.view[i]))):
                    for j, p in enumerate(PTS[i]):
                        if p is None: errs.append({"film": i, "view": A.view[i], "decode": dec, "side": j, "present": 0, "conf": pr[j][3]}); continue
                        d = math.hypot(pr[j][0] - p[0], pr[j][1] - p[1])
                        errs.append({"film": i, "view": A.view[i], "decode": dec, "side": j, "present": 1, "conf": pr[j][3],
                                     "err_r": d / p[2], "rad_rel_err": pr[j][2] / p[2] - 1})
        print(f"fold {k}: {len(tr)} train / {len(te)} test films, {time.perf_counter() - t:.0f}s")
    E = pd.DataFrame(errs); E.to_csv(L / ("locator_cv.csv" if MODE == "train" else "locator_cv2.csv"), index=False)
    for dec, g in E[E.present == 1].groupby("decode"):
        print(f"[{dec}] CV centre error / radius: median {g.err_r.median():.3f}, 90th pct {g.err_r.quantile(.9):.3f}, "
              f">0.5 radius: {(g.err_r > 0.5).mean():.1%}, >1 radius: {(g.err_r > 1).mean():.1%} | radius rel. error median {g.rad_rel_err.abs().median():.3f}")
    if MODE == "train":
        net = train(X, PTS, list(range(n)), epochs=50, log=print)
        torch.save(net.state_dict(), L / "locator.pt"); print("saved", L / "locator.pt")

elif MODE == "predict":
    net = Net(); net.load_state_dict(torch.load(L / "locator.pt")); net.eval()
    img = pd.read_csv(D / "images.csv"); rows = []; t = time.perf_counter()
    with torch.no_grad():
        for i, r in enumerate(img.itertuples()):
            im, s, ox, oy = cached(r.path); k = int(ORI[r.path])
            pr = decode_pair(net(torch.rot90(im, k, (0, 1))[None, None])[0], r.view)
            rec = {"nsin": r.nsin, "view": r.view, "fmt": r.fmt, "path": r.path, "k_upright": k}
            for side, (x, y, rad, conf) in zip(("R", "L"), pr):   # patient sides
                x, y = unrot_pt(x, y, k)
                rec.update({f"{side}_x": (x - ox) / s, f"{side}_y": (y - oy) / s, f"{side}_r": rad / s, f"{side}_conf": conf})
            rows.append(rec)
            if i % 100 == 0: print(i, f"{time.perf_counter() - t:.0f}s")
    pd.DataFrame(rows).to_csv(L / "pred.csv", index=False); print("saved", L / "pred.csv")
