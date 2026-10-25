"""Film orientation (0/90/180/270 deg) before femoral-head localisation.

Why: ~100 baseline AP pelvis films (mostly centre C06 TIFFs) are stored rotated by 90 deg (hips stacked vertically, spine
on the left), so 'screen-left hip = patient right' fails for them. Annotation (05a) showed 14/118 labelled pelvis films
rotated. There is no orientation tag in the TIFFs.

Method (RotNet-style self-supervision, no extra labels): films presumed upright = landscape whole-pelvis films + single-hip
films; each is rotated by a random k*90 deg and a ResNet-18 learns to predict k. Applied to every film, the predicted k is
the rotation that brings the film back upright. Validation: 20% of presumed-upright films held out (accuracy on all four
synthetic rotations) and the 14 rotated annotated films (their hip axis must become horizontal after correction).

usage: python -I 05r_orient.py <project dir>
writes: data/derived/landmarks/cache/<stem>.pt (384 letterbox, shared with 05b_locator), data/derived/landmarks/orient.csv
        (path, k_upright = number of np.rot90 / torch.rot90 counter-clockwise turns that make the film upright, p0..p3)
"""
import sys, pathlib, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
from joblib import Parallel, delayed

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; L = D / "landmarks"; CA = L / "cache"; CA.mkdir(exist_ok=True)
sys.path.insert(0, str(P))
I = pd.read_csv(D / "images.csv")


def cache_one(path):
    f = CA / (pathlib.Path(path).stem + ".pt")
    if f.exists(): return
    sys.path.insert(0, str(P))
    from render_lib import load
    from lib_img import letterbox
    torch.save(letterbox(load(path)[0]), f)


if __name__ == "__main__":
    t = time.perf_counter()
    Parallel(n_jobs=10, verbose=2)(delayed(cache_one)(p) for p in I.path)
    print(f"cache done {time.perf_counter() - t:.0f}s")
    torch.set_num_threads(12); torch.manual_seed(0); rng = np.random.default_rng(0)
    X = torch.stack([Fn.avg_pool2d(torch.load(CA / (pathlib.Path(p).stem + ".pt"))[0][None, None], 2)[0, 0] for p in I.path])  # (N,192,192)
    upright = ((I.view == "O") & (I.Columns > I.Rows)) | (I.view != "O")
    idx_up = np.where(upright)[0]; rng.shuffle(idx_up); n_va = len(idx_up) // 5
    va, tr = idx_up[:n_va], idx_up[n_va:]

    net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); net.fc = nn.Linear(512, 4)
    prep = lambda x: (x[:, None].repeat(1, 3, 1, 1) - 0.449) / 0.226
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); EP = 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * int(np.ceil(len(tr) / 32)))
    for ep in range(EP):
        net.train(); perm = rng.permutation(tr); tot = 0
        for b in range(0, len(perm), 32):
            xb = X[perm[b:b + 32]]; k = torch.from_numpy(rng.integers(0, 4, len(xb)))
            xb = torch.stack([torch.rot90(x, int(kk), (0, 1)) for x, kk in zip(xb, k)])
            xb = (xb.clamp(0, 1) ** float(rng.uniform(0.7, 1.4)))
            loss = Fn.cross_entropy(net(prep(xb)), k); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(xb)
        print(f"epoch {ep + 1} loss {tot / len(tr):.4f}")
    net.eval()
    with torch.no_grad():
        acc = []
        for k in range(4):
            pr = torch.cat([net(prep(torch.rot90(X[va[b:b + 64]], k, (1, 2)))).argmax(1) for b in range(0, len(va), 64)])
            acc.append(float((pr == k).float().mean()))
        print("held-out synthetic-rotation accuracy per k:", np.round(acc, 3))
        logits = torch.cat([net(prep(X[b:b + 64])) for b in range(0, len(X), 64)])
    pr = logits.softmax(1).numpy()
    # the classifier predicts the rotation k that was APPLIED to an upright film; undoing it needs (4 - k) % 4 more turns
    k_app = pr.argmax(1); O = pd.DataFrame({"path": I.path, "nsin": I.nsin, "view": I.view, "fmt": I.fmt, "k_applied": k_app,
                                            "k_upright": (4 - k_app) % 4, "conf": pr.max(1)})
    for j in range(4): O[f"p{j}"] = pr[:, j]
    O["portrait"] = (I.Rows > I.Columns).values
    O.to_csv(L / "orient.csv", index=False)
    print(pd.crosstab([O.view, O.portrait], O.k_upright))
    print("low-confidence (<0.8):", int((O.conf < 0.8).sum()))
    # check on annotated rotated films: after correction the hip axis must be horizontal
    A = pd.read_csv(L / "landmarks.csv").drop_duplicates("path", keep="last").merge(O[["path", "k_upright"]], on="path")
    o = A[(A.view == "O") & A.sl_x.notna() & A.sr_x.notna()]
    vert = (o.sr_x - o.sl_x).abs() / o.W < 0.2
    print("annotated pelvis films: vertical hip axis", int(vert.sum()), "| of those corrected by k=1/3:",
          int(o[vert].k_upright.isin([1, 3]).sum()), "| horizontal films with k!=0:", int((o[~vert].k_upright != 0).sum()))
