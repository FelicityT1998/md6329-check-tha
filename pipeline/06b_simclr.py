"""SimCLR continued pre-training of ImageNet ResNet-18 on the hip crops, with catastrophic-forgetting guards (plan 7.9.5).
Meant for Colab (GPU); runs on CPU too but slowly.

usage: python 06b_simclr.py <crop_dir> <out_dir> [--hips hips.parquet --folds folds.csv] [--epochs 100] [--bs 256]
writes: <out_dir>/simclr_ep{N}.pt (backbone state_dict, every 10 epochs), simclr_log.csv, simclr_best.pt

Guards against forgetting the ImageNet representation:
  - start from ImageNet weights; stem + layer1 frozen; layer-wise LR (layer2 0.25x, layer3 0.5x, layer4 1x of base 1e-4 ... projection head 10x)
  - L2-SP penalty alpha * ||theta - theta_ImageNet||^2 on trainable backbone weights (alpha = 1e-2)
  - monitor every 10 epochs: linear CKA between current and ImageNet features on 256 fixed crops, and (if labels are given)
    a linear probe on the frozen embedding: 3-fold inner CV inside repeat-0 / outer-0 TRAINING subjects only, log-loss.
    simclr_best.pt = checkpoint with the best probe log-loss (falls back to the last one without labels).
Augmentations: random resized crop (0.5-1.0), rotation +-10 deg, brightness/contrast/gamma, gaussian blur, gaussian noise.
No horizontal flip (all crops are in right-hip orientation; flipping would change anatomy direction).
"""
import argparse, pathlib, math, copy, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("crops"); ap.add_argument("out")
ap.add_argument("--hips"); ap.add_argument("--folds"); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--alpha", type=float, default=1e-2); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--tau", type=float, default=0.2)
A = ap.parse_args(); OUT = pathlib.Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)

files = sorted(pathlib.Path(A.crops).glob("*.png"))
X = torch.stack([torch.from_numpy(np.asarray(Image.open(f), dtype=np.float32) / 65535.0) for f in files])[:, None]   # (N,1,256,256)
keys = [f.stem for f in files]; print(len(X), "crops on", dev)
MEAN, STD = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None], torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
norm = lambda x: (x.repeat(1, 3, 1, 1) - MEAN) / STD

def augment(x):                                  # x (B,1,256,256) on device -> (B,1,224,224)
    B = len(x); s = torch.empty(B, device=dev).uniform_(0.7, 1.0); a = torch.empty(B, device=dev).uniform_(-10, 10) * math.pi / 180
    t = torch.empty(B, 2, device=dev).uniform_(-0.15, 0.15) * (1 - s[:, None])
    th = torch.stack([torch.stack([s * a.cos(), -s * a.sin(), t[:, 0]], 1), torch.stack([s * a.sin(), s * a.cos(), t[:, 1]], 1)], 1)
    y = Fn.grid_sample(x, Fn.affine_grid(th, (B, 1, 224, 224), align_corners=False), padding_mode="reflection", align_corners=False)
    y = y.clamp(0, 1) ** torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.4)
    y = (y - 0.5) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.3) + 0.5 + torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.1, 0.1)
    if np.random.rand() < 0.5: y = torchvision.transforms.functional.gaussian_blur(y, 9, float(np.random.uniform(0.1, 2.0)))
    return (y + torch.randn_like(y) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.04)).clamp(0, 1)

net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); net.fc = nn.Identity(); net = net.to(dev)
ref = copy.deepcopy(net).eval()
for m in (net.conv1, net.bn1, net.layer1):
    for p in m.parameters(): p.requires_grad = False
theta0 = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)).to(dev)
groups = [{"params": net.layer2.parameters(), "lr": A.lr * 0.25}, {"params": net.layer3.parameters(), "lr": A.lr * 0.5},
          {"params": net.layer4.parameters(), "lr": A.lr}, {"params": proj.parameters(), "lr": A.lr * 10}]
opt = torch.optim.AdamW(groups, weight_decay=0.0)   # decay replaced by L2-SP
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.epochs)

def nt_xent(z1, z2, tau):
    z = Fn.normalize(torch.cat([z1, z2]), dim=1); s = z @ z.t() / tau; n = len(z1)
    s.fill_diagonal_(-1e9); tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
    return Fn.cross_entropy(s, tgt)

def embed(model, idx, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            x = Fn.interpolate(X[idx[b:b + bs]].to(dev), size=(224, 224), mode="area"); out.append(model(norm(x)).cpu())
    return torch.cat(out).numpy()

def cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))

fixed = np.random.default_rng(0).choice(len(X), min(256, len(X)), replace=False); ref_feat = embed(ref, fixed)
probe = None
if A.hips and A.folds:   # linear probe on training subjects of repeat 0 / outer 0 only
    from sklearn.linear_model import LogisticRegression; from sklearn.metrics import log_loss; from sklearn.decomposition import PCA
    H = pd.read_parquet(A.hips); F = pd.read_csv(A.folds); F0 = F[F.repeat == 0].set_index("nsin")
    H["key"] = H.nsin.astype(str) + "_" + H.side; H = H[H.y10.notna() & H.key.isin(keys) & ~H.nsin.map(F0.outer_fold).eq(0)]
    pidx = np.array([keys.index(k) for k in H.key]); py = H.y10.values; pf = H.nsin.map(F0.inner_for_o0).values
    def probe(model):
        Z = embed(model, pidx); ps = np.zeros(len(Z))
        for j in range(3):
            tr, va = pf != j, pf == j
            pca = PCA(32).fit(Z[tr]); m = LogisticRegression(C=0.1, max_iter=2000).fit(pca.transform(Z[tr]), py[tr])
            ps[va] = m.predict_proba(pca.transform(Z[va]))[:, 1]
        return log_loss(py, ps)

log, best = [], (np.inf, None)
if probe: log.append({"epoch": 0, "loss": np.nan, "cka": 1.0, "probe_logloss": probe(net)}); print(log[-1])
for ep in range(1, A.epochs + 1):
    net.train(); net.layer1.eval(); net.bn1.eval(); perm = torch.randperm(len(X)); tot, t0 = 0.0, time.time()
    for b in range(0, len(perm) - A.bs // 2, A.bs):
        x = X[perm[b:b + A.bs]].to(dev)
        z1, z2 = proj(net(norm(augment(x)))), proj(net(norm(augment(x))))
        l2sp = sum(((p - theta0[n]) ** 2).sum() for n, p in net.named_parameters() if p.requires_grad)
        loss = nt_xent(z1, z2, A.tau) + A.alpha * l2sp
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    rec = {"epoch": ep, "loss": tot / max(1, len(perm) // A.bs), "sec": round(time.time() - t0)}
    if ep % 10 == 0 or ep == A.epochs:
        rec["cka"] = cka(embed(net, fixed), ref_feat)
        if probe: rec["probe_logloss"] = probe(net)
        torch.save(net.state_dict(), OUT / f"simclr_ep{ep}.pt")
        score = rec.get("probe_logloss", -ep)
        if score < best[0]: best = (score, ep); torch.save(net.state_dict(), OUT / "simclr_best.pt")
    log.append(rec); print(rec); pd.DataFrame(log).to_csv(OUT / "simclr_log.csv", index=False)
print("best epoch", best[1])
