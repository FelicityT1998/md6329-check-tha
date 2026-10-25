"""Image-branch inputs for the trainable residual head (08_imghead.py) and the AdaBN experiment (2026-10-23).

usage: python -I 07_fmaps.py <project dir> [fmap_dir=D:\\datasets\\CHECK\\fmaps]
reads : data/derived/crops.csv (qc_exclude rows skipped)
writes: <fmap_dir>/densenet_b3.npy   (n_hips, 1024, 14, 14) float16 - DenseNet-121 (ImageNet, frozen, eval-mode BN) up to
                                      denseblock3 at 224x224: mid-level maps that still keep 14x14 spatial detail of the joint
        <fmap_dir>/densenet_b3_index.csv
        data/derived/emb/emb_densenet121_adabn.npy (+ _index.csv) - same global embedding as 06_embed densenet121 but with
                                      AdaBN: every BatchNorm's running mean/var re-estimated on our hip crops (cumulative
                                      average, one pass, no labels) before embedding. Tests whether ImageNet BN statistics
                                      mismatch radiographs. Transductive (uses all crops, unlabeled) - disclosed in the report.
"""
import sys, pathlib, os
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn, torchvision
from PIL import Image

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; E = D / "emb"
FM = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(r"D:\datasets\CHECK\fmaps"); FM.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("EMB_THREADS", 12)))
C = pd.read_csv(D / "crops.csv"); C = C[~C.qc_exclude].reset_index(drop=True)
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def load(i):
    x = torch.from_numpy(np.asarray(Image.open(C.crop[i]), dtype=np.float32) / 65535.0)[None]
    x = Fn.interpolate(x[None], size=(224, 224), mode="area")[0]
    return (x.repeat(3, 1, 1) - MEAN) / STD


X = torch.stack([load(i) for i in range(len(C))]); print("crops", X.shape, flush=True)
net = torchvision.models.densenet121(weights="IMAGENET1K_V1").eval()

# (1) block-3 feature maps, ImageNet BN statistics
trunk = torch.nn.Sequential(*list(net.features.children())[:9])     # conv0 ... denseblock3
with torch.no_grad():
    maps = torch.cat([trunk(X[b:b + 32]).half() for b in range(0, len(X), 32)])
np.save(FM / "densenet_b3.npy", maps.numpy()); C[["nsin", "side", "fmt"]].to_csv(FM / "densenet_b3_index.csv", index=False)
print("maps", tuple(maps.shape), flush=True); del maps

# (2) AdaBN global embedding
for m in net.modules():
    if isinstance(m, torch.nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None; m.train()
with torch.no_grad():
    for b in range(0, len(X), 64): net.features(X[b:b + 64])
net.eval(); net.classifier = torch.nn.Identity()


def tta(x):
    out = [x]
    for ang, sc in [(5, 1.0), (-5, 1.0), (0, 0.95), (0, 1.05)]:
        a = np.radians(ang); th = torch.tensor([[np.cos(a) / sc, -np.sin(a) / sc, 0], [np.sin(a) / sc, np.cos(a) / sc, 0]], dtype=torch.float32)
        out.append(Fn.grid_sample(x[None], Fn.affine_grid(th[None], (1, 3, 224, 224), align_corners=False), padding_mode="border", align_corners=False)[0])
    return torch.stack(out)


with torch.no_grad():
    emb = np.stack([net(tta(X[i])).mean(0).numpy() for i in range(len(X))]).astype(np.float32)
np.save(E / "emb_densenet121_adabn.npy", emb); C[["nsin", "side", "fmt"]].to_csv(E / "emb_densenet121_adabn_index.csv", index=False)
print("adabn emb", emb.shape)
