"""Frozen image embeddings of the hip crops (plan 7.9 B4 (a)/(b)).

usage: python -I 06_embed.py <project dir> [weights=imagenet|<path to SimCLR state_dict>] [tag]
reads : data/derived/crops.csv (05_crop.py)
writes: data/derived/emb/emb_<tag>.npy (n_hips x 512, row order = crops.csv), emb_<tag>_index.csv

Each crop is embedded 5 times (identity + 4 mild test-time augmentations: +-5 deg rotation, 0.95/1.05 scale) and averaged.
Normalisation: crop already windowed to [0,1]; replicated to 3 channels, ImageNet mean/std.
"""
import sys, pathlib
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn, torchvision
from PIL import Image

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; E = D / "emb"; E.mkdir(exist_ok=True)
W = sys.argv[2] if len(sys.argv) > 2 else "imagenet"; TAG = sys.argv[3] if len(sys.argv) > 3 else W if W in ("imagenet", "imagenet_ms", "densenet121") else "simclr"
torch.set_num_threads(int(__import__('os').environ.get('EMB_THREADS', 12)))
C = pd.read_csv(D / "crops.csv")
if "qc_exclude" in C: C = C[~C.qc_exclude].reset_index(drop=True)   # visual QC failures (5 hips) -> treated as no image
net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); net.fc = torch.nn.Identity()
if W == "imagenet_ms":     # multi-scale: global-average-pooled layer3 (256) + layer4 (512); mid-level texture (joint space, trabeculae)
    class MS(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.b = base   # registered, so .eval() reaches its BatchNorms
        def forward(self, x):
            b = self.b; x = b.maxpool(b.relu(b.bn1(b.conv1(x)))); x = b.layer2(b.layer1(x)); x3 = b.layer3(x); x4 = b.layer4(x3)
            return torch.cat([x3.mean((2, 3)), x4.mean((2, 3))], 1)
    net = MS(net)
elif W == "densenet121":   # backbone of Tiulpin 2019 / von Schacky 2020 (1024-d)
    d = torchvision.models.densenet121(weights="IMAGENET1K_V1"); d.classifier = torch.nn.Identity(); net = d
elif W != "imagenet":
    sd = torch.load(W, map_location="cpu"); sd = {k.replace("backbone.", ""): v for k, v in sd.items() if not k.startswith(("proj", "fc"))}
    print("load", net.load_state_dict(sd, strict=False))
net.eval()
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def tta(x):
    out = [x]
    for ang, sc in [(5, 1.0), (-5, 1.0), (0, 0.95), (0, 1.05)]:
        a = np.radians(ang); th = torch.tensor([[np.cos(a) / sc, -np.sin(a) / sc, 0], [np.sin(a) / sc, np.cos(a) / sc, 0]], dtype=torch.float32)
        g = Fn.affine_grid(th[None], (1, 1, 224, 224), align_corners=False)
        out.append(Fn.grid_sample(x[None], g, padding_mode="border", align_corners=False)[0])
    return torch.stack(out)

embs = []
with torch.no_grad():
    for i, r in enumerate(C.itertuples()):
        x = torch.from_numpy(np.asarray(Image.open(r.crop), dtype=np.float32) / 65535.0)[None]
        x = Fn.interpolate(x[None], size=(224, 224), mode="area")[0]
        b = (tta(x).repeat(1, 3, 1, 1) - MEAN) / STD
        embs.append(net(b).mean(0).numpy())
        if i % 300 == 0: print(i, "/", len(C))
np.save(E / f"emb_{TAG}.npy", np.stack(embs).astype(np.float32))
C[["nsin", "side", "fmt"]].to_csv(E / f"emb_{TAG}_index.csv", index=False)
print("saved", E / f"emb_{TAG}.npy", np.stack(embs).shape)
