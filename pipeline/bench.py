"""Compute-feasibility micro-benchmarks on synthetic data shaped like the real task (CPU).
usage: python -I bench.py <one TIFF> <one DICOM>"""
import sys, time, pathlib
import numpy as np, torch, torch.nn as nn, torchvision
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from render_lib import load

torch.set_num_threads(14)
def t(f, n=1):
    f(); s = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - s) / n

# 1. raw image decode (dominates preprocessing)
for p in sys.argv[1:3]:
    s = time.perf_counter(); a, _ = load(p); print(f"decode {pathlib.Path(p).suffix}: {time.perf_counter() - s:.2f}s  shape {a.shape}")

# 2. CNNs: frozen embedding (inference) and fine-tuning step, 224x224, batch 32
x = torch.randn(32, 3, 224, 224)
for name in ["resnet18", "resnet50", "densenet121"]:
    m = getattr(torchvision.models, name)(weights=None)
    m.eval()
    with torch.no_grad(): inf = t(lambda: m(x), 3)
    m.train(); opt = torch.optim.AdamW(m.parameters(), 1e-4); y = torch.randint(0, 2, (32,))
    def step():
        opt.zero_grad(); loss = nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step()
    tr = t(step, 2)
    # last-block-only fine-tuning (freeze everything except final stage + head)
    for n_, p_ in m.named_parameters(): p_.requires_grad = n_.startswith(("layer4", "fc", "features.denseblock4", "classifier"))
    opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], 1e-4)
    tr_last = t(step, 2)
    print(f"{name}: infer {32 / inf:.0f} img/s | full fine-tune {32 / tr:.0f} img/s | last-block fine-tune {32 / tr_last:.0f} img/s")

# 3. FT-Transformer-sized model: 60 feature tokens, d=64, 3 blocks, 1,700 hips per epoch
class FTT(nn.Module):
    def __init__(s, f=60, d=64, L=3):
        super().__init__(); s.w = nn.Parameter(torch.randn(f, d)); s.b = nn.Parameter(torch.zeros(f, d)); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), L); s.head = nn.Linear(d, 1)
    def forward(s, x):
        tok = x[..., None] * s.w + s.b
        return s.head(s.enc(torch.cat([s.cls.expand(len(x), -1, -1), tok], 1))[:, 0]).squeeze(-1)
m = FTT(); opt = torch.optim.AdamW(m.parameters(), 1e-3)
X = torch.randn(1700, 60); Y = (torch.rand(1700) < .055).float()
def epoch():
    for i in range(0, 1700, 128):
        opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(X[i:i + 128]), Y[i:i + 128]).backward(); opt.step()
print(f"FT-Transformer: {t(epoch, 3):.2f}s / epoch (1,700 hips)")

# 4. LightGBM and elastic net on 1,700 x 80
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
Xn = np.random.randn(1700, 80); Yn = (np.random.rand(1700) < .055).astype(int)
print(f"LightGBM 500 trees: {t(lambda: lgb.LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=.03, verbose=-1).fit(Xn, Yn), 2):.2f}s")
print(f"Elastic-net logistic (saga): {t(lambda: LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.5, C=.1, max_iter=3000).fit(Xn, Yn), 2):.2f}s")
