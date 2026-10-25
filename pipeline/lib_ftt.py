"""FT-Transformer (Gorishniy et al. 2021) for hip-level THA prediction - plan 7.9 blocks B2-B6, B9.

B2 tokenizer : numeric value embedding (linear | periodic | piecewise-linear) + feature-ID embedding (+ optional group
               embedding); a learned per-feature [MISS] token replaces the value embedding when the value is missing;
               categorical features use an embedding table (index 0 = missing); [CLS] is prepended.
B3 encoder   : pre-LN blocks, multi-head self-attention + ReGLU feed-forward, attention / FFN / residual dropout.
B4/B5 image  : optional image embedding -> linear projection -> one extra token  (fusion="token")
               or FiLM modulation of the [CLS] representation                   (fusion="film").
B6 heads     : [CLS] -> 10-year binary logit and 10 discrete-time yearly hazard logits.
"""
import math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

# early-stopping criterion: default = validation binary BCE on hips with known 10-y status (~25 events per inner fold, noisy);
# FTT_ES=total -> the full loss (binary + lam * discrete-time NLL) on all validation hips, censored ones included.
ES_TOTAL = os.environ.get("FTT_ES") == "total"
# 2026-10-23 training-stability switches (defaults = original behaviour):
#   FTT_SCHED=warmcos   linear LR warm-up over the first 10 epochs, then cosine decay over the epoch budget
#   FTT_MIN_EPOCHS=n    early stopping may not fire before epoch n (no stopping while the model is still under-fitting)
#   FTT_PATIENCE=n      overrides the caller's patience
#   FTT_NORM=bn         BatchNorm over the token channel (statistics over batch x tokens) instead of LayerNorm in every block
SCHED = os.environ.get("FTT_SCHED", "none"); MIN_EPOCHS = int(os.environ.get("FTT_MIN_EPOCHS", 0))
PATIENCE_ENV = os.environ.get("FTT_PATIENCE"); NORM = os.environ.get("FTT_NORM", "ln")


class TokenBN(nn.Module):
    """BatchNorm1d applied to every token's d-dim vector, statistics pooled over batch and tokens."""
    def __init__(self, d):
        super().__init__(); self.bn = nn.BatchNorm1d(d)
    def forward(self, x):
        B, T, d = x.shape
        return self.bn(x.reshape(B * T, d)).reshape(B, T, d)


def make_norm(d):
    return TokenBN(d) if NORM == "bn" else nn.LayerNorm(d)


class NumTokenizer(nn.Module):
    def __init__(self, n, d, kind="linear", bins=None, n_freq=8):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.w = nn.Parameter(torch.randn(n, d) * 0.1)
        elif kind == "periodic":
            self.freq = nn.Parameter(torch.randn(n, n_freq) * 0.5)
            self.proj = nn.Parameter(torch.randn(n, 2 * n_freq, d) / math.sqrt(2 * n_freq))
        elif kind == "piecewise_linear":
            self.register_buffer("edges", torch.as_tensor(bins, dtype=torch.float32))    # (n, B+1) training-fold quantiles
            nb = self.edges.shape[1] - 1
            self.proj = nn.Parameter(torch.randn(n, nb, d) / math.sqrt(nb))
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, x):                                   # x (B, n)
        if self.kind == "linear":
            return x[..., None] * self.w + self.b
        if self.kind == "periodic":
            z = 2 * math.pi * x[..., None] * self.freq       # (B, n, k)
            return torch.einsum("bnk,nkd->bnd", torch.cat([z.sin(), z.cos()], -1), self.proj) + self.b
        lo, hi = self.edges[:, :-1], self.edges[:, 1:]       # PLE: fraction of each bin that x has passed
        enc = ((x[..., None] - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)
        return torch.einsum("bnk,nkd->bnd", enc, self.proj) + self.b


class Block(nn.Module):
    def __init__(self, d, heads, ffn_mult, attn_do, ffn_do, res_do):
        super().__init__()
        self.ln1, self.ln2 = make_norm(d), make_norm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_do, batch_first=True)
        h = max(4, int(d * ffn_mult))
        self.ff1, self.ff2 = nn.Linear(d, 2 * h), nn.Linear(h, d)
        self.ffn_do, self.res_do = nn.Dropout(ffn_do), nn.Dropout(res_do)
        self.last_attn = None

    def forward(self, x, keep_attn=False):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, need_weights=keep_attn, average_attn_weights=True)
        if keep_attn: self.last_attn = w.detach()
        x = x + self.res_do(a)
        u, g = self.ff1(self.ln2(x)).chunk(2, -1)
        return x + self.res_do(self.ff2(self.ffn_do(u * Fn.relu(g))))         # ReGLU


class FTT(nn.Module):
    def __init__(self, n_num, cat_cards, groups, d=32, n_blocks=2, heads=4, ffn_mult=4 / 3, attn_do=0.1, ffn_do=0.1, res_do=0.0,
                 num_tok="linear", bins=None, group_tokens=False, readout="cls", img_dim=0, fusion="none", n_years=10):
        super().__init__()
        self.num = NumTokenizer(n_num, d, num_tok, bins)
        self.fid = nn.Parameter(torch.randn(n_num + len(cat_cards), d) * 0.02)      # feature-ID embedding
        self.miss = nn.Parameter(torch.randn(n_num, d) * 0.02)                      # per-feature [MISS]
        self.cat = nn.ModuleList([nn.Embedding(c + 1, d) for c in cat_cards])
        self.group_tokens = group_tokens
        if group_tokens:
            self.register_buffer("gid", torch.as_tensor(groups, dtype=torch.long))
            self.gemb = nn.Embedding(int(max(groups)) + 2, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.fusion, self.readout = fusion, readout
        if img_dim and fusion == "token":
            self.img = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, d)); self.img_id = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if img_dim and fusion == "film":
            self.film = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, 2 * d))
        self.blocks = nn.ModuleList([Block(d, heads, ffn_mult, attn_do, ffn_do, res_do) for _ in range(n_blocks)])
        self.norm = nn.BatchNorm1d(d) if NORM == "bn" else nn.LayerNorm(d)             # readout is (B, d)
        self.head_bin, self.head_dt = nn.Linear(d, 1), nn.Linear(d, n_years)
        # 2026-10-22: with zero bias the yearly hazards start at 0.5 and, because early stopping watches the binary head,
        # training stops long before they reach the ~0.5%/year baseline (10-y CI mean 0.34 vs 0.05; AUROC 0.51).
        # FTT_DT_BIAS (e.g. -5.3 = logit of the KM yearly hazard) starts them at the baseline instead.
        import os
        if os.environ.get("FTT_DT_BIAS"): nn.init.constant_(self.head_dt.bias, float(os.environ["FTT_DT_BIAS"]))

    def tokens(self, xn, mask, xc, img=None):
        t = self.num(xn)
        t = torch.where(mask[..., None], self.miss.expand_as(t), t)
        if len(self.cat): t = torch.cat([t, torch.stack([e(xc[:, i]) for i, e in enumerate(self.cat)], 1)], 1)
        t = t + self.fid
        if self.group_tokens: t = t + self.gemb(self.gid)
        parts = [self.cls.expand(len(t), -1, -1), t]
        if img is not None and self.fusion == "token": parts.append(self.img(img)[:, None] + self.img_id)
        return torch.cat(parts, 1)

    def forward(self, xn, mask, xc, img=None, keep_attn=False):
        x = self.tokens(xn, mask, xc, img)
        for b in self.blocks: x = b(x, keep_attn)
        z = self.norm(x[:, 0] if self.readout == "cls" else x.mean(1))
        if img is not None and self.fusion == "film":
            g, b = self.film(img).chunk(2, -1); z = z * (1 + g) + b
        z = Fn.relu(z)
        return self.head_bin(z)[:, 0], self.head_dt(z)


def loss_fn(out_bin, out_dt, y, ybin_mask, t_obs, event, kind="bce_weighted", pos_weight=10.0, lam=0.5):
    """binary loss on hips with known 10-y status + lam x discrete-time NLL on all hips.
    t_obs: last year at risk (event year or censoring year, 0..10); event: 1 if THA at t_obs."""
    yb, ob = y[ybin_mask], out_bin[ybin_mask]
    if kind == "bce_weighted":
        lb = Fn.binary_cross_entropy_with_logits(ob, yb, pos_weight=torch.tensor(pos_weight))
    else:
        g = 1.0 if kind == "focal_g1" else 2.0
        p = torch.sigmoid(ob); bce = Fn.binary_cross_entropy_with_logits(ob, yb, reduction="none")
        pt = torch.where(yb == 1, p, 1 - p); alpha = torch.where(yb == 1, 0.75, 0.25)
        lb = (alpha * (1 - pt) ** g * bce).mean()
    if lam == 0: return lb
    T = out_dt.shape[1]; years = torch.arange(1, T + 1, device=out_dt.device)[None]
    at_risk = (years <= t_obs[:, None]).float()                                   # years 1..t_obs observed
    is_ev = ((years == t_obs[:, None]) & (event[:, None] == 1)).float()
    ll = -(Fn.binary_cross_entropy_with_logits(out_dt, is_ev, reduction="none") * at_risk).sum(1)
    return lb - lam * ll.mean() / T


def ple_bins(X, mask, n_bins=8):
    """per-feature quantile edges from training data (missing ignored); degenerate features get a unit bin."""
    edges = []
    for j in range(X.shape[1]):
        v = X[~mask[:, j], j]
        e = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1))) if len(v) else np.array([0.0, 1.0])
        if len(e) < 2: e = np.array([e[0], e[0] + 1])
        e = np.concatenate([e, np.full(n_bins + 1 - len(e), e[-1] + 1e-3 * np.arange(1, n_bins + 2 - len(e)))])
        edges.append(np.sort(e))
    return np.stack(edges)


def fit(model, data, idx_tr, idx_va=None, epochs=200, lr=1e-3, wd=1e-4, bs=256, patience=20, loss_kw=None, seed=0, img=None):
    """data: dict of tensors xn, mask, xc, y, ybin_mask, t_obs, event. Early stopping on the validation binary BCE.
    returns (best_epoch, history list)."""
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if SCHED == "warmcos":
        warm = 10
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    if PATIENCE_ENV: patience = int(PATIENCE_ENV)
    loss_kw = loss_kw or {}; hist, best, best_ep, best_state, bad = [], np.inf, epochs, None, 0
    take = lambda ix: (data["xn"][ix], data["mask"][ix], data["xc"][ix], None if img is None else img[ix])
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(idx_tr); tl = 0
        for b in range(0, len(perm), bs):
            ix = perm[b:b + bs]; ob, od = model(*take(ix))
            loss = loss_fn(ob, od, data["y"][ix], data["ybin_mask"][ix], data["t_obs"][ix], data["event"][ix], **loss_kw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item() * len(ix)
        rec = {"epoch": ep + 1, "train_loss": tl / len(idx_tr)}
        if idx_va is not None:
            vb = idx_va[data["ybin_mask"][idx_va].numpy().astype(bool)]
            model.eval()
            with torch.no_grad():
                ob, od = model(*take(idx_va if ES_TOTAL else vb))
                if ES_TOTAL:   # 2026-10-22: stop on the full training objective over ALL validation hips (censored ones feed the DT term)
                    v = loss_fn(ob, od, data["y"][idx_va], data["ybin_mask"][idx_va], data["t_obs"][idx_va], data["event"][idx_va], **loss_kw).item()
                else:
                    v = Fn.binary_cross_entropy_with_logits(ob, data["y"][vb]).item()
            rec["val_bce"] = v
            if v < best - 1e-4 and ep + 1 >= (10 if SCHED == "warmcos" else 0): best, best_ep, bad = v, ep + 1, 0; best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience and ep + 1 >= MIN_EPOCHS: hist.append(rec); break
        if SCHED == "warmcos": sched.step()
        hist.append(rec)
    if best_state is not None: model.load_state_dict(best_state)
    return best_ep, hist


def predict(model, data, idx, img=None):
    model.eval()
    with torch.no_grad():
        ob, od = model(data["xn"][idx], data["mask"][idx], data["xc"][idx], None if img is None else img[idx])
    p10 = torch.sigmoid(ob).numpy()
    surv = torch.cumprod(1 - torch.sigmoid(od), 1).numpy()                       # S(t) from yearly hazards
    return p10, 1 - surv                                                          # binary-head risk, cumulative incidence 1..10
