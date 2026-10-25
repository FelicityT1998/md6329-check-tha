"""Shared image helpers for the localisation pipeline (05r_orient, 05b_locator, 05_crop)."""
import numpy as np, torch, torch.nn.functional as Fn

S = 384   # letterbox size


def letterbox(a):
    """float image -> (S,S) in [0,1], plus the scale/offset to map back to original pixels."""
    lo, hi = np.percentile(a[::4, ::4], [1, 99.5]); a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    h, w = a.shape; s = S / max(h, w)
    t = torch.from_numpy(a)[None, None]
    t = Fn.interpolate(t, size=(round(h * s), round(w * s)), mode="area")[0, 0]
    out = torch.zeros(S, S); oy, ox = (S - t.shape[0]) // 2, (S - t.shape[1]) // 2
    out[oy:oy + t.shape[0], ox:ox + t.shape[1]] = t
    return out, s, ox, oy


def rot_pt(x, y, k, n=S):
    """coordinates of pixel (x, y) after torch.rot90(img, k, (0, 1)) of an n x n image (counter-clockwise turns)."""
    for _ in range(k % 4): x, y = y, n - 1 - x
    return x, y


def unrot_pt(x, y, k, n=S):
    return rot_pt(x, y, (4 - k) % 4, n)


# ---------------------------------------------------------------- hip-level embedding matrix + fold-wise PCA (13_m3, 14_m4, 21_probe)
def image_matrix(H, tag, D):
    """(n_hips, 512) float32 aligned to H rows (nsin, side); NaN rows = hip without a crop."""
    import pandas as pd
    E = np.load(D / f"emb/emb_{tag}.npy"); idx = pd.read_csv(D / f"emb/emb_{tag}_index.csv")
    pos = pd.Series(np.arange(len(idx)), index=pd.MultiIndex.from_frame(idx[["nsin", "side"]]))
    j = pos.reindex(pd.MultiIndex.from_frame(H[["nsin", "side"]])).values
    M = np.full((len(H), E.shape[1]), np.nan, np.float32); ok = ~np.isnan(j); M[ok] = E[j[ok].astype(int)]
    return M


class ImgPrep:
    """PCA to n components fitted on the training rows that have a crop, z-scored; missing crop -> 0 (+ indicator column)."""
    def __init__(self, n=32): self.n = n
    def fit(self, M):
        from sklearn.decomposition import PCA
        ok = ~np.isnan(M[:, 0]); self.pca = PCA(self.n, random_state=0).fit(M[ok])
        Z = self.pca.transform(M[ok]); self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6; return self
    def transform(self, M, indicator=True):
        ok = ~np.isnan(M[:, 0]); Z = np.zeros((len(M), self.n), np.float32)
        Z[ok] = (self.pca.transform(M[ok]) - self.mu) / self.sd
        return np.column_stack([Z, (~ok).astype(np.float32)]) if indicator else Z
