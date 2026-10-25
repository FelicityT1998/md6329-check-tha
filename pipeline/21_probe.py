"""Shortcut probes (plan 6.3 B): can the model inputs predict the acquisition setting instead of the hip?

usage: python -I 21_probe.py <project dir> [emb_tag=imagenet]
reads : landmarks/cache/*.pt (384 letterbox of each whole film), emb/emb_<tag>.npy (hip crops), hips.parquet, splits_B.csv
writes: data/derived/emb/emb_wholefilm.npy (+ _index.csv), data/derived/results/probe_<tag>.csv

Targets : centre (C01-C10, macro one-vs-rest AUROC), image format (TIFF vs DICOM), follow-up group (annual vs interval),
          and, for reference, the outcome itself (10-y THA, AUROC).
Inputs  : (1) whole-film ImageNet ResNet-18 embedding (what a naive image model sees: burned-in IDs, markers, collimation),
          (2) hip-crop embedding (our image input), (3) clinical table DESCLB (MICE), (4) crop embedding with the format
          and centre means removed inside each training fold (a simple 'de-shortcut' residualisation).
Probe   : L2 logistic regression (C=0.1) on standardised PCA-64 features, 5-fold CV grouped by subject.
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn, torchvision
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; L = D / "landmarks"; E = D / "emb"; R = D / "results"; R.mkdir(exist_ok=True)
TAG = sys.argv[2] if len(sys.argv) > 2 else "imagenet"
sys.path.insert(0, str(P))
from lib_tab import load, feature_spec, raw_frame, TabPrep
from lib_img import image_matrix

H, FD, F = load(); S = pd.read_csv(D / "splits_B.csv")
# ------------------------------------------------------------------ whole-film embedding (once)
wf = E / "emb_wholefilm.npy"
if not wf.exists():
    torch.set_num_threads(12)
    I = pd.read_csv(D / "images.csv"); ori = pd.read_csv(L / "orient.csv").set_index("path").k_upright
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1"); net.fc = torch.nn.Identity(); net.eval()
    MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    out = []
    with torch.no_grad():
        for p in I.path:
            im = torch.rot90(torch.load(L / "cache" / (pathlib.Path(p).stem + ".pt"))[0], int(ori[p]), (0, 1))
            x = Fn.interpolate(im[None, None], size=(224, 224), mode="area")[0].repeat(3, 1, 1)
            out.append(net(((x - MEAN) / STD)[None])[0].numpy())
    np.save(wf, np.stack(out).astype(np.float32))
    # per hip: the film used for that hip's crop if known, else the subject's first pelvis film
    I["k"] = range(len(I)); rows = []
    crops = pd.read_csv(D / "crops.csv") if (D / "crops.csv").exists() else None
    for h in H[["nsin", "side"]].itertuples():
        c = crops[(crops.nsin == h.nsin) & (crops.side == h.side)] if crops is not None else []
        f = I[I.path == c.src_path.iloc[0]] if len(c) else I[(I.nsin == h.nsin) & I.view.isin(["O", h.side])]
        rows.append({"nsin": h.nsin, "side": h.side, "film_row": int(f.k.iloc[0]) if len(f) else -1})
    pd.DataFrame(rows).to_csv(E / "emb_wholefilm_hipmap.csv", index=False)

WF = np.load(wf); hm = pd.read_csv(E / "emb_wholefilm_hipmap.csv")
WFH = np.full((len(H), WF.shape[1]), np.nan, np.float32); ok = hm.film_row.values >= 0; WFH[ok] = WF[hm.film_row.values[ok]]
CROP = image_matrix(H, TAG, D)
spec = feature_spec(FD, "DESCLB"); Xraw = raw_frame(H, spec)

TARGETS = {"centre": S.dom_centre.where(S.dom_centre.str.startswith("C")).values,
           "format": S.dom_fmt.where(S.dom_fmt != "none").values,
           "followup_group": S.dom_group.values,
           "outcome_THA10": H.y10.map({0.0: "no", 1.0: "yes"}).values}


def probe(Xfun, target, resid=False):
    yv = pd.Series(target); keep = yv.notna().values & Xfun(None).notna_mask
    idx = np.where(keep)[0]; yy = yv.values[idx]; classes = sorted(set(yy)); P_ = np.zeros((len(idx), len(classes)))
    for a, b in GroupKFold(5).split(idx, groups=H.nsin.values[idx]):
        Xa, Xb = Xfun((idx[a], idx[b]))
        if resid:   # remove format x centre means estimated on the training fold
            key = pd.Series(S.dom_fmt.values + "|" + S.dom_centre.values)
            mu = pd.DataFrame(Xa).groupby(key.values[idx[a]]).mean(); g = mu.mean()
            Xa = Xa - mu.reindex(key.values[idx[a]]).values; Xb = Xb - mu.reindex(key.values[idx[b]]).fillna(g).values
        sc = StandardScaler().fit(Xa); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xa), yy[a])
        P_[b] = m.predict_proba(sc.transform(Xb))
    if len(classes) == 2: return roc_auc_score(yy == classes[1], P_[:, 1]), len(idx)
    return roc_auc_score(yy, P_, multi_class="ovr", average="macro"), len(idx)


class Emb:
    def __init__(self, M, npc=64): self.M, self.npc = M, npc; self.notna_mask = ~np.isnan(M[:, 0])
    def __call__(self, ab):
        if ab is None: return self
        a, b = ab; p = PCA(self.npc, random_state=0).fit(self.M[a]); return p.transform(self.M[a]), p.transform(self.M[b])


class Tab:
    notna_mask = np.ones(len(H), bool)
    def __call__(self, ab):
        if ab is None: return self
        a, b = ab; pp = TabPrep(spec, "mice"); return pp.fit_transform(Xraw.iloc[a]).values, pp.transform(Xraw.iloc[b]).values


if __name__ == "__main__":
    inputs = {"whole_film_embedding": (Emb(WFH), False), "hip_crop_embedding": (Emb(CROP), False),
              "hip_crop_embedding_residualised": (Emb(CROP), True), "clinical_table_DESCLB": (Tab(), False)}
    rows = []
    for iname, (fn, resid) in inputs.items():
        for tname, t in TARGETS.items():
            if resid and tname in ("centre", "format"):
                continue   # the residualisation uses the centre/format labels themselves, so probing them is circular
            auc, n = probe(fn, t, resid); rows.append({"input": iname, "target": tname, "auroc": auc, "n": n})
            print(f"{iname:34s} {tname:16s} AUROC {auc:.3f}  (n={n})", flush=True)
    pd.DataFrame(rows).to_csv(R / f"probe_{TAG}.csv", index=False)
