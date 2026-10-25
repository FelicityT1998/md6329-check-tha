"""Shared tabular data access + fold-wise preprocessing (plan 7.4 / 7.9 B0-B1). Everything estimated from data is fitted
on the training rows passed to .fit() only.

Feature-set spec: a string of group letters, e.g. "D", "DESCLB", "DESCLBR"; I (image) is handled by the image scripts.
De-redundancy (03_eda C1-C3): ir_le25 / flex_le115 / nrs_now / womac_total are dropped from model inputs; every
side-specific feature enters as ipsilateral value + asymmetry (ipsilateral - contralateral) instead of the raw contralateral.
"""
import pathlib
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PowerTransformer

P = pathlib.Path(__file__).parent; D = P / "data/derived"
DROP = {"ir_le25", "flex_le115", "nrs_now", "womac_total"}


def load():
    H = pd.read_parquet(D / "hips.parquet"); FD = pd.read_csv(D / "features.csv"); F = pd.read_csv(D / "folds.csv")
    return H, FD, F


def feature_spec(FD, groups, main_only=True, asym=True):
    """-> list of dicts {name, src, src_c, kind, transform, group}; src_c set when the feature is an asymmetry term."""
    f = FD[FD.group.isin(list(groups)) | (FD.col == "side_right")]
    if main_only: f = f[f.main]
    out = []
    for r in f[f.laterality != "contra"].itertuples():
        if r.feature in DROP: continue
        out.append(dict(name=r.col, src=r.col, src_c=None, kind=r.kind, transform=r.transform, group=r.group))
        if r.laterality == "ipsi":
            cc = r.col + "_c"
            if asym and r.kind != "nom":
                out.append(dict(name=r.col + "_asym", src=r.col, src_c=cc, kind="cont" if r.kind == "cont" else "ord",
                                transform="none", group=r.group))
            elif not asym:
                out.append(dict(name=cc, src=cc, src_c=None, kind=r.kind, transform=r.transform, group=r.group))
    return out


def raw_frame(H, spec):
    """Build the model-input frame (before any fitted step): asymmetry terms and structural rules applied."""
    X = {}
    for s in spec:
        X[s["name"]] = H[s["src"]] - H[s["src_c"]] if s["src_c"] else H[s["src"]]
    X = pd.DataFrame(X, index=H.index)
    if "phys_load" in X and "not_working" in X:   # structural: non-workers have no occupational load -> lowest category
        X.loc[(H.not_working == 1) & X.phys_load.isna(), "phys_load"] = 1.0
    return X


class TabPrep:
    """fit on training rows -> transform any rows.
    impute: 'mice' (IterativeImputer + indicators) | 'median' (+ indicators) | 'none' (NaN kept, for GBDT / FTT)
    scale : z-score after transform + winsorisation (skipped when impute == 'none' and scale=False)."""

    def __init__(self, spec, impute="mice", scale=True, seed=0):
        self.spec, self.impute, self.scale, self.seed = spec, impute, scale, seed
        self.num = [s["name"] for s in spec if s["kind"] != "nom"]
        self.nom = [s["name"] for s in spec if s["kind"] == "nom"]
        self.tf = {s["name"]: s["transform"] for s in spec}

    def _transform_num(self, X, fit):
        X = X[self.num].astype(float).copy()
        for c in self.num:
            t = self.tf[c]
            if t == "log1p": X[c] = np.log1p(X[c].clip(lower=0))
            elif t == "yeojohnson":
                v = X[[c]].values
                if fit: self.pt_[c] = PowerTransformer("yeo-johnson", standardize=False).fit(v[~np.isnan(v[:, 0])])
                ok = ~np.isnan(v[:, 0]); out = v.copy(); out[ok] = self.pt_[c].transform(v[ok]); X[c] = out[:, 0]
        if fit: self.lo_, self.hi_ = X.quantile(0.005), X.quantile(0.995)
        return X.clip(self.lo_, self.hi_, axis=1)

    def fit(self, Xraw):
        self.pt_ = {}
        Xn = self._transform_num(Xraw, fit=True)
        self.cats_ = {c: sorted(Xraw[c].dropna().unique().tolist()) for c in self.nom}
        self.miss_cols_ = [c for c in self.num if Xn[c].isna().any()]
        if self.impute == "mice":
            self.imp_ = IterativeImputer(BayesianRidge(), max_iter=10, random_state=self.seed, skip_complete=True).fit(Xn)
        elif self.impute == "median":
            self.imp_ = SimpleImputer(strategy="median").fit(Xn)
        Xi = pd.DataFrame(self.imp_.transform(Xn), columns=self.num, index=Xn.index) if self.impute != "none" else Xn
        self.mu_, self.sd_ = Xi.mean(), Xi.std().replace(0, 1)
        return self

    def transform(self, Xraw):
        Xn = self._transform_num(Xraw, fit=False)
        miss = Xn[self.miss_cols_].isna().astype(float).add_suffix("__miss")
        if self.impute != "none": Xn = pd.DataFrame(self.imp_.transform(Xn), columns=self.num, index=Xn.index)
        if self.scale: Xn = (Xn - self.mu_) / self.sd_
        parts = [Xn]
        if self.impute != "none": parts.append(miss)
        for c in self.nom:
            for v in self.cats_[c][1:]:   # reference = first category; NaN -> all zeros + indicator
                parts.append(pd.DataFrame({f"{c}={v}": (Xraw[c] == v).astype(float).fillna(0)}, index=Xraw.index))
            parts.append(pd.DataFrame({f"{c}__miss": Xraw[c].isna().astype(float)}, index=Xraw.index))
        out = pd.concat(parts, axis=1)
        return out.loc[:, out.std() > 0] if getattr(self, "keep_", None) is None else out.reindex(columns=self.keep_, fill_value=0)

    def fit_transform(self, Xraw):
        out = self.fit(Xraw).transform(Xraw); self.keep_ = list(out.columns); return out

    def ftt_arrays(self, Xraw):
        """For the FT-Transformer: z-scored numeric with NaN -> 0 + missing mask, categorical integer codes (0 = missing)."""
        Xn = self._transform_num(Xraw, fit=False)
        mask = Xn.isna().values
        Xz = ((Xn - self.mu_) / self.sd_).fillna(0).values.astype(np.float32)
        cat = np.stack([Xraw[c].map({v: i + 1 for i, v in enumerate(self.cats_[c])}).fillna(0).astype(int).values for c in self.nom], 1) \
            if self.nom else np.zeros((len(Xraw), 0), int)
        return Xz, mask, cat


def fold_rows(H, F, repeat, outer, inner=None):
    """Indices into H for one split. Binary outcome rows only (y10 known). inner=None -> outer train/test;
    inner=j -> (inner-train, inner-val) inside the outer training set."""
    f = F[F.repeat == repeat].set_index("nsin")
    known = H.y10.notna()
    test_s = set(f.index[f.outer_fold == outer])
    is_test = H.nsin.isin(test_s)
    if inner is None:
        return np.where(known & ~is_test)[0], np.where(known & is_test)[0]
    col = f[f"inner_for_o{outer}"]
    val_s = set(col.index[col == inner])
    is_val = H.nsin.isin(val_s)
    return np.where(known & ~is_test & ~is_val)[0], np.where(known & ~is_test & is_val)[0]
