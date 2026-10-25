"""Metrics for pooled out-of-fold predictions (plan 7.9 B8/B10): AUPRC (primary), AUROC, Brier, calibration
intercept/slope, Platt recalibration, subject-clustered bootstrap CIs."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def calib(y, p):
    """calibration-in-the-large (intercept with slope fixed at 1) and calibration slope."""
    lp = logit(p)
    slope = LogisticRegression(C=1e6).fit(lp[:, None], y).coef_[0, 0]
    import statsmodels.api as sm
    icpt = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return icpt, slope


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    icpt, slope = calib(y, p)
    return {"n": len(y), "events": int(y.sum()), "auprc": average_precision_score(y, p), "auroc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, np.clip(p, EPS, 1 - EPS)), "cal_intercept": icpt, "cal_slope": slope}


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6).fit(logit(p)[:, None], y); return self
    def __call__(self, p):
        return self.m.predict_proba(logit(p)[:, None])[:, 1]


def select_score(p, y):
    """Model-selection score = log-loss after Platt recalibration on the same pooled inner OOF predictions.
    Raw log-loss would penalise class-weighted / focal configurations for their (fixable) probability shift."""
    return log_loss(y, np.clip(Platt().fit(p, y)(p), EPS, 1 - EPS))


def cluster_boot(df, pcol, n=2000, seed=0, keys=("auprc", "auroc", "brier", "cal_slope")):
    """df with columns nsin, y, <pcol>; resample subjects with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.values for k, g in df.groupby("nsin")}
    ids = np.array(list(groups)); out = []
    y, p = df.y.values, df[pcol].values
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.choice(ids, len(ids))])
        pos = df.index.get_indexer(idx)
        if y[pos].sum() < 2: continue
        m = metrics(y[pos], p[pos]); out.append({k: m[k] for k in keys})
    B = pd.DataFrame(out)
    return B.quantile([0.025, 0.975])


def summarise(oof, pcol="p", by_repeat=True):
    """oof: rows = hip x repeat with columns repeat, nsin, y, <pcol>. Metrics per repeat (each hip once) then mean +- sd."""
    rows = [dict(repeat=r, **metrics(g.y, g[pcol])) for r, g in oof.groupby("repeat")]
    R = pd.DataFrame(rows)
    return R, R.drop(columns="repeat").agg(["mean", "std"])
