"""Threshold metrics for the main models (report Table 3, 2026-10-25).

usage: python -I 92_threshold_table.py <project dir>
reads : data/derived/oof/<tag>.parquet (Platt-calibrated out-of-fold predictions, five repeats)
writes: data/derived/results/threshold_metrics.csv and report/sections/table_threshold.tex

Two operating points, each computed per repeat on the pooled out-of-fold predictions and then averaged over repeats:
  risk >= 10%  : the decision-curve threshold used in the report (calibrated probability), so the flagged fraction varies;
  top 10%      : the 10% of hips with the highest predicted risk (fixed workload, independent of calibration).
Metrics: sensitivity (recall), specificity, precision (PPV), F1 and Matthews correlation coefficient (MCC).
"""
import sys, pathlib
import numpy as np, pandas as pd

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"
MODELS = [("m0_DESCLB_mice", "M0 elastic net (clinical)"), ("m1_DESCLB", "M1 LightGBM"), ("m1b_DESCLB", "M1b EBM"),
          ("m2ens_DESCLB_lam1_dtinit_es_s3", "M2 FT-Transformer"), ("m3_head_res_-", "M3 image head (pixels only)"),
          ("m5_pix_logit_DESCLB", "M0 + pixel logit (final)")]


def point(y, flag):
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / (tp + fp) if tp + fp else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn)
    return dict(flagged=flag.mean(), sens=sens, spec=spec, ppv=ppv, f1=f1, mcc=mcc)


rows = []
for tag, label in MODELS:
    d = pd.read_parquet(D / f"oof/{tag}.parquet"); d = d[d.y.notna()]
    col = "p_cal" if "p_cal" in d else "p_raw"
    for op in ("risk>=10%", "top 10%"):
        per = []
        for _, g in d.groupby("repeat"):
            y, p = g.y.values.astype(int), g[col].values
            flag = p >= 0.10 if op == "risk>=10%" else p >= np.quantile(p, 0.90)
            per.append(point(y, flag))
        R = pd.DataFrame(per)
        rows.append(dict(model=label, tag=tag, op=op, n_repeats=len(R), **{k: R[k].mean() for k in R}, **{k + "_sd": R[k].std() for k in R}))
T = pd.DataFrame(rows); T.to_csv(D / "results/threshold_metrics.csv", index=False)
print(T[["model", "op", "flagged", "sens", "spec", "ppv", "f1", "mcc"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- LaTeX table
f = lambda v: f"{v:.2f}"
lines = []
for label in [m[1] for m in MODELS]:
    a = T[(T.model == label) & (T.op == "risk>=10%")].iloc[0]; b = T[(T.model == label) & (T.op == "top 10%")].iloc[0]
    bold = label.startswith("M0 + pixel")
    cells = [f"{100 * a.flagged:.0f}\\%", f(a.sens), f(a.ppv), f(a.mcc), f(b.sens), f(b.ppv), f(b.mcc)]
    if bold: cells = [f"\\textbf{{{c}}}" for c in cells]; label = f"\\textbf{{{label}}}"
    lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
tex = r"""\begin{table}[!tbp]
  \centering
  \caption{\textbf{Threshold metrics at two operating points.} Left: calibrated risk $\ge$10\%, the threshold of the
  decision curve (the share of hips flagged depends on calibration). Right: the 10\% of hips with the highest risk (a
  fixed workload of about 170 hips, independent of calibration). Sensitivity, precision (positive predictive value) and
  Matthews correlation coefficient (MCC) are computed per repeat on pooled out-of-fold predictions and averaged over five
  repeats; the event rate is 5.5\%. The image head is calibrated on its cross-fitted inner predictions.}
  \label{tab:threshold}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lccccccc@{}}
    \toprule
    & \multicolumn{4}{c}{Risk $\ge$ 10\%} & \multicolumn{3}{c}{Top 10\% of hips} \\
    \cmidrule(lr){2-5}\cmidrule(l){6-8}
    Model & Flagged & Sens. & Prec. & MCC & Sens. & Prec. & MCC \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
(P / "report/sections/table_threshold.tex").write_text(tex, encoding="utf-8")
print("-> report/sections/table_threshold.tex")
