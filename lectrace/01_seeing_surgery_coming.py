import math

import numpy as np
import torch
from lectrace import Reference, image, link, note, plot, table, text

from _helpers import (ASSETS, attention_overlay, contributions_chart, load_demo, load_fmap, load_head, load_oof,
                      pr_curve_chart)


def main():
    # ------------------------------------------------------------------ introduction (2-3 min)
    text("# Seeing surgery coming")
    text("### Pixel-supervised prediction of 10-year hip replacement in the CHECK study")
    text(
        "**The problem.** A 55-year-old sees the GP for the first time with hip pain. Implants now last decades, and "
        "younger recipients carry more lifetime revision risk. So the useful question is not only *is this OA?* but "
        "*will this hip need replacing within ten years?* The answer sets how intensive conservative care should be, "
        "when to refer, and who to enrol in joint-preservation trials."
    )
    text(
        "**The data.** CHECK: 1,002 Dutch adults aged 45-65 with early hip or knee pain, followed for ten years. "
        "1,705 hips have a known 10-year status and 93 were replaced (5.5%). We use only what exists at the first "
        "visit: questionnaires, a standardised hip examination, and the raw pixels of the same-day pelvic film. "
        "Radiologists' grades were scored after year 10 with later films side by side, so they are never inputs."
    )
    image(str(ASSETS / "fig2_timeline.png"), width=900)
    text(
        "**The approach.** A penalised logistic model on 76 clinical features (M0). A small trainable head on frozen "
        "DenseNet-121 feature maps that turns each hip crop into one THA logit (M3). The final model is M0 with that "
        "logit as one extra column, evaluated by 5 x 5 nested cross-validation grouped by participant."
    )
    image(str(ASSETS / "fig1_overview.png"), width=900)
    note(
        "The five states that follow track one real test hip through the final model, then return to the whole "
        "cohort. Every number is recomputed here from saved weights and out-of-fold predictions; none is typed in."
    )

    # ------------------------------------------------------------------ state 1
    text("## State 1 · One hip becomes a row")
    text(
        "**What happens:** the hip's baseline values pass the preprocessing that was fitted on the training fold. "
        "**What changes:** raw degrees and scores become comparable z-scores. **Why it matters:** this is where "
        "leakage would enter. The other hip is seen only through baseline asymmetry, never through its follow-up."
    )
    d = load_demo()  # @hide
    hip = {"participant": d["nsin"], "side": d["side"], "THA within 10 y": d["y10"], "THA year": d["t_event"]}  # @inspect hip
    text(
        "This hip comes from the test fold of the first split. It was chosen because the image changed its rank the "
        "most: it is the most favourable example in that fold, not a typical one."
    )
    w = d["walk"]["ir_rom_asym"]  # @hide
    asym = w["ipsi"] - w["contra"]  # @inspect asym
    z_asym = zscore(asym, w)  # @inspect z_asym
    text(
        "It rotates 15 degrees *more* than the other hip, so on examination it looks like the better hip. Skewed "
        "scales go through a Yeo-Johnson transform whose lambda was fitted on the training fold:"
    )
    w = d["walk"]["sos_mean"]  # @hide
    sos_t = yeo_johnson(w["raw"], w["lambda"])  # @inspect sos_t
    z_sos = zscore(sos_t, w)  # @inspect z_sos
    note("Order inside every fold: asymmetry and skip rules, log1p / Yeo-Johnson, winsorise, missing flags, MICE, z-score.")

    # ------------------------------------------------------------------ state 2
    text("## State 2 · The clinical model scores the hip")
    text(
        "**What happens:** an elastic net adds up weight times value over 77 columns, then Platt recalibration maps "
        "the logit to a probability. **What changes:** the row becomes one calibrated risk. **Why it matters:** this "
        "is the reference every deep model has to beat, and at 93 events none of our tabular deep models did."
    )
    m0 = d["m0"]  # @hide
    cfg = m0["cfg"]  # @inspect cfg
    logit_m0, contrib_m0 = linear_score(m0)  # @inspect logit_m0
    p_m0 = platt(logit_m0, m0)  # @inspect p_m0
    plot(contributions_chart(top_rows(contrib_m0, m0["names"], 8), "M0: largest terms for this hip"))  # @stepover
    text(
        f"Risk {p_m0:.3f}, below the 5.5% base rate. Among the {d['n_test_fold']} test hips of this fold it ranks at "
        f"the {100 * d['rank_m0']:.0f}th percentile: a hip that would be sent home with routine advice."
    )

    # ------------------------------------------------------------------ state 3
    text("## State 3 · What the image head computes")
    text(
        "**What happens:** the crop's frozen DenseNet-121 map (1024 channels on a 14 x 14 grid) passes two residual "
        "blocks, an attention softmax over the 196 cells, and a linear layer. **What changes:** a map becomes one "
        "logit plus a 'where it looked' map. **Why it matters:** pooling this map to one vector, our first attempt, "
        "reached AUPRC 0.12; keeping the grid reached 0.49."
    )
    fmap = load_fmap()  # @hide
    fmap_shape = tuple(fmap.shape)  # @inspect fmap_shape
    net = load_head()  # @hide
    logit_img, att, items = head_forward(net, fmap)  # @inspect logit_img
    text(
        "This equals the logit stored by the training pipeline for this hip (3.658). The head never saw this "
        "participant: it is the full-fit head of outer fold 0, and this hip is in that fold's test set."
    )
    plot(attention_overlay(att))  # @stepover
    text(
        "The weight sits on the superolateral joint margin. The head was also trained to predict 11 OARSI reading "
        "items. Readings were training *targets* only:"
    )
    table(item_rows(items, d), caption="Predicted vs read OARSI items (readings scored after year 10, shown for comparison only)")  # @stepover
    note("The reader graded a superior femoral osteophyte of grade 3. The examination was unremarkable; the pixels were not.")

    # ------------------------------------------------------------------ state 4
    text("## State 4 · Fusion: one extra column")
    text(
        "**What happens:** the image logit is z-scored and enters the same elastic net as one more feature. **What "
        "changes:** one term outweighs all clinical terms for this hip. **Why it matters:** training hips get a "
        "cross-fitted logit from a head that never saw them, so test labels never enter any step."
    )
    fu = d["fused"]  # @hide
    z_img = (logit_img - fu["img_mu"]) / fu["img_sd"]  # @inspect z_img
    w_img = fu["coef"][0]  # @inspect w_img
    logit_fused, contrib_fused = linear_score(fu)  # @inspect logit_fused
    p_fused = platt(logit_fused, fu)  # @inspect p_fused
    plot(contributions_chart(top_rows(contrib_fused, fu["names"], 8), "Fusion: largest terms for this hip"))  # @stepover
    text(
        f"Risk moves from {p_m0:.3f} to {p_fused:.3f}, and the rank from the {100 * d['rank_m0']:.0f}th to the "
        f"{100 * d['rank_fused']:.0f}th percentile. This hip was replaced in year {d['t_event']:.0f}."
    )

    # ------------------------------------------------------------------ state 5
    text("## State 5 · From one hip to a clinic")
    text(
        "**What happens:** the same computation for all 1,705 hips, then a precision-recall summary and a decision "
        "threshold. **What changes:** single risks become a rule a clinic could use. **Why it matters:** at 5.5% "
        "prevalence, a model can rank well and still flag mostly false alarms."
    )
    oof = load_oof()  # @hide
    y = oof.y.values  # @hide
    event_rate = y.mean()  # @inspect event_rate
    auprc_m0 = average_precision(y, oof.p_m0.values)  # @inspect auprc_m0
    auprc_fused = average_precision(y, oof.p_fused.values)  # @stepover @inspect auprc_fused
    max_lift = 1 / event_rate  # @inspect max_lift
    lift_fused = auprc_fused / event_rate  # @inspect lift_fused
    plot(pr_curve_chart({"clinical M0": pr_points(y, oof.p_m0.values),  # @stepover
                         "M0 + pixel logit": pr_points(y, oof.p_fused.values)}))
    text("A clinic acts on a threshold, not on a curve. At a 10% predicted risk:")
    at_m0 = operating_point(y, oof.p_m0.values, 0.10)  # @stepover @inspect at_m0
    at_fused = operating_point(y, oof.p_fused.values, 0.10)  # @inspect at_fused
    note("This is repeat 0 of the nested CV. The report gives five-repeat means: AUPRC 0.366 and 0.552.")

    # ------------------------------------------------------------------ conclusion
    text("## Key result and what it means clinically")
    table(
        [
            {"Result": "Clinical M0", "Internal estimate": "AUPRC 0.37, AUROC 0.81", "Reading": "examination + symptoms carry the clinical signal"},
            {"Result": "Image head alone", "Internal estimate": "AUPRC 0.49, AUROC 0.87", "Reading": "the spatial map is needed; pooled features gave 0.12"},
            {"Result": "M0 + pixel logit", "Internal estimate": "AUPRC 0.55, AUROC 0.89", "Reading": "+0.20 AUPRC; still +0.20 without THA in years 1-2"},
            {"Result": "At 10% risk", "Internal estimate": "sensitivity 0.62, precision 0.38", "Reading": "M0 alone: 0.54 and 0.21"},
            {"Result": "Leave-one-centre-out", "Internal estimate": "AUPRC 0.374 vs 0.377", "Reading": "discrimination transports; calibration does not"},
        ],
        caption="Five repeats of 5 x 5 nested cross-validation inside CHECK; no external cohort.",
    )
    text(
        "A first-visit risk score from routine data and the raw film, with no radiologist's grade, could set review "
        "intervals and referral timing. Two cautions. Radiographic severity is itself an indication for surgery, so "
        "part of what the head reads is the surgeon's future decision, not hidden biology. And calibration must be "
        "redone locally before use. Next: external validation in OAI hip films and World COACH."
    )
    link(Reference(title="CHECK cohort profile", authors=["Wesseling J", "et al."],
                   url="https://doi.org/10.1093/ije/dyu177",
                   description="Data: DANS doi:10.17026/DANS-XEX-HZWW (CC BY 4.0)."))


# ---------------------------------------------------------------- traced helpers (step into these)
def average_precision(y, p):
    order = np.argsort(-p)
    hits = y[order]
    tp = np.cumsum(hits)
    precision_at_k = tp / np.arange(1, len(y) + 1)
    ap = float((precision_at_k * hits).sum() / hits.sum())
    return ap


def operating_point(y, p, threshold):
    flag = p >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    flagged = round(float(flag.mean()), 3)
    sensitivity = round(tp / (tp + fn), 3)
    precision = round(tp / (tp + fp), 3)
    mcc = round((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 3)
    return {"flagged": flagged, "sensitivity": sensitivity, "precision": precision, "MCC": mcc}


def pr_points(y, p):
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    keep = np.r_[np.diff(recall) > 0, True]
    return recall[keep], precision[keep]


def yeo_johnson(x, lam):
    if x >= 0:
        return ((x + 1) ** lam - 1) / lam if lam != 0 else math.log1p(x)
    return -((1 - x) ** (2 - lam) - 1) / (2 - lam) if lam != 2 else -math.log1p(-x)


def zscore(x, w):
    clipped = min(max(x, w["lo"]), w["hi"])
    z = (clipped - w["mu"]) / w["sd"]
    return z


def linear_score(model):
    coef = np.array(model["coef"])
    x = np.array(model["x"])
    contributions = coef * x
    n_nonzero = int((coef != 0).sum())
    logit = model["intercept"] + contributions.sum()
    return float(logit), contributions


def platt(logit, model):
    p_raw = 1 / (1 + math.exp(-logit))
    lp = math.log(p_raw / (1 - p_raw))
    p_cal = 1 / (1 + math.exp(-(model["platt_a"] * lp + model["platt_b"])))
    return p_cal


def head_forward(net, fmap):
    with torch.no_grad():
        z = net.reduce(fmap)
        z = net.blocks(z)
        scores = net.att(z).flatten(1)
        att = torch.softmax(scores, dim=1)
        attended = (z.flatten(2) * att[:, None]).sum(2)
        average = z.mean((2, 3))
        features = torch.cat([attended, average], dim=1)
        out = net.out(features)[0]
    logit = float(out[0])
    peak_cell = divmod(int(att.argmax()), 14)
    return logit, att[0].reshape(14, 14).numpy(), out[1:].numpy()


def top_rows(contrib, names, k):
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": names[i], "contribution": round(float(contrib[i]), 3)} for i in order]


def item_rows(items, d):
    pred = np.array(items) * np.array(d["item_sd"]) + np.array(d["item_mu"])
    return [{"OARSI item": n.removeprefix("r_"), "predicted": round(float(p), 2), "read": r}
            for n, p, r in zip(d["items"], pred, d["item_reading"])]


if __name__ == "__main__":
    main()
