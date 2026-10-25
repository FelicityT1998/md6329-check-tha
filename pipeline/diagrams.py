"""Architecture / workflow diagrams for the report and plan section 7.9.

usage: python -I diagrams.py <project dir>   -> data/derived/figs/arch_*.png
Colour roles (reference palette, fixed order): tabular = blue, image = orange, heads/loss = aqua, adversarial = violet,
evaluation = yellow; neutral boxes are white with a hairline border.
"""
import sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = pathlib.Path(sys.argv[1]); OUT = P / "data/derived/figs"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
TINT = {BLUE: "#e3eefb", ORANGE: "#fde6dc", AQUA: "#dcf3ea", YELLOW: "#fcefd0", VIOLET: "#e6e3f8", None: "#ffffff"}
INK, INK2, MUTED, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF, "savefig.facecolor": SURF})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); return fig, ax


def box(ax, x, y, w, h, title, body="", role=None, fs=8.5, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=TINT[role],
                                ec=role or AXIS, lw=1.3 if role else 0.9, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=fs, color=INK, weight="bold")
    if body: ax.text(x + w / 2, y + h - 0.38, body, ha="center", va="top", fontsize=fs - 1.3, color=INK2, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a, b, side_a="r", side_b="l", label="", color=MUTED, rad=0.0):
    def pt(bx, s):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[s]
    p, q = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1, connectionstyle=f"arc3,rad={rad}"))
    if label: ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.08, label, fontsize=6.5, color=INK2, ha="center")


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight"); plt.close(fig); print("->", OUT / name)


# ------------------------------------------------------------------ 1 pipeline overview
fig, ax = canvas(15, 6.6)
a = box(ax, 0.2, 4.3, 2.4, 1.9, "CHECK T0 tables", "371 variables\n1,002 participants\n→ 2,004 hips", BLUE)
b = box(ax, 0.2, 1.6, 2.4, 1.9, "Baseline AP pelvis", "1,123 films\nTIFF 637 / DICOM 487\n10 recovered centres", ORANGE)
c = box(ax, 0.2, 0.1, 2.4, 1.1, "T1–T10 follow-up", "labels & censoring only", None, dashed=True)
d = box(ax, 3.2, 4.3, 2.6, 1.9, "B1 tabular prep", "structural skips → rules\ncentre-missing → excluded\nMICE / [MISS] · YJ / log1p\nz-score · asymmetry terms", BLUE)
e = box(ax, 3.2, 1.6, 2.6, 1.9, "B4 hip crops", "femoral-head locator\n(click labels → heatmap CNN)\n2.5×D box · 256² · L→R flip", ORANGE)
f = box(ax, 6.4, 4.3, 2.5, 1.9, "M0 · M1 · M1b", "elastic net\nLightGBM\nEBM (GA2M)", BLUE)
g = box(ax, 6.4, 1.6, 2.5, 1.9, "M3 image encoder", "ResNet-18\nImageNet → SimCLR\n(+L2-SP, frozen stem)", ORANGE)
h = box(ax, 9.5, 2.9, 2.6, 2.1, "M2 / M4 FT-Transformer", "tabular tokens + [I] token\n→ pre-LN encoder ×L\n→ binary + discrete-time heads", AQUA)
i = box(ax, 12.6, 4.1, 2.2, 2.1, "Evaluation", "5×5 nested grouped CV\nAUPRC · AUROC · calibration\ncluster bootstrap", YELLOW)
j = box(ax, 12.6, 1.4, 2.2, 2.3, "Experiments", "A incremental value\nB pseudo-external (centre,\n   format, sex, pain …)\nC SHAP · attention · Grad-CAM\nshortcut probes", YELLOW)
k = box(ax, 3.2, 0.1, 2.6, 1.1, "01 cohort / 04 folds", "y10 · time · event · folds", None, dashed=True)
for s, t in [(a, d), (b, e), (d, f), (e, g), (f, i)]: arrow(ax, s, t)
arrow(ax, d, h, "r", "l", rad=-0.15); arrow(ax, g, h, "r", "l", rad=0.15)
arrow(ax, h, i, "r", "l"); arrow(ax, h, j, "r", "l"); arrow(ax, c, k, "r", "l")
ax.text(0.2, 6.45, "End-to-end pipeline: baseline-only inputs, follow-up used for labels; every fitted step lives inside the training fold",
        fontsize=10, color=INK, weight="bold")
save(fig, "arch_pipeline.png")

# ------------------------------------------------------------------ 2 FT-Transformer block (B2 + B3 + B6)
fig, ax = canvas(14, 7.2)
ax.text(0.2, 7.0, "M2 FT-Transformer: feature tokenizer, encoder block and two heads", fontsize=10, weight="bold", color=INK)
feats = [("age", "0.42"), ("ir_rom", "−1.3"), ("ir_pain", "2"), ("womac_pain", "[MISS]"), ("hip_pain", "1"), ("employment", "cat")]
for n, (nm, v) in enumerate(feats):
    y0 = 6.0 - n * 0.8
    box(ax, 0.2, y0, 1.6, 0.6, nm, v, BLUE if v != "cat" else None, fs=7.5)
tk = box(ax, 2.4, 1.3, 2.9, 5.3, "B2 tokenizer", "numeric value embedding\n  linear  x·w_j + b_j\n  periodic  [sin, cos](2πc_j x)·W_j\n  piecewise-linear (fold quantile bins)\n\n+ feature-ID embedding e_j\n+ group embedding (D/E/S/C/L/B/R)\n\nmissing → learned [MISS]_j\ncategorical → embedding table\n\nprepend [CLS]", BLUE, fs=8)
for n in range(6): arrow(ax, (0.2, 6.0 - n * 0.8, 1.6, 0.6), tk)
seq = box(ax, 5.8, 1.3, 1.5, 5.3, "tokens", "[CLS]\nt₁\nt₂\nt₃\n…\nt_n\n([I])", None, fs=8)
arrow(ax, tk, seq)
blk = box(ax, 7.8, 1.3, 3.0, 5.3, "B3 encoder block × L", "LayerNorm\n↓\nMulti-head self-attention\n(heads 2–8, attn dropout)\n↓  + residual\nLayerNorm\n↓\nReGLU feed-forward\n(ffn_mult · d)\n↓  + residual\n\nNAS: d ∈ {16…64}, L ∈ {1,2,3}", AQUA, fs=8)
arrow(ax, seq, blk)
h1 = box(ax, 11.4, 4.2, 2.4, 1.6, "Binary head", "[CLS] → LN → ReLU → 1\nP(THA ≤ 10 y)\nweighted BCE / focal", AQUA, fs=7.8)
h2 = box(ax, 11.4, 2.1, 2.4, 1.6, "Discrete-time head", "[CLS] → 10 yearly hazards\nNLL incl. censored hips\nS(t) = Π(1 − h_k)", AQUA, fs=7.8)
arrow(ax, blk, h1, rad=-0.1); arrow(ax, blk, h2, rad=0.1)
ax.text(11.4, 1.5, "L = L_bin + λ·L_dt   (λ ∈ {0, 0.5, 1})\nPlatt recalibration on inner OOF", fontsize=7.5, color=INK2)
save(fig, "arch_ftt_block.png")

# ------------------------------------------------------------------ 3 image branch + anti-forgetting
fig, ax = canvas(14, 6.4)
ax.text(0.2, 6.2, "M3 image branch: localisation → crop → encoder; SimCLR continued pre-training with forgetting guards", fontsize=10, weight="bold", color=INK)
a = box(ax, 0.2, 3.4, 2.3, 2.2, "AP pelvis film", "TIFF scan / DICOM\nburned-in ID, masks,\ncalibration ball", ORANGE, fs=8)
b = box(ax, 3.0, 3.4, 2.4, 2.2, "Head locator", "ResNet-18 U-net, 384²\nheatmaps: screen-L/R centre\n+ radius map\ntrained on ~150 clicked films", ORANGE, fs=8)
c = box(ax, 5.9, 3.4, 2.3, 2.2, "Crop", "square 2.5 × diameter\n→ 256² · 16-bit\nlocal 1–99% window\nleft hips mirrored", ORANGE, fs=8)
d = box(ax, 8.7, 3.4, 2.5, 2.2, "Encoder", "ResNet-18\n(a) ImageNet frozen\n(b) SimCLR-adapted\n(c) last block fine-tuned", ORANGE, fs=8)
e = box(ax, 11.7, 3.4, 2.1, 2.2, "Embedding", "512-d → fold PCA-32\n→ linear to d_token\n= [I] token", ORANGE, fs=8)
for s, t in [(a, b), (b, c), (c, d), (d, e)]: arrow(ax, s, t)
g = box(ax, 5.9, 0.2, 7.9, 2.6, "Catastrophic-forgetting guards (SimCLR & fine-tuning)",
        "start from ImageNet weights · stem + layer1 frozen · layer-wise LR decay (1e-4 top, cosine)\n"
        "L2-SP penalty α‖θ − θ_ImageNet‖² instead of decay to zero · BatchNorm statistics frozen when fine-tuning\n"
        "monitor every 10 epochs: linear-probe log-loss (training-fold inner CV) + linear CKA vs ImageNet features → early stop\n"
        "SimCLR uses unlabeled T0 crops (transductive, disclosed); per-outer-fold pre-training as sensitivity", VIOLET, fs=8.3)
arrow(ax, g, d, "t", "b")
h = box(ax, 0.2, 0.2, 5.2, 2.6, "Shortcut controls", "crop excludes burned-in text, R/L markers, masks\ncalibration ball inpainted if inside the box\nintensity augmentations (gamma, contrast, noise, blur)\nprobe: predict centre / format from embedding", None, fs=8.3)
save(fig, "arch_image_branch.png")

# ------------------------------------------------------------------ 4 fusion variants + heads + adversary
fig, ax = canvas(14, 6.0)
ax.text(0.2, 5.8, "M4 fusion: image as a token (main) vs late stacking vs FiLM; optional centre adversary", fontsize=10, weight="bold", color=INK)
t1 = box(ax, 0.2, 3.3, 4.2, 2.1, "(main) [I] as a token", "[CLS] t₁ … t_n [I] → encoder\nattention maps show exam × image\ninteraction (FAIS hypothesis)", AQUA, fs=8.2)
t2 = box(ax, 4.9, 3.3, 4.2, 2.1, "late stacking", "p_tab (M2) , p_img (M3)\n→ logistic on inner OOF predictions\nstable; no interaction", None, fs=8.2)
t3 = box(ax, 9.6, 3.3, 4.2, 2.1, "FiLM", "[I] → (γ, β)\nz = z_CLS ⊙ (1 + γ) + β\nfew parameters", None, fs=8.2)
h = box(ax, 3.0, 0.3, 4.2, 2.2, "Heads (B6)", "binary 10-y logit\n+ discrete-time yearly hazards\nPlatt-calibrated", AQUA, fs=8.2)
g = box(ax, 8.2, 0.3, 4.6, 2.2, "Centre adversary (ablation G)", "gradient reversal → centre classifier\non [CLS] (or [I])\nalternative: centre × outcome balanced sampling", VIOLET, fs=8.2)
arrow(ax, t1, h, "b", "t"); arrow(ax, t1, g, "b", "t", rad=-0.2)
save(fig, "arch_fusion_heads.png")

# ------------------------------------------------------------------ 5 nested CV + HPO + two-stage NAS
fig, ax = canvas(14, 6.4)
ax.text(0.2, 6.2, "Resampling, hyper-parameter search and two-stage NAS (all splits by participant)", fontsize=10, weight="bold", color=INK)
a = box(ax, 0.2, 3.2, 3.0, 2.6, "Outer CV", "5 folds × 5 repeats\nstratified: THA count × format\n17–20 events per test fold\nevery hip tested once / repeat", YELLOW, fs=8)
b = box(ax, 3.8, 3.2, 3.0, 2.6, "Inner CV (3 folds)", "inside each outer train set\nselection: Platt-recalibrated\nlog-loss (not AUPRC)\nearly stopping: val BCE", YELLOW, fs=8)
c = box(ax, 7.4, 3.9, 3.0, 1.9, "Random search", "M0: 40 configs (C, l1_ratio, weights)\nM1: 60 configs (leaves, depth, lr, …)\nM1b: 4 configs\nidentical list in every fold", BLUE, fs=8)
d = box(ax, 7.4, 1.3, 3.0, 2.2, "NAS stage 1", "48 FT-Transformer architectures\n(3.7k–56k params)\nrepeat-0 / outer-0 inner folds\n→ top-3", AQUA, fs=8)
e = box(ax, 11.0, 1.3, 2.8, 2.2, "NAS stage 2", "top-3 × lr {3e-4, 1e-3}\n× wd {1e-5, 1e-4}\nnested in all 25 outer folds", AQUA, fs=8)
f = box(ax, 11.0, 3.9, 2.8, 1.9, "Refit", "outer train set\nmedian inner best epoch\n3 seeds averaged · Platt", YELLOW, fs=8)
g = box(ax, 0.2, 0.3, 6.6, 2.3, "Pooled out-of-fold predictions", "AUPRC (primary) · AUROC · calibration slope / intercept · Brier · decision curve\n95% CI: participant-clustered bootstrap (2,000)\nExperiment B: train on source domain, test on held-out centre / format / sex / pain / group", YELLOW, fs=8)
arrow(ax, a, b); arrow(ax, b, c); arrow(ax, b, d, rad=0.2); arrow(ax, d, e); arrow(ax, c, f); arrow(ax, e, f, "t", "b"); arrow(ax, f, g, "b", "r", rad=-0.3)
save(fig, "arch_cv_hpo_nas.png")
