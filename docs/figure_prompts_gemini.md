# Gemini 作图提示词（报告 v1，2026-10-24 更新）

用法：每段英文提示词整段粘贴给 Gemini 图像生成，一次只生成一张图。生成后逐字核对（图像模型容易编造标签、画错下标、×、≤）。保存为 `report/figs/` 下对应文件名，tex 会自动换掉占位 / 草稿图。

配色约定（与 `91_figs.py` / `97_report_figs.py` 一致）：表格 = 蓝 #2a78d6（底 #e3eefb），影像 = 橙 #eb6834（底 #fde6dc），输出头 / 损失 / 融合 = 绿 #1baf7a（底 #dcf3ea），时序准入 / 防泄漏 = 紫 #6250d6（底 #e6e3f8），箭头 #898781，文字 #0b0b0b / #52514e，底色 #fcfcfb。

**10-24 晚重新编号**（报告 v2）：总览移到 Intro 末尾成为 Fig 1，时间线移到 Methods 成为 Fig 2，新增 M0 图为 Fig 5，后面顺延。下面各节标题里的旧图号以这张表为准。

| 报告图号 | 文件名 | 状态 |
|---|---|---|
| Fig 1 总览（输入 → 架构 → 输出） | `fig1_overview.png` | 已生成（原 `fig2_overview.png`） |
| Fig 5 **M0 elastic net（新）** | `fig5_m0.png` | **新写**，见文末「Fig 5：M0」；生成前 tex 显示占位框 |
| Fig 6 FT-Transformer | `fig6_ftt.png` | 已生成（原 `fig5_ftt.png`） |
| Fig 7 影像分支 | `fig7_image_branch.png` | 已生成（原 `fig6_image_branch.png`） |

---

## Fig 2：总览（模型输入 + 架构 + 输出）

```
Create a clean, publication-quality overview diagram for a NeurIPS paper, landscape 16:9, flat vector style, background #fcfcfb, thin 1.5 px rounded rectangles, sans-serif font (Inter or Segoe UI), no 3D, no shadows, no clip-art, no people, no logos. Use ONLY the text given below, spelled exactly; do not add any other words or numbers. Colours: clinical/tabular blue #2a78d6 (fill #e3eefb), imaging orange #eb6834 (fill #fde6dc), fusion and outputs green #1baf7a (fill #dcf3ea), time-admissibility violet #6250d6 (fill #e6e3f8), arrows grey #898781, text #0b0b0b and #52514e.

Title (top-left, bold): "Predicting 10-year hip replacement at first presentation (CHECK study)"

A thin violet vertical bar on the far left labelled vertically "t₀ = first consultation · only values fixed at t₀ enter the model".

THREE COLUMNS, left to right, connected by straight arrows.

COLUMN 1 — "Inputs (one hip = one row)":
  Top blue card "Clinical tables · 76 features" with six small chips in a 3×2 grid: "D demographics", "E hip examination", "S symptoms", "C care & drugs", "L social & work", "B biochemistry".
  Bottom orange card "Baseline AP pelvic radiograph" with a simple grey line drawing of a pelvis and a small square crop around one femoral head, caption "pixels only · no radiologist grading".
  Below both, a dashed grey card: "Years 1–10 follow-up → labels only (THA, censoring)".

COLUMN 2 — "Models":
  Blue panel "Tabular branch" with four stacked rows: "M0 elastic net (reference)", "M1 LightGBM", "M1b EBM", "M2 FT-Transformer (feature tokens + self-attention, two heads)".
  Orange panel "Image branch (M3)" with a left-to-right mini pipeline: "frozen DenseNet-121 · 14×14 map" → "residual head + attention pooling" → "THA logit". Under the residual head, a small violet tag: "11 OARSI reading items = training targets only".
  A green box "Fusion" receiving one arrow from "M0 elastic net" and one arrow from "THA logit", with text "elastic net on clinical features + 1 cross-fitted image logit".

COLUMN 3 — "Outputs":
  Green card "P(THA within 10 years)" with a small calibration-curve icon and caption "Platt-calibrated".
  Green card "2 / 5 / 10-year cumulative risk" with a tiny rising step curve, caption "M2 discrete-time head".
  Green card "Where the model looked" with a tiny heat map over a hip crop, caption "attention map".
  Grey card "Use: conservative-care intensity · review interval · referral timing · trial enrichment".

Bottom strip, full width, light grey: "Evaluation: 5 × 5 nested cross-validation grouped by participant · AUPRC (primary), AUROC, calibration, decision curve · temporal audit · pseudo-external transport".

Generous white space, aligned panel tops, straight arrows with small heads, all text horizontal and legible at 50% zoom.
```

---

## Fig 5：FT-Transformer（M2）—— 改动说明

与草稿比改了 5 处：
1. ⟦LayerNorm⟧ 被画成了带方括号的字样 → 直接写 `LayerNorm`（主版本是 LN；BN 只作消融，没胜出）。
2. 特征数「~80」→「76 clinical features in 6 groups」。
3. 二分类头的 readout：选中的 3 个架构里 2 个用 token 均值、1 个用 [CLS] → 写 `[CLS] or token mean → norm → ReLU → linear`。
4. 损失写全：`weighted BCE (pos_weight 5) or focal (γ = 2)`。
5. **训练条最重要**：主版本 `m2ens_DESCLB_lam1_dtinit_es_s3` **没有**用 warm-up + cosine、min-epoch 40、patience 30（那是 v2 消融，没胜出）。主版本是 early stopping on the TOTAL validation loss, patience 20, 3 seeds。

```
Create a clean, publication-quality architecture diagram for a NeurIPS paper, landscape 16:9, flat vector style, background #fcfcfb, thin 1.5 px rounded rectangles, sans-serif font (Inter or Segoe UI), no 3D, no shadows, no clip-art, no people. Use ONLY the text given below, spelled exactly; do not add any other words, numbers or logos. Never draw square brackets around words unless they are written below. Colours: tabular components blue #2a78d6 (fill #e3eefb), output heads and losses green #1baf7a (fill #dcf3ea), training notes violet #6250d6 (fill #e6e3f8), arrows grey #898781, text #0b0b0b and #52514e.

Title (top-left, bold): "M2 · FT-Transformer for 10-year hip replacement risk"

Layout left → right in five columns, connected by arrows:

Column 1 — "One hip = one row": a vertical stack of six small blue cards, each with a feature name and an example value:
  "age  58", "internal rotation ROM  22°", "pain on internal rotation  2/4", "WOMAC pain  MISSING", "hip pain (same side)  yes", "employment  category".
  Under the stack, small grey caption: "76 clinical features in 6 groups: demographics, examination, symptoms, care, social, biochemistry (+ asymmetry = same side − other side)".

Column 2 — "Feature tokenizer" (large blue panel). Inside, three stacked mini-diagrams:
  (a) a number x flowing into "value embedding  x·w_j + b_j" producing a short horizontal bar of 6 coloured cells labelled "d-dim token";
  (b) a "+" merging "feature-ID embedding e_j" and "group embedding (D / E / S / C / L / B)";
  (c) a missing value icon (empty circle) mapped to "learned MISS_j token", and a category mapped to "embedding table".
  Above the token sequence a distinct dark-blue token labelled "CLS".

Column 3 — "Token sequence": a vertical column of small square tokens: "CLS", "t_age", "t_IR-ROM", "t_IR-pain", "t_WOMAC", "…", "t_n". Faint attention lines fan between tokens.

Column 4 — "Transformer encoder × L" (large panel, blue outline), showing ONE block in detail, bottom-to-top:
  "LayerNorm" → "Multi-head self-attention (4–8 heads)" → circle-plus "+ residual" → "LayerNorm" → "ReGLU feed-forward" → circle-plus "+ residual". Draw the two residual skip arrows clearly bypassing each sub-layer.
  Small note at the bottom of the panel: "L = 1–3 · d = 48–64 · 34k–54k parameters (two-stage NAS)".

Column 5 — two green output panels, both fed from one readout arrow labelled "CLS or token mean":
  Top: "Binary head" with text "readout → norm → ReLU → linear", "P(THA within 10 years)", "weighted BCE (pos_weight 5) or focal (γ = 2)".
  Bottom: "Discrete-time head" with text "10 yearly hazard logits h_1…h_10", "bias initialised at baseline hazard (logit −5.3)", "S(t) = Π (1 − h_k) → 2 / 5 / 10-year risk", and a tiny step-shaped survival curve icon.
  Between them a green formula box: "L = L_binary + λ · L_discrete-time   (λ = 1; censored hips enter L_discrete-time)".

Bottom strip, full width, violet: "Training: AdamW · early stopping on the TOTAL validation loss (censored hips included), patience 20 · 3 seeds × 3 inner-fold models averaged · Platt recalibration on inner out-of-fold predictions".

Keep generous white space, align panel tops, arrows straight with small heads, all text horizontal and legible at 50% zoom.
```

注：提示词里把 `[CLS]`、`[MISS]` 写成 `CLS`、`MISS_j`，是为了防 Gemini 再把方括号当占位符画乱；如果它画得干净，可以手动在 PPT 里补回方括号。

---

## Fig 6：影像分支（M3）—— 改动说明

与草稿比改了 3 处：
1. 「Pixel-derived features = predicted OARSI items + image THA logit」**与最终模型不符**：主融合模型只用 **THA logit** 一个特征（`m5_pix_logit_DESCLB`）；11 项预测值只做过对照（不更好）。
2. 加上交叉拟合的说明：训练行的 logit 来自没见过它的内折头。
3. 「Shortcut and bias controls」第 3 条措辞对齐结果：去中心×格式均值后结局探针不变（0.725 → 0.730）。

```
Create a clean, publication-quality architecture diagram for a NeurIPS paper, landscape 16:9, flat vector style, background #fcfcfb, thin 1.5 px rounded rectangles, sans-serif font, no 3D, no shadows, no people, no logos. Use ONLY the text given below, spelled exactly. Colours: image pipeline orange #eb6834 (fill #fde6dc), learned head and outputs green #1baf7a (fill #dcf3ea), tabular blue #2a78d6 (fill #e3eefb), controls violet #6250d6 (fill #e6e3f8), arrows grey #898781, text #0b0b0b / #52514e.

Title (top-left, bold): "M3 · Image branch: from a raw pelvic radiograph to one pixel-derived risk score"

TOP ROW (left → right, orange, arrows between):
 1. "AP pelvis radiograph" — a simple grey line drawing of a pelvis with both femoral heads; caption "TIFF scan or DICOM · 1,123 films".
 2. "Orientation fix" — the pelvis icon rotated 90° with a curved arrow back to upright; caption "4-way rotation classifier (self-supervised) · 98.5% accuracy · ~110 films corrected".
 3. "Femoral-head locator" — the upright pelvis with two small heat-map blobs on the femoral heads and a circle around each; caption "ResNet-18 U-net heat maps · two peaks → patient right / left · median error 0.27 head radius".
 4. "Hip crop" — a square crop showing one femoral head and acetabulum; caption "2.5 × head diameter · 256² · 16-bit · left hips mirrored".

MIDDLE ROW (encoder and head), drawn larger:
 5. "Frozen DenseNet-121 (ImageNet) up to dense block 3" — orange trapezoid narrowing left→right; output a grid "1024 × 14 × 14 feature map". Small violet tag: "frozen · BatchNorm statistics fixed".
 6. "Trainable residual head" — green panel containing, left → right: "1×1 conv 1024→64 + BatchNorm + ReLU", then two blocks each drawn as "3×3 conv · BN · ReLU · 3×3 conv · BN" with a curved skip arrow labelled "identity skip", then a split into "attention pooling (where to look)" and "global average pooling", joined by a bracket labelled "concatenated". Next to attention pooling a tiny 14×14 heat map over a hip crop, bright over the superior joint space.
 7. Two green output stacks from the head:
    "Auxiliary targets (training only): 11 OARSI reading items" — small text: "KL · JSN superior · JSN medial · osteophytes ×4 · sclerosis · cysts · flattening · buttressing";
    "THA logit".
    Under them: "loss = weighted BCE(THA) + masked MSE(11 items)".

BOTTOM ROW (fusion):
 8. Green box "Cross-fitted image THA logit" with small text "training hips: scored by a head trained on the other inner folds · test hips: head trained on the full training set" → arrow into
 9. Blue box "Elastic-net fusion", which also receives an arrow from a blue box "clinical features (76)" → output green box "P(THA within 10 years)".
 Dashed grey side note: "reading scores are training targets only; at prediction time the model sees pixels only".

RIGHT COLUMN, violet, narrow, titled "Shortcut and bias controls": three bullet lines
  "crop removes burned-in IDs, markers, collimation masks",
  "probe: embedding still predicts centre (AUROC 0.90) and format (0.93)",
  "outcome signal unchanged after removing centre × format means".

Keep the three rows aligned, consistent spacing, straight arrows, all text horizontal and readable at 50% zoom.
```

### 生成后核对清单
- Fig 2：6 个组字母和名称；「76」；融合只接 1 个 image logit。
- Fig 5：没有方括号乱码；训练条是 patience 20（不是 30）、没有 warm-up；参数量 34k–54k。
- Fig 6：第 8 格只有 THA logit；数字 1,123 / 98.5% / ~110 / 0.27 / 256² / 0.90 / 0.93。
- 逐字检查拼写；需要的话导出后在 PPT / Figma 里补字。

---

## Fig 5：M0 —— 临床 elastic net（基线模型 + 主模型的临床部分）（10-24 晚新写）

要点：M0 既是纯临床基线，也是最终融合模型的「骨架」——融合模型就是同一个 elastic net 多接一列交叉拟合的 image logit。图里要把这两种用法画成同一个学习器的上下两条路，而不是两个模型。数字已核对：40 组候选；C 1.5×10⁻³–9.6；内层 25 折里 20 折选了 ℓ1 比例 0.1–0.3（偏 ridge）。

```
Create a clean, publication-quality architecture diagram for a NeurIPS paper, landscape 16:9, flat vector style, background #fcfcfb, thin 1.5 px rounded rectangles, sans-serif font (Inter or Segoe UI), no 3D, no shadows, no clip-art, no people, no logos. Use ONLY the text given below, spelled exactly; do not add any other words or numbers. Never draw square brackets around words. Colours: clinical/tabular blue #2a78d6 (fill #e3eefb), image-derived input orange #eb6834 (fill #fde6dc), model outputs and calibration green #1baf7a (fill #dcf3ea), leakage-control notes violet #6250d6 (fill #e6e3f8), arrows grey #898781, text #0b0b0b and #52514e.

Title (top-left, bold): "M0 · Clinical elastic net: the reference model and the backbone of the final model"

Layout left → right in four columns, connected by straight arrows.

Column 1 — "Baseline clinical features (t₀ only)": a vertical stack of six small blue chips: "D demographics", "E hip examination", "S symptoms", "C care & drugs", "L social & work", "B biochemistry". Under the stack a small grey caption: "76 features · same side + asymmetry (same side − other side)".

Column 2 — "Fold-wise preprocessing" (tall blue panel). Six numbered steps, top to bottom, each a thin row with a tiny icon:
  "1  structural rules (skipped items, non-workers)"
  "2  log1p / Yeo-Johnson for skewed variables"
  "3  winsorise at 0.5 / 99.5 %"
  "4  missing indicators"
  "5  MICE imputation (BayesianRidge)"
  "6  z-score · one-hot"
  A violet tag attached to the panel: "fitted on training rows only · test rows only transformed".

Column 3 — "Penalised logistic regression" (central blue panel, drawn largest). Inside:
  the formula "minimise  log-loss + λ [ α‖w‖₁ + (1 − α)‖w‖₂² ]" written cleanly on one line,
  below it a small diagram of correlated features: two overlapping blue circles labelled "WOMAC pain" and "WOMAC function", with two short arrows of equal length into a node labelled "shared weight (ℓ2 part)", and one faded circle with a cross labelled "dropped feature (ℓ1 part)".
  A small grey box at the bottom of the panel: "40 candidate settings · C 0.0015–9.6 · α 0–1 · class weight on / off · chosen on 3 inner folds by Platt-recalibrated log-loss".

Column 4 — two green output panels stacked:
  Top: "Clinical model (M0)" → "Platt recalibration" → "P(THA within 10 years)".
  Bottom: "Final model (fusion)" → "Platt recalibration" → "P(THA within 10 years)".

TWO INPUT ROUTES into Column 3, drawn as two horizontal lanes:
  Upper lane (blue arrows only): preprocessed clinical features → penalised logistic regression → top output panel "Clinical model (M0)".
  Lower lane: the same blue arrow PLUS one orange arrow from a small orange box on the lower left labelled "cross-fitted image logit (one extra column)" → the same penalised logistic regression → bottom output panel "Final model (fusion)". Next to the orange box a tiny violet note: "each training hip scored by a head that never saw it".

Bottom strip, full width, light grey: "Evaluation: 5 × 5 nested cross-validation grouped by participant · the same folds for every model · AUPRC primary".

Generous white space, aligned panel tops, straight arrows with small heads, all text horizontal and legible at 50% zoom.
```

核对清单：公式里的 ‖w‖₁、‖w‖₂²、λ、α 没画错；「40」「0.0015–9.6」「3 inner folds」；下方车道只多一条橙色箭头（融合 = 同一个学习器 + 1 列）；没有出现 FT-Transformer 或树模型的字样。保存为 `report/figs/fig5_m0.png`，tex 会自动替换占位框。
---

## Fig 1 总览 —— 箭头修正（10-25，在 Gemini 里对现有图做「编辑」，不要重新生成）

已生成的 `fig1_overview.png` 有 3 处连线错误，文字都对：
1. 「Years 1–10 follow-up → labels only」的箭头指向了 Fusion 框，像是随访数据进了模型 → 改成一条**虚线**，指向底部的 Evaluation 条（标签只用于训练和评估）。
2. **Fusion 框没有连到输出**：最终的 P(THA within 10 years) 应该来自 Fusion → 加一条从 Fusion 右侧到「P(THA within 10 years)」卡片的绿色箭头。M0 到 P(THA) 的原箭头可保留（M0 是对照）。
3. 「Where the model looked / attention map」的连线是从表格分支引出的，错 → 应从 Image branch 的「residual head + attention pooling」引出。

```
Edit the attached diagram. Keep every word, colour, box and position exactly as they are; change only three connectors:
(1) Replace the solid arrow from the dashed box "Years 1–10 follow-up → labels only (THA, censoring)" into the "Fusion" box with a thin dashed grey arrow from that dashed box down into the grey bottom strip that begins "Evaluation:". No arrow may touch the "Fusion" box from the follow-up box.
(2) Add a green arrow from the right edge of the green "Fusion" box to the left edge of the green card "P(THA within 10 years)".
(3) Remove the grey connector that runs from the "Tabular branch" panel to the card "Where the model looked". Instead draw an orange arrow from the box "residual head + attention pooling" in the orange "Image branch (M3)" panel to the card "Where the model looked".
Do not add any new text.
```