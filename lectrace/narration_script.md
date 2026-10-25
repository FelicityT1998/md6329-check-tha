# Narration script: Seeing Surgery Coming (LecTrace, about 10 minutes)

Spoken text for the step-through lecture at http://localhost:7000 (or the GitHub Pages copy). **[→]** = press the
right arrow; **[⇧→]** = Shift+right (step over); **[u]** = step out. The numbers below are the traced values; they
come from repeat 0 of the nested cross-validation unless marked "report".

---

## Introduction (0:00–2:30)

**[title screen]**
A 55-year-old sees their GP for the first time with hip pain. Hip implants now last decades, and the younger the
patient, the more lifetime revision risk they carry. So the useful question at this visit is not only "is this
osteoarthritis?" It is "will this hip need replacing within ten years?" That answer decides how hard to push
conservative care, when to refer, and who to enrol in joint-preservation trials.

**[→ data, timeline figure]**
We use CHECK, a Dutch cohort of 1,002 adults aged 45 to 65 with early hip or knee complaints, followed for ten years.
1,705 hips have a known ten-year status, and 93 of them were replaced: 5.5%. We use only what exists at that first
visit: questionnaires, a standardised hip examination, and the raw pixels of the same-day pelvic radiograph. The
radiologists' OARSI grades were scored after year 10, with the later films side by side. So they are never model
inputs.

**[→ approach, overview figure]**
Three pieces. M0 is a penalised logistic regression on 76 clinical features. M3 is a small trainable head on frozen
DenseNet-121 feature maps that turns each hip crop into one replacement logit. The final model is M0 with that logit
added as one extra column. Everything is evaluated with five-by-five nested cross-validation, grouped by participant
so the two hips of one person never sit on opposite sides of a split.

**[→ note]**
The five states that follow track one real test hip through the final model, then return to the whole cohort.
Nothing on these pages is typed in; every number is recomputed from saved weights and predictions.

## State 1 · One hip becomes a row (2:30–3:45)

**[→ heading, then → the three-sentence summary]**
What happens here: the hip's baseline values go through preprocessing that was fitted on the training fold only. Raw
degrees and scores become comparable z-scores. This is the step where leakage would get in, so watch what the other
hip contributes.

**[→ `hip`]**
This is a left hip. It was replaced in year 7. It comes from the test fold of the first split, and I chose it because
the image changed its rank the most. It is the best example in its fold, not a typical one.

**[→ `asym`, → `z_asym`]**
Internal rotation of this hip minus that of the other hip: plus 15 degrees. This hip rotates *more* than its partner,
so on examination it looks like the better hip. After the fold's clipping and scaling that is z = 2.68. The other
hip enters only through this baseline difference, never through its own follow-up.

**[→ `sos_t`, → `z_sos`]**
Skewed questionnaire scales go through a Yeo-Johnson transform with lambda fitted on the training fold, then a
z-score: −0.65 here. Symptoms are mild.

## State 2 · The clinical model scores the hip (3:45–5:00)

**[→ heading and summary]**
The elastic net adds weight times value over 77 columns, then Platt recalibration maps the logit to a probability.
This is the baseline every deep model has to beat. With 93 events, none of our tabular deep models did.

**[→ `cfg`]**
The inner folds chose strong regularisation, C about 0.012, and an L1 ratio of 0.1, so mostly ridge.

**[→ `logit_m0`, → `p_m0`]**
The logit is −3.60. After recalibration the risk is 0.013, which is below the 5.5% base rate.

**[⇧→ contribution chart]**
Orange bars push risk up and blue bars push it down. The largest downward terms are the examination asymmetries: pain on
internal rotation and flexion range, each compared with the other hip. The rotation-range asymmetry from State 1 was shrunk to zero by the
elastic net, so it adds nothing here. Knee pain, age and the weekly pain score push it up, and they roughly
cancel the downward terms.

**[→ rank text]**
Among the 340 test hips of this fold it ranks at the 37th percentile. In clinic this hip would go home with routine
advice.

## State 3 · What the image head computes (5:00–7:00)

**[→ heading and summary]**
The crop's frozen DenseNet-121 map, 1,024 channels on a 14-by-14 grid, passes two residual blocks, an attention
softmax over the 196 cells, and a linear layer. Our first attempt pooled this map into one vector and reached AUPRC
0.12. Keeping the spatial grid reached 0.49.

**[→ `fmap_shape`]**
One by 1,024 by 14 by 14.

**[→ `logit_img` — step into `head_forward` with → if time allows]**
*(If stepping in:)* `reduce` cuts the 1,024 channels to 64. The blocks keep the 14-by-14 grid. `scores` is one number
per cell, and the softmax turns those into weights that sum to 1. The pooled vector is concatenated with the plain
average, and the linear layer outputs the THA logit plus 11 OARSI items. **[u]**

The logit is 3.658. That is exactly the value the training pipeline stored for this hip, and the head that produced
it never saw this participant.

**[⇧→ attention overlay]**
The weight sits on the superolateral joint margin.

**[→ text, ⇧→ OARSI table]**
The head was also trained to predict 11 OARSI reading items, as targets only. The predictions are on the left and
the post-year-10 readings on the right, for comparison. The reader graded a superior femoral osteophyte of grade 3.

**[→ note]**
The examination looked unremarkable, but the pixels did not.

## State 4 · Fusion: one extra column (7:00–8:00)

**[→ heading and summary]**
The image logit is z-scored and enters the same elastic net as one more feature. The training hips get a
cross-fitted logit from a head that never saw them, so test labels enter no step.

**[→ `z_img`, → `w_img`]**
Here z equals 4.19 and the weight is 0.55, so this single term adds about 2.3 to the logit.

**[→ `logit_fused`, → `p_fused`, ⇧→ chart]**
The logit moves from −3.60 to −1.36, and the risk from 0.013 to 0.379. In the chart, one bar now outweighs every
clinical term.

**[→ rank text]**
The rank moves from the 37th to the 95th percentile, and this hip was replaced in year 7.

## State 5 · From one hip to a clinic (8:00–9:30)

**[→ heading and summary]**
Now we run the same computation for all 1,705 hips. At 5.5% prevalence a model can rank well and still raise
mostly false alarms, so the metric matters.

**[→ `event_rate`]**
The event rate is 0.055. A random ranking scores an AUPRC equal to this number.

**[→ `auprc_m0`, then ⇧→ `auprc_fused`]**
*(Optionally step into `average_precision` once:)* sort by risk, take precision at each rank, and average it over
the true replacements. The clinical model scores 0.383 and the fusion 0.586.

**[→ `max_lift`, → `lift_fused`]**
The maximum lift is 1 over the event rate, about 18. The fusion gets 10.7 times chance.

**[⇧→ PR curve]**
The fusion curve stays above the clinical curve over the whole recall range, and the gap is widest at low recall,
where a referral rule would operate.

**[→ text, ⇧→ `at_m0`, → `at_fused`]**
A clinic acts on a threshold, not a curve. At 10% predicted risk, M0 flags 14% of hips and catches 55% of
replacements, with precision 0.21. The fusion flags fewer hips, 9%, yet catches 65%, with precision 0.37. MCC rises
from 0.28 to 0.45.

**[→ note]**
These are repeat-0 values. The report gives five-repeat means: AUPRC 0.366 and 0.552.

## Conclusion (9:30–10:30)

**[→ table]**
Here are the internal estimates over five repeats. The clinical model reaches AUPRC 0.37. The image head alone
reaches 0.49, and only because it keeps the spatial map. Adding the pixel logit gives 0.55, a gain of 0.20 that
holds when replacements in years 1 and 2 are excluded. With leave-one-centre-out validation, discrimination
transports but calibration does not.

**[→ closing text]**
So a first-visit risk score built from routine data and the raw film, with no radiologist's grade, could set review
intervals and referral timing. There are two cautions. First, radiographic severity is itself an indication for
surgery, so part of what the head reads is the surgeon's future decision rather than hidden biology. Second,
calibration has to be redone locally before use. Next is external validation in OAI hip films and World COACH.
Thank you.
