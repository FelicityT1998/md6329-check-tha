"""Phone-size teleprompter PDF for the LecTrace lecture: one page per action, Chinese key instructions + English speech.

usage: python -I teleprompter.py <out.pdf>
Press counts follow the trace's step-over rule (lectrace frontend): Shift+Right moves to the next step in the same
frame on a different line; hidden lines (@hide) still count as one press but change nothing on screen.
"""
import html, sys, pathlib

# (section, key action, what appears on screen, speech)   **x** = bold
PAGES = [
    ("准备", "打开网页", "",
     "浏览器打开 http://localhost:7000（或 https://felicityt1998.github.io/md6329-check-tha/）→ 点左侧讲座 → 按 F 全屏。"
     "<br><br>此时高亮在 <code>def main():</code>。<br><br>全程只用 <b>Shift+→</b>（下面写作 ⇧→）；只有 State 3 有一次 → 和一次 u。"
     "<br>按多了：<b>Shift+←</b> 退回一步。"),
    ("引入 0:00", "⇧→ × 3", "标题 + 「The problem」段",
     "A 55-year-old sees their GP for the first time with hip pain. Implants now last decades, and younger patients "
     "carry more revision risk. So the question is not only: is this OA? It is: **will this hip need replacing within "
     "ten years?** That decides how hard to push conservative care, and when to refer."),
    ("引入", "⇧→ × 2", "「The data」段 + 时间线图",
     "We use CHECK: about a thousand Dutch adults aged 45 to 65, followed for ten years. **1,705 hips, 93 replaced: "
     "5.5%.** We use only what exists at the first visit: questionnaires, a hip examination, and the same-day "
     "radiograph. The radiologists' grades were scored after year 10, so they are never inputs."),
    ("引入", "⇧→ × 2", "「The approach」段 + 总览图",
     "M0 is a penalised logistic regression on 76 clinical features. M3 is a small head on frozen DenseNet feature "
     "maps; it turns each hip crop into one logit. **The final model is M0 plus that logit as one extra column.** "
     "Everything is nested cross-validation, grouped by participant."),
    ("引入", "⇧→ × 1", "灰色 note",
     "Now we follow **one real test hip** through the final model. No number on these pages is typed in."),
    ("State 1 · 2:30", "⇧→ × 2", "State 1 标题 + 三句话",
     "**State 1.** The hip's raw values go through preprocessing fitted on the training fold only. This is where "
     "leakage would get in."),
    ("State 1", "⇧→ × 3", "右侧出现 hip（第 1 下画面不变，正常）",
     "This is a left hip, **replaced in year 7.** I picked it because the image changed its rank the most. It is the "
     "best example, not a typical one."),
    ("State 1", "⇧→ × 3", "asym = 15.0，z_asym = 2.68（第 1 下不变）",
     "It rotates **15 degrees more** than the other hip. So on examination it looks like the better hip. The other "
     "hip enters only through this baseline difference."),
    ("State 1", "⇧→ × 5", "sos_t、z_sos = −0.65，最后是灰色 note",
     "Skewed scales get a Yeo-Johnson transform, with lambda fitted on the training fold. Symptoms are mild: "
     "**z is minus 0.65.**"),
    ("State 2 · 3:45", "⇧→ × 2", "State 2 标题 + 三句话",
     "**State 2.** The clinical model adds weight times value over 77 columns, then recalibrates. This is the "
     "baseline. With 93 events, none of our deep tabular models beat it."),
    ("State 2", "⇧→ × 4", "cfg → logit_m0 = −3.60 → p_m0 = 0.013（第 1 下不变）",
     "The inner folds chose strong, mostly ridge regularisation. The logit is minus 3.6. After calibration the risk "
     "is **0.013, below the base rate.**"),
    ("State 2", "⇧→ × 2", "贡献条形图 + 百分位文字",
     "Examination asymmetries pull the risk down; knee pain and age push it up. This hip ranks at the **37th "
     "percentile.** It would go home with routine advice."),
    ("State 3 · 5:00", "⇧→ × 2", "State 3 标题 + 三句话",
     "**State 3: the image head.** A frozen DenseNet map, 1,024 channels on a 14 by 14 grid, goes through two "
     "residual blocks, attention, and a linear layer. Pooling the map to one vector gave AUPRC **0.12**. Keeping the "
     "grid gave **0.49**."),
    ("State 3", "⇧→ × 4", "fmap_shape；高亮停在 head_forward 这一行",
     "Here is the input: one by 1,024 by 14 by 14. Let's step inside the head."),
    ("State 3", "→ × 1，再 ⇧→ × 5", "进入 head_forward，停在 att = torch.softmax(...)",
     "reduce cuts 1,024 channels to 64. The residual blocks keep the grid. scores gives one number per cell. "
     "**Softmax turns them into weights that sum to one.**"),
    ("State 3", "u × 1（跳出函数）", "回到 main，logit_img = 3.658",
     "The logit is **3.658**: exactly the value the training pipeline saved. And this head never saw this "
     "participant."),
    ("State 3", "⇧→ × 1", "注意力叠加图",
     "The attention sits on the **upper outer joint margin**, where early hip OA usually starts."),
    ("State 3", "⇧→ × 3", "文字 → OARSI 表 → 灰色 note",
     "The head also learned to predict 11 OARSI items, as training targets only. The reader found a **grade-3 femoral "
     "osteophyte.** The examination looked fine. The pixels did not."),
    ("State 4 · 7:00", "⇧→ × 2", "State 4 标题 + 三句话",
     "**State 4: fusion.** The image logit enters the same elastic net as one more column. Training hips get a logit "
     "from a head that never saw them."),
    ("State 4", "⇧→ × 3", "z_img = 4.19，w_img = 0.55（第 1 下不变）",
     "The image z-score is 4.19 and its weight 0.55. **This one term adds about 2.3 to the logit.**"),
    ("State 4", "⇧→ × 2", "logit_fused = −1.36，p_fused = 0.379",
     "The risk moves from **0.013 to 0.379.**"),
    ("State 4", "⇧→ × 2", "融合贡献图 + 百分位文字",
     "One bar now outweighs every clinical term. The rank jumps from the **37th to the 95th percentile.** And this "
     "hip was replaced in year 7."),
    ("State 5 · 8:00", "⇧→ × 2", "State 5 标题 + 三句话",
     "**State 5.** Now all 1,705 hips. At 5.5% prevalence, a model can rank well and still raise mostly false "
     "alarms."),
    ("State 5", "⇧→ × 3", "event_rate = 0.055（前 2 下不变）",
     "A random model's AUPRC equals the event rate: **0.055.**"),
    ("State 5", "⇧→ × 2", "auprc_m0 = 0.383，auprc_fused = 0.586",
     "The clinical model reaches **0.38.** The fusion reaches **0.59.**"),
    ("State 5", "⇧→ × 2", "max_lift = 18.3，lift_fused = 10.7",
     "The best possible lift is about 18. The fusion reaches **10.7 times chance.**"),
    ("State 5", "⇧→ × 2", "PR 曲线 + 「A clinic acts on a threshold」",
     "The fusion curve stays above the clinical curve everywhere. But a clinic acts on a threshold, not a curve."),
    ("State 5", "⇧→ × 2", "at_m0、at_fused 两个字典",
     "At 10% risk, the fusion flags fewer hips, yet catches **65% of replacements instead of 55%.** Precision rises "
     "from 0.21 to 0.37."),
    ("State 5", "⇧→ × 1", "灰色 note",
     "These are repeat-0 numbers. The report averages five repeats: **0.37 and 0.55.**"),
    ("结论 9:30", "⇧→ × 2", "结论表",
     "To sum up. Clinical model: AUPRC 0.37. Adding the pixel logit: **0.55**, and the gain holds without early "
     "surgeries. Across centres, ranking transports; calibration does not."),
    ("结论", "⇧→ × 2", "结尾段 + 参考文献",
     "So a first-visit score from routine data and the raw film could guide review and referral. Two cautions: the "
     "film partly encodes the surgeon's own decision, and calibration must be redone locally. Next: external "
     "validation. **Thank you.**"),
]


def md(s):
    s = html.escape(s, quote=False).replace("&lt;br&gt;", "<br>").replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    s = s.replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")
    parts = s.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


CSS = """
@page { size: 90mm 180mm; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #111; color: #f2f2f2; }
.page { width: 90mm; height: 180mm; padding: 7mm 6mm; page-break-after: always; display: flex; flex-direction: column; background: #111; }
.top { display: flex; justify-content: space-between; font-size: 9pt; color: #9a9a9a; }
.key { margin-top: 4mm; background: #2a78d6; color: #fff; border-radius: 3mm; padding: 3mm; font-size: 20pt; font-weight: 700; text-align: center; }
.see { margin-top: 2mm; font-size: 10pt; color: #b8d3f5; text-align: center; min-height: 5mm; }
.say { margin-top: 6mm; font-size: 17pt; line-height: 1.45; }
.say b { color: #ffd166; }
.prep { margin-top: 6mm; font-size: 13pt; line-height: 1.6; }
code { font-family: Consolas, monospace; color: #b8d3f5; }
.foot { margin-top: auto; font-size: 8pt; color: #666; text-align: right; }
"""


def build(out):
    n = len(PAGES)
    body = []
    for i, (sec, key, see, say) in enumerate(PAGES):
        cls = "prep" if i == 0 else "say"
        body.append(f'<div class="page"><div class="top"><span>{html.escape(sec)}</span><span>{i}/{n - 1}</span></div>'
                    f'<div class="key">{html.escape(key)}</div><div class="see">{html.escape(see)}</div>'
                    f'<div class="{cls}">{md(say)}</div><div class="foot">Seeing Surgery Coming · LecTrace</div></div>')
    page = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    tmp = pathlib.Path(out).with_suffix(".html")
    tmp.write_text(page, encoding="utf8")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri())
        pg.pdf(path=out, prefer_css_page_size=True, print_background=True)
        b.close()
    lines = ["# Presentation script: Seeing Surgery Coming (LecTrace)", "",
             "One block per action. ⇧→ = Shift+Right (step over), → = step into, u = step out. "
             "Phone teleprompter version: `LecTrace_teleprompter_phone.pdf`.", ""]
    for i, (sec, key, see, say) in enumerate(PAGES[1:], 1):
        lines += [f"### {i}. {sec} · {key}", f"*On screen: {see}*", "", say.replace("<br>", " "), ""]
    pathlib.Path(out).with_name("presentation_script.md").write_text("\n".join(lines), encoding="utf8")
    words = sum(len(s.replace("**", "").split()) for _, _, _, s in PAGES[1:])
    print(f"{n} pages, {words} spoken words (~{words / 110:.1f} min at 110 wpm) -> {out}")


if __name__ == "__main__":
    build(sys.argv[1])
