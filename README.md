# Seeing Surgery Coming

Pixel-supervised multimodal prediction of 10-year hip replacement in the CHECK study.
MD6329 final project, Tian Fengxiao (G2606127G), Nanyang Technological University.

- **LecTrace lecture (live):** https://GITHUB_USER.github.io/REPO_NAME/
- **Lecture source:** [`lectrace/01_seeing_surgery_coming.py`](lectrace/01_seeing_surgery_coming.py)
- **Report source:** [`report/`](report/) (NeurIPS 2026 template; `python -I report/build_overleaf.py report` makes a single-file Overleaf upload)

## Layout

| Folder | Contents |
|---|---|
| `pipeline/` | Every script that produced a number or figure in the report, numbered in run order |
| `pipeline/logs/` | PowerShell launchers for the long multi-repeat runs |
| `pipeline/check_raw_tools/` | Helpers used to index and extract the DANS archive |
| `lectrace/` | The step-through lecture and the small derived assets it reads |
| `report/` | LaTeX source, bibliography and figures |
| `docs/` | Literature review and experiment plan, full results log, figure prompts (Chinese working notes) |

## Data

CHECK (Cohort Hip and Cohort Knee): Wesseling et al., *Int J Epidemiol* 2016, doi:10.1093/ije/dyu177.
Download from DANS, doi:10.17026/DANS-XEX-HZWW (CC BY 4.0). The CHECK-cohort study is funded by the Dutch Arthritis
Foundation. Raw tables and radiographs are **not** in this repository. `lectrace/assets/` holds only derived data for
one hip (crop, 14 x 14 DenseNet feature map, head weights, model coefficients) and the out-of-fold predictions of
repeat 0.

## Running the pipeline

Each script prints its own usage line in the docstring. Scripts take the project directory as the first argument and
read / write `data/derived/`. Order:

| Step | Scripts | Output |
|---|---|---|
| Cohort and centres | `01_cohort.py`, `02_centres.py` | `hips.parquet`, `subjects.parquet`, `centres.csv` |
| EDA | `03_eda.py`, `97_report_figs.py` | `figs/`, Table 1 forest plot |
| Resampling plan | `04_folds.py` | `folds.csv` (5 repeats x 5 outer x 3 inner, grouped by participant) |
| Image curation | `05a_annotate.py` -> `05r_orient.py` -> `05b_locator.py` -> `05_crop.py` | hip crops |
| Image features | `06_embed.py`, `06b_simclr.py` (GPU), `07_fmaps.py` | embeddings, DenseNet-121 block-3 maps |
| Image head | `08_imghead.py` | cross-fitted image THA logits |
| Tabular models | `10_m0.py` (elastic net), `11_m1.py` (LightGBM), `11b_ebm.py`, `12_m2.py` (FT-Transformer) | `oof/*.parquet` |
| Multimodal | `13_m3.py`, `14_m4.py`, `15_m2pix.py` | `oof/*.parquet` |
| Transport and probes | `20_expB.py`, `21_probe.py`, `95_temporal_audit.py` | Experiment B, shortcut probes |
| Results and figures | `90_results.py`, `91_figs.py`, `92_threshold_table.py`, `93_attn_fig.py`, `94_calib_dca.py`, `96_interp.py`, `99_attn_roi.py` | tables and figures in the report |
| Lecture assets | `98_lectrace_assets.py` | `lectrace/assets/` |

Shared code: `lib_tab.py` (fold-wise preprocessing), `lib_eval.py` (metrics, bootstrap), `lib_ftt.py`
(FT-Transformer), `lib_img.py` (localisation helpers).

## Running the lecture

```
cd lectrace
uv sync
uv run lectrace serve        # http://localhost:7000  (arrow keys step, Shift+arrow steps over, u steps out)
```

Every number in the lecture is recomputed at trace time from `lectrace/assets/`.

## AI assistance

An AI coding agent (Claude Opus 5.5 via Claude Code) wrote much of the pipeline code under the author's direction. See
the report's Software 3.0 statement.
