# Seeing surgery coming: a LecTrace lecture (MD6329)

A step-through lecture of the final model in the report: one real test hip passes through
preprocessing, the clinical elastic net (M0), the image head on frozen DenseNet-121 feature maps,
and the fusion model. Every number is recomputed at trace time from the files in `assets/`.

```
uv sync
uv run lectrace serve            # localhost:7000
uv run lectrace build            # static site in _site/
uv run lectrace init             # GitHub Actions workflow -> GitHub Pages
```

`assets/` is produced by `../98_lectrace_assets.py` from the full pipeline and contains only
derived data for one hip (crop, 14 x 14 feature map, model weights, coefficients) plus
out-of-fold predictions of repeat 0.

Data: CHECK study (Wesseling et al., Int J Epidemiol 2016, doi:10.1093/ije/dyu177), DANS
doi:10.17026/DANS-XEX-HZWW, CC BY 4.0. The CHECK-cohort study is funded by the Dutch Arthritis
Foundation.
