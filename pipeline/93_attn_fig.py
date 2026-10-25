"""Where does the trainable image head look? Attention-pooling maps of the 'res' head (08_imghead.py) on outer-test hips.

usage: python -I 93_attn_fig.py <project dir>
reads : imghead/head_res_o{k}.pt (k = 0..4, fitted on the outer-training set of repeat 0), fmaps/densenet_b3.npy, crops.csv,
        imghead/pred_res.parquet
writes: data/derived/figs/res_img_attention.png  (top row: 6 THA hips with the highest score; middle: 6 non-THA hips with the
        highest score (false positives); bottom: 6 THA hips with the lowest score (misses)); each crop with its 14x14
        attention map upsampled and overlaid, title = KL grade and predicted 10-y risk rank.
        data/derived/imghead/attention_centroid.csv  attention centre of mass per test hip (y, x in crop fractions)
"""
import sys, pathlib
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = pathlib.Path(sys.argv[1]); sys.argv = [sys.argv[0], str(P), "none"]
sys.path.insert(0, str(P))
import importlib.util
spec = importlib.util.spec_from_file_location("ih", P / "08_imghead.py"); ih = importlib.util.module_from_spec(spec); spec.loader.exec_module(ih)
D = P / "data/derived"; H = ih.H; C = pd.read_csv(D / "crops.csv").set_index(["nsin", "side"])
M = np.load(ih.FM / "densenet_b3.npy", mmap_mode="r")
pred = pd.read_parquet(D / "imghead/pred_res.parquet"); pred = pred[(pred.inner == -1) & pred.img_logit.notna()]
rows, atts = [], {}
for k in range(5):
    net = ih.Head("res"); net.load_state_dict(torch.load(D / f"imghead/head_res_o{k}.pt")); net.eval()
    r = pred[pred.outer == k].row.values
    with torch.no_grad():
        for b in range(0, len(r), 128):
            x = torch.from_numpy(M[ih.MAPROW[r[b:b + 128]]].astype(np.float32)); _, a = net(x, return_att=True)
            for rr, aa in zip(r[b:b + 128], a.numpy()): atts[rr] = aa.reshape(14, 14)
for rr, a in atts.items():
    yy, xx = np.mgrid[0:14, 0:14]; rows.append({"row": rr, "cy": float((a * yy).sum() / 13), "cx": float((a * xx).sum() / 13)})
pd.DataFrame(rows).to_csv(D / "imghead/attention_centroid.csv", index=False)

pred["y"] = H.y10.values[pred.row]; pred["kl"] = H.r_kl.values[pred.row]; pred = pred[pred.y.notna()].sort_values("img_logit", ascending=False)
pred["rank"] = np.arange(1, len(pred) + 1)
groups = [("THA, highest scores", pred[pred.y == 1].head(6)), ("no THA, highest scores (false positives)", pred[pred.y == 0].head(6)),
          ("THA, lowest scores (missed)", pred[pred.y == 1].tail(6))]
fig, axs = plt.subplots(3, 6, figsize=(13, 7.4))
for gi, (name, g) in enumerate(groups):
    for j, r in enumerate(g.itertuples()):
        ax = axs[gi, j]; ax.axis("off")
        img = np.asarray(Image.open(C.loc[(H.nsin.values[r.row], H.side.values[r.row]), "crop"]), dtype=np.float32) / 65535
        a = Fn.interpolate(torch.tensor(atts[r.row])[None, None], size=img.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(img, cmap="gray"); ax.imshow(a / a.max(), cmap="magma", alpha=0.45)
        ax.set_title(f"KL {r.kl:.0f} · rank {r.rank}/{len(pred)}", fontsize=8, color="#52514e")
    axs[gi, 0].text(-0.08, 0.5, name, transform=axs[gi, 0].transAxes, rotation=90, ha="right", va="center", fontsize=9, color="#0b0b0b")
fig.suptitle("Image head attention (residual head + OARSI auxiliary, outer-test hips, repeat 0)", x=0.02, ha="left", fontsize=11, color="#0b0b0b")
fig.tight_layout(); fig.savefig(D / "figs/res_img_attention.png", dpi=180)
c = pd.DataFrame(rows); print("attention centroid (fraction of crop, 0 = top/lateral): y", c.cy.mean().round(3), "x", c.cx.mean().round(3))
