"""Plumbing for 01_seeing_surgery_coming.py. The underscore keeps this file out of the trace."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ASSETS = Path(__file__).parent / "assets"
BLUE, ORANGE, GREEN, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"


def load_demo():
    return json.loads((ASSETS / "demo.json").read_text())


def load_oof():
    import pandas as pd
    return pd.read_csv(ASSETS / "oof_rep0.csv")


# ---------------------------------------------------------------- image head (same definition as 08_imghead.py)
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        return Fn.relu(x + self.b2(self.c2(Fn.relu(self.b1(self.c1(x))))))


class Head(nn.Module):
    def __init__(self, n_aux=11, c=64):
        super().__init__()
        self.drop2d, self.drop = nn.Dropout2d(0.1), nn.Dropout(0.3)
        self.reduce = nn.Sequential(nn.Conv2d(1024, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(c), ResBlock(c))
        self.att = nn.Conv2d(c, 1, 1)
        self.out = nn.Linear(2 * c, 1 + n_aux)


def load_head():
    net = Head()
    net.load_state_dict(torch.load(ASSETS / "head_res_o0.pt", map_location="cpu"))
    return net.eval()


class Summarised(torch.Tensor):
    """Tensor that the trace stores as shape + summary when large (the 1024 x 14 x 14 map would otherwise be
    serialised at every step, ~4 MB each). Ops keep the subclass, so the head's intermediates are summarised too."""

    def __lectrace__(self):
        t = self.detach().as_subclass(torch.Tensor)
        if t.numel() <= 256:
            return t
        return {"shape": list(t.shape), "mean": round(float(t.mean()), 4), "std": round(float(t.std()), 4),
                "min": round(float(t.min()), 4), "max": round(float(t.max()), 4)}


def load_fmap():
    fmap = torch.from_numpy(np.load(ASSETS / "fmap.npy").astype(np.float32))[None]   # (1, 1024, 14, 14)
    return fmap.as_subclass(Summarised)


# ---------------------------------------------------------------- figures
def attention_overlay(att):
    """att: (14, 14) attention weights -> crop with the map on top."""
    from PIL import Image
    crop = np.array(Image.open(ASSETS / "crop.png"))
    up = np.kron(att, np.ones((256 // 14 + 1, 256 // 14 + 1)))[:256, :256]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    axes[0].imshow(crop, cmap="gray"); axes[0].set_title("hip crop (lateral side left)", fontsize=9)
    axes[1].imshow(crop, cmap="gray"); axes[1].imshow(up, cmap="inferno", alpha=0.55)
    axes[1].set_title("attention pooling weights (14 x 14)", fontsize=9)
    for a in axes: a.axis("off")
    fig.tight_layout()
    return fig


def contributions_chart(rows, title):
    """rows: list of {feature, contribution} dicts -> horizontal bar chart."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title, "width": 420, "height": {"step": 22},
        "data": {"values": rows},
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "y": {"field": "feature", "type": "nominal", "sort": None, "title": None, "axis": {"labelFontSize": 12}},
            "x": {"field": "contribution", "type": "quantitative", "title": "w_j * x_j  (logit units)"},
            "color": {"condition": {"test": "datum.contribution > 0", "value": ORANGE}, "value": BLUE},
            "tooltip": [{"field": "feature"}, {"field": "contribution"}],
        },
    }


def pr_curve_chart(curves):
    """curves: {name: (recall, precision)} -> Vega-Lite line chart."""
    vals = [{"model": k, "recall": float(r), "precision": float(p)} for k, (R, Pr) in curves.items() for r, p in zip(R, Pr)]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Precision-recall, repeat 0 out-of-fold (1,705 hips, 93 THA)", "width": 420, "height": 280,
        "data": {"values": vals},
        "mark": {"type": "line", "interpolate": "step-after"},
        "encoding": {
            "x": {"field": "recall", "type": "quantitative"},
            "y": {"field": "precision", "type": "quantitative", "scale": {"domain": [0, 1]}},
            "color": {"field": "model", "type": "nominal", "scale": {"range": [BLUE, ORANGE]}},
        },
    }
