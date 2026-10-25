"""Click-annotation of femoral heads on baseline AP films (training data for the hip locator, plan 7.4 step 3).

usage:
  python -I 05a_annotate.py <project dir> sample [n=150]   -> data/derived/landmarks/sample.csv (stratified by centre x format x view)
  python -I 05a_annotate.py <project dir> label            -> opens a window; results appended to landmarks/landmarks.csv (resumable)

How to label one film (screen positions, NOT patient sides):
  1. click the CENTRE of the femoral head on the SCREEN-LEFT hip, then one point on the EDGE of that same head (gives radius)
  2. same for the SCREEN-RIGHT hip
     - single-hip film (APL/APR): label just the one hip shown, then press n
     - a hip that is not usable (prosthesis, cut off, not visible): press x to skip that hip
  3. keys:  n = save & next   b = back to the previously saved film (redo it; works after a restart too)
            u = undo last click   x = skip current hip   f = toggle 'R/L marker looks flipped'
            r = reset this film   q = quit (saves the current film only if it has clicks)
  The edge point can be anywhere on the head outline; the lateral-superior rim is usually clearest.
"""
import sys, pathlib, threading, queue
import numpy as np, pandas as pd

P = pathlib.Path(sys.argv[1]); D = P / "data/derived"; L = D / "landmarks"; L.mkdir(exist_ok=True)
MODE = sys.argv[2]
SHOW = 1100  # longest side of the displayed image (px); coordinates are saved in ORIGINAL pixel units

if MODE == "sample":
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    img = pd.read_csv(D / "images.csv")
    cen = pd.read_csv(D / "centres.csv").set_index("nsin").centre
    img["centre"] = img.nsin.map(cen).fillna("unassigned").str.replace(r"small\S*", "small", regex=True)
    img["stratum"] = img.centre + "|" + img.fmt + "|" + np.where(img.view == "O", "O", "single")
    rng = np.random.default_rng(20261021)
    # proportional allocation with a floor of 2 films per stratum, so every centre/format/view combination is seen
    w = img.stratum.value_counts()
    alloc = np.maximum(2, np.round(w / w.sum() * n)).astype(int).clip(upper=w)
    pick = pd.concat([g.sample(min(len(g), alloc[s]), random_state=int(rng.integers(1e9))) for s, g in img.groupby("stratum")])
    pick = pick.sample(frac=1, random_state=1)[["nsin", "view", "fmt", "centre", "path"]]
    pick.to_csv(L / "sample.csv", index=False)
    print(f"sampled {len(pick)} films from {img.stratum.nunique()} strata ->", L / "sample.csv")
    print(pick.groupby(["centre", "fmt"]).size().unstack(fill_value=0))
    sys.exit()

# ------------------------------------------------------------------ label mode
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(P))
from render_lib import load

S = pd.read_csv(L / "sample.csv")
OUT = L / "landmarks.csv"   # APPEND-ONLY: a re-labelled film gets a new row; every reader keeps the last row per path
def saved_rows():
    if not OUT.exists(): return pd.DataFrame(columns=["path"])
    R_ = pd.read_csv(OUT).drop_duplicates("path", keep="last")
    return R_[R_[["sl_x", "sr_x"]].notna().any(axis=1) | (R_.get("both_skipped", 0) == 1)]   # all-empty rows = not labelled
done = set(saved_rows().path)
history = list(saved_rows().path)          # saved films in order; 'b' walks back through it
todo = [r for r in S.itertuples() if r.path not in done]
print(f"{len(done)} films done, {len(todo)} to go")

def prep(path):
    a, _ = load(path)
    st = max(1, int(np.ceil(max(a.shape) / SHOW)))
    b = a[::st, ::st]
    lo, hi = np.percentile(b, [1, 99.5])
    return np.clip((b - lo) / (hi - lo + 1e-6), 0, 1), st, a.shape

# background prefetch: decoding an 85 MB TIFF takes ~0.3 s, keep two films ready
Q = queue.Queue(maxsize=2)
def worker():
    for r in todo:
        try: Q.put((r, *prep(r.path)))
        except Exception as e: Q.put((r, None, repr(e), None))
    Q.put(None)
threading.Thread(target=worker, daemon=True).start()

plt.rcParams["toolbar"] = "None"   # the zoom/pan toolbar would turn its clicks into landmarks
fig, ax = plt.subplots(figsize=(13, 9)); plt.subplots_adjust(0.01, 0.01, 0.99, 0.95)
state, pending = {}, []            # pending: (item, prefill) pushed back by 'b', shown before the prefetch queue
def new_film(item, prefill=None):
    r, im, st, shape = item
    state.update(item=item, r=r, st=st, shape=shape, clicks=[], skipped=[], flipped=False, artists=[])
    if prefill: state.update(clicks=list(prefill["clicks"]), skipped=list(prefill["skipped"]), flipped=prefill["flipped"])
    ax.clear(); ax.axis("off")
    if im is None:
        ax.set_title(f"READ ERROR {r.path}: {st}  - press n"); fig.canvas.draw_idle(); return
    ax.imshow(im, cmap="gray")
    redraw_marks()
def prefill_from_row(row, st):
    """saved row (original pixels) -> display clicks; the edge point is re-drawn at the lateral side of the saved circle."""
    clicks, skipped = [], []
    for side in ("sl", "sr"):
        x, y, rr = row[f"{side}_x"], row[f"{side}_y"], row[f"{side}_r"]
        if pd.isna(x): skipped.append(side); continue
        clicks += [(x / st, y / st), ((x + (-rr if side == "sl" else rr)) / st, y / st)]
    return {"clicks": clicks, "skipped": skipped, "flipped": bool(row.get("flipped", 0))}
def title():
    r = state["r"]; k = len(state["clicks"]) + 2 * len(state["skipped"])
    step = ["screen-LEFT hip: CENTRE", "screen-LEFT hip: EDGE", "screen-RIGHT hip: CENTRE", "screen-RIGHT hip: EDGE", "done - press n"][min(k, 4)]
    ax.set_title(f"[{len(done) + 1}/{len(S)}] {pathlib.Path(r.path).name}  view={r.view}  centre={r.centre}   next: {step}"
                 f"{'   FLIPPED' if state['flipped'] else ''}   (n next, b back, u undo, x skip hip, f flip, r reset, q quit)", fontsize=9)
    fig.canvas.draw_idle()
def redraw_marks():
    for a in state["artists"]: a.remove()
    state["artists"] = []
    pts = state["clicks"]
    for i, (x, y) in enumerate(pts):
        col = "#eb6834" if i % 2 == 0 else "#2a78d6"
        state["artists"] += ax.plot(x, y, "+", color=col, ms=14, mew=2)
        if i % 2 == 1:
            cx, cy = pts[i - 1]; rad = np.hypot(x - cx, y - cy)
            state["artists"].append(ax.add_patch(plt.Circle((cx, cy), rad, fill=False, color="#1baf7a", lw=1.5)))
    title()
def on_click(ev):
    if ev.inaxes != ax or ev.button != 1 or len(state["clicks"]) + 2 * len(state["skipped"]) >= 4: return
    state["clicks"].append((ev.xdata, ev.ydata)); redraw_marks()
def save():
    r, st, c = state["r"], state["st"], state["clicks"]
    rec = {"nsin": r.nsin, "view": r.view, "fmt": r.fmt, "centre": r.centre, "path": r.path, "H": state["shape"][0], "W": state["shape"][1],
           "flipped": int(state["flipped"])}
    # walk through the two screen sides in order, consuming clicks / skips
    ci = 0
    for side in ("sl", "sr"):
        if side in state["skipped"] or ci + 1 >= len(c):
            rec.update({f"{side}_x": np.nan, f"{side}_y": np.nan, f"{side}_r": np.nan}); continue
        (x0, y0), (x1, y1) = c[ci], c[ci + 1]; ci += 2
        rec.update({f"{side}_x": x0 * st, f"{side}_y": y0 * st, f"{side}_r": np.hypot(x1 - x0, y1 - y0) * st})
    rec["both_skipped"] = int(len(state["skipped"]) == 2)
    if not c and len(state["skipped"]) < 2: return            # nothing entered: do not write an empty row
    pd.DataFrame([rec]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)   # append only, never delete
    done.add(r.path)
    if r.path not in history: history.append(r.path)   # keep the original order so 'b' keeps walking backwards
def next_film():
    item, pre = pending.pop() if pending else (Q.get(), None)
    if item is None: plt.close(fig); return
    new_film(item, pre)
def go_back():
    """Show the previous saved film WITH its saved marks (edit them or just press n). Nothing is deleted: re-saving
    appends a new row that supersedes the old one. The current film (and any clicks on it) is queued to come next."""
    cur = state["r"].path if state.get("r") is not None else None
    back = [p for p in history if p != cur]
    if cur in history: back = history[:history.index(cur)]
    if not back: ax.set_title("no earlier saved film"); fig.canvas.draw_idle(); return
    prev = back[-1]
    pending.append((state["item"], {"clicks": state["clicks"], "skipped": state["skipped"], "flipped": state["flipped"]}))
    row = saved_rows().set_index("path").loc[prev]
    r = next(x for x in S.itertuples() if x.path == prev)
    ax.set_title("loading previous film ..."); fig.canvas.draw_idle(); fig.canvas.flush_events()
    try:
        item = (r, *prep(r.path)); new_film(item, prefill_from_row(row, item[2]))
    except Exception as e: new_film((r, None, repr(e), None))
def on_key(ev):
    if ev.key == "u" and state["clicks"]: state["clicks"].pop(); redraw_marks()
    elif ev.key == "x":
        k = len(state["clicks"])
        side = "sl" if k < 2 and "sl" not in state["skipped"] else "sr"
        if k % 2 == 1: state["clicks"].pop()
        state["skipped"].append(side); redraw_marks()
    elif ev.key == "f": state["flipped"] = not state["flipped"]; title()
    elif ev.key == "r": state.update(clicks=[], skipped=[]); redraw_marks()
    elif ev.key == "n": save(); next_film()
    elif ev.key == "b": go_back()
    elif ev.key == "q":
        if state["clicks"] or state["skipped"]: save()   # an untouched film is not saved, so it comes back next session
        plt.close(fig)
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
for k in ("keymap.quit", "keymap.fullscreen", "keymap.home", "keymap.back", "keymap.forward", "keymap.save", "keymap.xscale",
          "keymap.yscale", "keymap.grid", "keymap.pan", "keymap.zoom"):
    plt.rcParams[k] = []
if todo: next_film(); plt.show()
print(f"saved {len(done)} films -> {OUT}")
