"""STEP 1 - click-to-polygon annotator (human = click + hotkey, SAM = precise outline).

  left click      add INCLUDE point  -> live SAM preview (green)
  right click     add EXCLUDE point  -> refine preview
  b c w p s f     accept preview as Bottle / Bottle Cap / Wrapper / Polythene / Shoe, Jacket / Foam
  Esc             discard preview            u   undo last accepted polygon
  z / x           zoom in at cursor (x2) / reset zoom      h   toggle hint boxes
  ]  or  [        next / previous image (progress auto-saved)    q   save & quit
Output: work/annotations.json
"""
import argparse, glob, json, os
import cv2, numpy as np
from common import LABELS, KEY2NAME, hex2rgb
from seg import Segmenter, mask_to_polygon, detect_hints

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--work", default="work")
ap.add_argument("--sam", default="sam2.1_b.pt", help="'' = GrabCut only")
ap.add_argument("--win", type=int, default=1280, help="window width in px")
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
SAVE = os.path.join(a.work, "annotations.json")
files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
ann = json.load(open(SAVE)) if os.path.exists(SAVE) else {}
seg = Segmenter(a.sam)

def save():
    json.dump(ann, open(SAVE, "w"))

BAR = 64
state = dict(i=0, pts=[], labs=[], mask=None, zoom=1.0, cx=0, cy=0, mouse=(0, 0), hints=False, hintcache={})
cache = {}
def load(i):
    if i not in cache:
        cache.clear(); cache[i] = cv2.imread(files[i])
    return cache[i]

def view_rect(img):
    H, W = img.shape[:2]
    vw, vh = W / state["zoom"], H / state["zoom"]
    x0 = int(min(max(state["cx"] - vw / 2, 0), W - vw)); y0 = int(min(max(state["cy"] - vh / 2, 0), H - vh))
    return x0, y0, int(vw), int(vh)

def to_img(mx, my, img):
    x0, y0, vw, vh = view_rect(img); s = vw / a.win_w
    return int(x0 + mx * s), int(y0 + my * s)

def draw():
    img = load(state["i"]); H, W = img.shape[:2]
    vis = img.copy()
    name = os.path.basename(files[state["i"]])
    objs = ann.setdefault(name, {"width": W, "height": H, "objects": []})["objects"]
    for o in objs:
        col = hex2rgb(dict((n, h) for n, h, _ in LABELS)[o["label"]])[::-1]
        p = np.array(o["polygon"], np.float32).reshape(-1, 1, 2).astype(np.int32)
        ov = vis.copy(); cv2.fillPoly(ov, [p], col); vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
        cv2.polylines(vis, [p], True, col, 1)
    if state["hints"]:
        if name not in state["hintcache"]:
            state["hintcache"][name] = detect_hints(img)
        for b in state["hintcache"][name]:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
    if state["mask"] is not None:
        cs, _ = cv2.findContours(state["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 1)
    for (x, y), l in zip(state["pts"], state["labs"]):
        cv2.circle(vis, (x, y), 3, (0, 255, 0) if l else (0, 0, 255), -1)
    x0, y0, vw, vh = view_rect(img)
    out = cv2.resize(vis[y0:y0 + vh, x0:x0 + vw], (a.win_w, int(a.win_w * vh / vw)), interpolation=cv2.INTER_LINEAR)
    l1 = "%d/%d  %s   |   %d objects   |   zoom x%.0f%s" % (state["i"] + 1, len(files), name, len(objs), state["zoom"], "   |   HINTS ON" if state["hints"] else "")
    l2 = "L-click=add  R-click=exclude | b Bottle  c Cap  w Wrapper  p Polythene  s Shoe/Jacket  f Foam | Esc discard  u undo  z/x zoom  h hints  [ ] nav  q quit"
    bar = np.zeros((BAR, out.shape[1], 3), np.uint8)
    cv2.putText(bar, l1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, l2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    out = np.vstack([bar, out])
    cv2.imshow("annotate", out)

def resegment():
    img = load(state["i"])
    state["mask"] = seg.from_points(img, state["pts"], state["labs"]) if state["pts"] else None

def on_mouse(ev, mx, my, flags, _):
    my -= BAR
    if my < 0:
        return
    state["mouse"] = (mx, my)
    if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        img = load(state["i"])
        state["pts"].append(to_img(mx, my, img)); state["labs"].append(1 if ev == cv2.EVENT_LBUTTONDOWN else 0)
        resegment()
        draw()

def goto(i):
    state.update(i=max(0, min(len(files) - 1, i)), pts=[], labs=[], mask=None, zoom=1.0)
    img = load(state["i"]); state["cx"], state["cy"] = img.shape[1] / 2, img.shape[0] / 2

img0 = load(0); a.win_w = a.win
cv2.namedWindow("annotate", cv2.WINDOW_NORMAL); cv2.resizeWindow("annotate", a.win_w, int(a.win_w * 9 / 16) + BAR); cv2.setMouseCallback("annotate", on_mouse)
goto(0); draw()
while True:
    k = cv2.waitKey(30) & 0xFF
    if k == 255:
        continue
    ch = chr(k) if k < 128 else ""
    name = os.path.basename(files[state["i"]]); objs = ann[name]["objects"]
    if ch in KEY2NAME and state["mask"] is not None:
        poly = mask_to_polygon(state["mask"])
        if poly:
            objs.append({"polygon": poly, "label": KEY2NAME[ch]}); save()
        state.update(pts=[], labs=[], mask=None)
    elif k == 27: state.update(pts=[], labs=[], mask=None)
    elif ch == "u" and objs: objs.pop(); save()
    elif ch == "h": state["hints"] = not state["hints"]
    elif ch == "z":
        img = load(state["i"]); x, y = to_img(*state["mouse"], img); state.update(cx=x, cy=y, zoom=min(state["zoom"] * 2, 16))
    elif ch == "x":
        img = load(state["i"]); state.update(zoom=1.0, cx=img.shape[1] / 2, cy=img.shape[0] / 2)
    elif ch == "]": save(); goto(state["i"] + 1)
    elif ch == "[": save(); goto(state["i"] - 1)
    elif ch == "q": break
    draw()
save(); cv2.destroyAllWindows()
print("saved", SAVE, "-", sum(len(v["objects"]) for v in ann.values()), "objects in", len(ann), "images")