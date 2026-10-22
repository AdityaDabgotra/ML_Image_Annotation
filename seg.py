"""Segmentation helpers: SAM2 (ultralytics) if available, GrabCut fallback; hint detector."""
import cv2, numpy as np

def detect_hints(img, min_area=80, max_area=40000):
    """Cheap 'something odd here' boxes (bright/dark/off-hue blobs). Noisy - hints only."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.hypot(a, b)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = cv2.GaussianBlur(L, (0, 0), 25)
    white = (L > 200) & (L - bg > 45) & (chroma < 12)
    dark = (L < 45) & (bg - L > 60)
    off = (hsv[..., 1] > 140) & (hsv[..., 2] > 70) & ~((hsv[..., 0] > 15) & (hsv[..., 0] < 55))
    m = ((white | dark | off).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, ar in st[1:] if min_area <= ar <= max_area]

def grabcut(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W - 1, box[2]), min(H - 1, box[3])
    m = np.zeros((H, W), np.uint8)
    try:
        cv2.grabCut(img, m, (x1, y1, x2 - x1, y2 - y1), np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    return ((m == 1) | (m == 3)).astype(np.uint8)

def grabcut_points(img, pts, labs, r=40):
    """Fallback: seed GrabCut with the clicks (include=FG, exclude=BG); keep the blob containing the first click."""
    H, W = img.shape[:2]
    x, y = pts[0]
    x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(W, x + r), min(H, y + r)
    gm = np.full((H, W), cv2.GC_BGD, np.uint8)
    gm[y1:y2, x1:x2] = cv2.GC_PR_BGD
    for (px, py), l in zip(pts, labs):
        cv2.circle(gm, (px, py), 3, cv2.GC_FGD if l else cv2.GC_BGD, -1)
    cv2.circle(gm, (x, y), 9, cv2.GC_PR_FGD, -1)
    cv2.circle(gm, (x, y), 3, cv2.GC_FGD, -1)
    try:
        cv2.grabCut(img, gm, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    m = ((gm == 1) | (gm == 3)).astype(np.uint8)
    n, lbl = cv2.connectedComponents(m, connectivity=8)
    return (lbl == lbl[y, x]).astype(np.uint8) if lbl[y, x] > 0 else None

class Segmenter:
    def __init__(self, weights="sam2.1_b.pt"):
        self.sam = None
        if weights:
            try:
                from ultralytics import SAM
                self.sam = SAM(weights)
                print("Segmenter: using", weights)
            except Exception as e:
                print("SAM not available (%s) -> GrabCut fallback (lower quality)" % e)

    def from_points(self, img, pts, labs):
        """pts=[(x,y),...] labs=[1 (include) | 0 (exclude), ...] -> binary mask (H,W) or None"""
        H, W = img.shape[:2]
        if self.sam is not None:
            r = self.sam(img, points=[[list(p) for p in pts]], labels=[list(labs)], verbose=False)[0]
            if r.masks is None or not len(r.masks.data):
                return None
            m = r.masks.data[0].cpu().numpy().astype(np.uint8)
            return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return grabcut_points(img, pts, labs)

def mask_to_polygon(mask, eps=0.6):
    """Largest outer contour, lightly simplified -> flat [x1,y1,x2,y2,...] (None if degenerate)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 6:
        return None
    c = cv2.approxPolyDP(c, eps, True)
    if len(c) < 3:
        return None
    return [round(float(v), 2) for v in c.reshape(-1, 2).flatten()]
