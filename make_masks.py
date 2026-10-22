"""STEP 4 - CVAT-exported COCO JSON -> one RGB PNG mask per image (class colour on black, same size/name)."""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw
from common import NAME2HEX, hex2rgb

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True, help="the JSON exactly as exported by CVAT")
ap.add_argument("--out", default="rgb_images")
ap.add_argument("--prefix", default="", help="e.g. 2023BCSE001_  (enrolment number + underscore)")
ap.add_argument("--images", default=None, help="optional: folder of originals, to verify names/sizes")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
coco = json.load(open(a.json))
cat = {c["id"]: c["name"] for c in coco["categories"]}
bad = [n for n in cat.values() if n not in NAME2HEX]
assert not bad, "Labels not allowed by the assignment: %s" % bad
assert set(NAME2HEX) == set(cat.values()), "Label set differs from the 6 required: %s" % sorted(cat.values())
per = {}
for an in coco["annotations"]:
    per.setdefault(an["image_id"], []).append(an)
n_poly = 0
for im in coco["images"]:
    W, H = im["width"], im["height"]
    mask = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(mask)
    anns = sorted(per.get(im["id"], []), key=lambda x: -x["area"])      # big first, small objects stay visible on top
    for an in anns:
        seg = an["segmentation"]
        assert isinstance(seg, list), "RLE/brush mask found - assignment requires polygons only"
        for poly in seg:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            d.polygon(pts, fill=hex2rgb(NAME2HEX[cat[an["category_id"]]]))
            n_poly += 1
    stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
    mask.save(os.path.join(a.out, a.prefix + stem + ".png"))
    if a.images:
        o = Image.open(os.path.join(a.images, im["file_name"]))
        assert o.size == (W, H), "size mismatch " + im["file_name"]
print("images in JSON: %d | masks written: %d | polygons: %d" % (len(coco["images"]), len(os.listdir(a.out)), n_poly))
empty = [im["file_name"] for im in coco["images"] if im["id"] not in per]
if empty: print("WARNING - images with NO annotation (assignment says don't leave blank):", empty)
