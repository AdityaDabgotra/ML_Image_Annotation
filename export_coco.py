"""STEP 2 - annotations.json -> COCO 1.0 JSON in the shape CVAT imports (Upload annotations -> COCO 1.0)."""
import argparse, json, datetime
from common import LABELS, NAME2ID

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="work/annotations.json")
ap.add_argument("--out", default="work/instances_default.json")
a = ap.parse_args()
ann = json.load(open(a.ann))
coco = {"licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": str(datetime.date.today().year)},
        "categories": [{"id": NAME2ID[n], "name": n, "supercategory": ""} for n, _, _ in LABELS],
        "images": [], "annotations": []}
aid = 1
for iid, name in enumerate(sorted(ann), 1):
    v = ann[name]
    coco["images"].append({"id": iid, "width": v["width"], "height": v["height"], "file_name": name, "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0})
    for o in v["objects"]:
        xs, ys = o["polygon"][0::2], o["polygon"][1::2]
        # shoelace area
        area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs))))
        coco["annotations"].append({"id": aid, "image_id": iid, "category_id": NAME2ID[o["label"]], "segmentation": [o["polygon"]],
                                    "area": round(area, 2), "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                                    "iscrowd": 0, "attributes": {"occluded": False}})
        aid += 1
json.dump(coco, open(a.out, "w"), indent=1)
print("wrote", a.out, len(coco["images"]), "images,", len(coco["annotations"]), "polygons")
