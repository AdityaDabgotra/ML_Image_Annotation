# ML Assignment 1 - SAM-assisted polygon annotation pipeline

pip install -r requirements.txt     (first run auto-downloads sam2.1_b.pt)

1. python annotate.py --images path\to\group57
   click an object -> green SAM outline -> press b/c/w/p/s/f. Right-click = exclude point. z/x zoom, h hints, ] next, u undo.
   (progress autosaves to work/annotations.json; re-run to resume)
2. python export_coco.py                      -> work/instances_default.json
3. CVAT: New task -> Labels -> "Raw" tab -> paste labels_raw.json -> upload images.
   Actions -> Upload annotations -> COCO 1.0 -> work/instances_default.json.
   Skim every frame, fix anything loose/wrong, Save.
   Actions -> Export task dataset -> COCO 1.0, untick "Save images". Unzip: annotations/instances_default.json
   Rename it  <ENROL>_annotations.json  (do NOT edit its contents).
4. python make_masks.py --json <ENROL>_annotations.json --out rgb_images --prefix <ENROL>_ --images path\to\group57
5. Drive: json/<ENROL>_annotations.json  and  rgb_images/ (all masks)
