# SAM-Assisted Polygon Annotation Pipeline for River Plastic Debris

Semi-automatic annotation toolkit built for **Machine Learning (CST051), Assignment 1** at NIT Srinagar.
It turns a set of river-surface frames into precise polygon annotations (COCO 1.0) and per-image RGB class masks, with a human doing only *click + hotkey* and a segmentation model drawing the outline.

![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

The dataset is a sequence of near-duplicate drone/bridge frames of a river. The plastic items are **small, sparse, and camouflaged by dense algae**, so:

- Drawing every polygon by hand is slow and imprecise on tiny objects.
- Fully automatic detection is unreliable. A simple colour/contrast detector finds the real objects but also ~90 false positives per frame, and deciding *foam vs. polythene vs. wrapper* is a judgement call.

The pipeline therefore splits the work: **the human decides what and which class; the model decides exactly where the boundary is.**

## Pipeline

```
 images ──► annotate.py ──► export_coco.py ──► CVAT (import, review, fix) ──► CVAT export ──► make_masks.py ──► RGB masks
          click + hotkey     COCO 1.0 JSON      labels from labels_raw.json   COCO 1.0 JSON     (colour on black)
          SAM2 outlines                                                      (submitted as-is)
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `annotate.py` | OpenCV window: click an object, SAM2 proposes a tight outline, press a hotkey to assign the class. Progress autosaves. |
| 2 | `export_coco.py` | Converts saved polygons to a COCO 1.0 JSON in the shape CVAT imports. |
| 3 | CVAT (manual) | Create task with `labels_raw.json`, import the COCO file, review every frame, fix, export from CVAT. |
| 4 | `make_masks.py` | Reads the CVAT-exported JSON and draws each polygon filled with its class colour on a black background. |

## Classes

Names, spelling and colours are fixed by the assignment (note `Shoe, Jacket` is **one** label containing a comma).

| Class | Colour | Hex | Hotkey |
|-------|--------|-----|--------|
| Bottle | Red | `#FF0000` | `b` |
| Bottle Cap | Orange | `#FF9900` | `c` |
| Wrapper | Yellow | `#FFFF00` | `w` |
| Polythene | Green | `#00CC00` | `p` |
| Shoe, Jacket | Blue | `#0099FF` | `s` |
| Foam | Violet | `#FF66FF` | `f` |

## Repository layout

```
.
├── annotate.py          # interactive click-to-polygon annotator
├── seg.py               # SAM2 wrapper, GrabCut fallback, polygon extraction, hint detector
├── common.py            # label names, colours, hotkeys (single source of truth)
├── export_coco.py       # annotations.json -> COCO 1.0 for CVAT import
├── make_masks.py        # CVAT COCO export -> RGB mask PNGs
├── labels_raw.json      # paste into CVAT "Raw" label editor
├── requirements.txt
├── work/
│   └── annotations.json # autosaved polygons (the only non-regenerable working file)
└── json/
    └── <ENROL>_annotations.json   # final, unedited CVAT export
```

Ignored by git (regenerable): the dataset images, SAM weights (`*.pt`), `rgb_images/`, `work/instances_default.json`, `work/overlays/`.

## Setup

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

Place the dataset images in a sibling folder, e.g. `../group57/`. SAM2 weights (`sam2.1_b.pt`) are downloaded automatically by `ultralytics` on first run.

## Usage

### 1. Annotate

```bash
python annotate.py --images ../group57
```

| Input | Action |
|-------|--------|
| Left click | Add an *include* point, live SAM preview (green outline) |
| Right click | Add an *exclude* point to refine the outline |
| `b c w p s f` | Accept the preview with that class |
| `Esc` | Discard the preview |
| `u` | Undo the last accepted polygon |
| `z` / `x` | Zoom in at cursor (2× per press, up to 16×) / reset |
| `h` | Toggle hint boxes (noisy bright/dark/off-colour blobs, to avoid missing small items) |
| `]` / `[` | Next / previous image |
| `q` | Save and quit |

Progress is stored in `work/annotations.json`; re-running the command resumes where you stopped.

Options: `--sam ""` falls back to GrabCut (lower quality; for testing without weights), `--win 1600` changes the window width.

### 2. Export for CVAT

```bash
python export_coco.py          # -> work/instances_default.json
```

### 3. Review in CVAT

1. Create a task. In **Labels → Raw**, paste `labels_raw.json` (each label has `"type": "polygon"`).
2. Upload the original images (unrenamed, filenames must match).
3. **Actions → Upload annotations → COCO 1.0** → `work/instances_default.json`.
4. Review every frame, fix loose outlines or wrong classes, save.
5. **Actions → Export task dataset → COCO 1.0** (untick *Save images*). Keep `annotations/instances_default.json` unedited and rename it `<ENROL>_annotations.json`.

### 4. Generate RGB masks

```bash
python make_masks.py --json <ENROL>_annotations.json \
                     --out rgb_images --prefix <ENROL>_ --images ../group57
```

The script:
- asserts the label set is exactly the six required classes and rejects RLE/brush masks (polygons only),
- draws larger polygons first so small objects stay visible on top,
- writes one PNG per image, same size as the original, same name (plus prefix),
- verifies mask sizes against the originals and warns about images with no annotation.

## Design notes and limitations

- **Accuracy is not automatic.** SAM2 gives pixel-tight boundaries, but class decisions and missed-object checks are human. The CVAT review pass exists to catch outline and class errors.
- **The GrabCut fallback is weak.** In testing it outlined a slipper well but failed on a large white wrapper. Use SAM2 for real annotation.
- **Frames with no plastic** (e.g. wide river shots) produce all-black masks; the assignment says not to leave images blank, so clarify with the instructor how to handle these.
- Polygons are lightly simplified (`approxPolyDP`, ε = 0.6 px) to keep the JSON small without visibly changing the boundary.
- The GUI and SAM2 integration were developed against the ultralytics API; behaviour may vary slightly across `ultralytics` versions.

## Disclosure

Outlines in this project were proposed by a segmentation model (SAM2) and reviewed and corrected by the author in CVAT. The assignment brief asks for polygon annotations drawn in CVAT and states that copied or shared annotations are not accepted; model-assisted pre-annotation was used as a tool, and the instructor/TA should be consulted on whether it is acceptable for graded submission.

## License

MIT. Dataset images belong to their course providers and are not included in this repository.