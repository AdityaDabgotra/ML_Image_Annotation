"""Shared constants. Labels/colours are EXACTLY as required by the assignment."""
LABELS = [  # (name in CVAT, hex, review hotkey)
    ("Bottle",       "#FF0000", "b"),
    ("Bottle Cap",   "#FF9900", "c"),
    ("Wrapper",      "#FFFF00", "w"),
    ("Polythene",    "#00CC00", "p"),
    ("Shoe, Jacket", "#0099FF", "s"),   # ONE label whose name contains a comma
    ("Foam",         "#FF66FF", "f"),
]
NAME2HEX = {n: h for n, h, _ in LABELS}
NAME2ID = {n: i + 1 for i, (n, _, _) in enumerate(LABELS)}   # CVAT COCO ids start at 1
KEY2NAME = {k: n for n, _, k in LABELS}

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
