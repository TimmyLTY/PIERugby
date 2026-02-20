#!/usr/bin/env python3
"""
Phase 1: Prepare the tracking dict_file from detection output.

Creates the data structure expected by Deep-EIoU and downstream modules:
    dict_file (.npy dict):
        'bboxes'   : (N, 6) [frame, x1, y1, x2, y2, conf]
        'in_pitch' : (N,) bool — all True initially (no homography)

Usage:
    python setup_tracking.py
"""

import numpy as np
import os

BOXES_FILE = "boxesDFineFeats.npy"
DICT_FILE = "tracking_dict.npy"


def setup_tracking_dict(boxes_file: str, dict_file: str):
    boxes = np.load(boxes_file)
    print(f"Loaded {boxes_file}: {boxes.shape}")
    print(f"  Frames: {boxes[:, 0].min():.0f} → {boxes[:, 0].max():.0f}")
    print(f"  Detections: {len(boxes)}")

    track_dict = {
        'bboxes': boxes,
        # Without homography, mark all detections as "on pitch"
        'in_pitch': np.ones(len(boxes), dtype=bool),
    }

    np.save(dict_file, track_dict)
    print(f"\n✅ Created {dict_file}")
    print(f"   Keys: {list(track_dict.keys())}")


if __name__ == "__main__":
    if not os.path.exists(BOXES_FILE):
        print(f"ERROR: {BOXES_FILE} not found. Run DFinePlayer.py first.")
        exit(1)
    setup_tracking_dict(BOXES_FILE, DICT_FILE)
