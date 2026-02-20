#!/usr/bin/env python3
"""
Convert D-FINE detection output to standard MOT Challenge format.

MOT Detection format (det.txt):
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

For raw detections: id=-1, x=y=z=-1.
Frames are 1-indexed per MOT convention.

Usage:
    python convert_to_mot.py [--input boxesDFineFeats.npy] [--output det.txt]
"""

import numpy as np
import argparse


def convert_to_mot_format(boxes_file: str, output_file: str):
    """Convert (N,6) detection array to MOT Challenge det.txt.

    Input:  [frame_id, x1, y1, x2, y2, confidence]  (0-indexed frames)
    Output: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z  (1-indexed)
    """
    boxes = np.load(boxes_file)
    print(f"Loaded {boxes_file}: {boxes.shape[0]} detections, "
          f"frames {boxes[:, 0].min():.0f}-{boxes[:, 0].max():.0f}")

    with open(output_file, "w") as f:
        for row in boxes:
            frame_id = int(row[0]) + 1  # MOT = 1-indexed
            bb_left, bb_top = row[1], row[2]
            bb_width = row[3] - row[1]
            bb_height = row[4] - row[2]
            conf = row[5]
            f.write(f"{frame_id},-1,{bb_left:.2f},{bb_top:.2f},"
                    f"{bb_width:.2f},{bb_height:.2f},{conf:.6f},-1,-1,-1\n")

    # Verify
    with open(output_file) as f:
        lines = f.readlines()
    print(f"Saved {output_file} ({len(lines)} lines)")
    print(f"Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>")
    print(f"Sample: {lines[0].rstrip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert detection output to MOT format")
    parser.add_argument("--input", default="boxesDFineFeats.npy")
    parser.add_argument("--output", default="det.txt")
    args = parser.parse_args()
    convert_to_mot_format(args.input, args.output)
