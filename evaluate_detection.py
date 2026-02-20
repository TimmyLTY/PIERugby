#!/usr/bin/env python3
"""
Detection Evaluation Script for Rugby Video.

Evaluates D-FINE detection output quality through:
  - Detection statistics (count, confidence, box dimensions)
  - Temporal consistency (frame coverage, count stability)
  - Overlap / duplicate analysis (intra-frame IoU)
  - Visual spot-check (sample frame grid)

Outputs:
  - Console report
  - evaluation_samples.png : grid of sample frames with drawn detections
  - evaluation_summary.json : machine-readable metrics

Usage:
    python evaluate_detection.py
"""

import numpy as np
import cv2
import json
import os

# ──────────────────────────── Configuration ────────────────────────────
BOXES_FILE = "boxesDFineFeats.npy"
FEATS_FILE = "DFine_feats.npy"
VIDEO_FILE = "video/vid_3.mp4"
START_FRAME = 50  # must match DFinePlayer.py settings
NUM_SAMPLE_FRAMES = 8
OUTPUT_IMAGE = "evaluation_samples.png"
OUTPUT_JSON = "evaluation_summary.json"


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def analyze_detections(boxes, feats):
    """Compute detection statistics."""
    frames = boxes[:, 0].astype(int)
    coords = boxes[:, 1:5]
    scores = boxes[:, 5]

    unique_frames = np.unique(frames)
    counts_per_frame = np.array([np.sum(frames == f) for f in unique_frames])

    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    areas = widths * heights

    # Overlap / duplicate analysis
    duplicate_count = 0
    total_pairs = 0
    for f in unique_frames:
        mask = frames == f
        frame_boxes = coords[mask]
        n = len(frame_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_iou(frame_boxes[i], frame_boxes[j])
                total_pairs += 1
                if iou > 0.5:
                    duplicate_count += 1

    stats = {
        "total_detections": int(len(boxes)),
        "total_frames_with_detections": int(len(unique_frames)),
        "frame_range": [int(unique_frames.min()), int(unique_frames.max())],
        "feature_shape": list(feats.shape),
        "detections_per_frame": {
            "mean": round(float(counts_per_frame.mean()), 1),
            "std": round(float(counts_per_frame.std()), 1),
            "min": int(counts_per_frame.min()),
            "max": int(counts_per_frame.max()),
            "median": round(float(np.median(counts_per_frame)), 0),
        },
        "confidence_scores": {
            "mean": round(float(scores.mean()), 3),
            "std": round(float(scores.std()), 3),
            "min": round(float(scores.min()), 3),
            "max": round(float(scores.max()), 3),
            "median": round(float(np.median(scores)), 3),
            "q25": round(float(np.percentile(scores, 25)), 3),
            "q75": round(float(np.percentile(scores, 75)), 3),
        },
        "box_dimensions": {
            "width_mean": round(float(widths.mean()), 1),
            "width_std": round(float(widths.std()), 1),
            "height_mean": round(float(heights.mean()), 1),
            "height_std": round(float(heights.std()), 1),
            "area_mean": round(float(areas.mean()), 0),
            "aspect_ratio_mean": round(float((widths / np.clip(heights, 1, None)).mean()), 2),
        },
        "overlap_analysis": {
            "total_intra_frame_pairs": int(total_pairs),
            "high_iou_pairs_count": int(duplicate_count),
            "duplicate_rate_pct": round(
                float(duplicate_count / total_pairs * 100 if total_pairs > 0 else 0), 2
            ),
        },
    }

    # Temporal consistency
    expected_frames = set(range(int(unique_frames.min()), int(unique_frames.max()) + 1))
    detected_frames = set(unique_frames.tolist())
    missing_frames = sorted(expected_frames - detected_frames)

    stats["temporal_consistency"] = {
        "count_cv": round(float(
            counts_per_frame.std() / counts_per_frame.mean()
            if counts_per_frame.mean() > 0 else 0
        ), 3),
        "missing_frame_count": len(missing_frames),
        "missing_frame_pct": round(float(
            len(missing_frames) / len(expected_frames) * 100
            if len(expected_frames) > 0 else 0
        ), 1),
    }
    if len(missing_frames) <= 20 and missing_frames:
        stats["temporal_consistency"]["missing_frames"] = missing_frames

    return stats, counts_per_frame, unique_frames


def create_sample_visualization(boxes, video_file, start_frame, num_samples, output_path):
    """Draw detections on sample frames and save as a grid image."""
    frames_col = boxes[:, 0].astype(int)
    unique_frames = np.unique(frames_col)

    indices = np.linspace(0, len(unique_frames) - 1, num_samples, dtype=int)
    sample_frames = unique_frames[indices]

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_file}")
        return

    imgs = []
    for target_frame in sample_frames:
        actual_frame = int(target_frame) + start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        mask = frames_col == target_frame
        frame_boxes = boxes[mask, 1:5]
        frame_scores = boxes[mask, 5]

        for (x1, y1, x2, y2), score in zip(frame_boxes, frame_scores):
            color = (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{score:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        n_det = int(mask.sum())
        cv2.putText(frame, f"Frame {target_frame} | {n_det} dets",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        imgs.append(frame)

    cap.release()
    if not imgs:
        print("ERROR: No frames could be read from video.")
        return

    cols = (len(imgs) + 1) // 2
    h, w = imgs[0].shape[:2]
    scale = min(1.0, 1920 / (w * cols))
    new_w, new_h = int(w * scale), int(h * scale)

    grid_rows = []
    for row_idx in range(2):
        row_imgs = imgs[row_idx * cols : (row_idx + 1) * cols]
        resized = [cv2.resize(img, (new_w, new_h)) for img in row_imgs]
        while len(resized) < cols:
            resized.append(np.zeros((new_h, new_w, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(resized))

    grid = np.vstack(grid_rows)
    cv2.imwrite(output_path, grid)
    print(f"  Sample visualization saved: {output_path}")


def print_report(stats):
    """Print formatted evaluation report."""
    print("\n" + "=" * 65)
    print("  D-FINE RUGBY DETECTION — EVALUATION REPORT")
    print("=" * 65)

    print(f"\n  OVERVIEW")
    print(f"    Total detections:  {stats['total_detections']}")
    print(f"    Frames with dets:  {stats['total_frames_with_detections']}")
    print(f"    Frame range:       {stats['frame_range'][0]} -> {stats['frame_range'][1]}")
    print(f"    Feature shape:     {stats['feature_shape']}")

    dpf = stats["detections_per_frame"]
    print(f"\n  DETECTIONS PER FRAME")
    print(f"    Mean +/- Std:  {dpf['mean']} +/- {dpf['std']}")
    print(f"    Min / Max:     {dpf['min']} / {dpf['max']}")
    print(f"    Median:        {dpf['median']:.0f}")

    cs = stats["confidence_scores"]
    print(f"\n  CONFIDENCE SCORES")
    print(f"    Mean +/- Std:     {cs['mean']} +/- {cs['std']}")
    print(f"    Min / Max:        {cs['min']} / {cs['max']}")
    print(f"    Q25 / Med / Q75:  {cs['q25']} / {cs['median']} / {cs['q75']}")

    bd = stats["box_dimensions"]
    print(f"\n  BOX DIMENSIONS (pixels)")
    print(f"    Width:   {bd['width_mean']} +/- {bd['width_std']}")
    print(f"    Height:  {bd['height_mean']} +/- {bd['height_std']}")
    print(f"    Area:    {bd['area_mean']}")
    print(f"    Aspect ratio (w/h): {bd['aspect_ratio_mean']}")

    oa = stats["overlap_analysis"]
    print(f"\n  OVERLAP / DUPLICATE ANALYSIS")
    print(f"    Intra-frame pairs checked: {oa['total_intra_frame_pairs']}")
    print(f"    High IoU (>0.5) pairs:     {oa['high_iou_pairs_count']}")
    print(f"    Duplicate rate:            {oa['duplicate_rate_pct']}%")

    tc = stats["temporal_consistency"]
    print(f"\n  TEMPORAL CONSISTENCY")
    print(f"    Detection count CV:    {tc['count_cv']}")
    print(f"    Missing frames:        {tc['missing_frame_count']} ({tc['missing_frame_pct']}%)")
    if "missing_frames" in tc and tc["missing_frames"]:
        print(f"    Missing frame IDs:     {tc['missing_frames']}")

    print("\n" + "=" * 65)
    print("  NOTE: mAP/AP50/AP75 require ground truth annotations.")
    print("  This report evaluates internal consistency & quality.")
    print("=" * 65 + "\n")


def main():
    if not os.path.exists(BOXES_FILE):
        print(f"ERROR: {BOXES_FILE} not found. Run DFinePlayer.py first.")
        return
    if not os.path.exists(FEATS_FILE):
        print(f"ERROR: {FEATS_FILE} not found. Run DFinePlayer.py first.")
        return

    boxes = np.load(BOXES_FILE)
    feats = np.load(FEATS_FILE)
    print(f"Loaded {BOXES_FILE}: shape {boxes.shape}")
    print(f"Loaded {FEATS_FILE}: shape {feats.shape}")

    if boxes.shape[0] != feats.shape[0]:
        print(f"WARNING: box count ({boxes.shape[0]}) != feat count ({feats.shape[0]})")

    stats, counts_per_frame, unique_frames = analyze_detections(boxes, feats)
    print_report(stats)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Metrics saved: {OUTPUT_JSON}")

    if os.path.exists(VIDEO_FILE):
        create_sample_visualization(boxes, VIDEO_FILE, START_FRAME, NUM_SAMPLE_FRAMES, OUTPUT_IMAGE)
    else:
        print(f"WARNING: Video not found at {VIDEO_FILE}, skipping visualization.")


if __name__ == "__main__":
    main()
