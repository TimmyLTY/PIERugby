#!/usr/bin/env python3
"""
D-FINE Player Detection for Rugby Video.

Uses the D-FINE object detection model (COCO pre-trained) to detect persons
in rugby match video frames. Outputs bounding boxes + D-FINE features for
downstream tracking.

Outputs:
    - <save_box_name>.npy  : (N, 6) array [frame_id, x1, y1, x2, y2, confidence]
    - DFine_feats.npy      : (N, feat_dim) array of D-FINE last_hidden_state features
    - annotated_output.avi : video with drawn detection boxes

Based on original code by Eric Fenaux, adapted for Rugby.
"""

import numpy as np
import cv2
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import DFineForObjectDetection, AutoImageProcessor


def compute_query_indexes(pred_boxes, result_boxes, h, w):
    """Compute which query index each detected box corresponds to.

    This replicates the 'indexes' field from the old custom post-processing,
    allowing feature extraction from last_hidden_state.

    Args:
        pred_boxes: model raw output boxes (num_queries, 4) in cxcywh normalized
        result_boxes: post-processed boxes (num_dets, 4) in xyxy absolute pixels
        h, w: image dimensions
    Returns:
        indexes: array of query indices for each detection
    """
    # Convert pred_boxes (cxcywh normalized) to xyxy absolute
    cx, cy, bw, bh = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    pred_xyxy = torch.stack([
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ], dim=-1)

    # Match each result box to closest pred box
    indexes = []
    for rbox in result_boxes:
        dists = torch.sum((pred_xyxy - rbox.unsqueeze(0)) ** 2, dim=1)
        indexes.append(torch.argmin(dists).item())
    return indexes


def DFine_player(video_name: str, save_box_name: str, start_frame: int, end_frame: int):
    """Run D-FINE detection on a video segment."""

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    model_name = "ustc-community/dfine_x_coco"
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = DFineForObjectDetection.from_pretrained(model_name).to(device)
    print(f"Model loaded: {model_name}")

    # Find person label ID
    person_label = None
    for idx, name in model.config.id2label.items():
        if name.lower() == "person":
            person_label = int(idx)
            break
    print(f"Person label ID: {person_label}")

    # Open video
    vid = cv2.VideoCapture(video_name)
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_name} ({w}x{h}, {fps:.1f} fps)")

    batch_size = 8
    bboxes, feats = [], []

    video_out = "annotated_output.avi"
    video_writer = cv2.VideoWriter(
        video_out, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h)
    )

    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    length = 10_000 if end_frame == -1 else end_frame - start_frame
    start_time = time.time()
    i_frame = 0
    debug_done = False

    while i_frame < length:
        images = []
        for _ in range(batch_size):
            ret, frame = vid.read()
            if not ret:
                break
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if len(images) == 0:
            break

        inputs = image_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use OFFICIAL post-processing (compatible with Transformers 5.0)
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)] * len(images), threshold=0.5
        )

        for k, result in enumerate(results):
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()

            # Debug: first frame info
            if not debug_done:
                print(f"\n--- DEBUG (first frame) ---")
                print(f"  Raw detections: {len(labels)}")
                print(f"  Unique labels: {np.unique(labels)}")
                lbl_counts = dict(zip(*np.unique(labels, return_counts=True)))
                print(f"  Label counts: {lbl_counts}")
                if len(scores) > 0:
                    print(f"  Score range: {scores.min():.3f} - {scores.max():.3f}")
                print(f"  Result keys: {list(result.keys())}")
                debug_done = True

            # Filter: keep only persons
            if person_label is not None:
                humans = np.where(labels == person_label)[0]
                scores = scores[humans]
                boxes = boxes[humans]

            # Draw on frame
            frame_bgr = cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR)
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame_bgr, f"{s:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            video_writer.write(frame_bgr)

            # Collect detections
            if len(boxes) > 0:
                newdets = np.column_stack((np.ones(len(boxes)) * i_frame, boxes, scores))
                if len(bboxes) == 0:
                    bboxes = newdets.copy()
                else:
                    bboxes = np.vstack((bboxes, newdets))

                # Feature extraction via query index matching
                result_boxes_filtered = result["boxes"].cpu()
                if person_label is not None:
                    result_boxes_filtered = result_boxes_filtered[humans]
                pred_boxes_k = outputs.pred_boxes[k].cpu()
                indexes = compute_query_indexes(pred_boxes_k, result_boxes_filtered, h, w)
                last_hidden = outputs.last_hidden_state[k]
                frame_feats = last_hidden[indexes].cpu().numpy()
                if len(feats) == 0:
                    feats = frame_feats
                else:
                    feats = np.append(feats, frame_feats, axis=0)

            i_frame += 1

        if i_frame % 100 < batch_size:
            elapsed = time.time() - start_time
            print(f"  frame {i_frame}/{length}, {elapsed:.1f}s, {i_frame / elapsed:.1f} fps")

    vid.release()
    video_writer.release()

    if isinstance(bboxes, list):
        bboxes = np.empty((0, 6))
    if isinstance(feats, list):
        feats = np.empty((0, 0))

    # Clip boxes to image bounds
    if len(bboxes) > 0:
        bc = bboxes[:, 1:5]
        bc[:, [0, 2]] = np.clip(bc[:, [0, 2]], 0, w - 1)
        bc[:, [1, 3]] = np.clip(bc[:, [1, 3]], 0, h - 1)
        bboxes[:, 1:5] = bc

    np.save(save_box_name, bboxes)
    np.save('DFine_feats.npy', feats)
    print(f"\n✅ Detection complete:")
    print(f"   Boxes: {save_box_name} — shape {bboxes.shape}")
    print(f"   Features: DFine_feats.npy — shape {np.array(feats).shape}")
    print(f"   Video: {video_out}")


if __name__ == '__main__':
    video_in = "video/vid_3.mp4"
    boxes_file = "boxesDFineFeats.npy"
    DFine_player(video_in, boxes_file, start_frame=50, end_frame=950)
