#!/usr/bin/env python3
"""
Phase 3: Deep-EIoU Tracking — per-team multi-object tracking.

Runs Deep-EIoU tracker separately for each team to reduce identity switches.
Uses appearance embeddings (D-FINE or CLIP) + Kalman filter + EIoU distance.

Adapted from TacTic/MakeTracks/DeepEIoUteam.py

Usage:
    python run_tracking.py
"""

import sys
import os
import time
import argparse
import numpy as np
from scipy.spatial.distance import cdist

# Add Deep-EIoU to path
sys.path.insert(0, "/Users/timmy/PIE/Deep-EIoU/Deep-EIoU")
from tracker.Deep_EIoU import Deep_EIoU

DICT_FILE = "tracking_dict.npy"
EMBEDDS_FILE = "DFine_feats.npy"


def make_parser():
    parser = argparse.ArgumentParser("Deep-EIoU Rugby Tracker")

    # Tracking thresholds
    parser.add_argument("--track_high_thresh", type=float, default=0.6,
                        help="High confidence detection threshold")
    parser.add_argument("--track_low_thresh", type=float, default=0.1,
                        help="Low detection threshold for recovery")
    parser.add_argument("--new_track_thresh", type=float, default=0.7,
                        help="Threshold for creating new tracks")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="Frames to keep lost tracks (higher for rugby, players occlude often)")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="IoU matching threshold")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="Max aspect ratio for valid boxes")
    parser.add_argument("--min_box_area", type=float, default=10,
                        help="Min box area filter")
    parser.add_argument("--nms_thres", type=float, default=0.7,
                        help="NMS threshold")
    parser.add_argument("--mot20", default=False, action="store_true")

    # EIoU expansion — teacher says this needs tuning for rugby
    parser.add_argument("--init_expand_scale", type=float, default=0.7,
                        help="Expansion factor for EIoU distance")

    # Re-ID
    parser.add_argument("--with-reid", dest="with_reid", default=True,
                        action="store_true", help="Use appearance matching")
    parser.add_argument("--proximity_thresh", type=float, default=0.5,
                        help="Min IoU for Re-ID matching")
    parser.add_argument("--appearance_thresh", type=float, default=0.25,
                        help="Max cosine distance for Re-ID")

    return parser


def run_tracking(dict_file: str, embedds_file: str):
    args = make_parser().parse_args([])  # use defaults

    track_dict = np.load(dict_file, allow_pickle=True).item()

    # Clean previous tracking
    if 'track_ids' in track_dict:
        del track_dict['track_ids']

    boxes = track_dict['bboxes']
    with_embedds = np.where(track_dict['in_pitch'])[0]
    in_frame = boxes[:, 0].astype(np.int32)

    det_boxes = boxes[:, 1:]  # x1, y1, x2, y2, conf
    in_which_track = -np.ones(len(boxes), dtype=np.int32)

    embedds = np.load(embedds_file)
    print(f"Loaded embeddings: {embedds.shape}")

    # Filter to on-pitch detections
    in_frame_pitch = in_frame[with_embedds]
    boxes_pitch = det_boxes[with_embedds]
    in_which_track_pitch = -np.ones(len(with_embedds), dtype=np.int32)

    max_frame = in_frame_pitch.max()

    # Check if team classification is available and useful
    has_teams = ('team_id' in track_dict
                 and len(np.unique(track_dict['team_id'][track_dict['team_id'] > 0])) > 1)

    if has_teams:
        team_id_pitch = track_dict['team_id'][with_embedds]
        teams_to_track = [1, 2]
        print(f"Mode: per-team tracking")
    else:
        # Single tracker for all detections
        team_id_pitch = np.ones(len(with_embedds), dtype=np.int32)
        teams_to_track = [1]
        print(f"Mode: single tracker (no team split)")

    print(f"Tracking {len(with_embedds)} detections across {max_frame + 1} frames")
    print(f"  Parameters: track_buffer={args.track_buffer}, "
          f"expand_scale={args.init_expand_scale}, "
          f"appearance_thresh={args.appearance_thresh}")

    t0 = time.time()
    for iteam in teams_to_track:
        tracker = Deep_EIoU(args, frame_rate=30)
        track_id_offset = 0 if iteam == 1 else int(max(in_which_track_pitch.max(), 0)) + 1

        team_count = np.sum(team_id_pitch == iteam)
        team_label = f"Team {iteam}" if has_teams else "All"
        print(f"\n  {team_label}: {team_count} detections")

        for i_frame in range(max_frame + 1):
            in_this_frame_and_team = (in_frame_pitch == i_frame) & (team_id_pitch == iteam)
            in_this_frame = np.where(in_this_frame_and_team)[0]

            dets = boxes_pitch[in_this_frame]
            embedds_frame = embedds[in_this_frame]

            online_targets = tracker.update(dets, embedds_frame)

            online_ids = np.array([])
            online_scores = []

            for t in online_targets:
                tlwh = t.last_tlwh
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_ids = np.append(online_ids, t.track_id)
                    online_scores.append(t.score)

            online_ids += track_id_offset

            # Match tracked targets back to detections via confidence
            new_confs = np.array(online_scores)
            if len(new_confs) > 0 and len(in_this_frame) > 0:
                old_confs = boxes_pitch[in_this_frame, 4].reshape(-1, 1)
                pairdist = cdist(old_confs, new_confs.reshape(-1, 1), 'euclidean')
                pairs = np.where(pairdist == 0)
                if len(pairs[0]) > 0:
                    in_which_track_pitch[in_this_frame[pairs[0]]] = online_ids[pairs[1]]

        tracked_count = np.sum(in_which_track_pitch[team_id_pitch == iteam] >= 0)
        total_team = np.sum(team_id_pitch == iteam)
        pct = 100 * tracked_count / total_team if total_team > 0 else 0
        print(f"  {team_label}: {tracked_count}/{total_team} tracked ({pct:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n  Tracking time: {elapsed:.1f}s")

    # Write back
    in_which_track[with_embedds] = in_which_track_pitch
    track_dict['track_ids'] = in_which_track.astype(np.int16)
    np.save(dict_file, track_dict)

    # Summary
    valid_tracks = in_which_track[in_which_track >= 0]
    unique_ids = np.unique(valid_tracks)
    print(f"\n✅ Tracking complete:")
    print(f"   Total tracks: {len(unique_ids)}")
    print(f"   Tracked detections: {len(valid_tracks)}/{len(boxes)} "
          f"({100 * len(valid_tracks) / len(boxes):.1f}%)")
    print(f"   Saved to: {dict_file}")

    # Per-track stats
    track_lengths = [np.sum(valid_tracks == tid) for tid in unique_ids]
    print(f"   Track length: mean={np.mean(track_lengths):.0f}, "
          f"min={np.min(track_lengths)}, max={np.max(track_lengths)}")


if __name__ == "__main__":
    run_tracking(DICT_FILE, EMBEDDS_FILE)
