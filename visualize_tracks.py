#!/usr/bin/env python3
"""
Phase 4: Visualize tracking results on video.

Draws bounding boxes with track IDs and team colors on the video.

Usage:
    python visualize_tracks.py
"""

import numpy as np
import cv2
import os

DICT_FILE = "tracking_dict.npy"
VIDEO_FILE = "video/vid_3.mp4"
START_FRAME = 50
OUTPUT_VIDEO = "tracked_output.avi"

# Team colors (BGR)
TEAM_COLORS = {
    0: (128, 128, 128),  # unassigned: gray
    1: (0, 0, 255),      # team 1: red
    2: (255, 0, 0),      # team 2: blue
}


def visualize_tracks(dict_file: str, video_file: str, output_video: str,
                     start_frame: int = 50):
    track_dict = np.load(dict_file, allow_pickle=True).item()

    bboxes = track_dict['bboxes']
    frames = bboxes[:, 0].astype(int)
    boxes = bboxes[:, 1:5]
    confs = bboxes[:, 5]
    track_ids = track_dict.get('track_ids', -np.ones(len(bboxes)))
    team_ids = track_dict.get('team_id', np.zeros(len(bboxes)))

    vid = cv2.VideoCapture(video_file)
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    unique_frames = np.unique(frames)
    print(f"Drawing tracks on {len(unique_frames)} frames...")

    for i, target_frame in enumerate(unique_frames):
        ret, frame = vid.read()
        if not ret:
            break

        mask = frames == target_frame
        frame_boxes = boxes[mask]
        frame_ids = track_ids[mask]
        frame_teams = team_ids[mask]

        for (x1, y1, x2, y2), tid, team in zip(frame_boxes, frame_ids, frame_teams):
            tid = int(tid)
            team = int(team)

            if tid < 0:
                # Untracked detection — draw thin gray
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (180, 180, 180), 1)
                continue

            color = TEAM_COLORS.get(team, (128, 128, 128))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 6),
                          (int(x1) + tw + 4, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 2, int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Frame info
        n_tracked = int(np.sum(frame_ids >= 0))
        cv2.putText(frame, f"Frame {target_frame} | {n_tracked} tracked",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(unique_frames)} frames")

    vid.release()
    writer.release()
    print(f"\n✅ Tracking video saved: {output_video}")


if __name__ == "__main__":
    visualize_tracks(DICT_FILE, VIDEO_FILE, OUTPUT_VIDEO, START_FRAME)
