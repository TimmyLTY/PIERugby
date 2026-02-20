#!/usr/bin/env python3
"""
Phase 2: Team Classification using HAC on appearance embeddings.

Clusters detection embeddings into 2 teams using Agglomerative Clustering.
Adds 'team_id' (1 or 2) to the tracking dict.

Usage:
    python classify_teams.py
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


DICT_FILE = "tracking_dict.npy"
EMBEDDS_FILE = "DFine_feats.npy"


def classify_teams(dict_file: str, embedds_file: str, n_teams: int = 2):
    track_dict = np.load(dict_file, allow_pickle=True).item()
    embedds = np.load(embedds_file)

    with_embedds = np.where(track_dict['in_pitch'])[0]
    feats = embedds[with_embedds] if len(with_embedds) < len(embedds) else embedds

    print(f"Loaded {embedds_file}: {embedds.shape}")
    print(f"  On-pitch detections: {len(with_embedds)}")

    # Normalize features for cosine-like clustering
    feats_norm = normalize(feats, norm='l2')

    # HAC clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_teams,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(feats_norm)
    # Map to 1-indexed team IDs
    team_ids = labels + 1

    # Store in dict
    full_team_ids = np.zeros(len(track_dict['bboxes']), dtype=np.int32)
    full_team_ids[with_embedds] = team_ids
    track_dict['team_id'] = full_team_ids

    np.save(dict_file, track_dict)

    # Report
    for t in range(1, n_teams + 1):
        count = np.sum(team_ids == t)
        print(f"  Team {t}: {count} detections ({100 * count / len(team_ids):.1f}%)")

    print(f"\n✅ Team classification done, saved to {dict_file}")


if __name__ == "__main__":
    classify_teams(DICT_FILE, EMBEDDS_FILE)
