"""
candidates.py - for every GPS point, find the nearby arcs it could be on.

The classifier never sees the whole network: for each point we shortlist the
K closest arcs within RADIUS metres. These "candidates" are the rows the
classifier scores, and later the hidden states of the HMM.

Fast search: we sample points every STEP metres along every arc and put them
in a KD-tree. A ball query around a GPS point returns the arcs that pass
nearby; then we compute the exact point-to-segment distance and keep the K best.

If the true arc is not in a point's shortlist, no model can get that point
right. That fraction is the CANDIDATE MISS RATE - report it, and tune
RADIUS/K so it is low (e.g. < 2-3%) without exploding the number of rows.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import point_segment_distance

RADIUS = 50.0   # metres - search radius around each GPS point
K = 8           # max candidates kept per point
STEP = 10.0     # metres - sampling step along arcs for the KD-tree


class ArcIndex:
    """KD-tree over points sampled along every arc of one trip's network."""

    def __init__(self, node_xy, arcs_uv, step=STEP):
        self.node_xy, self.arcs_uv = node_xy, arcs_uv
        self.A = node_xy[arcs_uv[:, 0]]
        self.B = node_xy[arcs_uv[:, 1]]
        self.arc_len = np.linalg.norm(self.B - self.A, axis=1)
        n_samples = np.maximum(2, np.ceil(self.arc_len / step).astype(int) + 1)
        arc_of_sample = np.repeat(np.arange(len(arcs_uv)), n_samples)
        # fraction along the arc for each sample: 0 ... 1
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_samples])
        self.samples = self.A[arc_of_sample] + frac[:, None] * (self.B - self.A)[arc_of_sample]
        self.arc_of_sample = arc_of_sample
        self.tree = cKDTree(self.samples)
        self.step = step

    def candidates(self, P, radius=RADIUS, k=K):
        """Return a DataFrame with one row per (point, candidate arc):
        point_idx, arc_id, dist, proj_t. Every point gets at least 1 candidate
        (the nearest arc, even if farther than `radius`) so the HMM never has an
        empty step."""
        # half the sampling step is added so arcs passing between samples aren't missed
        hits = self.tree.query_ball_point(P, r=radius + self.step / 2)
        pt_idx, arc_idx = [], []
        for i, h in enumerate(hits):
            if not h:  # nothing nearby -> fall back to the single nearest arc
                _, j = self.tree.query(P[i])
                h = [j]
            arcs = np.unique(self.arc_of_sample[h])
            pt_idx.append(np.full(len(arcs), i))
            arc_idx.append(arcs)
        pt_idx = np.concatenate(pt_idx)
        arc_idx = np.concatenate(arc_idx)

        dist, t = point_segment_distance(P[pt_idx], self.A[arc_idx], self.B[arc_idx])
        df = pd.DataFrame({"point_idx": pt_idx, "arc_id": arc_idx, "dist": dist, "proj_t": t})
        # keep arcs inside the radius, but always keep each point's nearest one
        df = df.sort_values(["point_idx", "dist"])
        df["dist_rank"] = df.groupby("point_idx").cumcount()
        df = df[(df.dist <= radius) | (df.dist_rank == 0)]
        df = df[df.dist_rank < k].reset_index(drop=True)
        return df
