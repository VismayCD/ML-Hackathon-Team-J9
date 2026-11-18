"""
features.py - build the (point, candidate arc) feature table for one trip.

One row = "could GPS point i be on arc a?". Target `label` = 1 if a is the
point's ground-truth arc (from labels.py), else 0. Features only use the GPS
track and the road network - never the route - so the same function runs on
evaluator trips at prediction time (with_labels=False).

Feature            Why it helps
---------------------------------------------------------------------------
dist               closer arcs are more likely (GPS noise is ~5-20 m)
dist_rank          0 = closest candidate of this point
dist_minus_min     how much farther than the best candidate (relative evidence)
heading_diff       angle between driving direction and arc direction (0-180).
                   The ONLY thing that separates A->B from its twin B->A.
heading_reliable   heading is noise when the car is (almost) stopped
speed              m/s; fast points lie on major roads, slow ones near junctions
dt_prev            time gap to previous fix; big gaps = less reliable
arc_len            very short arcs are junction pieces
proj_t             where along the arc the point projects (0 start, 1 end)
n_cands            how crowded the area is (dense city vs open road)
deg_u, deg_v       connectivity of the arc's endpoints (junction complexity)
has_twin           1 if the reverse arc exists (two-way road)
"""
import numpy as np
import pandas as pd

from .candidates import ArcIndex, RADIUS, K
from .geo import LocalProjection, bearing_deg, angle_diff_deg
from .io import Trip
from .labels import align_points_to_route

MIN_SPEED_FOR_HEADING = 1.0   # m/s; below this, GPS heading is unreliable
MAX_LABEL_DIST = 50.0         # m; points farther than this from their label arc are "invalid"


def gps_motion(P, t):
    """Heading (deg), speed (m/s) and time gap (s) for each GPS point.
    Heading uses a central difference (point i-1 -> i+1) which is less noisy
    than a single step."""
    n = len(P)
    prev_i = np.maximum(np.arange(n) - 1, 0)
    next_i = np.minimum(np.arange(n) + 1, n - 1)
    d = P[next_i] - P[prev_i]
    dt = np.maximum(t[next_i] - t[prev_i], 1e-6)
    heading = bearing_deg(d[:, 0], d[:, 1])
    speed = np.linalg.norm(d, axis=1) / dt
    dt_prev = np.r_[0.0, np.diff(t)]
    return heading, speed, dt_prev


def build_trip_features(trip: Trip, with_labels=True, radius=RADIUS, k=K):
    """Returns (features DataFrame, context dict). The context (projection,
    arc index, point coordinates) is reused by the HMM and the API."""
    proj = LocalProjection.for_track(trip.track.lon, trip.track.lat)
    P = proj.to_xy(trip.track.lon, trip.track.lat)
    node_xy = proj.to_xy(trip.nodes.lon, trip.nodes.lat)
    arcs_uv = trip.arcs[["u", "v"]].to_numpy()

    index = ArcIndex(node_xy, arcs_uv)
    df = index.candidates(P, radius=radius, k=k)

    heading, speed, dt_prev = gps_motion(P, trip.track.t.to_numpy())
    A, B = index.A[df.arc_id], index.B[df.arc_id]
    arc_bearing = bearing_deg((B - A)[:, 0], (B - A)[:, 1])
    pi = df.point_idx.to_numpy()

    df["heading_diff"] = angle_diff_deg(heading[pi], arc_bearing)
    df["heading_reliable"] = (speed[pi] >= MIN_SPEED_FOR_HEADING).astype(int)
    df["speed"] = speed[pi]
    df["dt_prev"] = dt_prev[pi]
    df["arc_len"] = index.arc_len[df.arc_id]
    df["dist_minus_min"] = df.dist - df.groupby("point_idx").dist.transform("min")
    df["n_cands"] = df.groupby("point_idx").arc_id.transform("size")

    deg = np.bincount(arcs_uv.ravel(), minlength=len(node_xy))  # in + out degree per node
    df["deg_u"] = deg[arcs_uv[df.arc_id, 0]]
    df["deg_v"] = deg[arcs_uv[df.arc_id, 1]]
    twin = pd.MultiIndex.from_arrays([arcs_uv[:, 1], arcs_uv[:, 0]])
    has_twin = pd.MultiIndex.from_arrays([arcs_uv[:, 0], arcs_uv[:, 1]]).isin(twin)
    df["has_twin"] = has_twin[df.arc_id].astype(int)
    df.insert(0, "trip_id", trip.trip_id)

    ctx = {"proj": proj, "P": P, "index": index, "node_xy": node_xy, "arcs_uv": arcs_uv}
    if with_labels and trip.route is not None:
        label_arc, label_dist, _ = align_points_to_route(P, node_xy, arcs_uv, trip.route)
        df["label"] = (df.arc_id.to_numpy() == label_arc[pi]).astype(int)
        # distance from the point to its derived label arc; > MAX_LABEL_DIST m means
        # the route does not explain this point (e.g. GPS before the route starts),
        # so the label is unreliable -> excluded from training and scoring.
        df["label_dist"] = label_dist[pi]
        df["valid"] = (label_dist[pi] <= MAX_LABEL_DIST).astype(int)
        ctx["label_arc"], ctx["label_dist"] = label_arc, label_dist
    return df, ctx


FEATURE_COLS = ["dist", "dist_rank", "dist_minus_min", "heading_diff", "heading_reliable",
                "speed", "dt_prev", "arc_len", "proj_t", "n_cands", "deg_u", "deg_v", "has_twin"]
