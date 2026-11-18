"""
labels.py - turn the .route arc SEQUENCE into one ground-truth arc PER GPS POINT.

Problem: .route lists the arcs the vehicle drove, in order (e.g. 105 arcs), but
the track has one row per second (e.g. 1,045 points). There is no given
point -> arc mapping, so we build one.

Method: monotonic alignment by dynamic programming.
  - Point i may only be assigned to route position j(i), with j never going
    backwards in time: j(0) <= j(1) <= ... (the vehicle drives the route in order).
  - Among all such monotone assignments, choose the one with the smallest total
    point-to-arc distance.
  DP:  D[i, j] = cost[i, j] + min_{k <= j} D[i-1, k]
  The "min over k <= j" is a running (cumulative) minimum, so each row is O(m).

Why not just "nearest route arc"? A route can pass the same street twice or
use both directions of one road (A->B and later B->A). Nearest-arc would mix
these up; the order constraint fixes it.

Limitation (state this in the report): these labels are derived, not given.
Where GPS is far from every route arc (tunnels, bad fixes) the label is still
the best monotone match but may be wrong; we flag such points (label_dist large).
"""
import numpy as np

from .geo import point_segment_distance


def align_points_to_route(P, node_xy, arcs_uv, route):
    """P: (n,2) GPS points in metres. node_xy: (N,2). arcs_uv: (M,2) int.
    route: (m,) arc ids. Returns (label_arc (n,), label_dist (n,), route_pos (n,))."""
    n, m = len(P), len(route)
    A = node_xy[arcs_uv[route, 0]]          # start node of each route arc, (m,2)
    B = node_xy[arcs_uv[route, 1]]          # end node of each route arc,   (m,2)

    # back-pointer table: for each (i, j) the k (<= j) we came from. uint16 saves memory.
    bp_dtype = np.uint16 if m < 65535 else np.uint32
    back = np.empty((n, m), dtype=bp_dtype)
    prev = None
    for i in range(n):
        Pi = np.broadcast_to(P[i], (m, 2))
        cost, _ = point_segment_distance(Pi, A, B)
        if prev is None:
            cur = cost
            back[i] = np.arange(m)
        else:
            # running minimum of prev and where it was achieved
            run_min = np.minimum.accumulate(prev)
            is_new_min = np.r_[True, prev[1:] < run_min[:-1]]
            arg = np.maximum.accumulate(np.where(is_new_min, np.arange(m), 0))
            cur = cost + run_min
            back[i] = arg
        prev = cur

    # backtrack from the best end position
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    label_arc = route[pos]
    label_dist, _ = point_segment_distance(P, node_xy[arcs_uv[label_arc, 0]], node_xy[arcs_uv[label_arc, 1]])
    return label_arc, label_dist, pos
