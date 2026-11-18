"""
hmm.py - HMM + Viterbi route reconstruction.

Why: the classifier scores each GPS point independently, so consecutive points
can jump between parallel or opposite-direction roads that are not connected.
The HMM adds two pieces of knowledge: time order and road connectivity.

HMM definition
  Hidden state at step i : which candidate arc point i is on (its candidates
                           from candidates.py - so states = candidate arcs).
  Emission  log P(obs_i | arc) = EMISSION_WEIGHT * log(classifier probability)
                           (or, for the "no classifier" baseline, a Gaussian
                           on the GPS-to-arc distance).
  Transition log P(arc_b at i | arc_a at i-1)
        = -| network_distance(a -> b) - straight_line_distance(GPS_{i-1}, GPS_i) | / BETA
    i.e. a move is likely when the distance driven along the roads matches how
    far the GPS point moved (Newson & Krumm, 2009). network_distance is the
    shortest path from the projection on arc a to the projection on arc b.
  Unreachable moves (no path within MAX_ROUTE_M) get UNREACHABLE_LOGP instead
  of -infinity, so decoding never dies; long time gaps (> GAP_S seconds)
  use a flat transition (the chain "restarts" because we cannot know the path).

Viterbi (log space):
  delta_i(b) = emission_i(b) + max_a [ delta_{i-1}(a) + transition(a -> b) ]
  keep the argmax a as a back-pointer, then backtrack from the best final state.
"""
import heapq

import numpy as np

BETA = 2.0               # metres; tolerance for driven-vs-GPS distance mismatch (tune it!)
EMISSION_WEIGHT = 1.0     # relative weight of the classifier vs the transition
SIGMA = 10.0              # metres; GPS noise for the distance-only baseline emission
UNREACHABLE_LOGP = -50.0  # log-prob for impossible transitions (instead of -inf)
MAX_ROUTE_M = 500.0       # metres; search limit for shortest paths between steps
GAP_S = 30.0              # seconds; bigger time gaps -> flat transition


class RoadGraph:
    """Directed road graph with arc lengths and cached bounded Dijkstra."""

    def __init__(self, node_xy, arcs_uv):
        self.arcs_uv = arcs_uv
        self.arc_len = np.linalg.norm(node_xy[arcs_uv[:, 1]] - node_xy[arcs_uv[:, 0]], axis=1)
        self.out = [[] for _ in range(len(node_xy))]          # adjacency: node -> [(next_node, length)]
        for (u, v), L in zip(arcs_uv, self.arc_len):
            self.out[u].append((v, L))
        self._cache = {}

    def dist_from(self, src):
        """Shortest distances from node `src` to every node within MAX_ROUTE_M (dict)."""
        if src in self._cache:
            return self._cache[src]
        dist, heap = {src: 0.0}, [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > MAX_ROUTE_M:
                continue
            for v, L in self.out[u]:
                nd = d + L
                if nd < dist.get(v, np.inf) and nd <= MAX_ROUTE_M:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = dist
        return dist

    def route_distance(self, a, ta, b, tb):
        """Driving distance from fraction ta along arc a to fraction tb along arc b."""
        La, Lb = self.arc_len[a], self.arc_len[b]
        if a == b:
            return abs(tb - ta) * La                          # stayed on the same arc
        d_nodes = self.dist_from(self.arcs_uv[a, 1]).get(self.arcs_uv[b, 0])
        if d_nodes is None:
            return None                                       # not reachable within MAX_ROUTE_M
        return (1 - ta) * La + d_nodes + tb * Lb             # rest of a + path + start of b


def viterbi(cands, P, t, graph, use_classifier=True, beta=None, emission_weight=None):
    """cands: DataFrame with point_idx, arc_id, proj_t, dist and (if use_classifier) proba.
    P: (n,2) GPS points in metres, t: (n,) timestamps. Returns predicted arc per point (n,)."""
    # read the module-level values at call time, so changing hmm.BETA takes effect
    beta = BETA if beta is None else beta
    emission_weight = EMISSION_WEIGHT if emission_weight is None else emission_weight
    cands = cands.sort_values(["point_idx", "dist"])
    groups = {i: g for i, g in cands.groupby("point_idx")}
    n = len(P)
    steps = [groups[i] for i in range(n)]                     # every point has >= 1 candidate

    def emission(g):
        if use_classifier:
            return emission_weight * np.log(np.clip(g.proba.to_numpy(), 1e-6, 1.0))
        return -0.5 * (g.dist.to_numpy() / SIGMA) ** 2

    delta = emission(steps[0])
    back = [None]
    for i in range(1, n):
        prev, cur = steps[i - 1], steps[i]
        a_arcs, a_t = prev.arc_id.to_numpy(), prev.proj_t.to_numpy()
        b_arcs, b_t = cur.arc_id.to_numpy(), cur.proj_t.to_numpy()
        gps_d = float(np.linalg.norm(P[i] - P[i - 1]))
        trans = np.zeros((len(a_arcs), len(b_arcs)))
        if t[i] - t[i - 1] <= GAP_S:                           # normal step: score each move
            for x, (a, ta) in enumerate(zip(a_arcs, a_t)):
                for z, (b, tb) in enumerate(zip(b_arcs, b_t)):
                    rd = graph.route_distance(a, ta, b, tb)
                    trans[x, z] = UNREACHABLE_LOGP if rd is None else -abs(rd - gps_d) / beta
        # else: big GPS gap -> flat transitions (trans stays 0)
        scores = delta[:, None] + trans                       # (prev states, current states)
        back.append(scores.argmax(axis=0))
        delta = scores.max(axis=0) + emission(cur)

    # backtrack
    state = int(delta.argmax())
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = steps[i].arc_id.to_numpy()[state]
        if i > 0:
            state = int(back[i][state])
    return path
