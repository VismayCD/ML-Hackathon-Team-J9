"""
pipeline.py - the complete prediction pipeline used by the API and the report.

predict_trip(): raw trip folder -> candidates -> classifier -> HMM -> route.
It loads the trip WITHOUT its route (with_route=False), so ground truth can
never influence a prediction.

score_trip(): compares a prediction with the reference route. Called by the
API on the server side only; the reference route itself is never returned.
"""
import time

import numpy as np

from .features import FEATURE_COLS, build_trip_features
from .hmm import RoadGraph, viterbi
from .io import load_trip
from .labels import align_points_to_route
from .metrics import route_continuity


def predict_trip(trip_dir, model, use_hmm=True):
    t0 = time.time()
    trip = load_trip(trip_dir, with_route=False)              # <- no ground truth here
    cands, ctx = build_trip_features(trip, with_labels=False)
    cands["proba"] = model.predict_proba(cands[FEATURE_COLS].to_numpy())[:, 1]

    # classifier-only answer: best-scoring candidate for each point
    clf_arcs = cands.loc[cands.groupby("point_idx").proba.idxmax()].sort_values("point_idx").arc_id.to_numpy()

    hmm_arcs = None
    if use_hmm:
        graph = RoadGraph(ctx["node_xy"], ctx["arcs_uv"])
        hmm_arcs = viterbi(cands, ctx["P"], trip.track.t.to_numpy(), graph)

    final = hmm_arcs if hmm_arcs is not None else clf_arcs
    # collapse consecutive repeats -> the predicted route as an arc sequence
    route_seq = final[np.r_[True, final[1:] != final[:-1]]]
    return {
        "trip": trip, "ctx": ctx, "clf_arcs": clf_arcs, "hmm_arcs": hmm_arcs,
        "final_arcs": final, "route_seq": route_seq, "seconds": time.time() - t0,
    }


def arc_lonlat_paths(trip, arc_ids):
    """[[lat, lon], [lat, lon]] per arc - Folium/Leaflet order is (lat, lon)!"""
    u = trip.arcs.u.to_numpy()[arc_ids]
    v = trip.arcs.v.to_numpy()[arc_ids]
    nodes = trip.nodes
    lat, lon = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    return [[[float(lat[a]), float(lon[a])], [float(lat[b]), float(lon[b])]] for a, b in zip(u, v)]


def score_trip(pred, route, max_label_dist=50.0):
    """Server-side scoring against the reference route (never sent to the UI)."""
    ctx = pred["ctx"]
    label_arc, label_dist, _ = align_points_to_route(ctx["P"], ctx["node_xy"], ctx["arcs_uv"], route)
    valid = label_dist <= max_label_dist
    out = {"n_points": int(len(label_arc)), "n_scored_points": int(valid.sum())}
    for key, arcs in [("classifier", pred["clf_arcs"]), ("hmm", pred["hmm_arcs"])]:
        if arcs is None:
            continue
        out[f"{key}_top1_acc"] = float((arcs[valid] == label_arc[valid]).mean())
        out[f"{key}_route_continuity"] = route_continuity(arcs, ctx["arcs_uv"])
    # sequence level: overlap between predicted arc set and reference arc set
    pred_set, ref_set = set(pred["route_seq"].tolist()), set(route.tolist())
    tp = len(pred_set & ref_set)
    prec = tp / max(len(pred_set), 1); rec = tp / max(len(ref_set), 1)
    out.update({"route_precision": prec, "route_recall": rec,
                "route_f1": 2 * prec * rec / max(prec + rec, 1e-9)})
    return out
