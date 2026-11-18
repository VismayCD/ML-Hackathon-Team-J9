"""
scripts/eval_hmm.py - classifier-only vs classifier + HMM/Viterbi on the 10
student-test trips (report section 5), plus a small grid over the decoder's
BETA parameter (allowed: decoder parameters may be tuned on train/student-test).

Run from the repo root:   python scripts/eval_hmm.py
Outputs:
  experiments/hmm_beta_grid.csv       <- mean HMM Top-1 for each BETA
  experiments/hmm_comparison.csv      <- per trip: classifier vs HMM vs distance-only HMM
  experiments/plots/hmm_vs_classifier.png
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pathsnap.hmm as hmm                                   # noqa: E402
from pathsnap.features import FEATURE_COLS, build_trip_features  # noqa: E402
from pathsnap.io import list_trips, load_trip                # noqa: E402
from pathsnap.labels import align_points_to_route            # noqa: E402
from pathsnap.metrics import route_continuity                # noqa: E402
from pathsnap.pipeline import predict_trip, score_trip       # noqa: E402

model = joblib.load(ROOT / "models/final_model.joblib")
trips = list_trips(ROOT / "data/raw/student_test")

# ---- 1. BETA grid: how strictly should driven distance match GPS distance? ----
grid = []
for beta in [2, 5, 10, 20, 50]:
    hmm.BETA = beta
    accs = [score_trip(predict_trip(d, model), load_trip(d).route)["hmm_top1_acc"] for d in trips]
    grid.append({"beta": beta, "mean_hmm_top1": float(np.mean(accs))})
    print(f"BETA={beta:>3}: mean HMM top-1 = {np.mean(accs):.4f}")
grid = pd.DataFrame(grid)
grid.to_csv(ROOT / "experiments/hmm_beta_grid.csv", index=False)
hmm.BETA = float(grid.loc[grid.mean_hmm_top1.idxmax(), "beta"])
print("Chosen BETA =", hmm.BETA, "-> copy this value into src/pathsnap/hmm.py so the API uses it")

# ---- 2. per-trip comparison with the chosen BETA ----
rows = []
for d in trips:
    trip = load_trip(d)
    pred = predict_trip(d, model)
    s = score_trip(pred, trip.route)
    # ablation: HMM with distance-only emissions (no classifier) - shows what the classifier adds
    cands, ctx = build_trip_features(load_trip(d, with_route=False), with_labels=False)
    dist_only = hmm.viterbi(cands, ctx["P"], trip.track.t.to_numpy(), hmm.RoadGraph(ctx["node_xy"], ctx["arcs_uv"]),
                            use_classifier=False)
    lab, ldist, _ = align_points_to_route(ctx["P"], ctx["node_xy"], ctx["arcs_uv"], trip.route)
    valid = ldist <= 50
    rows.append({"trip_id": d.name, **s,
                 "distonly_hmm_top1_acc": float((dist_only[valid] == lab[valid]).mean()),
                 "distonly_hmm_route_continuity": route_continuity(dist_only, ctx["arcs_uv"]),
                 "seconds": pred["seconds"]})
res = pd.DataFrame(rows)
res.to_csv(ROOT / "experiments/hmm_comparison.csv", index=False)
print(res.round(4).to_string(index=False))
print("\nMEAN:\n", res.drop(columns="trip_id").mean().round(4).to_string())

# ---- 3. plot ----
(ROOT / "experiments/plots").mkdir(parents=True, exist_ok=True)
x = np.arange(len(res)); w = 0.27
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w, res.classifier_top1_acc, w, label="classifier only")
ax.bar(x, res.hmm_top1_acc, w, label="classifier + HMM")
ax.bar(x + w, res.distonly_hmm_top1_acc, w, label="distance-only HMM")
ax.set_xticks(x); ax.set_xticklabels(res.trip_id, rotation=45); ax.set_ylabel("Top-1 accuracy")
ax.set_title("Student-test trips: effect of HMM/Viterbi decoding"); ax.legend(); fig.tight_layout()
fig.savefig(ROOT / "experiments/plots/hmm_vs_classifier.png", dpi=150)
