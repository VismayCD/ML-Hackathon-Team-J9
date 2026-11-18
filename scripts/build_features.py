"""
scripts/build_features.py - run the data pipeline on every train and
student-test trip and save one feature table per split.

Run from the repo root:   python scripts/build_features.py
Outputs:
  data/processed/features_train.parquet   (80 trips)
  data/processed/features_test.parquet    (10 student-test trips)
  experiments/label_quality.csv           (per-trip label distance + miss rate, for the report)
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import load_trip, list_trips          # noqa: E402
from pathsnap.features import build_trip_features      # noqa: E402

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
Path("experiments").mkdir(exist_ok=True)

quality = []
for split, out_name in [("train", "features_train.parquet"), ("student_test", "features_test.parquet")]:
    frames = []
    for trip_dir in list_trips(RAW / split):
        t0 = time.time()
        trip = load_trip(trip_dir)
        df, ctx = build_trip_features(trip)
        frames.append(df)
        has_true = df.groupby("point_idx").label.max()
        quality.append({
            "split": split, "trip_id": trip.trip_id, "n_points": len(trip.track),
            "n_route_arcs": len(trip.route), "n_candidate_rows": len(df),
            "median_label_dist_m": float(pd.Series(ctx["label_dist"]).median()),
            "invalid_point_share": float((ctx["label_dist"] > 50).mean()),
            "candidate_miss_rate": float(1 - has_true.mean()),
            "seconds": round(time.time() - t0, 2),
        })
        print(f"{split:12s} {trip.trip_id}  points={len(trip.track):6d}  rows={len(df):7d}")
    pd.concat(frames, ignore_index=True).to_parquet(OUT / out_name, index=False)

pd.DataFrame(quality).to_csv("experiments/label_quality.csv", index=False)
print("done")
