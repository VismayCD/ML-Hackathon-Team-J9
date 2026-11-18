"""
scripts/eda.py - dataset exploration (report section 2). Uses train + student-test only.

Run from the repo root:   python scripts/eda.py
Outputs (experiments/eda/):
  inventory.csv            one row per trip: counts, file sizes, gaps, extent, connectivity
  summary.txt              overall numbers to quote in the report
  points_per_trip.png      histogram of GPS points per trip
  sampling_intervals.png   distribution of time steps between fixes (log scale)
  network_size.png         nodes vs arcs per trip
  speed_distribution.png   GPS speeds (sanity check for outliers)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.geo import LocalProjection   # noqa: E402
from pathsnap.io import list_trips          # noqa: E402

OUT = ROOT / "experiments/eda"
OUT.mkdir(parents=True, exist_ok=True)


def raw(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


rows, all_dt, all_speed = [], [], []
for split in ["train", "student_test"]:
    for d in list_trips(ROOT / "data/raw" / split):
        tid = d.name
        files = {ext: d / f"{tid}.{ext}" for ext in ["track", "nodes", "arcs", "route"]}
        track = raw(files["track"], ["lon", "lat", "t"])          # RAW, before any cleaning
        nodes = raw(files["nodes"], ["lon", "lat"])
        arcs = raw(files["arcs"], ["u", "v"])
        route = raw(files["route"], ["arc"]) if files["route"].exists() else pd.DataFrame(columns=["arc"])

        dt = np.diff(track.t.to_numpy())
        proj = LocalProjection.for_track(track.lon, track.lat)
        P = proj.to_xy(track.lon, track.lat)
        step_m = np.linalg.norm(np.diff(P, axis=0), axis=1)
        speed = step_m / np.maximum(dt, 1e-6)
        all_dt.append(dt); all_speed.append(speed[dt > 0])

        # connectivity of the road network (weakly connected = ignoring arc direction)
        n = len(nodes)
        g = coo_matrix((np.ones(len(arcs)), (arcs.u, arcs.v)), shape=(n, n))
        n_comp, comp = connected_components(g, directed=True, connection="weak")
        deg = np.bincount(np.r_[arcs.u, arcs.v], minlength=n)
        pairs = set(zip(arcs.u, arcs.v))
        two_way = np.mean([(v, u) in pairs for u, v in pairs])

        rows.append({
            "split": split, "trip_id": tid,
            **{f"{k}_kb": round(f.stat().st_size / 1024, 1) if f.exists() else None for k, f in files.items()},
            "n_points": len(track), "n_nodes": n, "n_arcs": len(arcs), "n_route_arcs": len(route),
            "missing_values": int(track.isna().sum().sum()),
            "invalid_coords": int((~track.lon.between(-180, 180) | ~track.lat.between(-90, 90)).sum()),
            "non_increasing_t": int((dt <= 0).sum()),
            "share_1s_steps": float(np.mean(dt == 1)), "max_gap_s": float(dt.max()) if len(dt) else 0,
            "gaps_over_10s": int((dt > 10).sum()),
            "duration_min": float((track.t.iloc[-1] - track.t.iloc[0]) / 60),
            "length_km": float(step_m.sum() / 1000),
            "extent_km": float(np.ptp(P, axis=0).max() / 1000),
            "max_speed_kmh": float(speed[dt > 0].max() * 3.6) if (dt > 0).any() else 0,
            "center_lat": float(track.lat.mean()), "center_lon": float(track.lon.mean()),
            "network_components": int(n_comp),
            "largest_component_share": float(np.bincount(comp).max() / n),
            "isolated_nodes": int((deg == 0).sum()), "two_way_share": float(two_way),
        })
        print(split, tid, len(track), "points")

inv = pd.DataFrame(rows)
inv.to_csv(OUT / "inventory.csv", index=False)
dt_all, sp_all = np.concatenate(all_dt), np.concatenate(all_speed)

with open(OUT / "summary.txt", "w") as f:
    for split, g in inv.groupby("split"):
        f.write(f"[{split}] trips={len(g)} points={g.n_points.sum()} "
                f"points/trip median={g.n_points.median():.0f} min={g.n_points.min()} max={g.n_points.max()}\n")
    f.write(f"share of 1-second steps: {np.mean(dt_all == 1):.4f}\n")
    f.write(f"time gaps > 10 s: {(dt_all > 10).sum()}  largest gap: {dt_all.max():.0f} s\n")
    f.write(f"missing values: {inv.missing_values.sum()}  invalid coords: {inv.invalid_coords.sum()}  "
            f"non-increasing timestamps: {inv.non_increasing_t.sum()}\n")
    f.write(f"speed > 200 km/h (likely GPS jumps): {(sp_all * 3.6 > 200).sum()} steps\n")
    f.write(f"networks with >1 component: {(inv.network_components > 1).sum()}  "
            f"isolated nodes total: {inv.isolated_nodes.sum()}\n")
    f.write(f"two-way arc share (mean): {inv.two_way_share.mean():.3f}\n")
print(open(OUT / "summary.txt").read())

plt.figure(figsize=(7, 4)); plt.hist(inv.n_points, bins=30); plt.xlabel("GPS points per trip"); plt.ylabel("trips")
plt.title("GPS points per trip (train + student-test)"); plt.tight_layout(); plt.savefig(OUT / "points_per_trip.png", dpi=150)
plt.figure(figsize=(7, 4)); plt.hist(dt_all, bins=np.arange(0, 31), log=True); plt.xlabel("seconds between fixes")
plt.ylabel("count (log)"); plt.title("Sampling intervals"); plt.tight_layout(); plt.savefig(OUT / "sampling_intervals.png", dpi=150)
plt.figure(figsize=(6, 5)); plt.scatter(inv.n_nodes, inv.n_arcs, c=(inv.split == "train"), cmap="coolwarm")
plt.xscale("log"); plt.yscale("log"); plt.xlabel("nodes"); plt.ylabel("arcs"); plt.title("Road network size per trip")
plt.tight_layout(); plt.savefig(OUT / "network_size.png", dpi=150)
plt.figure(figsize=(7, 4)); plt.hist(np.clip(sp_all * 3.6, 0, 200), bins=80, log=True); plt.xlabel("speed km/h (clipped at 200)")
plt.title("GPS step speeds"); plt.tight_layout(); plt.savefig(OUT / "speed_distribution.png", dpi=150)
