"""
scripts/plot_trip_static.py - static (matplotlib) figure of one trip for the
report's EDA section: the road network (grey), the raw GPS observations (blue)
and the reference route (green). Complements the interactive Folium map.

Run from the repo root:
  python scripts/plot_trip_static.py train 00000005
Output: experiments/eda/trip_<split>_<id>.png
Only train / student_test trips may be plotted (never eval: it draws the route).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import load_trip  # noqa: E402

split, tid = sys.argv[1], sys.argv[2]
if split == "eval":
    sys.exit("refusing to plot evaluator trips (would show the reference route)")
trip = load_trip(ROOT / "data/raw" / split / tid)
lon, lat = trip.track.lon.to_numpy(), trip.track.lat.to_numpy()
nlon, nlat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()

# keep arcs near the track so the figure is readable (bounding box + ~300 m)
pad_lat = 0.003; pad_lon = 0.003 / max(np.cos(np.radians(lat.mean())), 0.2)
inside = lambda x, y: (x > lon.min() - pad_lon) & (x < lon.max() + pad_lon) & (y > lat.min() - pad_lat) & (y < lat.max() + pad_lat)
near = inside(nlon[u], nlat[u]) | inside(nlon[v], nlat[v])
segs = np.stack([np.c_[nlon[u[near]], nlat[u[near]]], np.c_[nlon[v[near]], nlat[v[near]]]], axis=1)

fig, ax = plt.subplots(figsize=(9, 8))
ax.add_collection(LineCollection(segs, colors="#9a9a9a", linewidths=0.8, label=f"road network ({near.sum()} arcs)"))
if trip.route is not None:
    r = trip.route
    rsegs = np.stack([np.c_[nlon[u[r]], nlat[u[r]]], np.c_[nlon[v[r]], nlat[v[r]]]], axis=1)
    ax.add_collection(LineCollection(rsegs, colors="green", linewidths=3.5, alpha=0.55, label=f"reference route ({len(r)} arcs)"))
ax.scatter(lon, lat, s=3, c="blue", label=f"GPS observations ({len(lon)} points)", zorder=3)
ax.plot(lon[0], lat[0], "^", color="darkgreen", ms=10, label="start", zorder=4)
ax.plot(lon[-1], lat[-1], "s", color="black", ms=8, label="end", zorder=4)
ax.set_aspect(1 / np.cos(np.radians(lat.mean())))   # equal metres on both axes
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"{split} trip {tid}: GPS trajectory, road network and reference route")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False); ax.autoscale_view(); fig.tight_layout()
out = ROOT / f"experiments/eda/trip_{split}_{tid}.png"
fig.savefig(out, dpi=150)
print("saved", out)
