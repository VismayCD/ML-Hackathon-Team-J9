"""
io.py - load one PathSnap trip from disk.

Each trip folder (e.g. data/raw/train/00000005/) holds up to four files, all
whitespace-separated with NO header row:
    <id>.track : lon  lat  t        -> GPS observations, t in seconds (~1 Hz)
    <id>.nodes : lon  lat           -> road-network nodes, row index = node id
    <id>.arcs  : from_node to_node  -> directed road segments, row index = arc id
    <id>.route : arc_id             -> ground-truth arcs in travel order (may be absent)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame            # columns: lon, lat, t
    nodes: pd.DataFrame            # columns: lon, lat   (index = node id)
    arcs: pd.DataFrame             # columns: u, v       (index = arc id)
    route: Optional[np.ndarray]    # ground-truth arc ids in order, or None


def _read(path: Path, names):
    # sep=r"\s+" handles both tabs and spaces; header=None because files have no header
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="c")


def load_trip(trip_dir, with_route: bool = True) -> Trip:
    """Read the four files of one trip. Set with_route=False in the prediction
    pipeline so ground truth can never leak into the model's inputs."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"])
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"])
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"]).astype(np.int64)

    # Basic cleaning: drop rows with missing/invalid coordinates, sort by time,
    # and drop duplicate timestamps (keeps the first fix for each second).
    track = track.dropna()
    track = track[(track.lon.between(-180, 180)) & (track.lat.between(-90, 90))]
    track = track.sort_values("t").drop_duplicates("t").reset_index(drop=True)

    route = None
    route_path = trip_dir / f"{tid}.route"
    if with_route and route_path.exists() and route_path.stat().st_size > 0:
        route = _read(route_path, ["arc"])["arc"].to_numpy(np.int64)
    return Trip(tid, track, nodes, arcs, route)


def list_trips(split_dir):
    """All trip folders inside data/raw/<split>/, sorted by id."""
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())
