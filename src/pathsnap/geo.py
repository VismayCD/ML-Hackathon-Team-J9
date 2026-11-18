"""
geo.py - geometry helpers, all in METRES.

Why a local projection: lon/lat are angles, so Euclidean distance in degrees is
wrong (1 degree of longitude shrinks with cos(latitude)). Every trip covers only
a few km, so we use a local equirectangular projection centred on the trip:
    x = R * (lon - lon0) * cos(lat0)      y = R * (lat - lat0)     (radians)
Over a few km the error vs true geodesic distance is well under 1%, which is
far smaller than GPS noise (5-20 m). Trips are worldwide, so the centre (lon0,
lat0) is chosen per trip, never globally.
"""
import numpy as np

EARTH_R = 6_371_000.0  # mean Earth radius in metres


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180.0  # metres per degree lon
        self.ky = EARTH_R * np.pi / 180.0                              # metres per degree lat

    @classmethod
    def for_track(cls, lon, lat):
        # centre on the GPS track (the area we care about)
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def to_xy(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])

    def to_lonlat(self, xy):
        xy = np.asarray(xy, float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def point_segment_distance(P, A, B):
    """Distance from points P to segments A->B (all arrays of shape (n, 2), row-wise).
    Returns (dist, t) where t in [0, 1] is how far along the segment the closest
    point lies (0 = at A, 1 = at B)."""
    AB = B - A
    AP = P - A
    seg_len2 = np.einsum("ij,ij->i", AB, AB)
    # guard zero-length arcs (A == B) to avoid divide-by-zero
    t = np.where(seg_len2 > 0, np.einsum("ij,ij->i", AP, AB) / np.maximum(seg_len2, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = A + t[:, None] * AB
    return np.linalg.norm(P - closest, axis=1), t


def bearing_deg(dx, dy):
    """Compass-style bearing of a vector in degrees [0, 360): 0 = north, 90 = east."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
