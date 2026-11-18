"""
scripts/make_maps.py - interactive Folium maps (report sections 2 and 6).

Layers (each can be switched on/off with the layer control, top right):
  grey   road network near the trip
  blue   raw GPS observations
  green  reference route (only in these offline report maps - NEVER in the UI)
  orange classifier-only prediction
  red    final prediction (classifier + HMM)

Run from the repo root:
  python scripts/make_maps.py student_test 00000052        -> maps/student_test_00000052.html
  python scripts/make_maps.py train 00000005 --no-model    -> GPS + network + reference only (EDA map)
Open the .html in a browser, then screenshot it for the PDF.
"""
import sys
from pathlib import Path

import folium
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import load_trip                         # noqa: E402
from pathsnap.pipeline import arc_lonlat_paths, predict_trip  # noqa: E402

split, tid = sys.argv[1], sys.argv[2]
use_model = "--no-model" not in sys.argv
trip_dir = ROOT / "data/raw" / split / tid
trip = load_trip(trip_dir)
lat, lon = trip.track.lat.to_numpy(), trip.track.lon.to_numpy()

fmap = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles=None, max_zoom=19)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ", name="Basemap (Esri light grey)", max_native_zoom=16, max_zoom=19,
).add_to(fmap)   # keyless tiles: OSM/CARTO block pages opened as local files

# road network: only arcs with an end inside the track's bounding box (+ ~300 m margin) to keep the file small
pad_lat, pad_lon = 0.003, 0.003 / max(np.cos(np.radians(lat.mean())), 0.2)
nl, no = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
inside = (nl > lat.min() - pad_lat) & (nl < lat.max() + pad_lat) & (no > lon.min() - pad_lon) & (no < lon.max() + pad_lon)
near = np.where(inside[trip.arcs.u.to_numpy()] | inside[trip.arcs.v.to_numpy()])[0]
net = folium.FeatureGroup(name=f"Road network ({len(near)} arcs nearby)")
for seg in arc_lonlat_paths(trip, near):
    folium.PolyLine(seg, color="#888888", weight=2, opacity=0.6).add_to(net)
net.add_to(fmap)

gps = folium.FeatureGroup(name="GPS observations")
for la, lo in zip(lat[:: max(1, len(lat) // 2000)], lon[:: max(1, len(lon) // 2000)]):
    folium.CircleMarker([la, lo], radius=2, color="blue", fill=True, fill_opacity=0.8).add_to(gps)
gps.add_to(fmap)

if trip.route is not None:
    ref = folium.FeatureGroup(name="Reference route (ground truth)")
    for seg in arc_lonlat_paths(trip, trip.route):
        folium.PolyLine(seg, color="green", weight=7, opacity=0.6).add_to(ref)
    ref.add_to(fmap)

if use_model:
    model = joblib.load(ROOT / "models/final_model.joblib")
    pred = predict_trip(trip_dir, model)
    clf_seq = pred["clf_arcs"][np.r_[True, pred["clf_arcs"][1:] != pred["clf_arcs"][:-1]]]
    clf = folium.FeatureGroup(name="Classifier-only prediction", show=False)
    for seg in arc_lonlat_paths(trip, clf_seq):
        folium.PolyLine(seg, color="orange", weight=4, opacity=0.9).add_to(clf)
    clf.add_to(fmap)
    hmm = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
    for seg in arc_lonlat_paths(trip, pred["route_seq"]):
        folium.PolyLine(seg, color="red", weight=3, opacity=0.9).add_to(hmm)
    hmm.add_to(fmap)

folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(fmap)
folium.LayerControl(collapsed=False).add_to(fmap)
fmap.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

legend = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 10px;
            border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>Trip {tid}</b><br>
<span style="color:#888888">&#9644;</span> road network<br>
<span style="color:blue">&#9679;</span> GPS observations<br>
<span style="color:green">&#9644;</span> reference route<br>
<span style="color:orange">&#9644;</span> classifier only<br>
<span style="color:red">&#9644;</span> classifier + HMM
</div>""".replace("{tid}", tid)
fmap.get_root().html.add_child(folium.Element(legend))

out = ROOT / "maps" / f"{split}_{tid}.html"
out.parent.mkdir(exist_ok=True)
fmap.save(str(out))
print("saved", out)
