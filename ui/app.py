"""
ui/app.py - evaluator-facing UI (runs on LAPTOP 2), built with Streamlit.

Flow: enter Laptop 1's API address -> the UI lists the ten evaluation trips
(GET /trips) -> the evaluator picks one -> the UI sends the trip id
(POST /predict) -> it shows the metrics and draws the GPS points and the
predicted route on a Folium map. The UI never reads any data files itself.

Run on Laptop 2 (from the repo root):   streamlit run ui/app.py
"""
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - GPS map-matching demo")

api = st.sidebar.text_input("API address (Laptop 1)", value="http://localhost:8000",
                            help="e.g. http://192.168.43.17:8000 - Laptop 1's IP address")

try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
except Exception as e:  # show a readable error instead of crashing
    st.error(f"Cannot reach the API at {api}: {e}")
    st.stop()

trip_id = st.sidebar.selectbox("Evaluation trip", trips)
if st.sidebar.button("Run map-matching", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API..."):
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=300)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()      # keep the result when the map is panned/zoomed

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']}")
    c = st.columns(5)
    c[0].metric("Top-1 accuracy (HMM)", f"{m.get('hmm_top1_acc', float('nan')):.3f}")
    c[1].metric("Top-1 accuracy (classifier only)", f"{m.get('classifier_top1_acc', float('nan')):.3f}")
    c[2].metric("Route continuity (HMM)", f"{m.get('hmm_route_continuity', float('nan')):.3f}")
    c[3].metric("Route F1", f"{m.get('route_f1', float('nan')):.3f}")
    c[4].metric("Inference time (s)", m.get("inference_seconds"))

    gps = res["gps"]
    fmap = folium.Map(tiles=None, max_zoom=19)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ", name="Basemap (Esri light grey)", max_native_zoom=16, max_zoom=19,
    ).add_to(fmap)   # keyless tiles: OSM/CARTO block pages opened as local files
    gps_layer = folium.FeatureGroup(name="GPS observations (blue)")
    for lat, lon in gps[:: max(1, len(gps) // 1500)]:            # thin very long tracks for speed
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True).add_to(gps_layer)
    route_layer = folium.FeatureGroup(name="Predicted route (red)")
    for seg in res["predicted_route_paths"]:
        folium.PolyLine(seg, color="red", weight=5, opacity=0.8).add_to(route_layer)
    gps_layer.add_to(fmap); route_layer.add_to(fmap)
    folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="black")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                     [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st_folium(fmap, width=1200, height=650, returned_objects=[])
    with st.expander("Raw metrics returned by the API"):
        st.json(m)
