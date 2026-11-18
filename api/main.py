"""
api/main.py - FastAPI backend (runs on LAPTOP 1).

Endpoints
  GET  /health              -> {"status": "ok"}  (quick connectivity check from Laptop 2)
  GET  /trips               -> the trip ids available in EVAL_DIR (the 10 evaluation trips)
  POST /predict {"trip_id"} -> runs the full pipeline on that trip and returns
                               GPS points, the predicted route geometry and metrics.

Ground truth rule: the reference route is read ONLY inside score_trip() on this
server to compute accuracy. It is never part of the response, and the model
only ever sees the trip loaded with with_route=False.

Run (from the repo root, on Laptop 1):
  Windows PowerShell:  $env:EVAL_DIR="data/raw/eval";  uvicorn api.main:app --host 0.0.0.0 --port 8000
  For testing before demo day, leave EVAL_DIR unset -> uses data/raw/student_test.
Every request is logged in this terminal - show these logs to the evaluator.
"""
import logging
import os
import sys
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import load_trip                                      # noqa: E402
from pathsnap.pipeline import arc_lonlat_paths, predict_trip, score_trip  # noqa: E402

EVAL_DIR = ROOT / os.environ.get("EVAL_DIR", "data/raw/student_test")
MODEL_PATH = ROOT / os.environ.get("MODEL_PATH", "models/final_model.joblib")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
# allow the UI on Laptop 2 (another origin) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

log.info("Loading model from %s", MODEL_PATH)
MODEL = joblib.load(MODEL_PATH)                 # loaded once at start-up, reused for every request
log.info("Serving trips from %s", EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "eval_dir": str(EVAL_DIR.name)}


@app.get("/trips")
def trips():
    ids = sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    # only accept ids that are actually listed in EVAL_DIR (blocks "../train/..." tricks)
    allowed = {p.name for p in EVAL_DIR.iterdir() if p.is_dir()}
    if req.trip_id not in allowed:
        raise HTTPException(status_code=404, detail=f"unknown trip {req.trip_id}")
    trip_dir = EVAL_DIR / req.trip_id
    log.info("POST /predict trip=%s : running pipeline", req.trip_id)
    t0 = time.time()
    pred = predict_trip(trip_dir, MODEL, use_hmm=True)
    trip = pred["trip"]

    metrics = {}
    reference = load_trip(trip_dir).route        # server-side only
    if reference is not None:
        metrics = score_trip(pred, reference)
    metrics["inference_seconds"] = round(time.time() - t0, 3)

    log.info("trip=%s points=%d classifier_top1=%s hmm_top1=%s time=%.2fs", req.trip_id, len(trip.track),
             metrics.get("classifier_top1_acc"), metrics.get("hmm_top1_acc"), metrics["inference_seconds"])
    return {
        "trip_id": req.trip_id,
        "gps": trip.track[["lat", "lon"]].round(6).values.tolist(),         # [lat, lon] for Leaflet
        "predicted_route_arcs": [int(a) for a in pred["route_seq"]],
        "predicted_route_paths": arc_lonlat_paths(trip, pred["route_seq"]),   # list of [[lat,lon],[lat,lon]]
        "metrics": metrics,                                                   # numbers only, no reference route
    }
