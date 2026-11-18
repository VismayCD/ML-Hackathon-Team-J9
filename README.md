# PathSnap - GPS map-matching and route reconstruction (Team J-9)

UE24CS352A Machine Learning hackathon, PES University.
Team: Vismay, Vishnu, Ajay, Vishwas.

PathSnap matches noisy GPS points to a road network: for every GPS point it
shortlists nearby road segments (arcs), scores them with a classifier, and uses
an HMM with Viterbi decoding to produce a connected route. A FastAPI server
(Laptop 1) runs the pipeline; a Streamlit UI (Laptop 2) shows the result.

## Pipeline

```
.track/.nodes/.arcs  ->  local metre projection  ->  candidate arcs (<= 8 within 50 m)
  ->  features (distance, heading difference, speed, ...)  ->  classifier probability
  ->  HMM/Viterbi over candidates (road connectivity + distance consistency)
  ->  predicted arc per GPS point  ->  API response  ->  UI map
```

## Repository layout

| Path | What it is |
| --- | --- |
| `src/pathsnap/io.py` | load a trip's four files |
| `src/pathsnap/geo.py` | metre projection, point-to-segment distance, bearings |
| `src/pathsnap/labels.py` | per-GPS-point ground-truth arc from the `.route` sequence (monotonic DP alignment) |
| `src/pathsnap/candidates.py` | KD-tree candidate generation |
| `src/pathsnap/features.py` | feature table per (point, candidate) |
| `src/pathsnap/metrics.py` | Top-1, Macro F1, log loss, miss rate, route continuity |
| `src/pathsnap/hmm.py` | HMM + Viterbi decoder |
| `src/pathsnap/pipeline.py` | `predict_trip()` and server-side `score_trip()` |
| `scripts/eda.py` | dataset exploration tables and plots |
| `scripts/build_features.py` | builds `data/processed/*.parquet` |
| `scripts/train_models.py` | 7 models, GroupKFold CV, student-test evaluation, plots |
| `scripts/tune_optuna.py` | Bayesian optimisation (Optuna TPE) + final model export |
| `scripts/eval_hmm.py` | classifier-only vs classifier + HMM |
| `scripts/make_maps.py` | Folium maps |
| `scripts/plot_trip_static.py` | static figure of one trip (GPS, road network, reference route) |
| `api/main.py` | FastAPI server (Laptop 1) |
| `ui/app.py` | Streamlit UI (Laptop 2) |
| `experiments/` | every CSV, log and plot used in the report |
| `models/` | trained models + `config.json` (`final_model.joblib` = tuned HGB used by the API; `RandomForest.joblib` and `KNN.joblib` are git-ignored for size and rebuilt by `train_models.py`) |
| `maps/` | Folium HTML maps (EDA map for train trip 00000005, result maps for student-test trips) |
| `report/` | final technical report (`9_Report.pdf`) |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Reproduce all results (run from the repo root, in this order)

```bash
python scripts/eda.py                               # experiments/eda/
python scripts/build_features.py                    # data/processed/*.parquet
python scripts/train_models.py                      # experiments/model_comparison.csv, plots, models/*.joblib
python scripts/tune_optuna.py                       # experiments/optuna_trials.csv, models/final_model.joblib
python scripts/eval_hmm.py                          # experiments/hmm_comparison.csv
python scripts/make_maps.py student_test 00000052   # maps/student_test_00000052.html
python scripts/make_maps.py train 00000005 --no-model   # maps/train_00000005.html (EDA map)
python scripts/plot_trip_static.py train 00000005   # experiments/eda/trip_train_00000005.png
```

On Windows, `run_all.bat` runs the first six steps in order.

## Run the demo (two laptops on the same Wi-Fi / hotspot)

Laptop 1 (API):
```bat
REM Command Prompt; copy the evaluator trip folders into data\raw\eval\ first
.venv\Scripts\activate.bat
set EVAL_DIR=data/raw/eval
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

PowerShell: `$env:EVAL_DIR="data/raw/eval"`. Leave `EVAL_DIR` unset to serve the 10 student-test trips.
Find Laptop 1's IP with `ipconfig` (Wi-Fi adapter IPv4). If Laptop 2 cannot connect, allow port 8000 in the
Windows firewall (as Administrator):
`netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000`

Laptop 2 (UI):
```bat
.venv\Scripts\activate.bat
streamlit run ui/app.py
REM in the sidebar enter http://<Laptop-1-IP>:8000
```

## Data rules followed

- Model fitting and CV use only the 80 training trips (GroupKFold by trip).
- The 10 student-test trips are used for model comparison.
- Evaluator trips are used only at demo time; their reference routes are read
  only inside the API's server-side scoring and never sent to the UI or model.

## AI assistance

Starter code was drafted with an AI assistant (Claude) and then run, checked
and modified by the team. All results in the report come from running these
scripts on our machines.
