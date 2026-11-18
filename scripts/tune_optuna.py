"""
scripts/tune_optuna.py - Bayesian hyperparameter optimisation (Optuna, TPE sampler)
for the chosen model (HistGradientBoosting by default - change MODEL_NAME if the
comparison picks a different winner and adapt `suggest_params`).

Objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold CV on
the 80 TRAIN trips only. The student-test trips are used only AFTER tuning to
compare untuned vs tuned. Evaluator trips are never touched.

Run from the repo root:   python scripts/tune_optuna.py          (N_TRIALS below)
Outputs:
  experiments/optuna_trials.csv          <- every trial: params + score (the "search record")
  experiments/optuna_history.png         <- optimisation history plot
  experiments/tuned_vs_untuned.csv       <- section 4 comparison table
  models/final_model.joblib              <- tuned model refit on all train trips
  models/config.json                     <- best params + feature list (+ metadata)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURE_COLS   # noqa: E402
from pathsnap.metrics import evaluate         # noqa: E402

SEED = 42
N_TRIALS = 20          # 20 trials x 3 folds ~ 15-30 min on a laptop; lower to 10 if short on time
MODEL_NAME = "HistGradientBoosting"

train = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
test = pd.read_parquet(ROOT / "data/processed/features_test.parquet")
train, test = train[train.valid == 1].reset_index(drop=True), test[test.valid == 1].reset_index(drop=True)
X, y, groups = train[FEATURE_COLS].to_numpy(), train.label.to_numpy(), train.trip_id.to_numpy()
N_FOLDS = 3            # 3 group folds during tuning to keep it fast (still group-aware)
FOLDS = list(GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups)))).split(X, y, groups))


def suggest_params(trial):
    """Search space - ranges chosen around sklearn defaults, wide enough to
    allow both simpler (fewer leaves, more regularisation) and more complex trees."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }


def cv_score(params):
    top1, f1 = [], []
    for tr, va in FOLDS:
        m = HistGradientBoostingClassifier(random_state=SEED, **params).fit(X[tr], y[tr])
        r = evaluate(train.iloc[va], m.predict_proba(X[va])[:, 1])
        top1.append(r["top1_acc"]); f1.append(r["macro_f1"])
    return float(np.mean(top1)), float(np.mean(f1))


def objective(trial):
    top1, f1 = cv_score(suggest_params(trial))
    trial.set_user_attr("cv_macro_f1", f1)   # stored in the trial record too
    return top1


def fit_and_test(params, label):
    """Refit on all train trips, score on student-test, time inference."""
    cv_top1, cv_f1 = cv_score(params)
    m = HistGradientBoostingClassifier(random_state=SEED, **params)
    t0 = time.time(); m.fit(X, y); train_time = time.time() - t0
    t0 = time.time(); proba = m.predict_proba(test[FEATURE_COLS].to_numpy())[:, 1]; infer = time.time() - t0
    r = evaluate(test, proba)
    return m, {"model": label, "cv_top1": cv_top1, "cv_macro_f1": cv_f1, "test_top1_acc": r["top1_acc"],
               "test_macro_f1": r["macro_f1"], "test_log_loss": r["log_loss"],
               "train_time_s": train_time, "inference_time_s": infer}


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="pathsnap_hgb")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"Optimisation took {time.time() - t0:.0f} s; best CV top-1 = {study.best_value:.4f}")
    print("Best params:", study.best_params)

    (ROOT / "experiments").mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(ROOT / "experiments/optuna_trials.csv", index=False)
    vals = [t.value for t in study.trials]
    plt.figure(figsize=(7, 4)); plt.plot(vals, "o", alpha=.6, label="trial")
    plt.plot(np.maximum.accumulate(vals), "-", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.title("Optuna TPE optimisation history"); plt.legend()
    plt.tight_layout(); plt.savefig(ROOT / "experiments/optuna_history.png", dpi=150); plt.close()

    untuned_params = {"max_iter": 300, "learning_rate": 0.1}       # same as train_models.py
    untuned, r_untuned = fit_and_test(untuned_params, "untuned")
    tuned, r_tuned = fit_and_test(study.best_params, "tuned")
    pd.DataFrame([r_untuned, r_tuned]).to_csv(ROOT / "experiments/tuned_vs_untuned.csv", index=False)
    print(pd.DataFrame([r_untuned, r_tuned]).round(4).to_string(index=False))

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(tuned, ROOT / "models/final_model.joblib")
    joblib.dump(untuned, ROOT / "models/untuned_model.joblib")
    json.dump({"model": MODEL_NAME, "best_params": study.best_params, "feature_cols": FEATURE_COLS,
               "n_trials": N_TRIALS, "seed": SEED, "best_cv_top1": study.best_value},
              open(ROOT / "models/config.json", "w"), indent=2)
    print("Saved models/final_model.joblib and models/config.json")
