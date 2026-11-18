"""
scripts/train_models.py - train and compare 7 models from different families.

For every model:
  1. 5-fold GroupKFold CV on the 80 TRAIN trips (groups = trip_id, so one trip
     never appears in both the training and validation part of a fold).
  2. Refit on all train trips, evaluate on the 10 STUDENT-TEST trips.
  3. Record CV Macro F1 (mean +/- std), test Top-1, Macro F1, precision, recall,
     log loss, route continuity (classifier only, before HMM), train + inference time.

Run from the repo root:   python scripts/train_models.py
Outputs:
  experiments/model_comparison.csv   <- the report's comparison table (section 3.1)
  experiments/cv_folds.csv           <- per-fold scores (stability across folds)
  experiments/plots/*.png            <- comparison plots, MLP loss curve, confusion matrix
  models/<name>.joblib               <- every trained model
"""
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                     # save plots without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURE_COLS                       # noqa: E402
from pathsnap.io import load_trip                                # noqa: E402
from pathsnap.metrics import evaluate, top1_predictions, route_continuity  # noqa: E402

SEED = 42
N_FOLDS = 5
MAX_ROWS_SLOW = 200_000   # KNN / MLP are slow on millions of rows -> train them on a random subsample

# ---- the models we compare (scaling is inside a pipeline so it is fitted on training folds only) ----
MODELS = {
    "LogisticRegression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)), False),
    "GaussianNB":         (GaussianNB(), False),
    "KNN":                (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), True),
    "DecisionTree":       (DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED), False),
    "RandomForest":       (RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=SEED), False),
    "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED), False),
    "MLP":                (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                         max_iter=200, random_state=SEED)), True),
}


def maybe_subsample(X, y, slow):
    if slow and len(X) > MAX_ROWS_SLOW:
        idx = np.random.default_rng(SEED).choice(len(X), MAX_ROWS_SLOW, replace=False)
        return X[idx], y[idx]
    return X, y


def main():
    train = pd.read_parquet(ROOT / "data/processed/features_train.parquet")
    test = pd.read_parquet(ROOT / "data/processed/features_test.parquet")
    # only score/train on points whose derived label is trustworthy (see features.py)
    train, test = train[train.valid == 1].reset_index(drop=True), test[test.valid == 1].reset_index(drop=True)
    X, y, groups = train[FEATURE_COLS].to_numpy(), train.label.to_numpy(), train.trip_id.to_numpy()
    Xt = test[FEATURE_COLS].to_numpy()

    # arcs of each student-test trip, to measure route continuity of classifier-only predictions
    test_arcs = {tid: load_trip(ROOT / "data/raw/student_test" / tid, with_route=False).arcs[["u", "v"]].to_numpy()
                 for tid in test.trip_id.unique()}

    (ROOT / "experiments/plots").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    rows, fold_rows = [], []
    n_folds = min(N_FOLDS, len(np.unique(groups)))

    for name, (model, slow) in MODELS.items():
        print(f"\n=== {name} ===")
        # ---------- 1. group-aware cross-validation ----------
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
            Xtr, ytr = maybe_subsample(X[tr], y[tr], slow)
            model.fit(Xtr, ytr)
            m = evaluate(train.iloc[va], model.predict_proba(X[va])[:, 1])
            fold_rows.append({"model": name, "fold": fold, **m})
            print(f"  fold {fold}: top1={m['top1_acc']:.4f}  macroF1={m['macro_f1']:.4f}")
        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])

        # ---------- 2. refit on all train trips, evaluate on student-test ----------
        Xtr, ytr = maybe_subsample(X, y, slow)
        t0 = time.time(); model.fit(Xtr, ytr); train_time = time.time() - t0
        t0 = time.time(); proba = model.predict_proba(Xt)[:, 1]; infer_time = time.time() - t0
        m = evaluate(test, proba)

        top1 = top1_predictions(test.assign(proba=proba)).sort_values(["trip_id", "point_idx"])
        continuity = np.mean([route_continuity(g.arc_id.to_numpy(), test_arcs[tid]) for tid, g in top1.groupby("trip_id")])

        rows.append({
            "model": name, "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1_mean": cv.top1_acc.mean(),
            "test_top1_acc": m["top1_acc"], "test_macro_f1": m["macro_f1"], "test_precision": m["precision"],
            "test_recall": m["recall"], "test_log_loss": m["log_loss"], "test_route_continuity": continuity,
            "train_time_s": train_time, "inference_time_s": infer_time, "train_rows": len(Xtr),
        })
        print(f"  TEST top1={m['top1_acc']:.4f} macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f} "
              f"continuity={continuity:.4f}")
        joblib.dump(model, ROOT / f"models/{name}.joblib", compress=3)   # compressed to stay under GitHub limits

        # MLP is the only model with an epoch-wise history -> plot its curves (real, from training)
        if name == "MLP":
            mlp = model[-1]
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(mlp.loss_curve_, label="training loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("log loss")
            ax2 = ax1.twinx(); ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy")
            ax2.set_ylabel("validation accuracy"); fig.legend(loc="upper right"); ax1.set_title("MLP training history")
            fig.tight_layout(); fig.savefig(ROOT / "experiments/plots/mlp_training_curve.png", dpi=150); plt.close(fig)

    res = pd.DataFrame(rows).sort_values("test_top1_acc", ascending=False)
    res.to_csv(ROOT / "experiments/model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT / "experiments/cv_folds.csv", index=False)
    print("\n", res.round(4).to_string(index=False))

    # ---------- 3. comparison plots (all from the real numbers above) ----------
    for col, title, fname in [("cv_macro_f1_mean", "CV Macro F1 (5-fold GroupKFold, train trips)", "cv_macro_f1.png"),
                              ("test_top1_acc", "Student-test Top-1 accuracy", "test_top1.png"),
                              ("test_macro_f1", "Student-test Macro F1", "test_macro_f1.png"),
                              ("test_log_loss", "Student-test log loss (lower is better)", "test_log_loss.png")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        err = res.cv_macro_f1_std if col == "cv_macro_f1_mean" else None
        ax.barh(res.model, res[col], xerr=err, color="tab:blue"); ax.set_title(title); ax.invert_yaxis()
        fig.tight_layout(); fig.savefig(ROOT / f"experiments/plots/{fname}", dpi=150); plt.close(fig)

    # confusion matrix of the best model on student-test candidates
    best = res.iloc[0].model
    best_model = joblib.load(ROOT / f"models/{best}.joblib")
    ConfusionMatrixDisplay.from_predictions(test.label, (best_model.predict_proba(Xt)[:, 1] >= 0.5).astype(int),
                                            display_labels=["not true arc", "true arc"])
    plt.title(f"Confusion matrix - {best} (student-test)"); plt.tight_layout()
    plt.savefig(ROOT / "experiments/plots/confusion_matrix_best.png", dpi=150); plt.close()
    print(f"\nBest by student-test Top-1: {best}")


if __name__ == "__main__":
    main()
