"""
metrics.py - every number in the report comes from these functions.

Two levels:
  * candidate level (binary rows "is this the true arc?"): Macro F1, precision,
    recall, log loss, confusion matrix.
  * observation level (one decision per GPS point): Top-1 accuracy, candidate
    miss rate, route continuity.
Primary selection metric: observation-level Top-1 accuracy, because the task
is "pick the right arc for each GPS point"; Macro F1 is reported alongside
because the candidate classes are imbalanced (~1 positive per 5-8 rows).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top1_predictions(df: pd.DataFrame, score_col="proba"):
    """Highest-scoring candidate per point -> DataFrame [trip_id, point_idx, arc_id, label]."""
    idx = df.groupby(["trip_id", "point_idx"])[score_col].idxmax()
    return df.loc[idx, ["trip_id", "point_idx", "arc_id", "label"]].reset_index(drop=True)


def candidate_miss_rate(df: pd.DataFrame):
    """Share of GPS points whose true arc is NOT among their candidates."""
    has_true = df.groupby(["trip_id", "point_idx"]).label.max()
    return float(1.0 - has_true.mean())


def route_continuity(pred_arcs, arcs_uv):
    """Share of consecutive predicted arcs that are the same arc or physically
    connected (end node of the first == start node of the next)."""
    a, b = np.asarray(pred_arcs[:-1]), np.asarray(pred_arcs[1:])
    if len(a) == 0:
        return 1.0
    ok = (a == b) | (arcs_uv[a, 1] == arcs_uv[b, 0])
    return float(ok.mean())


def evaluate(df: pd.DataFrame, proba: np.ndarray, threshold=0.5):
    """All candidate- and observation-level metrics for scored rows.
    df must contain trip_id, point_idx, arc_id, label (only 'valid' rows)."""
    df = df.assign(proba=proba)
    y, yhat = df.label.to_numpy(), (proba >= threshold).astype(int)
    top1 = top1_predictions(df)
    return {
        "top1_acc": float(top1.label.mean()),   # misses count as wrong
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "miss_rate": candidate_miss_rate(df),
    }
