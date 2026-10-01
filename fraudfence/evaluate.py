"""Evaluate the detector against labeled synthetic data.

Time-based split: train on the first 70% of the timeline, evaluate on the
last 30%. Prints a metrics table and saves a text report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .detector import FLAG_THRESHOLD, score_dataframe, train_model
from .features import engineer_features


def evaluate(df: pd.DataFrame, threshold: float = FLAG_THRESHOLD,
             contamination: float = 0.03) -> dict:
    df = engineer_features(df)
    cutoff = df["timestamp"].quantile(0.7)
    train, test = df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]
    model = train_model(train, contamination=contamination)
    scored = score_dataframe(test, model, threshold=threshold)

    y_true = scored["is_fraud"].to_numpy(dtype=int)
    y_pred = scored["flagged"].to_numpy(dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": threshold,
        "n_train": len(train), "n_test": len(test),
        "fraud_in_test": int(y_true.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "scored": scored, "model": model,
    }


def format_report(res: dict) -> str:
    lines = [
        "FraudFence evaluation report",
        "===========================",
        f"Train size: {res['n_train']}   Test size: {res['n_test']}   "
        f"Fraud in test: {res['fraud_in_test']}",
        f"Risk threshold: {res['threshold']:.0f}",
        "",
        f"True positives : {res['tp']}",
        f"False positives: {res['fp']}",
        f"False negatives: {res['fn']}",
        f"True negatives : {res['tn']}",
        "",
        f"Precision: {res['precision']:.3f}",
        f"Recall   : {res['recall']:.3f}",
        f"F1 score : {res['f1']:.3f}",
    ]
    return "\n".join(lines)


def run(df: pd.DataFrame, out_path: str | Path | None = None) -> dict:
    res = evaluate(df)
    report = format_report(res)
    print(report)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n")
        print(f"\nReport saved to {out_path}")
    return res
