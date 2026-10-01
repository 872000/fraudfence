"""Fraud detector: z-score anomaly signals + IsolationForest -> 0-100 risk score.

The final risk score is a weighted blend of signals so it stays explainable:

    risk = 100 * clip(w_iso * iso_anomaly
                      + w_amount * amount_z_signal
                      + w_velocity * velocity_signal
                      + w_hour * hour_signal
                      + w_category * category_signal
                      + w_country * country_signal)

Each flagged transaction gets human-readable "top reasons" derived from the
strongest contributing signals.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import FEATURE_COLS, engineer_features

WEIGHTS = {
    "iso": 0.35,
    "amount": 0.25,
    "velocity": 0.15,
    "hour": 0.10,
    "category": 0.10,
    "country": 0.05,
}

FLAG_THRESHOLD = 50.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class FraudModel:
    iso: IsolationForest
    iso_mean: float
    iso_std: float
    feature_cols: list = field(default_factory=lambda: list(FEATURE_COLS))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "FraudModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def train_model(df: pd.DataFrame, contamination: float = 0.03,
                random_state: int = 42) -> FraudModel:
    """Fit the IsolationForest on engineered features.

    ``df`` should be training-period transactions with features engineered
    (use engineer_features first). Fitting on mostly-normal data works best.
    """
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=random_state, n_jobs=-1,
    )
    iso.fit(X)
    scores = iso.score_samples(X)
    return FraudModel(iso=iso, iso_mean=float(scores.mean()),
                      iso_std=float(scores.std() or 1.0))


def _signals(df: pd.DataFrame, model: FraudModel):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    raw = model.iso.score_samples(X)
    iso_anomaly = _sigmoid(-(raw - model.iso_mean) / max(model.iso_std, 1e-9))
    amount_sig = np.clip(np.abs(df["amount_z"].to_numpy()) / 4.0, 0, 1)
    velocity_sig = np.clip(df["prior_1h_count"].to_numpy() / 3.0, 0, 1)
    hour_sig = df["hour_unusualness"].to_numpy()
    category_sig = df["is_new_category"].to_numpy()
    country_sig = df["is_new_country"].to_numpy()
    return {
        "iso": iso_anomaly, "amount": amount_sig, "velocity": velocity_sig,
        "hour": hour_sig, "category": category_sig, "country": country_sig,
    }


def score_dataframe(df: pd.DataFrame, model: FraudModel,
                    threshold: float = FLAG_THRESHOLD) -> pd.DataFrame:
    """Attach risk_score (0-100), flagged, and top_reasons to each row."""
    out = df.copy()
    sig = _signals(out, model)
    risk = sum(WEIGHTS[k] * sig[k] for k in WEIGHTS)
    out["risk_score"] = np.clip(risk, 0, 1) * 100.0
    out["flagged"] = out["risk_score"] >= threshold
    out["top_reasons"] = [_explain(out.iloc[i], sig, i) for i in range(len(out))]
    return out


def _explain(row: pd.Series, sig: dict, i: int) -> str:
    parts: list[tuple[float, str]] = []
    az = float(row["amount_z"])
    if sig["amount"][i] > 0.25:
        direction = "above" if az > 0 else "below"
        parts.append((WEIGHTS["amount"] * sig["amount"][i],
                      f"amount {abs(az):.1f}x std-dev {direction} customer's norm"))
    if sig["velocity"][i] > 0.3:
        parts.append((WEIGHTS["velocity"] * sig["velocity"][i],
                      f"{int(row['prior_1h_count'])} txns in the last hour (velocity burst)"))
    if sig["hour"][i] > 0.5:
        parts.append((WEIGHTS["hour"] * sig["hour"][i],
                      f"unusual hour ({int(row['hour']):02d}:00) for this customer"))
    if sig["category"][i] > 0.5:
        parts.append((WEIGHTS["category"] * sig["category"][i],
                      f"first-ever purchase in category '{row['category']}'"))
    if sig["country"][i] > 0.5:
        parts.append((WEIGHTS["country"] * sig["country"][i],
                      f"first-ever transaction from {row['country']}"))
    if sig["iso"][i] > 0.6:
        parts.append((WEIGHTS["iso"] * sig["iso"][i],
                      "overall pattern differs from normal behavior (IsolationForest)"))
    parts.sort(reverse=True)
    top = [p[1] for p in parts[:3]]
    return "; ".join(top) if top else "mild anomaly signals"
