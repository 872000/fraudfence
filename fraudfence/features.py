"""Feature engineering for fraud detection.

All per-customer statistics are computed **causally**: each transaction is
scored using only the customer's *prior* transactions, never the future.
This mirrors production scoring, where a transaction must be judged with
the history available at that moment.

Features per transaction:
  * log_amount               - log of the amount
  * hour / dayofweek         - when it happened
  * amount_z                 - from-scratch z-score of log_amount vs the
                               customer's own prior spending
  * hour_unusualness         - 1 - fraction of prior txns within +/-2h
  * prior_1h_count           - velocity: txns by this customer in last 60 min
  * mins_since_last          - gap since previous txn
  * is_new_category          - category never used before by this customer
  * is_new_country           - country never seen before for this customer
  * amount_vs_median         - amount / customer's prior median amount
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "log_amount", "hour", "dayofweek", "amount_z", "hour_unusualness",
    "prior_1h_count", "mins_since_last", "is_new_category",
    "is_new_country", "amount_vs_median",
]

_MIN_HISTORY = 3


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns to a transaction dataframe.

    The dataframe must contain txn_id, customer_id, timestamp, amount,
    category, country. Rows are processed in timestamp order per customer.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feats = {c: np.zeros(len(df)) for c in FEATURE_COLS}

    # per-customer running history: list of (timestamp, log_amount, category, country, hour)
    history: dict[str, list] = defaultdict(list)

    for i, row in df.iterrows():
        cust = row["customer_id"]
        ts = row["timestamp"].to_pydatetime()
        log_amt = float(np.log(max(row["amount"], 0.01)))
        hour = int(ts.hour)
        hist = history[cust]

        if len(hist) >= _MIN_HISTORY:
            past_log = np.array([h[1] for h in hist])
            mean, std = past_log.mean(), past_log.std()
            std = std if std > 1e-6 else 1e-6
            amount_z = (log_amt - mean) / std
            median_amt = float(np.median(np.exp(past_log)))
            hour_unusual = 1.0 - float(
                np.mean([abs(h[4] - hour) <= 2 for h in hist]))
            cats_seen = {h[2] for h in hist}
            countries_seen = {h[3] for h in hist}
            is_new_cat = 1.0 if row["category"] not in cats_seen else 0.0
            is_new_cty = 1.0 if row["country"] not in countries_seen else 0.0
            one_hour_ago = ts.timestamp() - 3600
            prior_1h = sum(1 for h in hist if h[0].timestamp() >= one_hour_ago)
            mins_since = (ts - hist[-1][0]).total_seconds() / 60.0
        else:
            amount_z, median_amt = 0.0, float(np.exp(log_amt))
            hour_unusual, is_new_cat, is_new_cty = 0.0, 0.0, 0.0
            prior_1h, mins_since = 0, 24 * 60.0

        feats["log_amount"][i] = log_amt
        feats["hour"][i] = hour
        feats["dayofweek"][i] = ts.weekday()
        feats["amount_z"][i] = float(np.clip(amount_z, -6, 6))
        feats["hour_unusualness"][i] = hour_unusual
        feats["prior_1h_count"][i] = float(min(prior_1h, 20))
        feats["mins_since_last"][i] = float(np.log1p(min(max(mins_since, 0), 60 * 24 * 30)))
        feats["is_new_category"][i] = is_new_cat
        feats["is_new_country"][i] = is_new_cty
        feats["amount_vs_median"][i] = float(
            np.log1p(row["amount"] / max(median_amt, 0.01)))

        hist.append((ts, log_amt, row["category"], row["country"], hour))

    for c in FEATURE_COLS:
        df[c] = feats[c]
    return df
