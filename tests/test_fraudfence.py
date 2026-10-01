"""Real tests for FraudFence (no mocks of the code under test)."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from fraudfence import detector, evaluate, features, review as review_store
from fraudfence.detector import FLAG_THRESHOLD
from fraudfence.generator import generate


@pytest.fixture(scope="module")
def txns():
    return generate(n_customers=40, n_days=20, fraud_rate=0.03, seed=7)


@pytest.fixture(scope="module")
def scored(txns):
    df = features.engineer_features(pd.DataFrame(txns))
    cutoff = df["timestamp"].quantile(0.7)
    model = detector.train_model(df[df["timestamp"] <= cutoff])
    return detector.score_dataframe(df[df["timestamp"] > cutoff], model)


def test_generator_labels_sane(txns):
    fraud = [t for t in txns if t["is_fraud"]]
    legit = [t for t in txns if not t["is_fraud"]]
    assert len(fraud) > 0 and len(legit) > 0
    assert all(t["fraud_type"] in
               {"odd_hour", "amount_spike", "merchant_mismatch", "velocity_burst"}
               for t in fraud)
    assert all(t["fraud_type"] == "" for t in legit)
    # timestamps sorted
    assert [t["timestamp"] for t in txns] == sorted(t["timestamp"] for t in txns)


def test_generator_is_seeded():
    a = generate(n_customers=10, n_days=5, seed=123)
    b = generate(n_customers=10, n_days=5, seed=123)
    assert [t["txn_id"] for t in a] == [t["txn_id"] for t in b]
    assert [t["amount"] for t in a] == [t["amount"] for t in b]


def test_zscore_flags_obvious_outlier():
    # one customer, 30 small normal txns then one huge one
    rows = []
    for i in range(30):
        rows.append({"txn_id": f"T{i:05d}", "customer_id": "C0001",
                     "timestamp": f"2026-02-{(i % 28) + 1:02d}T12:00:00",
                     "amount": 25.0 + i * 0.5, "merchant": "FreshMart",
                     "category": "groceries", "country": "US"})
    rows.append({"txn_id": "TOUT", "customer_id": "C0001",
                 "timestamp": "2026-03-01T12:00:00", "amount": 9999.0,
                 "merchant": "FreshMart", "category": "groceries", "country": "US"})
    df = features.engineer_features(pd.DataFrame(rows))
    outlier = df[df["txn_id"] == "TOUT"].iloc[0]
    assert outlier["amount_z"] > 3.0, f"amount_z={outlier['amount_z']}"

    model = detector.train_model(df)
    scored = detector.score_dataframe(df, model)
    risk = float(scored[scored["txn_id"] == "TOUT"]["risk_score"].iloc[0])
    assert risk >= FLAG_THRESHOLD, f"risk={risk}"
    assert "amount" in scored[scored["txn_id"] == "TOUT"]["top_reasons"].iloc[0]


def test_risk_score_bounds(scored):
    assert scored["risk_score"].between(0, 100).all()
    assert scored["flagged"].dtype == bool
    assert scored["top_reasons"].str.len().gt(0).all()


def test_evaluate_metrics_sane(scored):
    df = scored.copy()
    # evaluate() re-engineers; feed it raw-shaped rows instead
    raw = df[["txn_id", "customer_id", "timestamp", "amount", "merchant",
              "category", "country", "is_fraud"]].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    res = evaluate.evaluate(raw)
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["f1"] <= 1.0
    assert res["tp"] + res["fn"] == res["fraud_in_test"]
    assert res["tp"] + res["fp"] + res["fn"] + res["tn"] == res["n_test"]
    # detector should beat random guessing on recall
    assert res["recall"] > 0.3, f"recall={res['recall']}"


def test_review_queue_round_trip(tmp_path, scored):
    db = tmp_path / "cases.db"
    conn = review_store.connect(db)
    new = review_store.upsert_cases(conn, scored)
    assert new > 0
    # idempotent re-insert
    assert review_store.upsert_cases(conn, scored) == 0

    cases = review_store.list_cases(conn, status="open")
    assert len(cases) == new
    # ranked by risk desc
    risks = [c["risk_score"] for c in cases]
    assert risks == sorted(risks, reverse=True)

    txn_id = cases[0]["txn_id"]
    assert review_store.decide(conn, txn_id, "fraud", reviewer="tester")
    assert review_store.decide(conn, "NOPE", "legit") is False
    remaining = review_store.list_cases(conn, status="open")
    assert all(c["txn_id"] != txn_id for c in remaining)
    stats = review_store.stats(conn)
    assert stats.get("fraud", 0) == 1
    conn.close()


def test_model_save_load_round_trip(tmp_path, txns):
    df = features.engineer_features(pd.DataFrame(txns).head(500))
    model = detector.train_model(df)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = detector.FraudModel.load(path)
    s1 = detector.score_dataframe(df, model)["risk_score"].to_numpy()
    s2 = detector.score_dataframe(df, loaded)["risk_score"].to_numpy()
    assert (s1 == s2).all()
