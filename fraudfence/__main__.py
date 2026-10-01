"""FraudFence CLI.

Usage:
    python -m fraudfence generate --out data/transactions.csv --seed 42
    python -m fraudfence score --in data/transactions.csv --out data/scored.csv
    python -m fraudfence review list
    python -m fraudfence review decide T0001234 fraud
    python -m fraudfence evaluate --in data/demo_transactions.csv
    python -m fraudfence demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import review as review_store
from .detector import FLAG_THRESHOLD, FraudModel, score_dataframe, train_model
from .evaluate import run as run_evaluate
from .features import engineer_features
from .generator import generate, save_csv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "cases.db"
MODEL_PATH = ROOT / "data" / "model.pkl"


def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"error: file not found: {path}")
    return pd.read_csv(p)


def cmd_generate(args: argparse.Namespace) -> None:
    txns = generate(n_customers=args.customers, n_days=args.days,
                    fraud_rate=args.fraud_rate, seed=args.seed)
    path = save_csv(txns, args.out)
    n_fraud = sum(1 for t in txns if t["is_fraud"])
    print(f"Generated {len(txns)} transactions ({n_fraud} fraud) -> {path}")


def cmd_score(args: argparse.Namespace) -> None:
    df = _load_csv(args.input)
    df = engineer_features(df)
    if MODEL_PATH.exists():
        model = FraudModel.load(MODEL_PATH)
    else:  # ad-hoc: fit on the same file when no trained model exists
        model = train_model(df)
    scored = score_dataframe(df, model, threshold=args.threshold)
    cols = ["txn_id", "customer_id", "timestamp", "amount", "merchant",
            "category", "country", "risk_score", "flagged", "top_reasons"]
    if "is_fraud" in scored.columns:
        cols.append("is_fraud")
    scored[cols].to_csv(args.out, index=False)
    n_flag = int(scored["flagged"].sum())
    print(f"Scored {len(scored)} transactions, {n_flag} flagged "
          f"(threshold {args.threshold:.0f}) -> {args.out}")
    print("\nTop 5 riskiest:")
    top = scored.sort_values("risk_score", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"  {r['txn_id']}  risk={r['risk_score']:.0f}  "
              f"${r['amount']:.2f} @ {r['merchant']}  -> {r['top_reasons']}")


def _ensure_cases(db_path: str, csv_path: str | None, threshold: float):
    conn = review_store.connect(db_path)
    if csv_path:
        df = _load_csv(csv_path)
        df = engineer_features(df)
        model = FraudModel.load(MODEL_PATH) if MODEL_PATH.exists() else train_model(df)
        scored = score_dataframe(df, model, threshold=threshold)
    else:
        # score demo data bundled with the repo
        demo = ROOT / "data" / "demo_transactions.csv"
        df = _load_csv(str(demo))
        df = engineer_features(df)
        model = FraudModel.load(MODEL_PATH) if MODEL_PATH.exists() else train_model(df)
        scored = score_dataframe(df, model, threshold=threshold)
    new = review_store.upsert_cases(conn, scored)
    if new:
        print(f"Added {new} new cases to the review queue.")
    return conn


def cmd_review(args: argparse.Namespace) -> None:
    if args.action == "list":
        conn = _ensure_cases(args.db, args.input, args.threshold)
        cases = review_store.list_cases(conn, status=args.status, limit=args.limit)
        if not cases:
            print("No cases found.")
            return
        print(f"{'TXN':<10} {'RISK':>5} {'AMOUNT':>10}  {'MERCHANT':<22} {'STATUS':<6} TOP REASON")
        print("-" * 110)
        for c in cases:
            reason = (c["reasons"] or "").split(";")[0][:55]
            print(f"{c['txn_id']:<10} {c['risk_score']:>5.0f} "
                  f"${c['amount']:>9.2f}  {c['merchant']:<22} {c['status']:<6} {reason}")
    elif args.action == "decide":
        conn = review_store.connect(args.db)
        ok = review_store.decide(conn, args.txn_id, args.verdict, reviewer=args.reviewer)
        print(f"{'Updated' if ok else 'Case not found:'} {args.txn_id}")
    elif args.action == "stats":
        conn = review_store.connect(args.db)
        for status, count in review_store.stats(conn).items():
            print(f"{status:<6} {count}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    df = _load_csv(args.input)
    run_evaluate(df, out_path=args.report)


def cmd_demo(args: argparse.Namespace) -> None:
    print("== FraudFence demo: generate -> train -> score -> evaluate -> review ==\n")
    txns = generate(seed=42)
    demo_csv = ROOT / "data" / "demo_transactions.csv"
    save_csv(txns, demo_csv)
    print(f"1. Generated {len(txns)} labeled transactions -> {demo_csv.name}")

    df = pd.DataFrame(txns)
    df = engineer_features(df)
    cutoff = df["timestamp"].quantile(0.7)
    model = train_model(df[df["timestamp"] <= cutoff])
    model.save(MODEL_PATH)
    print("2. Trained IsolationForest on first 70% of timeline (model saved)")

    scored = score_dataframe(df, model)
    scored_csv = ROOT / "data" / "demo_scored.csv"
    scored[["txn_id", "customer_id", "timestamp", "amount", "merchant",
            "category", "risk_score", "flagged", "top_reasons",
            "is_fraud"]].to_csv(scored_csv, index=False)
    print(f"3. Scored all transactions -> {scored_csv.name}")

    res = run_evaluate(pd.DataFrame(txns),
                       out_path=ROOT / "data" / "evaluation_report.txt")
    print("\n4. Evaluation complete (see above + evaluation_report.txt)")

    conn = review_store.connect(DEFAULT_DB)
    new = review_store.upsert_cases(conn, scored)
    print(f"5. Review queue: {new} new cases in {DEFAULT_DB.name} "
          f"(`python -m fraudfence review list` to inspect)")
    print(f"\nDemo done. F1={res['f1']:.3f} on held-out data.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraudfence",
                                description="FraudFence: transaction fraud detection")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate synthetic labeled transactions")
    g.add_argument("--out", default="data/transactions.csv")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--customers", type=int, default=120)
    g.add_argument("--days", type=int, default=30)
    g.add_argument("--fraud-rate", type=float, default=0.02)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="Risk-score a transaction CSV")
    s.add_argument("--in", dest="input", default="data/transactions.csv")
    s.add_argument("--out", default="data/scored.csv")
    s.add_argument("--threshold", type=float, default=FLAG_THRESHOLD)
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("review", help="Case-review queue")
    r.add_argument("action", choices=["list", "decide", "stats"])
    r.add_argument("txn_id", nargs="?", help="transaction id (for decide)")
    r.add_argument("verdict", nargs="?", choices=["fraud", "legit"])
    r.add_argument("--db", default=str(DEFAULT_DB))
    r.add_argument("--input", default=None, help="CSV to (re)score into the queue")
    r.add_argument("--status", default=None, choices=["open", "fraud", "legit"])
    r.add_argument("--limit", type=int, default=25)
    r.add_argument("--reviewer", default="")
    r.add_argument("--threshold", type=float, default=FLAG_THRESHOLD)
    r.set_defaults(func=cmd_review)

    e = sub.add_parser("evaluate", help="Evaluate detector vs labels")
    e.add_argument("--in", dest="input", default="data/demo_transactions.csv")
    e.add_argument("--report", default="data/evaluation_report.txt")
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("demo", help="End-to-end demo on bundled synthetic data")
    d.set_defaults(func=cmd_demo)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
