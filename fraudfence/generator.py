"""Synthetic transaction stream generator (seeded, fully offline).

Produces realistic-looking card transactions for a set of customers with
per-customer spending profiles, then injects *labeled* fraud of four kinds:

  * odd_hour      - transaction at an unusual hour for the customer
  * amount_spike  - amount far above the customer's normal spend
  * merchant_mismatch - merchant category the customer never uses
  * velocity_burst - many transactions in a few minutes (card testing)

Every row carries ``is_fraud`` (0/1) and ``fraud_type`` so the detector can
be evaluated against ground truth.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CATEGORIES = [
    "groceries", "fuel", "dining", "travel", "shopping",
    "entertainment", "utilities", "health", "online_retail",
]

MERCHANTS = {
    "groceries": ["FreshMart", "GreenGrocer", "DailyFoods"],
    "fuel": ["PetroStop", "GasPlus", "FuelDepot"],
    "dining": ["Cafe Luna", "Bistro 9", "Noodle House"],
    "travel": ["SkyWays Air", "RailLink", "CityStay Hotels"],
    "shopping": ["Urban Outfitters+", "MegaMall", "Corner Store"],
    "entertainment": ["Cineplex+", "GameZone", "StreamFlix"],
    "utilities": ["PowerGrid Co", "CityWater", "NetCom"],
    "health": ["MediCare Pharmacy", "FitLife Gym", "DentalCare"],
    "online_retail": ["ShopWeb", "QuickBuy", "DealDash"],
}

COUNTRIES = ["US", "US", "US", "CA", "UK", "DE"]


@dataclass
class CustomerProfile:
    customer_id: str
    mean_log_amount: float
    std_log_amount: float
    categories: list[str]
    home_country: str
    peak_hours: list[int] = field(default_factory=lambda: [9, 12, 18, 19])


def make_customers(rng: np.random.Generator, n_customers: int) -> list[CustomerProfile]:
    customers = []
    for i in range(n_customers):
        n_cats = int(rng.integers(3, 6))
        cats = list(rng.choice(CATEGORIES, size=n_cats, replace=False))
        customers.append(
            CustomerProfile(
                customer_id=f"C{i:04d}",
                mean_log_amount=float(rng.normal(3.4, 0.6)),  # ~$30 typical
                std_log_amount=float(rng.uniform(0.4, 0.9)),
                categories=cats,
                home_country=str(rng.choice(COUNTRIES)),
                peak_hours=sorted(rng.choice(range(7, 23), size=4, replace=False).tolist()),
            )
        )
    return customers


def _normal_amount(rng: np.random.Generator, profile: CustomerProfile) -> float:
    return float(np.exp(rng.normal(profile.mean_log_amount, profile.std_log_amount)))


def _normal_hour(rng: np.random.Generator, profile: CustomerProfile) -> int:
    peak = int(rng.choice(profile.peak_hours))
    return int(np.clip(rng.normal(peak, 2.5), 0, 23))


def _normal_txn(rng: np.random.Generator, profile: CustomerProfile, txn_id: int,
                ts: datetime, fraud_type: str | None = None) -> dict:
    """Build one transaction; apply a fraud transformation when requested."""
    category = str(rng.choice(profile.categories))
    merchant = str(rng.choice(MERCHANTS[category]))
    amount = _normal_amount(rng, profile)
    hour = _normal_hour(rng, profile)
    country = profile.home_country

    if fraud_type == "odd_hour":
        hour = int(rng.choice([1, 2, 3, 4, 23]))
        amount = _normal_amount(rng, profile) * float(rng.uniform(2.0, 4.0))
    elif fraud_type == "amount_spike":
        amount = float(np.exp(profile.mean_log_amount + 4.5 * profile.std_log_amount))
    elif fraud_type == "merchant_mismatch":
        other = [c for c in CATEGORIES if c not in profile.categories]
        category = str(rng.choice(other))
        merchant = str(rng.choice(MERCHANTS[category]))
        amount = _normal_amount(rng, profile) * float(rng.uniform(3.0, 6.0))
        country = str(rng.choice([c for c in COUNTRIES if c != profile.home_country]))
    elif fraud_type == "velocity_burst":
        amount = float(rng.uniform(1.0, 5.0))  # small card-testing charges
        hour = _normal_hour(rng, profile)

    return {
        "txn_id": f"T{txn_id:07d}",
        "customer_id": profile.customer_id,
        "timestamp": ts.replace(hour=hour, minute=int(rng.integers(0, 60)),
                                second=int(rng.integers(0, 60))).isoformat(),
        "amount": round(amount, 2),
        "merchant": merchant,
        "category": category,
        "country": country,
        "is_fraud": 1 if fraud_type else 0,
        "fraud_type": fraud_type or "",
    }


def generate(n_customers: int = 120, n_days: int = 30, fraud_rate: float = 0.02,
             seed: int = 42) -> list[dict]:
    """Generate a labeled synthetic transaction stream.

    Returns a list of transaction dicts sorted by timestamp.
    """
    rng = np.random.default_rng(seed)
    customers = make_customers(rng, n_customers)
    start = datetime(2026, 1, 5, 0, 0, 0)

    txns: list[dict] = []
    txn_id = 0
    for profile in customers:
        n_txns = int(rng.integers(n_days // 2, n_days * 3))
        day_offsets = np.sort(rng.integers(0, n_days, size=n_txns))
        burst_state = 0
        for day in day_offsets:
            txn_id += 1
            ts = start + timedelta(days=int(day))
            fraud_type: str | None = None
            # velocity bursts: emit a cluster of small frauds together
            if burst_state > 0:
                fraud_type = "velocity_burst"
                burst_state -= 1
                ts = ts + timedelta(minutes=int(rng.integers(0, 8)))
            elif rng.random() < fraud_rate:
                roll = rng.random()
                if roll < 0.30:
                    fraud_type = "odd_hour"
                elif roll < 0.60:
                    fraud_type = "amount_spike"
                elif roll < 0.85:
                    fraud_type = "merchant_mismatch"
                else:
                    fraud_type = "velocity_burst"
                    burst_state = int(rng.integers(3, 7))
            txns.append(_normal_txn(rng, profile, txn_id, ts, fraud_type))

    txns.sort(key=lambda t: t["timestamp"])
    return txns


def save_csv(txns: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(txns[0].keys()))
        writer.writeheader()
        writer.writerows(txns)
    return path
