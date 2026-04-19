"""
Market-validate GRI category weights using actual stock market reactions.

Instead of using our model-derived weights (which are circular — derived from
labels we created), compute weights from actual S&P 500 volatility following
events of each category.

The idea: if armed_conflict events consistently cause larger stock market
moves than political_transitions events, armed_conflict should get a higher
weight — and that weight comes from market data, not our assumptions.

Method:
1. For each event in event_studies, get the category and avg |CAR|
2. Compute mean absolute stock reaction by category
3. Normalize to weights that sum to 8 (one per category)
4. Compare to our model-derived weights

Usage:
    python index/market_validate_weights.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection
from models.exposure_scorer.train import EVENT_TO_CATEGORY


def compute_market_weights():
    conn = get_db_connection()

    # Get all event studies with category
    df = pd.read_sql("""
        SELECT event_id, ticker, car_1_5, car_1_30
        FROM event_studies
        WHERE car_1_5 IS NOT NULL
    """, conn)
    conn.close()

    # Map event_id to category
    df["category"] = df["event_id"].map(EVENT_TO_CATEGORY)
    df = df[df["category"].notna()]

    # Compute mean absolute reaction by category
    # |CAR| is the signal — larger absolute moves = more market impact
    df["abs_car_5"] = df["car_1_5"].abs()
    df["abs_car_30"] = df["car_1_30"].abs()

    category_impact = df.groupby("category").agg(
        mean_abs_car5=("abs_car_5", "mean"),
        mean_abs_car30=("abs_car_30", "mean"),
        median_abs_car5=("abs_car_5", "median"),
        max_abs_car5=("abs_car_5", "max"),
        n_observations=("abs_car_5", "count"),
        n_events=("event_id", "nunique"),
    ).sort_values("mean_abs_car5", ascending=False)

    # Normalize to weights (max = 1.0)
    max_impact = category_impact["mean_abs_car5"].max()
    category_impact["market_weight"] = (category_impact["mean_abs_car5"] / max_impact).round(3)

    # Our model-derived weights
    model_weights = {
        "armed_conflict_instability": 1.0,
        "sanctions_financial_restrictions": 0.9,
        "technology_controls": 0.85,
        "trade_policy_actions": 0.8,
        "resource_energy_disruptions": 0.8,
        "regulatory_sovereignty_shifts": 0.6,
        "political_transitions_volatility": 0.5,
        "institutional_alliance_realignment": 0.4,
    }
    category_impact["model_weight"] = category_impact.index.map(model_weights)
    category_impact["weight_gap"] = (category_impact["market_weight"] - category_impact["model_weight"]).abs()

    # Report
    print("=" * 90)
    print("MARKET-VALIDATED GRI CATEGORY WEIGHTS")
    print("=" * 90)
    print(f"\nMethod: Mean |CAR_1_5| per category across {len(df)} event-study observations")
    print(f"         Normalized so max category = 1.0\n")

    print(f"{'Category':40s} {'Market Wt':>10s} {'Model Wt':>10s} {'Gap':>8s} {'|CAR5| mean':>12s} {'Events':>8s} {'Obs':>8s}")
    print("-" * 90)
    for cat, row in category_impact.iterrows():
        gap_marker = " **" if row["weight_gap"] > 0.2 else ""
        print(f"{cat:40s} {row['market_weight']:>9.3f} {row['model_weight']:>9.1f} "
              f"{row['weight_gap']:>7.3f}{gap_marker} {row['mean_abs_car5']:>11.4f} "
              f"{int(row['n_events']):>7d} {int(row['n_observations']):>7d}")

    # Overall correlation
    corr = category_impact[["market_weight", "model_weight"]].corr().iloc[0, 1]
    print(f"\nCorrelation between market and model weights: {corr:.3f}")

    if corr > 0.7:
        print("  GOOD: Model weights are broadly aligned with market evidence.")
    elif corr > 0.4:
        print("  MODERATE: Some alignment, but notable divergences exist.")
    else:
        print("  POOR: Model weights diverge significantly from market evidence.")

    # Biggest divergences
    print(f"\nBIGGEST DIVERGENCES (|gap| > 0.15):")
    for cat, row in category_impact[category_impact["weight_gap"] > 0.15].iterrows():
        direction = "overweighted" if row["model_weight"] > row["market_weight"] else "underweighted"
        print(f"  {cat}: {direction} by {row['weight_gap']:.2f} "
              f"(model={row['model_weight']:.1f}, market={row['market_weight']:.3f})")

    # Suggested corrected weights
    print(f"\nSUGGESTED CORRECTED WEIGHTS (blend: 50% market + 50% model):")
    for cat, row in category_impact.iterrows():
        blended = (row["market_weight"] + row["model_weight"]) / 2
        print(f"  {cat:40s} {blended:.3f} (was {row['model_weight']:.1f})")

    return category_impact


if __name__ == "__main__":
    compute_market_weights()
