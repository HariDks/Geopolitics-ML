"""
Geopolitical Risk Index (GRI) — A daily composite risk score.

Methodology:
  The GRI is a 0-100 score computed daily from 3 signals:

  1. EVENT VOLUME (40% weight)
     - Count of geopolitical events per day, normalized against a 90-day rolling average
     - Spikes in event volume indicate escalation
     - Each event category is weighted by its average corporate impact severity
       from our trained model (armed_conflict weights more than political_transitions)

  2. EVENT SEVERITY (40% weight)
     - Average Goldstein scale (negative = more conflictual) weighted by event count
     - Goldstein ranges from -10 (war) to +10 (cooperation)
     - Normalized to 0-100 scale where lower Goldstein = higher risk

  3. CATEGORY CONCENTRATION (20% weight)
     - Herfindahl index of event categories
     - High concentration in one category = focused crisis (higher risk)
     - Even spread across categories = normal geopolitical noise (lower risk)
     - e.g., if 90% of events are armed_conflict, something specific is happening

  Sub-indices (one per event category):
     Each of the 8 categories gets its own 0-100 score based on volume + severity
     within that category alone.

  The composite GRI = weighted average of the 3 signals, scaled to 0-100.

Usage:
    python index/compute_index.py                          # compute for all available dates
    python index/compute_index.py --start 2024-01-01       # from specific date
    python index/compute_index.py --output index/gri.csv   # save to file
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("gri")

# Category weights based on average corporate impact severity from Model 2 training
# Higher weight = this category has stronger corporate impact when it occurs
CATEGORY_WEIGHTS = {
    "armed_conflict_instability": 1.0,       # wars, terrorism — highest direct impact
    "sanctions_financial_restrictions": 0.9,  # asset freezes, SWIFT — severe financial impact
    "technology_controls": 0.85,              # chip bans — concentrated but high impact
    "trade_policy_actions": 0.8,              # tariffs — broad but slower onset
    "resource_energy_disruptions": 0.8,       # OPEC, commodity shocks
    "regulatory_sovereignty_shifts": 0.6,     # regulations — structural, slow
    "political_transitions_volatility": 0.5,  # elections, coups — variable impact
    "institutional_alliance_realignment": 0.4, # treaties — long-term, low immediate
}

ROLLING_WINDOW = 90  # days for normalization baseline


def compute_daily_signals(conn, start_date: str = "2022-01-01") -> pd.DataFrame:
    """Compute raw daily signals from event data."""
    query = """
        SELECT event_date, event_category,
               COUNT(*) as event_count,
               AVG(CASE WHEN goldstein_scale IS NOT NULL THEN goldstein_scale END) as avg_goldstein,
               AVG(CASE WHEN severity_estimate IS NOT NULL THEN severity_estimate END) as avg_severity,
               SUM(CASE WHEN severity_estimate >= 4 THEN 1 ELSE 0 END) as high_severity_count
        FROM geopolitical_events
        WHERE event_date >= ?
        AND source IN ('gdelt', 'acled')
        GROUP BY event_date, event_category
        ORDER BY event_date, event_category
    """
    df = pd.read_sql(query, conn, params=(start_date,))
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def compute_gri(daily_signals: pd.DataFrame) -> pd.DataFrame:
    """Compute the Geopolitical Risk Index from daily signals."""

    # Aggregate to daily totals
    daily = daily_signals.groupby("event_date").agg(
        total_events=("event_count", "sum"),
        avg_goldstein=("avg_goldstein", "mean"),
        high_severity=("high_severity_count", "sum"),
    ).reset_index()

    # Weighted event count (by category importance)
    weighted_counts = daily_signals.copy()
    weighted_counts["weight"] = weighted_counts["event_category"].map(CATEGORY_WEIGHTS).fillna(0.5)
    weighted_counts["weighted_events"] = weighted_counts["event_count"] * weighted_counts["weight"]
    daily_weighted = weighted_counts.groupby("event_date")["weighted_events"].sum().reset_index()
    daily = daily.merge(daily_weighted, on="event_date", how="left")

    # Category concentration (Herfindahl index)
    def herfindahl(group):
        shares = group["event_count"] / group["event_count"].sum()
        return (shares ** 2).sum()

    hhi = daily_signals.groupby("event_date").apply(herfindahl, include_groups=False).reset_index()
    hhi.columns = ["event_date", "category_hhi"]
    daily = daily.merge(hhi, on="event_date", how="left")

    daily = daily.sort_values("event_date").reset_index(drop=True)

    # ── Signal 1: Volume Score (0-100) ──
    # Normalize weighted event count against 90-day rolling average
    daily["vol_rolling_mean"] = daily["weighted_events"].rolling(ROLLING_WINDOW, min_periods=30).mean()
    daily["vol_rolling_std"] = daily["weighted_events"].rolling(ROLLING_WINDOW, min_periods=30).std()
    daily["vol_zscore"] = (daily["weighted_events"] - daily["vol_rolling_mean"]) / daily["vol_rolling_std"].clip(lower=1)
    # Z-score to 0-100: z=0 → 50, z=+2 → 85, z=-2 → 15
    daily["volume_score"] = (50 + daily["vol_zscore"] * 17.5).clip(0, 100)

    # ── Signal 2: Severity Score (0-100) ──
    # Goldstein: -10 (war) to +10 (cooperation) → invert and scale
    # More negative Goldstein = higher risk
    daily["severity_score"] = ((daily["avg_goldstein"] * -1 + 10) / 20 * 100).clip(0, 100)

    # ── Signal 3: Concentration Score (0-100) ──
    # HHI: 0.125 (perfectly even across 8 categories) to 1.0 (all one category)
    # Higher HHI = more concentrated = more crisis-like
    daily["concentration_score"] = ((daily["category_hhi"] - 0.125) / (1.0 - 0.125) * 100).clip(0, 100)

    # ── Composite GRI ──
    daily["gri"] = (
        0.40 * daily["volume_score"] +
        0.40 * daily["severity_score"] +
        0.20 * daily["concentration_score"]
    ).round(1)

    # ── Sub-indices per category ──
    for cat in CATEGORY_WEIGHTS.keys():
        cat_df = daily_signals[daily_signals["event_category"] == cat].copy()
        cat_daily = cat_df.groupby("event_date").agg(
            cat_events=("event_count", "sum"),
            cat_goldstein=("avg_goldstein", "mean"),
        ).reset_index()

        cat_daily = cat_daily.sort_values("event_date")
        cat_daily["cat_vol_mean"] = cat_daily["cat_events"].rolling(ROLLING_WINDOW, min_periods=30).mean()
        cat_daily["cat_vol_std"] = cat_daily["cat_events"].rolling(ROLLING_WINDOW, min_periods=30).std()
        cat_daily["cat_vol_z"] = (cat_daily["cat_events"] - cat_daily["cat_vol_mean"]) / cat_daily["cat_vol_std"].clip(lower=1)
        cat_daily["cat_vol_score"] = (50 + cat_daily["cat_vol_z"] * 17.5).clip(0, 100)
        cat_daily["cat_sev_score"] = ((cat_daily["cat_goldstein"] * -1 + 10) / 20 * 100).clip(0, 100)
        cat_daily[f"gri_{cat[:15]}"] = (0.5 * cat_daily["cat_vol_score"] + 0.5 * cat_daily["cat_sev_score"]).round(1)

        daily = daily.merge(
            cat_daily[["event_date", f"gri_{cat[:15]}"]],
            on="event_date", how="left",
        )

    return daily


def find_drivers(daily_signals: pd.DataFrame, target_date: pd.Timestamp) -> list[dict]:
    """Find the top events driving the index on a given date."""
    day_data = daily_signals[daily_signals["event_date"] == target_date].sort_values(
        "event_count", ascending=False
    )
    drivers = []
    for _, row in day_data.iterrows():
        drivers.append({
            "category": row["event_category"],
            "events": int(row["event_count"]),
            "avg_goldstein": round(row["avg_goldstein"], 2) if row["avg_goldstein"] else None,
            "high_severity": int(row["high_severity_count"]),
        })
    return drivers


@click.command()
@click.option("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--output", default=None, help="Save GRI to CSV")
def main(start, output):
    """Compute the Geopolitical Risk Index."""
    conn = get_db_connection()
    logger.info(f"Computing GRI from {start}...")

    daily_signals = compute_daily_signals(conn, start)
    gri = compute_gri(daily_signals)

    conn.close()

    # Drop warmup period (need 90 days for rolling stats)
    gri = gri[gri["vol_rolling_mean"].notna()].copy()

    logger.info(f"Computed GRI for {len(gri)} days")

    # Summary
    latest = gri.iloc[-1]
    print(f"\n{'='*70}")
    print(f"GEOPOLITICAL RISK INDEX (GRI)")
    print(f"{'='*70}")
    print(f"Date range: {gri['event_date'].min().date()} to {gri['event_date'].max().date()}")
    print(f"Latest GRI: {latest['gri']:.1f}/100")
    print(f"  Volume score:        {latest['volume_score']:.1f}")
    print(f"  Severity score:      {latest['severity_score']:.1f}")
    print(f"  Concentration score: {latest['concentration_score']:.1f}")

    # Stats
    print(f"\nHistorical range: {gri['gri'].min():.1f} — {gri['gri'].max():.1f}")
    print(f"Mean: {gri['gri'].mean():.1f}, Std: {gri['gri'].std():.1f}")

    # Recent trend
    print(f"\nLast 7 days:")
    for _, row in gri.tail(7).iterrows():
        bar = "█" * int(row["gri"] / 2)
        print(f"  {row['event_date'].date()} | GRI={row['gri']:5.1f} | {bar}")

    # Peak days
    print(f"\nTop 5 highest-risk days:")
    for _, row in gri.nlargest(5, "gri").iterrows():
        drivers = find_drivers(daily_signals, row["event_date"])
        top_cat = drivers[0]["category"] if drivers else "?"
        print(f"  {row['event_date'].date()} | GRI={row['gri']:5.1f} | driven by {top_cat} ({drivers[0]['events']} events)")

    # Sub-indices latest
    print(f"\nSub-indices (latest):")
    sub_cols = [c for c in gri.columns if c.startswith("gri_")]
    for col in sorted(sub_cols, key=lambda c: latest.get(c, 0) or 0, reverse=True):
        val = latest.get(col)
        if val and not np.isnan(val):
            name = col.replace("gri_", "").replace("_", " ")
            print(f"  {name:30s} {val:5.1f}")

    if output:
        out_cols = ["event_date", "gri", "volume_score", "severity_score",
                    "concentration_score", "total_events", "avg_goldstein"] + sub_cols
        gri[out_cols].to_csv(output, index=False)
        logger.info(f"Saved to {output}")
        print(f"\nSaved {len(gri)} days to {output}")


if __name__ == "__main__":
    main()
