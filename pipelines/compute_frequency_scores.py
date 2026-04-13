"""
Time-weighted frequency score computation.

Reads the geopolitical_events database, applies exponential decay weighting
(λ per category), normalizes to 1-5 scale, and produces an updated
priority matrix with data-driven frequency scores alongside the expert-assigned
severity scores from Phase 1.

Usage:
    python pipelines/compute_frequency_scores.py
    python pipelines/compute_frequency_scores.py --reference-date 2025-01-01
    python pipelines/compute_frequency_scores.py --output data/processed/live_matrix.json

This is the Week 3-4 validation step — compare computed scores against
the expert scores in config/priority_matrix.json to validate taxonomy mappings.
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import get_db_connection, get_logger, load_lambda_rates

logger = get_logger("compute_frequency_scores")

ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"

CATEGORY_ORDER = [
    "trade_policy_actions",
    "sanctions_financial_restrictions",
    "armed_conflict_instability",
    "regulatory_sovereignty_shifts",
    "technology_controls",
    "resource_energy_disruptions",
    "political_transitions_volatility",
    "institutional_alliance_realignment",
]

CHANNEL_ORDER = [
    "procurement_supply_chain",
    "revenue_market_access",
    "capital_allocation_investment",
    "regulatory_compliance_cost",
    "logistics_operations",
    "innovation_ip",
    "workforce_talent",
    "reputation_stakeholder",
    "financial_treasury",
    "cybersecurity_it",
]


# ─── Time-weighted frequency ──────────────────────────────────────────────────

def compute_time_weighted_frequency(
    event_dates: list[date],
    category: str,
    lambda_rates: dict[str, float],
    reference_date: date,
) -> float:
    """
    Compute decay-weighted frequency score for a set of events.
    Formula: sum(e^(-λ * t)) where t = years since event.

    Args:
        event_dates: list of event dates
        category: event category string (for lambda lookup)
        lambda_rates: dict of category → λ
        reference_date: date to compute weights from

    Returns:
        raw_weighted_score: float (sum of decay weights)
    """
    lam = lambda_rates.get(category, 0.5)
    total = 0.0
    for ev_date in event_dates:
        t_years = (reference_date - ev_date).days / 365.25
        if t_years < 0:
            t_years = 0  # future events (shouldn't happen) get full weight
        total += np.exp(-lam * t_years)
    return total


def normalize_to_1_5(raw_scores: list[float]) -> list[int]:
    """
    Normalize raw weighted scores to 1-5 scale using percentile bins.
    P80+ = 5, P60-80 = 4, P40-60 = 3, P20-40 = 2, P0-20 = 1

    If all scores are 0, returns all 1s.
    """
    arr = np.array(raw_scores, dtype=float)
    if arr.max() == 0:
        return [1] * len(raw_scores)

    p20, p40, p60, p80 = np.percentile(arr, [20, 40, 60, 80])
    result = []
    for score in arr:
        if score >= p80:
            result.append(5)
        elif score >= p60:
            result.append(4)
        elif score >= p40:
            result.append(3)
        elif score >= p20:
            result.append(2)
        else:
            result.append(1)
    return result


# ─── Query events ─────────────────────────────────────────────────────────────

def query_events_by_category(conn, category: str) -> list[date]:
    """Return all event dates for a given category from the database."""
    rows = conn.execute(
        "SELECT event_date FROM geopolitical_events WHERE event_category = ?",
        (category,),
    ).fetchall()
    dates = []
    for row in rows:
        try:
            dates.append(date.fromisoformat(row[0]))
        except (ValueError, TypeError):
            pass
    return dates


def count_events_by_category(conn) -> dict[str, int]:
    """Return event count per category for diagnostics."""
    rows = conn.execute(
        "SELECT event_category, COUNT(*) as cnt FROM geopolitical_events GROUP BY event_category"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_live_matrix(reference_date: date) -> dict:
    """
    Compute the live priority matrix using time-weighted frequency scores
    from actual database events + expert severity scores from Phase 1.

    Returns dict with computed frequency scores, expert severity scores,
    priority scores, and comparison stats.
    """
    conn = get_db_connection()
    lambda_rates = load_lambda_rates()

    # Load expert scores from Phase 1
    with open(CONFIG_DIR / "priority_matrix.json") as f:
        expert_matrix = json.load(f)

    expert_frequency = expert_matrix["frequency_scores"]
    expert_severity = expert_matrix["severity_scores"]

    event_counts = count_events_by_category(conn)
    logger.info(f"Event counts in DB: {event_counts}")
    total_events = sum(event_counts.values())
    logger.info(f"Total events in DB: {total_events:,}")

    # Compute raw time-weighted scores per category
    raw_scores_per_category = {}
    for category in CATEGORY_ORDER:
        event_dates = query_events_by_category(conn, category)
        raw = compute_time_weighted_frequency(event_dates, category, lambda_rates, reference_date)
        raw_scores_per_category[category] = raw
        logger.info(f"  {category}: {len(event_dates)} events → raw score {raw:.2f}")

    conn.close()

    # Normalize all 8 category scores to 1-5
    raw_values = [raw_scores_per_category[cat] for cat in CATEGORY_ORDER]
    normalized = normalize_to_1_5(raw_values)
    computed_frequency_by_category = dict(zip(CATEGORY_ORDER, normalized))

    # Build computed frequency scores matrix (same score for all channels within a category,
    # since the database doesn't yet have channel-level event tagging).
    # NOTE: This is a simplification. Full channel-level scoring requires the
    # corporate outcome dataset from Weeks 5-7.
    computed_frequency = {}
    for category in CATEGORY_ORDER:
        score = computed_frequency_by_category[category]
        computed_frequency[category] = [score] * 10

    # Compute priority scores using computed frequency + expert severity
    computed_priority = {}
    for category in CATEGORY_ORDER:
        freq_row = computed_frequency[category]
        sev_row = expert_severity[category]
        computed_priority[category] = [f * s for f, s in zip(freq_row, sev_row)]

    # Comparison: how much do computed vs expert frequency scores diverge?
    comparison = {}
    for category in CATEGORY_ORDER:
        expert_freq = expert_frequency[category]
        expert_mean = sum(expert_freq) / len(expert_freq)
        computed_mean = computed_frequency_by_category[category]
        divergence_pct = abs(computed_mean - expert_mean) / max(expert_mean, 0.1) * 100
        comparison[category] = {
            "expert_frequency_mean": round(expert_mean, 2),
            "computed_frequency_score": computed_mean,
            "divergence_pct": round(divergence_pct, 1),
            "flag_for_review": divergence_pct > 20,
            "event_count_in_db": event_counts.get(category, 0),
        }

    categories_flagged = [cat for cat, v in comparison.items() if v["flag_for_review"]]
    if categories_flagged:
        logger.warning(
            f"Categories with >20% divergence from expert scores (review needed): "
            f"{categories_flagged}"
        )

    return {
        "_metadata": {
            "computed_at": datetime.now().isoformat(),
            "reference_date": reference_date.isoformat(),
            "total_events_in_db": total_events,
            "note": (
                "Frequency scores computed from database events. Severity scores are "
                "expert-assigned from Phase 1. Channel-level frequency differentiation "
                "requires corporate outcome data (Weeks 5-7)."
            ),
        },
        "impact_channel_order": CHANNEL_ORDER,
        "computed_frequency_scores": computed_frequency,
        "expert_frequency_scores": expert_frequency,
        "expert_severity_scores": expert_severity,
        "computed_priority_scores": computed_priority,
        "expert_priority_scores": expert_matrix["priority_scores"],
        "validation_comparison": comparison,
        "raw_weighted_scores": {cat: round(raw_scores_per_category[cat], 3) for cat in CATEGORY_ORDER},
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--reference-date", default=None, help="Reference date for decay calc (default: today)")
@click.option("--output", default=None, help="Output JSON path (default: data/processed/live_matrix.json)")
def main(reference_date: str | None, output: str | None) -> None:
    """Compute time-weighted frequency scores from event database and produce live priority matrix."""
    ref_date = date.fromisoformat(reference_date) if reference_date else date.today()
    out_path = Path(output) if output else ROOT_DIR / "data" / "processed" / "live_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing live priority matrix (reference date: {ref_date})")
    result = compute_live_matrix(ref_date)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Written to {out_path}")

    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total events in DB: {result['_metadata']['total_events_in_db']:,}")
    print()
    print(f"{'Category':<40} {'Expert Freq':>12} {'Computed':>10} {'Diverge%':>10} {'Flag':>6}")
    print("-" * 82)
    for cat in CATEGORY_ORDER:
        comp = result["validation_comparison"][cat]
        flag = "⚠ REVIEW" if comp["flag_for_review"] else "OK"
        print(
            f"{cat:<40} {comp['expert_frequency_mean']:>12.1f} "
            f"{comp['computed_frequency_score']:>10d} "
            f"{comp['divergence_pct']:>9.1f}% "
            f"{flag:>10}"
        )

    print()
    flagged = [cat for cat, v in result["validation_comparison"].items() if v["flag_for_review"]]
    if flagged:
        print(f"⚠  {len(flagged)} categories need review — check taxonomy mappings for:")
        for cat in flagged:
            ev_count = result["validation_comparison"][cat]["event_count_in_db"]
            print(f"   - {cat} ({ev_count} events in DB)")
    else:
        print("✓  All categories within 20% of expert scores.")


if __name__ == "__main__":
    main()
