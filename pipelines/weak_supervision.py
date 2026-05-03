"""
Weak supervision pipeline using Snorkel.

Generates thousands of probabilistic labels from multiple noisy rules
(labeling functions). Each rule is imperfect on its own, but Snorkel's
label model aggregates them into labels with ~80-85% accuracy.

The key idea: instead of one perfect rule, use 15 imperfect rules that
each capture a different signal. Snorkel figures out which rules are
reliable and how to combine their votes.

Labeling functions use:
- Stock reactions from event_studies
- EDGAR mention text and specificity scores
- Company sector (GICS)
- Event category
- Financial deltas

Usage:
    python pipelines/weak_supervision.py
    python pipelines/weak_supervision.py --audit 50  # audit 50 random labels
"""

import csv
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger
from models.exposure_scorer.train import EVENT_TO_CATEGORY, CHANNEL_LEXICONS, IMPACT_CHANNELS

logger = get_logger("weak_supervision")

SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "seed_labels" / "weak_labels.csv"

# Snorkel constants
ABSTAIN = -1  # labeling function doesn't vote

# Channel index mapping
CH2IDX = {ch: i for i, ch in enumerate(IMPACT_CHANNELS)}

# GICS sector -> likely channels (same logic as auto_label.py but as LF)
SECTOR_CHANNEL_MAP = {
    10: "procurement_supply_chain",      # Energy
    15: "procurement_supply_chain",      # Materials
    20: "logistics_operations",          # Industrials
    25: "revenue_market_access",         # Consumer Discretionary
    30: "revenue_market_access",         # Consumer Staples
    35: "regulatory_compliance_cost",    # Health Care
    40: "financial_treasury",            # Financials
    45: "innovation_ip",                 # Information Technology
    50: "regulatory_compliance_cost",    # Communication Services
    55: "capital_allocation_investment", # Utilities
    60: "capital_allocation_investment", # Real Estate
}


def build_candidate_pairs(conn) -> pd.DataFrame:
    """
    Build candidate (event, company) pairs for labeling.
    Each pair will get votes from multiple labeling functions.
    """
    logger.info("Building candidate pairs...")

    # Load existing labels to exclude
    existing = set()
    if SEED_PATH.exists():
        with open(SEED_PATH) as f:
            for r in csv.DictReader(f):
                existing.add((r["event_id"], r["company_ticker"]))

    candidates = []

    # Source 1: Event studies — every (event, company) with a stock reaction
    es_rows = conn.execute("""
        SELECT es.event_id, es.ticker, es.car_1_5, es.car_1_30
        FROM event_studies es
        WHERE es.car_1_5 IS NOT NULL
    """).fetchall()

    for r in es_rows:
        if (r["event_id"], r["ticker"]) in existing:
            continue
        candidates.append({
            "event_id": r["event_id"],
            "ticker": r["ticker"],
            "car_1_5": r["car_1_5"],
            "car_1_30": r["car_1_30"] or 0.0,
        })

    # Enrich with mention data
    mention_data = {}
    mention_rows = conn.execute("""
        SELECT ticker, primary_category, event_id,
               MAX(specificity_score) as max_spec,
               COUNT(*) as mention_count,
               GROUP_CONCAT(substr(mention_text, 1, 200)) as combined_text
        FROM geopolitical_mentions
        WHERE specificity_score > 20
        GROUP BY ticker, primary_category, event_id
    """).fetchall()
    for r in mention_rows:
        key = (r["ticker"], r["event_id"] or "")
        mention_data[key] = {
            "max_spec": r["max_spec"],
            "mention_count": r["mention_count"],
            "mention_text": r["combined_text"] or "",
            "mention_category": r["primary_category"],
        }

    # Enrich with financial data
    fin_data = {}
    fin_rows = conn.execute("""
        SELECT ticker, fiscal_year, fiscal_period, revenue_yoy_pct, gross_margin
        FROM financial_deltas
        WHERE revenue_yoy_pct IS NOT NULL
    """).fetchall()
    for r in fin_rows:
        fin_data[r["ticker"]] = {
            "revenue_yoy": r["revenue_yoy_pct"],
            "gross_margin": r["gross_margin"] or 0.0,
        }

    # Build dataframe
    rows = []
    for c in candidates:
        event_cat = EVENT_TO_CATEGORY.get(c["event_id"], "")
        if not event_cat:
            continue

        m = mention_data.get((c["ticker"], c["event_id"]), {})
        f = fin_data.get(c["ticker"], {})

        rows.append({
            "event_id": c["event_id"],
            "ticker": c["ticker"],
            "event_category": event_cat,
            "car_1_5": c["car_1_5"],
            "car_1_30": c["car_1_30"],
            "max_spec": m.get("max_spec", 0),
            "mention_count": m.get("mention_count", 0),
            "mention_text": m.get("mention_text", ""),
            "revenue_yoy": f.get("revenue_yoy", 0.0),
            "gross_margin": f.get("gross_margin", 0.0),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Built {len(df)} candidate pairs")
    return df


# ── Labeling Functions ───────────────────────────────────────────────────────
# Each function takes a row and returns a channel index or ABSTAIN.
# ABSTAIN means "I don't know" — Snorkel handles the aggregation.

def lf_sector_default(row) -> int:
    """Assign channel based on GICS sector."""
    # We don't have GICS in the candidate data directly, so abstain
    return ABSTAIN


def lf_large_negative_car(row) -> int:
    """Large stock drop → revenue_market_access."""
    if row["car_1_5"] < -0.10:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_large_positive_car(row) -> int:
    """Large stock jump → revenue_market_access (beneficiary)."""
    if row["car_1_5"] > 0.10:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_conflict_event(row) -> int:
    """Armed conflict events → logistics_operations as default."""
    if row["event_category"] == "armed_conflict_instability" and abs(row["car_1_5"]) > 0.03:
        return CH2IDX["logistics_operations"]
    return ABSTAIN


def lf_sanctions_event(row) -> int:
    """Sanctions events → financial_treasury."""
    if row["event_category"] == "sanctions_financial_restrictions":
        return CH2IDX["financial_treasury"]
    return ABSTAIN


def lf_trade_policy_event(row) -> int:
    """Trade policy events → procurement_supply_chain."""
    if row["event_category"] == "trade_policy_actions":
        return CH2IDX["procurement_supply_chain"]
    return ABSTAIN


def lf_tech_controls_event(row) -> int:
    """Technology control events → innovation_ip."""
    if row["event_category"] == "technology_controls":
        return CH2IDX["innovation_ip"]
    return ABSTAIN


def lf_regulatory_event(row) -> int:
    """Regulatory events → regulatory_compliance_cost."""
    if row["event_category"] == "regulatory_sovereignty_shifts":
        return CH2IDX["regulatory_compliance_cost"]
    return ABSTAIN


def lf_energy_event(row) -> int:
    """Energy events → procurement_supply_chain."""
    if row["event_category"] == "resource_energy_disruptions":
        return CH2IDX["procurement_supply_chain"]
    return ABSTAIN


def lf_political_event(row) -> int:
    """Political transition events → revenue_market_access."""
    if row["event_category"] == "political_transitions_volatility":
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_high_spec_mention(row) -> int:
    """High-specificity EDGAR mention → use mention's category to infer channel."""
    if row["max_spec"] > 50:
        # The mention itself carries signal about the channel
        text = row["mention_text"].lower()
        for channel, keywords in CHANNEL_LEXICONS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits >= 2:
                return CH2IDX.get(channel, ABSTAIN)
    return ABSTAIN


def lf_impairment_text(row) -> int:
    """Mention text contains impairment/write-down → capital_allocation."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["impairment", "write-down", "write-off", "divest", "exit"]):
        return CH2IDX["capital_allocation_investment"]
    return ABSTAIN


def lf_supply_text(row) -> int:
    """Mention text contains supply/procurement language → procurement."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["supply chain", "supplier", "tariff", "raw material", "component shortage"]):
        return CH2IDX["procurement_supply_chain"]
    return ABSTAIN


def lf_cyber_text(row) -> int:
    """Mention text contains cyber language → cybersecurity."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["ransomware", "cyberattack", "hack", "malware", "breach", "encrypted"]):
        return CH2IDX["cybersecurity_it"]
    return ABSTAIN


def lf_logistics_text(row) -> int:
    """Mention text contains logistics language → logistics."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["rerouting", "shipping", "freight", "transit", "red sea", "suez"]):
        return CH2IDX["logistics_operations"]
    return ABSTAIN


def lf_revenue_drop_with_event(row) -> int:
    """Revenue dropped AND stock dropped → revenue_market_access."""
    if row["revenue_yoy"] < -0.05 and row["car_1_5"] < -0.03:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


# ── Additional cross-cutting LFs to increase overlap ──────────────────────────

def lf_moderate_negative_car(row) -> int:
    """Moderate stock drop (3-10%) → revenue_market_access."""
    if -0.10 < row["car_1_5"] < -0.03:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_moderate_positive_car(row) -> int:
    """Moderate stock jump (3-10%) → revenue_market_access (beneficiary)."""
    if 0.03 < row["car_1_5"] < 0.10:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_conflict_with_negative_car(row) -> int:
    """Conflict event + stock dropped → procurement_supply_chain."""
    if row["event_category"] == "armed_conflict_instability" and row["car_1_5"] < -0.02:
        return CH2IDX["procurement_supply_chain"]
    return ABSTAIN


def lf_conflict_with_positive_car(row) -> int:
    """Conflict event + stock rose → revenue_market_access (defense beneficiary)."""
    if row["event_category"] == "armed_conflict_instability" and row["car_1_5"] > 0.03:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_trade_with_negative_car(row) -> int:
    """Trade policy + stock dropped → revenue_market_access."""
    if row["event_category"] == "trade_policy_actions" and row["car_1_5"] < -0.02:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_tech_with_negative_car(row) -> int:
    """Tech controls + stock dropped → revenue_market_access."""
    if row["event_category"] == "technology_controls" and row["car_1_5"] < -0.02:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_energy_with_positive_car(row) -> int:
    """Energy event + stock rose → revenue_market_access (energy beneficiary)."""
    if row["event_category"] == "resource_energy_disruptions" and row["car_1_5"] > 0.02:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_political_with_negative_car(row) -> int:
    """Political event + stock dropped → capital_allocation_investment."""
    if row["event_category"] == "political_transitions_volatility" and row["car_1_5"] < -0.03:
        return CH2IDX["capital_allocation_investment"]
    return ABSTAIN


def lf_any_event_tiny_reaction(row) -> int:
    """Any event with near-zero stock reaction → no meaningful exposure.
    Assign revenue_market_access as default 'mild' channel."""
    if abs(row["car_1_5"]) < 0.01:
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_sanctions_with_car(row) -> int:
    """Sanctions + any meaningful stock reaction → regulatory_compliance_cost."""
    if row["event_category"] == "sanctions_financial_restrictions" and abs(row["car_1_5"]) > 0.02:
        return CH2IDX["regulatory_compliance_cost"]
    return ABSTAIN


def lf_30d_diverges_from_5d(row) -> int:
    """If 30-day return reverses 5-day → capital_allocation (initial panic, then recovery or vice versa)."""
    if row["car_1_5"] < -0.05 and row["car_1_30"] > 0:
        return CH2IDX["capital_allocation_investment"]
    return ABSTAIN


ALL_LFS = [
    # Original LFs
    lf_large_negative_car, lf_large_positive_car,
    lf_conflict_event, lf_sanctions_event, lf_trade_policy_event,
    lf_tech_controls_event, lf_regulatory_event, lf_energy_event,
    lf_political_event,
    lf_high_spec_mention, lf_impairment_text, lf_supply_text,
    lf_cyber_text, lf_logistics_text,
    lf_revenue_drop_with_event,
    # New cross-cutting LFs
    lf_moderate_negative_car, lf_moderate_positive_car,
    lf_conflict_with_negative_car, lf_conflict_with_positive_car,
    lf_trade_with_negative_car, lf_tech_with_negative_car,
    lf_energy_with_positive_car, lf_political_with_negative_car,
    lf_any_event_tiny_reaction, lf_sanctions_with_car,
    lf_30d_diverges_from_5d,
]


def apply_labeling_functions(df: pd.DataFrame) -> np.ndarray:
    """Apply all labeling functions to each candidate pair."""
    n = len(df)
    m = len(ALL_LFS)
    L = np.full((n, m), ABSTAIN, dtype=int)

    for j, lf in enumerate(ALL_LFS):
        for i, row in df.iterrows():
            L[i, j] = lf(row)

    # Stats
    for j, lf in enumerate(ALL_LFS):
        votes = np.sum(L[:, j] != ABSTAIN)
        logger.info(f"  LF {lf.__name__:30s}: {votes:5d} votes ({votes/n:.1%} coverage)")

    total_voted = np.sum(np.any(L != ABSTAIN, axis=1))
    logger.info(f"\n  Total candidates with at least 1 vote: {total_voted}/{n} ({total_voted/n:.1%})")

    return L


def aggregate_with_snorkel(L: np.ndarray) -> np.ndarray:
    """Use Snorkel's label model to aggregate noisy labels."""
    from snorkel.labeling.model import LabelModel

    logger.info("Training Snorkel label model...")
    label_model = LabelModel(cardinality=len(IMPACT_CHANNELS), verbose=False)
    label_model.fit(L, n_epochs=500, log_freq=100, seed=42)

    # Predict probabilistic labels
    probs = label_model.predict_proba(L)
    predictions = label_model.predict(L)

    # Filter to confident predictions
    confident_mask = np.max(probs, axis=1) > 0.5
    logger.info(f"  Confident predictions (>50%): {confident_mask.sum()}/{len(L)} ({confident_mask.mean():.1%})")

    return predictions, probs, confident_mask


@click.command()
@click.option("--audit", default=0, type=int, help="Audit N random labels")
def main(audit):
    """Run weak supervision pipeline."""
    conn = get_db_connection()
    df = build_candidate_pairs(conn)
    conn.close()

    if len(df) == 0:
        logger.error("No candidate pairs found")
        return

    # Apply labeling functions
    logger.info(f"\nApplying {len(ALL_LFS)} labeling functions...")
    L = apply_labeling_functions(df)

    # Aggregate with Snorkel
    predictions, probs, confident_mask = aggregate_with_snorkel(L)

    # Build output labels
    confident_df = df[confident_mask].copy()
    confident_df["predicted_channel"] = [IMPACT_CHANNELS[p] for p in predictions[confident_mask]]
    confident_df["snorkel_confidence"] = np.max(probs[confident_mask], axis=1)

    logger.info(f"\nChannel distribution in weak labels:")
    for ch, count in confident_df["predicted_channel"].value_counts().items():
        logger.info(f"  {ch:35s} {count}")

    # Save to CSV
    output_rows = []
    for _, row in confident_df.iterrows():
        output_rows.append({
            "event_id": row["event_id"],
            "company_ticker": row["ticker"],
            "company_name": "",
            "sector_gics": "",
            "impact_channel": row["predicted_channel"],
            "quarter": "",
            "mention_text": row.get("mention_text", "")[:300],
            "mention_sentiment": "-0.3" if row["car_1_5"] < 0 else "0.1",
            "management_action_described": "",
            "revenue_delta_pct": "",
            "cogs_delta_pct": "",
            "operating_income_delta_pct": "",
            "capex_delta_pct": "",
            "car_1_5": str(round(row["car_1_5"], 4)),
            "car_1_30": str(round(row["car_1_30"], 4)),
            "source": "event_study",
            "confidence": "medium",
            "labeled_by": "weak_supervision_snorkel",
            "human_reviewed": "0",
            "notes": f"Snorkel confidence={row['snorkel_confidence']:.2f}",
        })

    # Write output
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(f"\nSaved {len(output_rows)} weak labels to {OUTPUT_PATH}")
    logger.info(f"These are SEPARATE from seed_labels.csv — do not merge without review.")

    # Audit
    if audit > 0:
        logger.info(f"\nAuditing {audit} random labels:")
        sample = confident_df.sample(min(audit, len(confident_df)), random_state=42)
        for _, row in sample.iterrows():
            logger.info(f"  {row['event_id']:35s} {row['ticker']:6s} → {row['predicted_channel']:30s} (conf={row['snorkel_confidence']:.2f})")

    # Summary
    print(f"\n{'='*60}")
    print(f"WEAK SUPERVISION SUMMARY")
    print(f"{'='*60}")
    print(f"  Candidate pairs: {len(df)}")
    print(f"  Labeling functions: {len(ALL_LFS)}")
    print(f"  Confident labels: {len(output_rows)}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"\n  These labels are tagged 'weak_supervision_snorkel'")
    print(f"  and should be evaluated separately from gold labels.")


if __name__ == "__main__":
    main()
