"""
Weak supervision pipeline v2 — two-stage design.

Stage 1: Filter candidates by stock reaction (did the company actually react?)
Stage 2: Classify channel using Snorkel with three independent signal types:
  - Event category (what type of geopolitical event?)
  - Company sector / GICS (what industry is the company in?)
  - EDGAR text (what language appears in the company's filings?)

Key fix over v1: stock-reaction LFs are no longer channel voters.
They were answering "was the company affected?" but guessing "revenue_market_access"
as the channel, creating systematic conflicts with event-category LFs.
Now stock data filters in Stage 1, and only channel-informed LFs vote in Stage 2.

Usage:
    python pipelines/weak_supervision_v2.py
    python pipelines/weak_supervision_v2.py --audit 50
    python pipelines/weak_supervision_v2.py --car-threshold 0.03
"""

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger
from models.exposure_scorer.train import EVENT_TO_CATEGORY, CHANNEL_LEXICONS, IMPACT_CHANNELS

logger = get_logger("weak_supervision_v2")

SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "seed_labels" / "weak_labels.csv"
GICS_CACHE_PATH = ROOT_DIR / "data" / "mappings" / "ticker_gics.json"

# Snorkel constants
ABSTAIN = -1

# Channel index mapping
CH2IDX = {ch: i for i, ch in enumerate(IMPACT_CHANNELS)}

# GICS sector → likely channel
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


def load_gics_cache() -> dict:
    """Load ticker → GICS sector code mapping."""
    if GICS_CACHE_PATH.exists():
        with open(GICS_CACHE_PATH) as f:
            return json.load(f)
    logger.warning(f"GICS cache not found at {GICS_CACHE_PATH}")
    return {}


def build_candidate_pairs(conn, car_threshold: float) -> pd.DataFrame:
    """
    Stage 1: Build candidate pairs filtered by stock reaction.

    Only includes (event, company) pairs where abs(car_1_5) > threshold.
    This replaces the stock-reaction labeling functions — stock data
    is used as a filter, not a channel voter.
    """
    logger.info(f"Stage 1: Building candidates with |CAR| > {car_threshold:.1%}...")

    # Load existing labels to exclude
    existing = set()
    if SEED_PATH.exists():
        with open(SEED_PATH) as f:
            for r in csv.DictReader(f):
                existing.add((r["event_id"], r["company_ticker"]))

    # Load GICS cache
    gics_cache = load_gics_cache()

    candidates = []

    # Event studies filtered by stock reaction threshold
    es_rows = conn.execute("""
        SELECT es.event_id, es.ticker, es.car_1_5, es.car_1_30
        FROM event_studies es
        WHERE es.car_1_5 IS NOT NULL
          AND ABS(es.car_1_5) > ?
    """, (car_threshold,)).fetchall()

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
        gics = gics_cache.get(c["ticker"], 0)

        rows.append({
            "event_id": c["event_id"],
            "ticker": c["ticker"],
            "event_category": event_cat,
            "car_1_5": c["car_1_5"],
            "car_1_30": c["car_1_30"],
            "gics_sector": gics,
            "max_spec": m.get("max_spec", 0),
            "mention_count": m.get("mention_count", 0),
            "mention_text": m.get("mention_text", ""),
            "revenue_yoy": f.get("revenue_yoy", 0.0),
            "gross_margin": f.get("gross_margin", 0.0),
        })

    df = pd.DataFrame(rows)
    logger.info(f"  Candidates after stock filter: {len(df)} (from {len(es_rows)} total event studies)")
    logger.info(f"  Tickers with GICS data: {(df['gics_sector'] > 0).sum()}/{len(df)}")
    return df


# ── Stage 2: Labeling Functions (channel-informed only) ─────────────────────
# Three signal types: event category, company sector, EDGAR text.
# No stock-reaction LFs — stock data was used in Stage 1.

# Signal type 1: Event category → channel
def lf_conflict_event(row) -> int:
    """Armed conflict events → logistics_operations."""
    if row["event_category"] == "armed_conflict_instability":
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
    """Energy/resource events → procurement_supply_chain."""
    if row["event_category"] == "resource_energy_disruptions":
        return CH2IDX["procurement_supply_chain"]
    return ABSTAIN


def lf_political_event(row) -> int:
    """Political transition events → revenue_market_access."""
    if row["event_category"] == "political_transitions_volatility":
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


# Signal type 2: Company sector (GICS) → channel
def lf_sector(row) -> int:
    """Assign channel based on GICS sector."""
    gics = row.get("gics_sector", 0)
    if gics in SECTOR_CHANNEL_MAP:
        channel = SECTOR_CHANNEL_MAP[gics]
        return CH2IDX.get(channel, ABSTAIN)
    return ABSTAIN


# Signal type 3: EDGAR text → channel
def lf_high_spec_mention(row) -> int:
    """High-specificity EDGAR mention → use lexicon to infer channel."""
    if row["max_spec"] > 50:
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


def lf_revenue_text(row) -> int:
    """Mention text contains revenue/market language → revenue_market_access."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["revenue decline", "lost sales", "boycott", "market access", "consumer sentiment"]):
        return CH2IDX["revenue_market_access"]
    return ABSTAIN


def lf_regulatory_text(row) -> int:
    """Mention text contains regulatory language → regulatory_compliance."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["compliance", "regulation", "data localization", "GDPR", "CFIUS", "local content"]):
        return CH2IDX["regulatory_compliance_cost"]
    return ABSTAIN


def lf_financial_text(row) -> int:
    """Mention text contains financial/sanctions language → financial_treasury."""
    text = row["mention_text"].lower()
    if any(kw in text for kw in ["sanctions", "OFAC", "asset freeze", "SWIFT", "blocked entity", "SDN"]):
        return CH2IDX["financial_treasury"]
    return ABSTAIN


ALL_LFS = [
    # Event category (7 LFs)
    lf_conflict_event, lf_sanctions_event, lf_trade_policy_event,
    lf_tech_controls_event, lf_regulatory_event, lf_energy_event,
    lf_political_event,
    # Sector (1 LF)
    lf_sector,
    # Text (8 LFs)
    lf_high_spec_mention, lf_impairment_text, lf_supply_text,
    lf_cyber_text, lf_logistics_text, lf_revenue_text,
    lf_regulatory_text, lf_financial_text,
]

# LF group indices for audit
EVENT_LF_INDICES = set(range(0, 7))
SECTOR_LF_INDICES = {7}
TEXT_LF_INDICES = set(range(8, 16))


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


def aggregate_with_snorkel(L: np.ndarray) -> tuple:
    """Use Snorkel's label model to aggregate noisy labels."""
    from snorkel.labeling.model import LabelModel

    logger.info("Training Snorkel label model...")
    label_model = LabelModel(cardinality=len(IMPACT_CHANNELS), verbose=False)
    label_model.fit(L, n_epochs=500, log_freq=100, seed=42)

    probs = label_model.predict_proba(L)
    predictions = label_model.predict(L)

    confident_mask = np.max(probs, axis=1) > 0.5
    logger.info(f"  Confident predictions (>50%): {confident_mask.sum()}/{len(L)} ({confident_mask.mean():.1%})")

    return predictions, probs, confident_mask


def audit_label_sources(L: np.ndarray, confident_mask: np.ndarray) -> dict:
    """Audit what signal types drove each confident label."""
    categories = {
        "event_only": 0,
        "sector_only": 0,
        "text_only": 0,
        "event_plus_sector_agree": 0,
        "event_plus_sector_disagree": 0,
        "event_plus_text": 0,
        "sector_plus_text": 0,
        "all_three": 0,
        "other": 0,
    }

    for i in np.where(confident_mask)[0]:
        voting_lfs = set(j for j in range(len(ALL_LFS)) if L[i, j] != ABSTAIN)

        has_event = bool(voting_lfs & EVENT_LF_INDICES)
        has_sector = bool(voting_lfs & SECTOR_LF_INDICES)
        has_text = bool(voting_lfs & TEXT_LF_INDICES)

        # Get actual votes to check agreement
        event_votes = set(L[i, j] for j in voting_lfs & EVENT_LF_INDICES)
        sector_votes = set(L[i, j] for j in voting_lfs & SECTOR_LF_INDICES)

        if has_event and has_sector and has_text:
            categories["all_three"] += 1
        elif has_event and has_text:
            categories["event_plus_text"] += 1
        elif has_sector and has_text:
            categories["sector_plus_text"] += 1
        elif has_event and has_sector:
            # Check if they agree
            if event_votes & sector_votes:
                categories["event_plus_sector_agree"] += 1
            else:
                categories["event_plus_sector_disagree"] += 1
        elif has_event:
            categories["event_only"] += 1
        elif has_sector:
            categories["sector_only"] += 1
        elif has_text:
            categories["text_only"] += 1
        else:
            categories["other"] += 1

    return categories


@click.command()
@click.option("--audit", default=0, type=int, help="Audit N random labels")
@click.option("--car-threshold", default=0.03, type=float, help="Min |CAR| for Stage 1 filter")
def main(audit, car_threshold):
    """Run two-stage weak supervision pipeline."""
    conn = get_db_connection()
    df = build_candidate_pairs(conn, car_threshold)
    conn.close()

    if len(df) == 0:
        logger.error("No candidate pairs found")
        return

    # Stage 2: Apply channel-only labeling functions
    logger.info(f"\nStage 2: Applying {len(ALL_LFS)} labeling functions (event + sector + text)...")
    L = apply_labeling_functions(df)

    # Aggregate with Snorkel
    predictions, probs, confident_mask = aggregate_with_snorkel(L)

    # Audit signal sources
    logger.info("\n--- SIGNAL SOURCE AUDIT ---")
    sources = audit_label_sources(L, confident_mask)
    total = confident_mask.sum()
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info(f"  {src:35s} {count:5d}  ({count/max(total,1):.1%})")

    # Build output
    confident_df = df[confident_mask].copy()
    confident_df["predicted_channel"] = [IMPACT_CHANNELS[p] for p in predictions[confident_mask]]
    confident_df["snorkel_confidence"] = np.max(probs[confident_mask], axis=1)

    logger.info(f"\nChannel distribution in weak labels:")
    for ch, count in confident_df["predicted_channel"].value_counts().items():
        logger.info(f"  {ch:35s} {count}")

    # Save to CSV (backward-compatible format)
    output_rows = []
    for _, row in confident_df.iterrows():
        output_rows.append({
            "event_id": row["event_id"],
            "company_ticker": row["ticker"],
            "company_name": "",
            "sector_gics": str(row.get("gics_sector", "")),
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
            "labeled_by": "weak_supervision_v2_snorkel",
            "human_reviewed": "0",
            "notes": f"Snorkel confidence={row['snorkel_confidence']:.2f}",
        })

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Save probability distributions for soft-label training
    # Each row: event_id, ticker, snorkel_confidence, prob_channel_0, ..., prob_channel_9
    PROBS_PATH = OUTPUT_PATH.parent / "weak_labels_probs.csv"
    prob_rows = []
    confident_indices = np.where(confident_mask)[0]
    for idx in confident_indices:
        row = df.iloc[idx]
        p = probs[idx]
        prob_row = {
            "event_id": row["event_id"],
            "company_ticker": row["ticker"],
            "sector_gics": str(row.get("gics_sector", "")),
            "event_category": row["event_category"],
            "car_1_5": str(round(row["car_1_5"], 4)),
            "car_1_30": str(round(row["car_1_30"], 4)),
            "mention_text": row.get("mention_text", "")[:300],
            "snorkel_confidence": str(round(float(np.max(p)), 4)),
            "predicted_channel": IMPACT_CHANNELS[predictions[idx]],
        }
        # Add probability for each channel
        for ch_idx, ch_name in enumerate(IMPACT_CHANNELS):
            prob_row[f"prob_{ch_name}"] = str(round(float(p[ch_idx]), 4))
        prob_rows.append(prob_row)

    prob_fieldnames = list(prob_rows[0].keys()) if prob_rows else []
    with open(PROBS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prob_fieldnames)
        writer.writeheader()
        writer.writerows(prob_rows)

    logger.info(f"Saved {len(prob_rows)} probability distributions to {PROBS_PATH}")

    logger.info(f"\nSaved {len(output_rows)} weak labels to {OUTPUT_PATH}")

    # Audit sample
    if audit > 0:
        logger.info(f"\nAuditing {audit} random labels:")
        sample = confident_df.sample(min(audit, len(confident_df)), random_state=42)
        for _, row in sample.iterrows():
            logger.info(f"  {row['event_id']:45s} {row['ticker']:6s} → {row['predicted_channel']:30s} (conf={row['snorkel_confidence']:.2f})")

    # Summary
    print(f"\n{'='*60}")
    print(f"WEAK SUPERVISION v2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Stage 1 (stock filter): |CAR| > {car_threshold:.1%}")
    print(f"  Candidates after filter: {len(df)}")
    print(f"  Labeling functions: {len(ALL_LFS)} (event:{len(EVENT_LF_INDICES)} + sector:{len(SECTOR_LF_INDICES)} + text:{len(TEXT_LF_INDICES)})")
    print(f"  Confident labels: {len(output_rows)}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
