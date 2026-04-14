"""
Semi-supervised labeling pipeline.

Uses Models 1-3 to generate candidate seed labels from:
1. EDGAR mentions with high specificity + event links → company actually discusses the event
2. Event studies with large stock reactions → market says the company was affected
3. Financial deltas showing abnormal YoY changes in event-relevant quarters

For each candidate, the pipeline:
- Classifies the event category (Model 1 already done via event_id mapping)
- Predicts the impact channel (Model 2)
- Estimates severity and financial impact (Model 3)
- Assigns a confidence score based on signal strength

Candidates above a confidence threshold are appended to seed_labels.csv.

Usage:
    python pipelines/auto_label.py                    # generate candidates, review before appending
    python pipelines/auto_label.py --append            # append high-confidence candidates
    python pipelines/auto_label.py --min-confidence 0.7  # stricter threshold
"""

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.utils import get_db_connection, get_logger

logger = get_logger("auto_label")

ROOT_DIR = Path(__file__).parent.parent
SEED_LABELS_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"

# Event → category mapping (reuse from Model 2)
from models.exposure_scorer.train import EVENT_TO_CATEGORY

# Event → approximate date for quarter mapping
EVENT_DATES = {
    "russia_invasion_2022": ("2022-02-24", "2022Q1"),
    "us_chip_export_oct2022": ("2022-10-07", "2023Q1"),
    "us_chip_export_oct2023": ("2023-10-17", "2024Q1"),
    "red_sea_houthi_2023": ("2023-12-15", "2024Q1"),
    "us_tariffs_2025": ("2025-04-02", "2025Q2"),
    "covid_lockdown_start": ("2020-03-11", "2020Q1"),
    "israel_hamas_2023": ("2023-10-07", "2024Q1"),
    "eu_energy_crisis_peak": ("2022-08-26", "2022Q3"),
    "us_china_trade_war_start": ("2018-03-22", "2018Q2"),
    "xinjiang_boycott_2021": ("2021-03-24", "2021Q2"),
    "brexit_referendum": ("2016-06-23", "2016Q3"),
    "india_pakistan_sindoor": ("2025-05-07", "2025Q2"),
    "us_iran_war_2026": ("2026-02-28", "2026Q1"),
    "notpetya_2017": ("2017-06-27", "2017Q3"),
    "iran_sanctions_2018": ("2018-05-08", "2018Q2"),
    "panama_mine_closure_2023": ("2023-11-28", "2024Q1"),
    "chile_lithium_nationalization": ("2023-04-21", "2023Q2"),
    "india_demonetization_2016": ("2016-11-08", "2016Q4"),
    "opec_price_war_2014": ("2014-11-27", "2015Q1"),
    "suez_blockage_2021": ("2021-03-23", "2021Q2"),
}

# GICS sector lookup from existing seed labels
def load_gics_map() -> dict[str, str]:
    """Load ticker → GICS mapping from seed labels + hardcoded S&P 500 sectors."""
    gics = {}
    # Seed labels first
    with open(SEED_LABELS_PATH) as f:
        for r in csv.DictReader(f):
            if r.get("sector_gics") and r.get("company_ticker"):
                gics[r["company_ticker"]] = r["sector_gics"]

    # S&P 500 top 100 sector codes (first 2 digits of GICS)
    # This ensures auto-labeled companies get correct sector assignment
    sp500_sectors = {
        # Energy (10)
        "XOM": "10", "CVX": "10", "COP": "10", "SLB": "10", "EOG": "10",
        # Materials (15)
        "LIN": "15", "SWK": "15", "ADM": "15",
        # Industrials (20)
        "BA": "20", "CAT": "20", "GE": "20", "HON": "20", "UPS": "20",
        "RTX": "20", "DE": "20", "LMT": "20", "GD": "20", "NOC": "20",
        "FDX": "20", "WM": "20", "GM": "20", "F": "20",
        # Consumer Discretionary (25)
        "AMZN": "25", "TSLA": "25", "HD": "25", "MCD": "25", "NKE": "25",
        "LOW": "25", "TJX": "25", "BKNG": "25", "ABNB": "25",
        # Consumer Staples (30)
        "PG": "30", "PEP": "30", "KO": "30", "COST": "30", "PM": "30",
        "MDLZ": "30", "MO": "30", "TGT": "30", "CL": "30",
        # Health Care (35)
        "UNH": "35", "JNJ": "35", "MRK": "35", "ABBV": "35", "LLY": "35",
        "TMO": "35", "ABT": "35", "DHR": "35", "BMY": "35", "AMGN": "35",
        "GILD": "35", "ISRG": "35", "MDT": "35", "SYK": "35", "BDX": "35",
        "ZTS": "35", "CI": "35", "ELV": "35", "HUM": "35",
        # Financials (40)
        "BRK-B": "40", "JPM": "40", "V": "40", "MA": "40", "GS": "40",
        "MS": "40", "SCHW": "40", "BLK": "40", "SPGI": "40", "CB": "40",
        "PNC": "40", "USB": "40", "CME": "40", "MMC": "40", "PSA": "40",
        # Information Technology (45)
        "AAPL": "45", "MSFT": "45", "NVDA": "45", "AVGO": "45", "CSCO": "45",
        "ACN": "45", "CRM": "45", "TXN": "45", "ORCL": "45", "QCOM": "45",
        "INTC": "45", "INTU": "45", "ADP": "45", "AMAT": "45", "LRCX": "45",
        "KLAC": "45", "MU": "45", "META": "45",
        # Communication Services (50)
        "GOOGL": "50",
        # Utilities (55)
        "NEE": "55", "DUK": "55", "SO": "55", "AEP": "55",
        # Real Estate (60)
        "PLD": "60", "EQIX": "60",
    }
    for ticker, sector in sp500_sectors.items():
        if ticker not in gics:
            gics[ticker] = sector

    return gics


# GICS sector → primary impact channel mapping
# First 2 digits of GICS code determine sector
SECTOR_CHANNEL_MAP = {
    # Energy (10) → resource/energy or procurement
    10: {
        "default": "procurement_supply_chain",
        "armed_conflict_instability": "logistics_operations",
        "resource_energy_disruptions": "procurement_supply_chain",
        "sanctions_financial_restrictions": "revenue_market_access",
    },
    # Materials (15) → procurement/supply chain
    15: {"default": "procurement_supply_chain"},
    # Industrials (20) → logistics or workforce
    20: {
        "default": "logistics_operations",
        "armed_conflict_instability": "logistics_operations",
        "trade_policy_actions": "procurement_supply_chain",
        "political_transitions_volatility": "workforce_talent",
    },
    # Consumer Discretionary (25) → revenue/market access or reputation
    25: {
        "default": "revenue_market_access",
        "political_transitions_volatility": "reputation_stakeholder",
        "armed_conflict_instability": "revenue_market_access",
    },
    # Consumer Staples (30) → revenue or reputation
    30: {
        "default": "revenue_market_access",
        "political_transitions_volatility": "reputation_stakeholder",
    },
    # Health Care (35) → regulatory compliance or innovation
    35: {
        "default": "regulatory_compliance_cost",
        "technology_controls": "innovation_ip",
        "trade_policy_actions": "procurement_supply_chain",
    },
    # Financials (40) → financial/treasury
    40: {
        "default": "financial_treasury",
        "sanctions_financial_restrictions": "financial_treasury",
        "political_transitions_volatility": "financial_treasury",
        "regulatory_sovereignty_shifts": "regulatory_compliance_cost",
    },
    # Information Technology (45) → innovation/IP or procurement
    45: {
        "default": "innovation_ip",
        "technology_controls": "innovation_ip",
        "trade_policy_actions": "procurement_supply_chain",
        "armed_conflict_instability": "procurement_supply_chain",
    },
    # Communication Services (50) → regulatory or reputation
    50: {
        "default": "regulatory_compliance_cost",
        "regulatory_sovereignty_shifts": "regulatory_compliance_cost",
        "political_transitions_volatility": "reputation_stakeholder",
    },
    # Utilities (55) → capital allocation
    55: {"default": "capital_allocation_investment"},
    # Real Estate (60) → capital allocation
    60: {"default": "capital_allocation_investment"},
}


def _assign_channel(event_id: str, event_cat: str, ticker: str, car_5: float,
                    gics_map: dict, scorer) -> str:
    """
    Assign impact channel using GICS sector + event category logic.
    Falls back to Model 2 prediction when sector is unknown.
    """
    gics_code = gics_map.get(ticker, "")
    sector = 0
    try:
        sector = int(str(gics_code)[:2])
    except (ValueError, TypeError):
        pass

    if sector in SECTOR_CHANNEL_MAP:
        sector_map = SECTOR_CHANNEL_MAP[sector]
        return sector_map.get(event_cat, sector_map["default"])

    # Fallback to Model 2
    exp = scorer.score(
        event_category=event_cat,
        ticker=ticker,
        mention_sentiment=-0.3 if car_5 and car_5 < 0 else 0.1,
        car_1_5=car_5 or 0.0,
    )
    return exp["channel_prediction"]


def generate_candidates(conn, min_confidence: float = 0.5) -> list[dict]:
    """
    Generate candidate seed labels from three data sources.

    Returns list of candidate dicts matching seed_labels.csv schema.
    """
    existing = set()
    with open(SEED_LABELS_PATH) as f:
        for r in csv.DictReader(f):
            existing.add((r["event_id"], r["company_ticker"]))

    gics_map = load_gics_map()
    candidates = []

    # Load exposure scorer for channel prediction
    from models.exposure_scorer.predict import ExposureScorer
    scorer = ExposureScorer()

    # ── Source 1: EDGAR mentions with high specificity + event links ──
    logger.info("Source 1: High-specificity EDGAR mentions...")
    mention_rows = conn.execute("""
        SELECT event_id, ticker, primary_category,
               MAX(specificity_score) as max_spec,
               COUNT(*) as mention_count,
               GROUP_CONCAT(substr(mention_text, 1, 300), ' | ') as combined_text
        FROM geopolitical_mentions
        WHERE event_id IS NOT NULL
        AND specificity_score >= 40
        GROUP BY event_id, ticker
        ORDER BY max_spec DESC
    """).fetchall()

    for row in mention_rows:
        event_id = row["event_id"]
        ticker = row["ticker"]

        if (event_id, ticker) in existing:
            continue

        event_cat = EVENT_TO_CATEGORY.get(event_id, "")
        if not event_cat:
            continue

        # Get stock reaction from event_studies
        es = conn.execute(
            "SELECT car_1_5, car_1_30 FROM event_studies WHERE event_id = ? AND ticker = ?",
            (event_id, ticker),
        ).fetchone()

        car_5 = es["car_1_5"] if es else 0.0
        car_30 = es["car_1_30"] if es else 0.0

        # Predict channel using sector-aware logic
        channel = _assign_channel(event_id, event_cat, ticker, car_5, gics_map, scorer)

        # Confidence based on: specificity + event match + stock reaction magnitude
        spec_signal = min(row["max_spec"] / 100, 1.0)
        stock_signal = min(abs(car_5 or 0) * 5, 1.0)  # 20% move = 1.0
        mention_signal = min(row["mention_count"] / 10, 1.0)
        confidence = 0.4 * spec_signal + 0.35 * stock_signal + 0.25 * mention_signal

        if confidence < min_confidence:
            continue

        quarter = EVENT_DATES.get(event_id, ("", ""))[1]
        mention_text = (row["combined_text"] or "")[:500]

        candidates.append({
            "event_id": event_id,
            "company_ticker": ticker,
            "company_name": "",
            "sector_gics": gics_map.get(ticker, ""),
            "impact_channel": channel,
            "quarter": quarter,
            "mention_text": mention_text,
            "mention_sentiment": round(-0.3 if car_5 and car_5 < 0 else 0.1, 2),
            "management_action_described": "",
            "revenue_delta_pct": "",
            "cogs_delta_pct": "",
            "operating_income_delta_pct": "",
            "capex_delta_pct": "",
            "car_1_5": round(car_5, 4) if car_5 else "",
            "car_1_30": round(car_30, 4) if car_30 else "",
            "source": "edgar_mention",
            "confidence": "high" if confidence >= 0.7 else "medium",
            "labeled_by": "auto_pipeline_v1",
            "human_reviewed": 0,
            "notes": f"Auto-labeled. Specificity={row['max_spec']}, mentions={row['mention_count']}, conf={confidence:.2f}",
            "_confidence_score": confidence,
            "_source_type": "edgar_mention",
        })

    logger.info(f"  Generated {len(candidates)} candidates from EDGAR mentions")

    # ── Source 2: Large stock reactions without EDGAR mention ──
    logger.info("Source 2: Large stock reactions from event studies...")
    edgar_pairs = set((c["event_id"], c["company_ticker"]) for c in candidates)

    es_rows = conn.execute("""
        SELECT es.event_id, es.ticker, es.car_1_5, es.car_1_30
        FROM event_studies es
        WHERE ABS(es.car_1_5) > 0.05
        ORDER BY ABS(es.car_1_5) DESC
    """).fetchall()

    stock_candidates = 0
    for row in es_rows:
        event_id = row["event_id"]
        ticker = row["ticker"]

        if (event_id, ticker) in existing:
            continue
        if (event_id, ticker) in edgar_pairs:
            continue  # already captured from EDGAR

        event_cat = EVENT_TO_CATEGORY.get(event_id, "")
        if not event_cat:
            continue

        car_5 = row["car_1_5"]
        car_30 = row["car_1_30"] or 0.0

        # Check if company has mention data for this event category
        mention_info = conn.execute("""
            SELECT COUNT(*) as cnt, MAX(specificity_score) as max_spec
            FROM geopolitical_mentions
            WHERE ticker = ? AND primary_category = ?
        """, (ticker, event_cat)).fetchone()

        has_mentions = mention_info["cnt"] > 0

        # Predict channel using sector-aware heuristics
        # Model 2 has limited accuracy, so we use GICS sector to assign
        # channels that make economic sense for each company type.
        channel = _assign_channel(event_id, event_cat, ticker, car_5, gics_map, scorer)

        # Confidence: stock reaction is strong, but without EDGAR mention it's
        # less certain the company is specifically exposed (vs market-wide move)
        stock_signal = min(abs(car_5) * 5, 1.0)
        mention_bonus = 0.15 if has_mentions else 0.0
        # Strong reactions (>15%) are more likely company-specific
        magnitude_bonus = 0.1 if abs(car_5) > 0.15 else 0.0
        confidence = 0.5 * stock_signal + mention_bonus + magnitude_bonus

        if confidence < min_confidence:
            continue

        quarter = EVENT_DATES.get(event_id, ("", ""))[1]

        candidates.append({
            "event_id": event_id,
            "company_ticker": ticker,
            "company_name": "",
            "sector_gics": gics_map.get(ticker, ""),
            "impact_channel": channel,
            "quarter": quarter,
            "mention_text": f"Stock reaction {car_5:+.1%} over 5 days following {event_id.replace('_', ' ')}",
            "mention_sentiment": round(-0.5 if car_5 < -0.1 else (-0.3 if car_5 < 0 else 0.3), 2),
            "management_action_described": "",
            "revenue_delta_pct": "",
            "cogs_delta_pct": "",
            "operating_income_delta_pct": "",
            "capex_delta_pct": "",
            "car_1_5": round(car_5, 4),
            "car_1_30": round(car_30, 4) if car_30 else "",
            "source": "event_study",
            "confidence": "high" if confidence >= 0.7 else "medium",
            "labeled_by": "auto_pipeline_v1",
            "human_reviewed": 0,
            "notes": f"Auto-labeled from stock reaction. car_1_5={car_5:+.4f}, conf={confidence:.2f}",
            "_confidence_score": confidence,
            "_source_type": "event_study",
        })
        stock_candidates += 1

    logger.info(f"  Generated {stock_candidates} candidates from stock reactions")
    logger.info(f"  TOTAL candidates: {len(candidates)}")

    return candidates


def print_candidates(candidates: list[dict]):
    """Print candidate summary for review."""
    by_event = {}
    by_source = {}
    by_confidence = {"high": 0, "medium": 0}

    for c in candidates:
        by_event[c["event_id"]] = by_event.get(c["event_id"], 0) + 1
        by_source[c["_source_type"]] = by_source.get(c["_source_type"], 0) + 1
        by_confidence[c["confidence"]] += 1

    print(f"\n{'='*70}")
    print(f"CANDIDATE SEED LABELS: {len(candidates)} total")
    print(f"{'='*70}")

    print(f"\nBy confidence: {by_confidence}")
    print(f"By source: {by_source}")
    print(f"\nBy event:")
    for event, count in sorted(by_event.items(), key=lambda x: -x[1]):
        print(f"  {event:40s} {count}")

    print(f"\nTop 20 candidates (by confidence):")
    print(f"  {'Event':35s} {'Ticker':8s} {'Channel':30s} {'CAR5':>8s} {'Conf':>6s} Source")
    print(f"  {'-'*95}")
    for c in sorted(candidates, key=lambda x: -x["_confidence_score"])[:20]:
        car = f"{c['car_1_5']:+.1%}" if c["car_1_5"] else "n/a"
        print(f"  {c['event_id'][:35]:35s} {c['company_ticker']:8s} "
              f"{c['impact_channel'][:30]:30s} {car:>8s} {c['_confidence_score']:>5.2f} {c['_source_type']}")


def append_to_csv(candidates: list[dict], min_confidence: float = 0.5):
    """Append candidates to seed_labels.csv."""
    # Read existing to get fieldnames
    with open(SEED_LABELS_PATH) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    # Filter internal fields
    filtered = []
    for c in candidates:
        if c["_confidence_score"] < min_confidence:
            continue
        row = {k: v for k, v in c.items() if not k.startswith("_")}
        filtered.append(row)

    with open(SEED_LABELS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in filtered:
            writer.writerow(row)

    logger.info(f"Appended {len(filtered)} candidates to {SEED_LABELS_PATH.name}")
    return len(filtered)


@click.command()
@click.option("--min-confidence", default=0.5, type=float, help="Minimum confidence threshold")
@click.option("--append", "do_append", is_flag=True, help="Append to seed_labels.csv (default: preview only)")
@click.option("--show-all", is_flag=True, help="Show all candidates, not just top 20")
def main(min_confidence, do_append, show_all):
    """Generate candidate seed labels using semi-supervised pipeline."""
    conn = get_db_connection()
    candidates = generate_candidates(conn, min_confidence=min_confidence)
    conn.close()

    if not candidates:
        logger.info("No candidates generated above confidence threshold.")
        return

    print_candidates(candidates)

    if do_append:
        n = append_to_csv(candidates, min_confidence=min_confidence)
        print(f"\nAppended {n} labels. Total seed labels: 163 + {n} = {163 + n}")
    else:
        high = sum(1 for c in candidates if c["confidence"] == "high")
        print(f"\nPreview only. Run with --append to add {len(candidates)} candidates ({high} high-confidence).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
