"""
Model 4: Strategy Recommender — Retrieval-based recommendation engine.

Phase 2A: Retrieval-based approach using Phase 1's top 34 cell documentation.
Given (event_category, impact_channel, company_context), returns ranked strategies.

Steps:
  1. Extract strategy archetypes from geopolitical_muscle_matrix_v2.xlsx
  2. Parse each archetype into individual atomic strategies
  3. Store in strategies table with metadata
  4. Build a scoring function for ranking

Usage:
    python models/strategy_recommender/build.py              # populate strategies table
    python models/strategy_recommender/build.py --show       # show all strategies
"""

import json
import re
import sys
from pathlib import Path

import click
import openpyxl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.utils import get_db_connection, get_logger

logger = get_logger("strategy_recommender")

ROOT_DIR = Path(__file__).parent.parent.parent
EXCEL_PATH = ROOT_DIR / "geopolitical_muscle_matrix_v2.xlsx"
MODEL_DIR = Path(__file__).parent / "saved"

# Map Excel column names to our taxonomy
EVENT_NAME_MAP = {
    "Trade Policy Actions": "trade_policy_actions",
    "Sanctions & Financial Restrictions": "sanctions_financial_restrictions",
    "Sanctions & Financial Restrict": "sanctions_financial_restrictions",
    "Armed Conflict & Instability": "armed_conflict_instability",
    "Regulatory & Sovereignty Shifts": "regulatory_sovereignty_shifts",
    "Technology Controls": "technology_controls",
    "Resource & Energy Disruptions": "resource_energy_disruptions",
    "Political Transitions & Volatility": "political_transitions_volatility",
    "Institutional & Alliance Realignment": "institutional_alliance_realignment",
}

CHANNEL_NAME_MAP = {
    "Procurement & Supply Chain": "procurement_supply_chain",
    "Revenue & Market Access": "revenue_market_access",
    "Capital Allocation & Investment": "capital_allocation_investment",
    "Regulatory Compliance Cost": "regulatory_compliance_cost",
    "Logistics & Operations": "logistics_operations",
    "Innovation & IP": "innovation_ip",
    "Workforce & Talent": "workforce_talent",
    "Reputation & Stakeholder Mgmt": "reputation_stakeholder",
    "Financial & Treasury": "financial_treasury",
    "Cybersecurity & IT Infrastructure": "cybersecurity_it",
}

# Categorize strategies by type
STRATEGY_CATEGORY_KEYWORDS = {
    "mitigate": ["diversif", "hedg", "buffer", "safety stock", "alternative", "dual-sourc",
                 "insurance", "backup", "contingency", "pre-position", "stockpil"],
    "hedge": ["hedg", "currency", "commodity", "insurance", "option", "swap",
              "forward", "escrow", "deferral"],
    "exit": ["exit", "divestiture", "withdraw", "wind down", "playbook",
             "disposition", "write-down", "repatriation"],
    "capture": ["capture", "opportunity", "adjacent market", "demand capture",
                "market entry", "re-entry", "new market"],
    "engage": ["engagement", "government relation", "regulatory shaping", "stakeholder",
               "communication", "lobbying", "proactive", "transparency", "humanitarian"],
    "monitor": ["monitor", "scanning", "horizon", "tracking", "early warning",
                "scenario planning", "assessment", "audit", "screening"],
}

# Cost and time estimates by strategy type
COST_ESTIMATES = {
    "mitigate": ("medium", "3-12 months"),
    "hedge": ("low-medium", "1-3 months"),
    "exit": ("high", "3-18 months"),
    "capture": ("medium-high", "6-18 months"),
    "engage": ("low", "1-6 months"),
    "monitor": ("low", "1-3 months"),
}


def classify_strategy(strategy_text: str) -> str:
    """Classify a strategy into one of 6 categories based on keywords."""
    text_lower = strategy_text.lower()
    best_cat = "mitigate"
    best_score = 0
    for cat, keywords in STRATEGY_CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


def extract_strategies_from_excel() -> list[dict]:
    """Extract and parse strategy archetypes from the Phase 1 Excel."""
    wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
    ws = wb["Top Priority Cells"]

    rows = list(ws.iter_rows(min_row=4, values_only=True))
    strategies = []
    strat_id = 0

    for row in rows:
        event_name = str(row[1]).strip() if row[1] else ""
        channel_name = str(row[2]).strip() if row[2] else ""
        score = row[5]
        archetypes_text = str(row[8]).strip() if row[8] else ""
        historical = str(row[9]).strip() if row[9] else ""
        transmission = str(row[6]).strip() if row[6] else ""
        indicators = str(row[7]).strip() if row[7] else ""

        event_cat = EVENT_NAME_MAP.get(event_name, "")
        channel = CHANNEL_NAME_MAP.get(channel_name, "")

        if not event_cat or not channel or not archetypes_text:
            continue

        # Split archetypes by semicolons into individual strategies
        individual_strats = [s.strip() for s in archetypes_text.split(";") if s.strip()]

        for strat_text in individual_strats:
            strat_id += 1
            cat = classify_strategy(strat_text)
            cost, timeline = COST_ESTIMATES.get(cat, ("medium", "3-12 months"))

            strategies.append({
                "strategy_id": f"STRAT-{strat_id:04d}",
                "event_category": event_cat,
                "impact_channel": channel,
                "strategy_name": strat_text,
                "strategy_category": cat,
                "description": f"{strat_text}. Context: {transmission[:200]}",
                "typical_cost_range": cost,
                "implementation_time": timeline,
                "prerequisites": json.dumps([]),
                "historical_precedents": json.dumps([historical[:300]] if historical else []),
                "success_conditions": f"Requires proactive implementation before event escalation. Priority score: {score}/25.",
                "precedent_count": 1 if historical else 0,
                "precedent_success_rate": 0.7,  # baseline estimate
                "priority_score": score,
            })

    wb.close()
    return strategies


def populate_strategies_table(conn, strategies: list[dict]) -> int:
    """Insert strategies into the database."""
    # Clear existing
    conn.execute("DELETE FROM strategies")

    stored = 0
    for s in strategies:
        conn.execute(
            """INSERT OR REPLACE INTO strategies
               (strategy_id, event_category, impact_channel, strategy_name,
                strategy_category, description, typical_cost_range,
                implementation_time, prerequisites, historical_precedents,
                success_conditions, precedent_count, precedent_success_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                s["strategy_id"], s["event_category"], s["impact_channel"],
                s["strategy_name"], s["strategy_category"], s["description"],
                s["typical_cost_range"], s["implementation_time"],
                s["prerequisites"], s["historical_precedents"],
                s["success_conditions"], s["precedent_count"],
                s["precedent_success_rate"],
            ),
        )
        stored += 1

    conn.commit()
    return stored


@click.command()
@click.option("--show", is_flag=True, help="Show all strategies without inserting")
def main(show):
    """Extract strategies from Phase 1 Excel and populate the database."""
    strategies = extract_strategies_from_excel()
    logger.info(f"Extracted {len(strategies)} individual strategies from {EXCEL_PATH.name}")

    # Summary by category
    cats = {}
    for s in strategies:
        cats[s["strategy_category"]] = cats.get(s["strategy_category"], 0) + 1
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat:15s} {cnt}")

    # Summary by event x channel
    cells = set()
    for s in strategies:
        cells.add((s["event_category"], s["impact_channel"]))
    logger.info(f"Covering {len(cells)} event-channel cells")

    if show:
        for s in strategies:
            print(f"{s['strategy_id']} | {s['event_category'][:25]:25s} | "
                  f"{s['impact_channel'][:25]:25s} | {s['strategy_category']:10s} | "
                  f"{s['strategy_name']}")
        return

    conn = get_db_connection()
    stored = populate_strategies_table(conn, strategies)
    conn.close()

    logger.info(f"Stored {stored} strategies in database")

    # Save config
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump({
            "total_strategies": len(strategies),
            "cells_covered": len(cells),
            "categories": cats,
            "source": "geopolitical_muscle_matrix_v2.xlsx (Top Priority Cells)",
        }, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
