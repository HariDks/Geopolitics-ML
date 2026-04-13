"""
Data preparation pipeline for model training.

Three steps:
1. compute_financial_deltas — De-cumulate XBRL quarterly data and compute YoY deltas
2. link_mentions_to_events — Map EDGAR geopolitical mentions to specific event_ids
3. score_mention_specificity — Filter boilerplate vs. genuine geopolitical mentions

Usage:
    python pipelines/data_prep.py --step deltas
    python pipelines/data_prep.py --step link
    python pipelines/data_prep.py --step specificity
    python pipelines/data_prep.py --step all
"""

import json
import re
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("data_prep")


# ─── Step 1: Financial Deltas ───────────────────────────────────────────────────

# Map fiscal_period to quarter number for ordering and de-cumulation
PERIOD_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}

FINANCIAL_COLS = ["revenue", "cogs", "operating_income", "net_income", "eps_diluted"]


def compute_financial_deltas(conn) -> int:
    """
    Create a financial_deltas table with:
    - Standalone quarterly figures (de-cumulated from YTD XBRL data)
    - Year-over-year delta percentages for same fiscal period
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Computing financial deltas")
    logger.info("=" * 60)

    conn.execute("DROP TABLE IF EXISTS financial_deltas")
    conn.execute("""
        CREATE TABLE financial_deltas (
            ticker              TEXT NOT NULL,
            fiscal_year         INTEGER NOT NULL,
            fiscal_period       TEXT NOT NULL,
            quarter             TEXT NOT NULL,
            period_end          TEXT,
            filed               TEXT,

            -- Standalone quarterly figures (de-cumulated)
            revenue_standalone          REAL,
            cogs_standalone             REAL,
            operating_income_standalone REAL,
            net_income_standalone       REAL,
            eps_diluted_standalone      REAL,

            -- Year-over-year delta (same period, prior year)
            revenue_yoy_pct             REAL,
            cogs_yoy_pct                REAL,
            operating_income_yoy_pct    REAL,
            net_income_yoy_pct          REAL,

            -- Gross margin (revenue - cogs) / revenue
            gross_margin                REAL,
            gross_margin_prior_year     REAL,
            gross_margin_delta_pp       REAL,  -- percentage point change

            PRIMARY KEY (ticker, fiscal_year, fiscal_period)
        )
    """)

    # Fetch all financials ordered for processing
    rows = conn.execute("""
        SELECT ticker, fiscal_year, fiscal_period, quarter, period_end, filed,
               revenue, cogs, operating_income, net_income, eps_diluted
        FROM company_financials
        ORDER BY ticker, fiscal_year, fiscal_period
    """).fetchall()

    # Group by ticker
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        t = r["ticker"]
        if t not in by_ticker:
            by_ticker[t] = []
        by_ticker[t].append(dict(r))

    stored = 0
    for ticker, records in by_ticker.items():
        # Group by fiscal year for de-cumulation
        by_year: dict[int, dict[str, dict]] = {}
        for rec in records:
            fy = rec["fiscal_year"]
            fp = rec["fiscal_period"]
            if fy not in by_year:
                by_year[fy] = {}
            by_year[fy][fp] = rec

        for fy, periods in sorted(by_year.items()):
            for fp in ["Q1", "Q2", "Q3", "FY"]:
                if fp not in periods:
                    continue
                rec = periods[fp]

                standalone = {}
                for col in FINANCIAL_COLS:
                    cum_val = rec.get(col)
                    if cum_val is None:
                        standalone[col] = None
                        continue

                    if fp == "FY":
                        # FY is always the full-year annual figure — never de-cumulate
                        standalone[col] = cum_val
                    elif fp == "Q1":
                        standalone[col] = cum_val
                    else:
                        # Try de-cumulation for Q2/Q3, but validate it makes sense
                        prev_fp = "Q1" if fp == "Q2" else "Q2"
                        prev_val = periods.get(prev_fp, {}).get(col)
                        if prev_val is not None and cum_val > prev_val > 0:
                            # Looks cumulative (Q2 > Q1), de-cumulate
                            standalone[col] = cum_val - prev_val
                        else:
                            # Not cumulative or prev missing — use raw value
                            standalone[col] = cum_val

                # YoY delta: compare raw same-period values across years
                # This avoids de-cumulation errors — same-period raw values are
                # directly comparable (both are either cumulative or standalone)
                yoy = {}
                prior_rec = by_year.get(fy - 1, {}).get(fp)
                if prior_rec:
                    for col in ["revenue", "cogs", "operating_income", "net_income"]:
                        current_raw = rec.get(col)
                        prior_raw = prior_rec.get(col)
                        if current_raw is not None and prior_raw and prior_raw != 0:
                            yoy[col] = (current_raw - prior_raw) / abs(prior_raw)
                        else:
                            yoy[col] = None
                else:
                    for col in ["revenue", "cogs", "operating_income", "net_income"]:
                        yoy[col] = None

                # Gross margin from standalone figures
                rev_s = standalone.get("revenue")
                cogs_s = standalone.get("cogs")
                gm = None
                if rev_s and rev_s != 0 and cogs_s is not None:
                    gm = (rev_s - cogs_s) / abs(rev_s)

                # Prior year gross margin (same approach)
                gm_prior = None
                if prior_rec:
                    pr = prior_rec.get("revenue")
                    pc = prior_rec.get("cogs")
                    if pr and pr != 0 and pc is not None:
                        # For FY, use raw; for Q, use same logic
                        if fp == "FY" or fp == "Q1":
                            gm_prior = (pr - pc) / abs(pr)
                        else:
                            prev_fp = "Q1" if fp == "Q2" else "Q2"
                            prior_by_year = by_year.get(fy - 1, {})
                            prev_pr = prior_by_year.get(prev_fp, {}).get("revenue")
                            prev_pc = prior_by_year.get(prev_fp, {}).get("cogs")
                            if prev_pr is not None and pr > prev_pr > 0:
                                pr_s = pr - prev_pr
                                pc_s = pc - (prev_pc or 0)
                            else:
                                pr_s, pc_s = pr, pc
                            if pr_s and pr_s != 0:
                                gm_prior = (pr_s - pc_s) / abs(pr_s)

                gm_delta = None
                if gm is not None and gm_prior is not None:
                    gm_delta = gm - gm_prior

                conn.execute(
                    """INSERT OR REPLACE INTO financial_deltas
                       (ticker, fiscal_year, fiscal_period, quarter, period_end, filed,
                        revenue_standalone, cogs_standalone, operating_income_standalone,
                        net_income_standalone, eps_diluted_standalone,
                        revenue_yoy_pct, cogs_yoy_pct, operating_income_yoy_pct, net_income_yoy_pct,
                        gross_margin, gross_margin_prior_year, gross_margin_delta_pp)
                       VALUES (?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?,?)""",
                    (
                        ticker, fy, fp, rec["quarter"], rec.get("period_end"), rec.get("filed"),
                        standalone.get("revenue"), standalone.get("cogs"),
                        standalone.get("operating_income"), standalone.get("net_income"),
                        standalone.get("eps_diluted"),
                        yoy.get("revenue"), yoy.get("cogs"),
                        yoy.get("operating_income"), yoy.get("net_income"),
                        gm, gm_prior, gm_delta,
                    ),
                )
                stored += 1

    conn.commit()

    # Create useful indices
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_ticker ON financial_deltas(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_year ON financial_deltas(fiscal_year)")

    logger.info(f"Stored {stored} financial delta records for {len(by_ticker)} companies")

    # Quick validation
    sample = conn.execute("""
        SELECT ticker, fiscal_year, fiscal_period,
               ROUND(revenue_standalone/1e9, 1) as rev_B,
               ROUND(revenue_yoy_pct * 100, 1) as rev_yoy_pct,
               ROUND(gross_margin * 100, 1) as gm_pct,
               ROUND(gross_margin_delta_pp * 100, 1) as gm_delta_pp
        FROM financial_deltas
        WHERE ticker = 'AAPL' AND revenue_yoy_pct IS NOT NULL
        ORDER BY fiscal_year, fiscal_period
        LIMIT 8
    """).fetchall()

    logger.info("Sample AAPL deltas:")
    for s in sample:
        logger.info(
            f"  {s['ticker']} {s['fiscal_year']}{s['fiscal_period']}: "
            f"Rev ${s['rev_B']}B, YoY {s['rev_yoy_pct']}%, "
            f"GM {s['gm_pct']}%, GM delta {s['gm_delta_pp']}pp"
        )

    return stored


# ─── Step 2: Link Mentions to Events ────────────────────────────────────────────

# Major events with date ranges and keyword signatures for matching
EVENT_SIGNATURES = {
    "russia_invasion_2022": {
        "date_start": "2022-02-24",
        "date_end": "2023-12-31",
        "categories": ["armed_conflict_instability", "sanctions_financial_restrictions"],
        "keywords": ["russia", "ukraine", "invasion", "rosneft", "kremlin", "moscow",
                      "russian", "ukrainian", "donbas", "crimea", "nord stream"],
    },
    "us_chip_export_oct2022": {
        "date_start": "2022-10-07",
        "date_end": "2024-06-30",
        "categories": ["technology_controls", "trade_policy_actions"],
        "keywords": ["export control", "chip", "semiconductor", "huawei", "smic",
                      "a100", "h100", "china", "bis rule", "advanced computing"],
    },
    "us_chip_export_oct2023": {
        "date_start": "2023-10-17",
        "date_end": "2025-06-30",
        "categories": ["technology_controls", "trade_policy_actions"],
        "keywords": ["export control", "chip", "semiconductor", "october 2023",
                      "updated rule", "advanced computing", "a800", "h800"],
    },
    "red_sea_houthi_2023": {
        "date_start": "2023-12-15",
        "date_end": "2025-06-30",
        "categories": ["armed_conflict_instability", "resource_energy_disruptions"],
        "keywords": ["red sea", "houthi", "suez", "cape of good hope", "shipping",
                      "bab el-mandeb", "yemen", "freight", "reroute"],
    },
    "us_tariffs_2025": {
        "date_start": "2025-01-01",
        "date_end": "2026-12-31",
        "categories": ["trade_policy_actions"],
        "keywords": ["tariff", "section 301", "section 232", "trade war", "customs",
                      "import duty", "reciprocal tariff", "universal tariff"],
    },
    "covid_lockdown_start": {
        "date_start": "2020-01-01",
        "date_end": "2022-06-30",
        "categories": ["armed_conflict_instability", "political_transitions_volatility"],
        "keywords": ["covid", "pandemic", "lockdown", "quarantine", "coronavirus",
                      "supply chain disruption", "shutdown"],
    },
    "israel_hamas_2023": {
        "date_start": "2023-10-07",
        "date_end": "2025-12-31",
        "categories": ["armed_conflict_instability"],
        "keywords": ["israel", "hamas", "gaza", "october 7", "palestinian",
                      "middle east conflict", "tel aviv"],
    },
    "eu_energy_crisis_peak": {
        "date_start": "2022-02-24",
        "date_end": "2023-12-31",
        "categories": ["resource_energy_disruptions"],
        "keywords": ["energy crisis", "natural gas", "gas price", "lng",
                      "energy security", "energy cost", "nord stream", "gas supply"],
    },
    "us_china_trade_war_start": {
        "date_start": "2018-03-01",
        "date_end": "2025-12-31",
        "categories": ["trade_policy_actions", "technology_controls"],
        "keywords": ["china tariff", "trade war", "us-china", "section 301",
                      "trade tension", "decoupling", "derisking"],
    },
    "xinjiang_boycott_2021": {
        "date_start": "2021-01-01",
        "date_end": "2023-12-31",
        "categories": ["trade_policy_actions", "institutional_alliance_realignment"],
        "keywords": ["xinjiang", "uyghur", "forced labor", "cotton ban",
                      "human rights", "boycott china"],
    },
    "brexit_referendum": {
        "date_start": "2016-06-23",
        "date_end": "2025-12-31",
        "categories": ["institutional_alliance_realignment", "political_transitions_volatility"],
        "keywords": ["brexit", "european union exit", "uk departure", "eu withdrawal"],
    },
    "india_pakistan_sindoor": {
        "date_start": "2025-05-01",
        "date_end": "2025-12-31",
        "categories": ["armed_conflict_instability"],
        "keywords": ["india", "pakistan", "sindoor", "kashmir", "loc",
                      "india-pakistan", "indo-pak"],
    },
    "us_iran_war_2026": {
        "date_start": "2026-02-28",
        "date_end": "2026-12-31",
        "categories": ["armed_conflict_instability", "resource_energy_disruptions"],
        "keywords": ["iran", "strait of hormuz", "hormuz", "persian gulf",
                      "iran war", "epic fury", "tehran"],
    },
    "notpetya_2017": {
        "date_start": "2017-06-27",
        "date_end": "2018-06-30",
        "categories": ["armed_conflict_instability"],
        "keywords": ["notpetya", "petya", "cyberattack", "ransomware", "malware",
                      "maersk", "merck", "cyber"],
    },
    "iran_sanctions_2018": {
        "date_start": "2018-05-08",
        "date_end": "2023-12-31",
        "categories": ["sanctions_financial_restrictions"],
        "keywords": ["iran sanction", "jcpoa", "iran deal", "iran nuclear",
                      "iranian sanctions", "iran oil"],
    },
    "panama_mine_closure_2023": {
        "date_start": "2023-11-01",
        "date_end": "2025-12-31",
        "categories": ["political_transitions_volatility"],
        "keywords": ["panama", "cobre panama", "first quantum", "mine closure",
                      "mining concession"],
    },
    "chile_lithium_nationalization": {
        "date_start": "2023-04-01",
        "date_end": "2025-12-31",
        "categories": ["political_transitions_volatility", "resource_energy_disruptions"],
        "keywords": ["chile", "lithium", "nationalization", "sqm", "albemarle",
                      "critical mineral"],
    },
    "india_demonetization_2016": {
        "date_start": "2016-11-08",
        "date_end": "2017-12-31",
        "categories": ["political_transitions_volatility"],
        "keywords": ["demonetization", "india cash", "rupee", "modi",
                      "cash ban", "banknote"],
    },
    "opec_price_war_2014": {
        "date_start": "2014-11-01",
        "date_end": "2016-12-31",
        "categories": ["resource_energy_disruptions"],
        "keywords": ["opec", "oil price", "crude", "oil war", "oil glut",
                      "saudi oil"],
    },
    "suez_blockage_2021": {
        "date_start": "2021-03-23",
        "date_end": "2021-06-30",
        "categories": ["armed_conflict_instability"],
        "keywords": ["suez", "ever given", "suez canal", "suez blockage",
                      "shipping disruption"],
    },
}


def link_mentions_to_events(conn) -> int:
    """
    For each geopolitical mention, try to match it to a specific event_id
    based on filing date overlap + keyword matching in the mention text.

    Adds event_id and match_score columns to geopolitical_mentions.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Linking mentions to events")
    logger.info("=" * 60)

    # Add columns if they don't exist
    for col, coltype in [("event_id", "TEXT"), ("event_match_score", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE geopolitical_mentions ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # Column already exists

    # Reset existing links
    conn.execute("UPDATE geopolitical_mentions SET event_id = NULL, event_match_score = NULL")

    mentions = conn.execute("""
        SELECT id, ticker, filing_date, primary_category, mention_text
        FROM geopolitical_mentions
    """).fetchall()

    linked = 0
    for m in mentions:
        mention_text_lower = (m["mention_text"] or "").lower()
        filing_date = m["filing_date"] or ""

        best_event = None
        best_score = 0.0

        for event_id, sig in EVENT_SIGNATURES.items():
            # Check date range: filing must be after event start
            if filing_date < sig["date_start"]:
                continue

            # Check category match
            cat_match = m["primary_category"] in sig["categories"]

            # Count keyword hits
            keyword_hits = sum(1 for kw in sig["keywords"] if kw in mention_text_lower)

            if keyword_hits == 0:
                continue

            # Score: keyword_hits * category_bonus
            score = keyword_hits * (1.5 if cat_match else 0.8)

            if score > best_score:
                best_score = score
                best_event = event_id

        if best_event and best_score >= 1.5:  # At least one keyword + category match
            conn.execute(
                "UPDATE geopolitical_mentions SET event_id = ?, event_match_score = ? WHERE id = ?",
                (best_event, round(best_score, 2), m["id"]),
            )
            linked += 1

    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gm_event ON geopolitical_mentions(event_id)")

    logger.info(f"Linked {linked} of {len(mentions)} mentions to events")

    # Summary
    summary = conn.execute("""
        SELECT event_id, COUNT(*) as cnt, ROUND(AVG(event_match_score), 1) as avg_score
        FROM geopolitical_mentions
        WHERE event_id IS NOT NULL
        GROUP BY event_id
        ORDER BY cnt DESC
    """).fetchall()

    for s in summary:
        logger.info(f"  {s['event_id']}: {s['cnt']} mentions (avg score {s['avg_score']})")

    return linked


# ─── Step 3: Mention Specificity Scoring ─────────────────────────────────────────

# Patterns that indicate boilerplate risk language (generic, not event-specific)
BOILERPLATE_PATTERNS = [
    r"may\s+adversely\s+affect",
    r"could\s+adversely\s+affect",
    r"risks?\s+include",
    r"subject\s+to\s+risks?",
    r"no\s+assurance",
    r"there\s+can\s+be\s+no",
    r"we\s+cannot\s+predict",
    r"may\s+be\s+materially",
    r"could\s+have\s+a\s+material",
    r"unforeseen\s+developments",
    r"general\s+economic\s+conditions",
    r"competitive\s+factors",
    r"characterized\s+by\s+rapid",
    r"evolving\s+industry\s+standards",
    r"equity\s+award",
    r"stock\s+option",
    r"diluted\s+net\s+income\s+per\s+share",
    r"anti-dilutive",
    r"foreign\s+currency\s+forward\s+contract",
    r"accumulated\s+other\s+comprehensive",
    r"fair\s+value\s+of\s+partially\s+vested",
    r"unearned\s+revenue\s+related\s+to",
    r"lease\s+obligation",
]

# Patterns that indicate genuine, specific geopolitical content
SPECIFICITY_PATTERNS = [
    # Named countries/regions in conflict context
    r"\b(?:russia|ukraine|china|iran|israel|gaza|taiwan|north\s+korea)\b",
    # Named policies/events
    r"\b(?:section\s+301|section\s+232|entity\s+list|chips\s+act|jcpoa|nafta|usmca)\b",
    # Specific financial impact language
    r"\$\s*[\d,.]+\s*(?:million|billion|M|B)",
    r"\d+(?:\.\d+)?%\s*(?:decline|decrease|increase|growth|impact|reduction)",
    # Named organizations
    r"\b(?:OFAC|BIS|WTO|NATO|OPEC|EU\s+Commission)\b",
    # Specific event references
    r"\b(?:tariff|sanction|embargo|export\s+control|import\s+ban)\b",
    r"\b(?:red\s+sea|suez|strait\s+of\s+hormuz|south\s+china\s+sea)\b",
    r"\b(?:invasion|coup|ceasefire|blockade|annexation)\b",
    # Management action language (signals real impact)
    r"\b(?:we\s+(?:exited|divested|wrote\s+down|impaired|relocated|suspended))\b",
    r"\b(?:rerouted|diversified|restructured|accelerated|stockpiled)\b",
]


def score_mention_specificity(conn) -> int:
    """
    Score each mention from 0-100 on specificity:
    - 0-30: Boilerplate (generic risk language, not event-specific)
    - 31-60: Semi-specific (mentions a keyword but in generic context)
    - 61-100: Specific (names countries, policies, dollar amounts, actions)

    Updates geopolitical_mentions with specificity_score and updates
    corporate_impacts confidence accordingly.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Scoring mention specificity")
    logger.info("=" * 60)

    # Add column if needed
    try:
        conn.execute("ALTER TABLE geopolitical_mentions ADD COLUMN specificity_score INTEGER")
    except Exception:
        pass

    mentions = conn.execute("""
        SELECT id, mention_text, keyword_count, primary_category
        FROM geopolitical_mentions
    """).fetchall()

    boilerplate_count = 0
    semi_count = 0
    specific_count = 0

    for m in mentions:
        text = m["mention_text"] or ""
        text_lower = text.lower()

        # Count boilerplate pattern matches
        bp_hits = sum(1 for pat in BOILERPLATE_PATTERNS if re.search(pat, text_lower))

        # Count specificity pattern matches
        sp_hits = sum(1 for pat in SPECIFICITY_PATTERNS if re.search(pat, text_lower))

        # Base score from keyword count (1-8 mapped to 10-40)
        kw_score = min((m["keyword_count"] or 1) * 10, 40)

        # Adjust: specificity adds points, boilerplate subtracts
        score = kw_score + (sp_hits * 15) - (bp_hits * 12)

        # Bonus for having a linked event
        # (checked separately to avoid querying in loop)

        # Clamp to 0-100
        score = max(0, min(100, score))

        conn.execute(
            "UPDATE geopolitical_mentions SET specificity_score = ? WHERE id = ?",
            (score, m["id"]),
        )

        if score <= 30:
            boilerplate_count += 1
        elif score <= 60:
            semi_count += 1
        else:
            specific_count += 1

    conn.commit()

    # Bonus for event-linked mentions
    conn.execute("""
        UPDATE geopolitical_mentions
        SET specificity_score = MIN(100, specificity_score + 15)
        WHERE event_id IS NOT NULL
    """)
    conn.commit()

    logger.info(f"Scored {len(mentions)} mentions:")
    logger.info(f"  Boilerplate (0-30):    {boilerplate_count} ({boilerplate_count*100//len(mentions)}%)")
    logger.info(f"  Semi-specific (31-60): {semi_count} ({semi_count*100//len(mentions)}%)")
    logger.info(f"  Specific (61-100):     {specific_count} ({specific_count*100//len(mentions)}%)")

    # Update corporate_impacts confidence based on specificity
    # Low specificity → low confidence, high specificity → medium confidence
    conn.execute("""
        UPDATE corporate_impacts
        SET confidence = 'medium'
        WHERE source = 'edgar_nlp'
        AND impact_id IN (
            SELECT 'AUTO-' || gm.ticker || '-' || gm.id
            FROM geopolitical_mentions gm
            WHERE gm.specificity_score >= 50
        )
    """)

    # Distribution by category
    dist = conn.execute("""
        SELECT primary_category,
               COUNT(*) as total,
               SUM(CASE WHEN specificity_score > 30 THEN 1 ELSE 0 END) as non_boilerplate,
               ROUND(AVG(specificity_score), 1) as avg_score
        FROM geopolitical_mentions
        GROUP BY primary_category
        ORDER BY avg_score DESC
    """).fetchall()

    logger.info("\nSpecificity by category:")
    for d in dist:
        logger.info(
            f"  {d['primary_category'][:35]:35s} "
            f"total={d['total']:5d}  non-bp={d['non_boilerplate']:5d}  "
            f"avg_score={d['avg_score']}"
        )

    return len(mentions)


# ─── CLI ─────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--step", required=True,
              type=click.Choice(["deltas", "link", "specificity", "all"]),
              help="Which prep step to run")
def main(step: str) -> None:
    """Run data preparation steps for model training."""
    conn = get_db_connection()

    if step in ("deltas", "all"):
        n = compute_financial_deltas(conn)
        logger.info(f"Financial deltas: {n} records\n")

    if step in ("link", "all"):
        n = link_mentions_to_events(conn)
        logger.info(f"Mentions linked: {n}\n")

    if step in ("specificity", "all"):
        n = score_mention_specificity(conn)
        logger.info(f"Mentions scored: {n}\n")

    conn.close()
    logger.info("Data prep complete.")


if __name__ == "__main__":
    main()
