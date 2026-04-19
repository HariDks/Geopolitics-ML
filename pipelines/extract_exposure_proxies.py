"""
Extract exposure proxy features from 10-K filing text.

Mines EDGAR mentions (geopolitical_mentions table) for indirect signals
about a company's geographic, supply chain, and asset exposure.

Extracts 5 proxy scores per company:
1. geo_mention_density — how often the company mentions specific regions
2. facility_concentration — mentions of plants/factories/mines in specific locations
3. single_source_risk — language suggesting concentrated supplier dependence
4. asset_exit_history — mentions of impairments, write-downs, exits
5. route_sensitivity — mentions of shipping routes, logistics chokepoints

Usage:
    python pipelines/extract_exposure_proxies.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("exposure_proxies")

OUTPUT_PATH = ROOT_DIR / "data" / "mappings" / "company_exposure_proxies.json"

# ── Region keyword groups ────────────────────────────────────────────────────

REGION_KEYWORDS = {
    "China": ["china", "chinese", "prc", "greater china", "mainland china", "beijing",
              "shanghai", "shenzhen", "hong kong", "taiwan", "taipei"],
    "Russia": ["russia", "russian", "moscow", "kremlin", "rosneft", "gazprom"],
    "Europe": ["europe", "european", "eu ", "european union", "gdpr", "emea",
               "uk ", "united kingdom", "germany", "france", "brexit"],
    "MiddleEast": ["middle east", "iran", "iraq", "saudi", "uae", "qatar",
                   "strait of hormuz", "persian gulf", "israel", "gaza"],
    "India": ["india", "indian", "mumbai", "delhi", "modi"],
    "LatAm": ["latin america", "brazil", "mexico", "argentina", "chile",
              "colombia", "venezuela", "peru"],
    "Africa": ["africa", "african", "nigeria", "south africa", "kenya",
               "sudan", "ethiopia", "mozambique", "congo", "drc"],
    "SoutheastAsia": ["southeast asia", "asean", "vietnam", "indonesia",
                      "thailand", "philippines", "myanmar", "malaysia", "singapore"],
    "Japan_Korea": ["japan", "japanese", "korea", "korean"],
}

# ── Exposure proxy patterns ──────────────────────────────────────────────────

FACILITY_PATTERNS = [
    r"\bplant[s]?\b", r"\bfactor(?:y|ies)\b", r"\bmine[s]?\b", r"\bfaciliti(?:y|es)\b",
    r"\bwarehouse[s]?\b", r"\bstore[s]?\b", r"\boffice[s]?\b", r"\brefineri(?:y|es)\b",
    r"\bdata center[s]?\b", r"\bassembly\b", r"\bmanufacturing\b",
]

SINGLE_SOURCE_PATTERNS = [
    r"substantially all\s+(?:of\s+)?(?:our\s+)?manufacturing",
    r"single[\s-]?sourc(?:e|ing)",
    r"sole[\s-]?sourc(?:e|ing)",
    r"limited[\s-]?source",
    r"contract manufactur(?:er|ing)",
    r"foundry partner",
    r"primary supplier",
    r"concentrated in",
    r"substantially dependent",
    r"rely (?:primarily|substantially|exclusively) on",
    r"one or a limited number of",
]

ASSET_EXIT_PATTERNS = [
    r"impairment", r"write[\s-]?down", r"write[\s-]?off",
    r"asset disposal", r"divest(?:iture|ed|ing)",
    r"exit(?:ed|ing)?\s+(?:from\s+)?(?:our|the)\s+\w+\s+(?:operations|business|market)",
    r"suspend(?:ed|ing)\s+operations",
    r"ceased operations", r"closed?\s+(?:our|the)\s+\w+\s+(?:plant|facility|mine|office)",
    r"goodwill impairment", r"asset impairment",
]

ROUTE_PATTERNS = [
    r"red sea", r"suez canal", r"strait of hormuz", r"cape of good hope",
    r"panama canal", r"south china sea", r"bab[- ]el[- ]mandeb",
    r"malacca strait", r"shipping route[s]?", r"freight cost[s]?",
    r"reroute|rerouting", r"transit time", r"shipping disruption",
    r"logistics route", r"supply route", r"trade route",
]

REVENUE_CONCENTRATION_PATTERNS = [
    r"(\d{1,3})%\s+of\s+(?:our\s+)?(?:total\s+)?(?:net\s+)?(?:revenue|sales|net sales)",
    r"(?:revenue|sales)\s+(?:from|in)\s+(?:the\s+)?(\w+)\s+(?:region|segment|market)",
    r"(?:greater china|china|europe|americas?|japan|india|middle east)\s+(?:represented|accounted for|contributed)\s+(?:approximately\s+)?(\d{1,3})%",
]


def compute_proxies_for_company(conn, ticker: str) -> dict:
    """Compute all exposure proxy scores for a company from its EDGAR mentions."""
    rows = conn.execute(
        "SELECT mention_text FROM geopolitical_mentions WHERE ticker = ?",
        (ticker,),
    ).fetchall()

    if not rows:
        return None

    # Concatenate all mention text
    all_text = " ".join(r["mention_text"] for r in rows).lower()
    n_paras = len(rows)

    # ── 1. Geographic mention density ──
    geo_density = {}
    for region, keywords in REGION_KEYWORDS.items():
        count = sum(all_text.count(kw) for kw in keywords)
        geo_density[region] = round(count / max(n_paras, 1), 3)

    # Top mentioned regions
    top_regions = sorted(geo_density.items(), key=lambda x: -x[1])[:5]

    # ── 2. Facility concentration ──
    facility_mentions = defaultdict(int)
    for region, keywords in REGION_KEYWORDS.items():
        for kw in keywords:
            # Find paragraphs mentioning BOTH a facility term AND a region
            for r in rows:
                para = r["mention_text"].lower()
                if kw in para:
                    for fp in FACILITY_PATTERNS:
                        if re.search(fp, para):
                            facility_mentions[region] += 1
                            break

    total_facility = sum(facility_mentions.values())
    facility_score = round(total_facility / max(n_paras, 1), 3)

    # ── 3. Single-source risk ──
    single_source_hits = sum(
        1 for pat in SINGLE_SOURCE_PATTERNS
        if re.search(pat, all_text, re.IGNORECASE)
    )
    single_source_score = round(single_source_hits / len(SINGLE_SOURCE_PATTERNS), 3)

    # ── 4. Asset exit history ──
    asset_exit_hits = sum(
        1 for pat in ASSET_EXIT_PATTERNS
        if re.search(pat, all_text, re.IGNORECASE)
    )
    asset_exit_score = round(asset_exit_hits / len(ASSET_EXIT_PATTERNS), 3)

    # ── 5. Route sensitivity ──
    route_hits = sum(
        1 for pat in ROUTE_PATTERNS
        if re.search(pat, all_text, re.IGNORECASE)
    )
    route_score = round(route_hits / len(ROUTE_PATTERNS), 3)

    # ── Revenue concentration mentions ──
    rev_conc_mentions = []
    for pat in REVENUE_CONCENTRATION_PATTERNS:
        matches = re.findall(pat, all_text, re.IGNORECASE)
        rev_conc_mentions.extend(matches)

    return {
        "n_paragraphs": n_paras,
        "geo_mention_density": geo_density,
        "top_regions": [{"region": r, "density": d} for r, d in top_regions if d > 0],
        "facility_concentration_score": facility_score,
        "facility_by_region": dict(facility_mentions),
        "single_source_risk_score": single_source_score,
        "asset_exit_score": asset_exit_score,
        "route_sensitivity_score": route_score,
        "revenue_concentration_mentions": len(rev_conc_mentions),
    }


@click.command()
@click.option("--limit", default=None, type=int)
def main(limit):
    """Extract exposure proxies for all companies in the mentions table."""
    conn = get_db_connection()

    tickers = [r["ticker"] for r in conn.execute(
        "SELECT DISTINCT ticker FROM geopolitical_mentions ORDER BY ticker"
    ).fetchall()]

    if limit:
        tickers = tickers[:limit]

    logger.info(f"Extracting exposure proxies for {len(tickers)} companies...")

    results = {}
    for i, ticker in enumerate(tickers):
        proxies = compute_proxies_for_company(conn, ticker)
        if proxies:
            results[ticker] = proxies
            top = proxies["top_regions"][:3]
            top_str = ", ".join(f"{r['region']}={r['density']:.2f}" for r in top)
            logger.info(
                f"  [{i+1}/{len(tickers)}] {ticker:6s} "
                f"facility={proxies['facility_concentration_score']:.2f} "
                f"single_src={proxies['single_source_risk_score']:.2f} "
                f"asset_exit={proxies['asset_exit_score']:.2f} "
                f"route={proxies['route_sensitivity_score']:.2f} "
                f"top_geo: {top_str}"
            )

    conn.close()

    output = {"_description": "Exposure proxies extracted from EDGAR 10-K/10-Q text", **results}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nExtracted proxies for {len(results)} companies")
    logger.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
