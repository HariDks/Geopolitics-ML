"""
SEC EDGAR ingestion pipeline for corporate outcome data.

Three components:
1. XBRL Company Facts — quarterly financials (revenue, COGS, operating income)
2. 10-K/10-Q text — Risk Factors + MD&A for geopolitical mention extraction
3. EFTS search — find filings mentioning specific geopolitical terms

Key finding: Earnings call TRANSCRIPTS are NOT in EDGAR. Companies don't file them.
Instead, we use 10-K/10-Q text which contains structured geopolitical risk language
in Item 1A (Risk Factors) and Item 2 (MD&A).

EDGAR rate limit: 10 req/sec. User-Agent header required.
XBRL frames endpoint: one call per concept per quarter = very efficient for bulk.

Usage:
    python pipelines/ingest_edgar.py --step financials --start 2022 --end 2025
    python pipelines/ingest_edgar.py --step filings --tickers AAPL,MSFT,NVDA
    python pipelines/ingest_edgar.py --step search --query "tariff"
    python pipelines/ingest_edgar.py --step all --start 2022 --end 2025
"""

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterator

import click
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("ingest_edgar")

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "edgar"

# EDGAR endpoints
EDGAR_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_XBRL_FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{frame}.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# Rate limiting: 10 req/sec max
EDGAR_RATE_LIMIT = 0.11  # seconds between requests (slightly over 10/sec)

# Reuse a single TCP connection
_session = requests.Session()
_session.headers.update({
    "User-Agent": os.getenv("EDGAR_USER_AGENT", "GeopoliticsML research@example.com"),
    "Accept-Encoding": "gzip, deflate",
})

# XBRL concept names for key financials (with fallbacks)
FINANCIAL_CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "Revenue",
        "SalesRevenueNet",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
    ],
}

# Geopolitical keywords for EFTS search and text extraction
GEO_KEYWORDS = {
    "trade_policy_actions": [
        "tariff", "trade war", "customs duty", "import ban", "export restriction",
        "trade policy", "trade agreement", "Section 301", "Section 232",
        "countervailing duty", "anti-dumping", "trade remedy",
    ],
    "sanctions_financial_restrictions": [
        "sanction", "OFAC", "entity list", "embargo", "SWIFT",
        "SDN list", "blocked persons", "asset freeze",
    ],
    "armed_conflict_instability": [
        "conflict", "war", "Red Sea", "shipping disruption", "military",
        "invasion", "armed conflict", "terrorism", "piracy",
    ],
    "regulatory_sovereignty_shifts": [
        "regulatory change", "data localization", "foreign investment screening",
        "local content", "FDI restriction", "nationalization", "CFIUS",
    ],
    "technology_controls": [
        "export control", "chip ban", "technology restriction", "CHIPS Act",
        "dual-use", "BIS", "entity list", "semiconductor restriction",
    ],
    "resource_energy_disruptions": [
        "energy crisis", "commodity price", "OPEC", "oil price",
        "energy security", "critical mineral", "supply shock",
    ],
    "political_transitions_volatility": [
        "election", "regime change", "political instability", "coup",
        "constitutional crisis", "populist",
    ],
    "institutional_alliance_realignment": [
        "WTO", "trade bloc", "NATO", "treaty withdrawal",
        "multilateral", "alliance", "Brexit",
    ],
}

# Flatten all keywords for text scanning
ALL_GEO_KEYWORDS = []
for keywords in GEO_KEYWORDS.values():
    ALL_GEO_KEYWORDS.extend(keywords)
ALL_GEO_KEYWORDS = list(set(ALL_GEO_KEYWORDS))


# ─── Rate-limited request ─────────────────────────────────────────────────────

def _edgar_get(url: str, params: dict | None = None) -> requests.Response:
    """Make a rate-limited GET request to EDGAR."""
    time.sleep(EDGAR_RATE_LIMIT)
    response = _session.get(url, params=params, timeout=30)
    if response.status_code == 429:
        logger.warning("Rate limited by EDGAR, waiting 10 seconds...")
        time.sleep(10)
        response = _session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response


# ─── Step 1: S&P 500 CIK mapping ──────────────────────────────────────────────

def load_sp500_tickers() -> list[str]:
    """Load S&P 500 tickers. Uses a static list supplemented by yfinance if available."""
    # Core S&P 500 — top 100 by weight (covers ~70% of index)
    # Full list can be fetched from Wikipedia or yfinance
    top_tickers = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
        "CRM", "DHR", "LIN", "NEE", "TXN", "BMY", "PM", "UPS", "RTX", "AMGN",
        "HON", "ORCL", "QCOM", "MS", "GS", "SCHW", "LOW", "INTC", "ELV", "SPGI",
        "BA", "CAT", "GE", "INTU", "BLK", "DE", "ISRG", "GILD", "MDT", "ADP",
        "SYK", "ADM", "BKNG", "PLD", "MMC", "CB", "CI", "MDLZ", "TJX", "DUK",
        "ZTS", "SO", "CL", "BDX", "CME", "PNC", "NOC", "USB", "MO", "TGT",
        "EQIX", "FDX", "GM", "F", "NKE", "LMT", "COP", "SLB", "EOG", "PSA",
        "AEP", "GD", "LRCX", "KLAC", "AMAT", "MU", "ABNB", "WM", "HUM", "SWK",
    ]
    return top_tickers


def build_cik_map() -> dict[str, str]:
    """Build ticker → zero-padded CIK mapping from EDGAR."""
    logger.info("Building ticker → CIK mapping from EDGAR...")
    response = _edgar_get(EDGAR_CIK_URL)
    data = response.json()

    cik_map = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        cik_map[ticker] = cik

    logger.info(f"  Loaded {len(cik_map)} ticker → CIK mappings")
    return cik_map


# ─── Step 2: XBRL financial data ──────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def fetch_company_facts(cik: str) -> dict | None:
    """Fetch all XBRL facts for a company."""
    url = EDGAR_XBRL_FACTS_URL.format(cik=cik)
    try:
        response = _edgar_get(url)
        return response.json()
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def extract_quarterly_financials(facts: dict, ticker: str) -> list[dict]:
    """
    Extract quarterly financial data from XBRL company facts.
    Returns list of dicts with: ticker, quarter, revenue, cogs, operating_income, etc.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    # Collect data points by (fiscal_year, fiscal_period)
    quarters: dict[str, dict] = {}

    for metric_name, concept_names in FINANCIAL_CONCEPTS.items():
        for concept in concept_names:
            concept_data = us_gaap.get(concept, {})
            units = concept_data.get("units", {})

            # Try USD first, then USD/shares for EPS
            unit_key = "USD" if metric_name != "eps_diluted" else "USD/shares"
            entries = units.get(unit_key, [])

            for entry in entries:
                form = entry.get("form", "")
                fp = entry.get("fp", "")
                fy = entry.get("fy")

                # Only quarterly data from 10-Q, or annual from 10-K
                if form == "10-Q" and fp in ("Q1", "Q2", "Q3"):
                    key = f"{fy}{fp}"
                elif form == "10-K" and fp == "FY":
                    key = f"{fy}FY"
                else:
                    continue

                if key not in quarters:
                    quarters[key] = {
                        "ticker": ticker,
                        "fiscal_year": fy,
                        "fiscal_period": fp,
                        "quarter": key,
                        "filed": entry.get("filed"),
                        "period_end": entry.get("end"),
                    }

                # Only set if not already set (first matching concept wins)
                if metric_name not in quarters[key]:
                    quarters[key][metric_name] = entry.get("val")

            if any(metric_name in q for q in quarters.values()):
                break  # Found data with this concept name, skip fallbacks

    return list(quarters.values())


def fetch_all_financials(tickers: list[str], cik_map: dict, start_year: int, end_year: int) -> list[dict]:
    """Fetch quarterly financials for all tickers via XBRL API."""
    all_financials = []

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker)
        if not cik:
            continue

        logger.info(f"  [{i+1}/{len(tickers)}] Fetching XBRL for {ticker} (CIK {cik})")

        facts = fetch_company_facts(cik)
        if not facts:
            continue

        quarters = extract_quarterly_financials(facts, ticker)

        # Filter to date range
        for q in quarters:
            fy = q.get("fiscal_year")
            if fy and start_year <= fy <= end_year:
                all_financials.append(q)

    logger.info(f"Total quarterly records: {len(all_financials)}")
    return all_financials


# ─── Step 3: Filing text download and parsing ─────────────────────────────────

def fetch_recent_filings(cik: str) -> list[dict]:
    """Fetch recent filing metadata for a company from EDGAR submissions API."""
    url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    try:
        response = _edgar_get(url)
        data = response.json()
    except Exception:
        return []

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    filings = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for j in range(len(forms)):
        filings.append({
            "form": forms[j] if j < len(forms) else "",
            "filing_date": dates[j] if j < len(dates) else "",
            "accession": accessions[j] if j < len(accessions) else "",
            "primary_document": primary_docs[j] if j < len(primary_docs) else "",
        })

    return filings


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
def download_filing_text(cik: str, accession: str, document: str) -> str | None:
    """Download the text/HTML of a specific filing document."""
    accession_clean = accession.replace("-", "")
    url = EDGAR_FILING_URL.format(cik=cik.lstrip("0"), accession=accession_clean, document=document)
    try:
        response = _edgar_get(url)
        return response.text
    except Exception as e:
        logger.warning(f"  Failed to download filing: {e}")
        return None


def _parse_filing_text(html_text: str) -> str:
    """Parse HTML/XML filing to plain text, suppressing warnings."""
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_section(text: str, start_patterns: list[str], end_patterns: list[str],
                     min_length: int = 2000) -> str:
    """
    Extract a section from filing text between start and end patterns.
    Skips Table of Contents matches (short sections) by requiring min_length.
    Searches for ALL matches and takes the longest one (the actual section, not ToC).
    """
    best = ""
    for start_pat in start_patterns:
        for match in re.finditer(start_pat, text, re.IGNORECASE):
            remaining = text[match.end():]
            # Find the end boundary
            for end_pat in end_patterns:
                end_match = re.search(end_pat, remaining[min_length:], re.IGNORECASE)
                if end_match:
                    section = remaining[:end_match.start() + min_length].strip()
                    if len(section) > len(best):
                        best = section
                    break
            # If no end pattern found, take up to 50K chars
            if not best and len(remaining) > min_length:
                best = remaining[:50000].strip()

    return best[:50000]


def extract_risk_factors(html_text: str) -> str:
    """Extract Item 1A Risk Factors section from a 10-K/10-Q HTML filing."""
    text = _parse_filing_text(html_text)
    return _extract_section(
        text,
        start_patterns=[
            r"Item\s*1A[\.\s\n]*Risk\s*Factors\n",
            r"RISK\s*FACTORS\n",
        ],
        end_patterns=[
            r"\nItem\s*1B",
            r"\nItem\s*1C",
            r"\nItem\s*2[\.\s]",
            r"\nPart\s*II",
            r"\nUNRESOLVED\s*STAFF",
            r"\nPROPERTIES",
        ],
        min_length=2000,
    )


def extract_mda(html_text: str) -> str:
    """Extract MD&A section from a 10-K/10-Q HTML filing."""
    text = _parse_filing_text(html_text)
    return _extract_section(
        text,
        start_patterns=[
            r"Item\s*7[\.\s\n]*Management.s\s*Discussion",
            r"Item\s*2[\.\s\n]*Management.s\s*Discussion",
            r"MANAGEMENT.S\s*DISCUSSION",
        ],
        end_patterns=[
            r"\nItem\s*7A",
            r"\nItem\s*3[\.\s]",
            r"\nItem\s*8[\.\s]",
            r"\nQUANTITATIVE\s*AND\s*QUALITATIVE",
            r"\nFINANCIAL\s*STATEMENTS",
        ],
        min_length=2000,
    )


# ─── Step 4: Geopolitical mention extraction ──────────────────────────────────

def extract_geopolitical_mentions(text: str, ticker: str, filing_date: str, form_type: str) -> list[dict]:
    """
    Scan filing text for geopolitical keywords and extract context windows.
    Returns list of mention dicts with category, text excerpt, and metadata.
    """
    mentions = []
    text_lower = text.lower()

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

    for para in paragraphs:
        para_lower = para.lower()
        matched_categories = set()

        for category, keywords in GEO_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in para_lower:
                    matched_categories.add(category)
                    break

        if matched_categories:
            mentions.append({
                "ticker": ticker,
                "filing_date": filing_date,
                "form_type": form_type,
                "mention_text": para[:500],
                "categories": list(matched_categories),
                "primary_category": list(matched_categories)[0],
                "keyword_count": sum(
                    1 for kw in ALL_GEO_KEYWORDS if kw.lower() in para_lower
                ),
            })

    return mentions


# ─── Step 5: Store results ────────────────────────────────────────────────────

def store_financials(conn, financials: list[dict]) -> int:
    """Store quarterly financials in a new table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS company_financials (
            ticker TEXT NOT NULL,
            quarter TEXT NOT NULL,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            period_end TEXT,
            filed TEXT,
            revenue REAL,
            cogs REAL,
            operating_income REAL,
            net_income REAL,
            eps_diluted REAL,
            PRIMARY KEY (ticker, quarter)
        )
    """)

    stored = 0
    for f in financials:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO company_financials
                   (ticker, quarter, fiscal_year, fiscal_period, period_end, filed,
                    revenue, cogs, operating_income, net_income, eps_diluted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f.get("ticker"), f.get("quarter"), f.get("fiscal_year"),
                    f.get("fiscal_period"), f.get("period_end"), f.get("filed"),
                    f.get("revenue"), f.get("cogs"), f.get("operating_income"),
                    f.get("net_income"), f.get("eps_diluted"),
                ),
            )
            stored += 1
        except Exception:
            pass

    conn.commit()
    return stored


def store_mentions(conn, mentions: list[dict]) -> int:
    """Store geopolitical mentions in a new table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geopolitical_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            filing_date TEXT,
            form_type TEXT,
            mention_text TEXT,
            primary_category TEXT,
            categories TEXT,
            keyword_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    stored = 0
    for m in mentions:
        conn.execute(
            """INSERT INTO geopolitical_mentions
               (ticker, filing_date, form_type, mention_text, primary_category,
                categories, keyword_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                m.get("ticker"), m.get("filing_date"), m.get("form_type"),
                m.get("mention_text"), m.get("primary_category"),
                json.dumps(m.get("categories", [])), m.get("keyword_count"),
            ),
        )
        stored += 1

    conn.commit()
    return stored


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--step", required=True,
              type=click.Choice(["financials", "filings", "search", "all"]),
              help="Which step to run")
@click.option("--start", default=2022, type=int, help="Start fiscal year")
@click.option("--end", default=2025, type=int, help="End fiscal year")
@click.option("--tickers", default=None, help="Comma-separated tickers (default: S&P 500 top 100)")
@click.option("--limit", default=None, type=int, help="Limit number of tickers to process")
@click.option("--query", default=None, help="Search query for EFTS (step=search)")
def main(step: str, start: int, end: int, tickers: str | None, limit: int | None, query: str | None) -> None:
    """Ingest SEC EDGAR data for corporate outcome analysis."""

    # Validate User-Agent
    ua = os.getenv("EDGAR_USER_AGENT", "")
    if not ua or ua == "YourName your_email@example.com":
        logger.error(
            "EDGAR_USER_AGENT must be set in .env with your name and email. "
            "Format: 'YourName your.email@domain.com'"
        )
        raise SystemExit(1)

    _session.headers["User-Agent"] = ua

    # Build CIK map
    cik_map = build_cik_map()

    # Get ticker list
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = load_sp500_tickers()

    if limit:
        ticker_list = ticker_list[:limit]

    logger.info(f"Processing {len(ticker_list)} tickers, years {start}-{end}")

    conn = get_db_connection()

    if step in ("financials", "all"):
        logger.info("=" * 60)
        logger.info("STEP: Fetching XBRL quarterly financials")
        logger.info("=" * 60)
        financials = fetch_all_financials(ticker_list, cik_map, start, end)
        stored = store_financials(conn, financials)
        logger.info(f"Stored {stored} quarterly financial records")

    if step in ("filings", "all"):
        logger.info("=" * 60)
        logger.info("STEP: Downloading and parsing 10-K/10-Q filings")
        logger.info("=" * 60)
        total_mentions = 0

        for i, ticker in enumerate(ticker_list):
            cik = cik_map.get(ticker)
            if not cik:
                continue

            logger.info(f"  [{i+1}/{len(ticker_list)}] {ticker}")

            filings = fetch_recent_filings(cik)
            # Filter to 10-K and 10-Q in date range
            relevant = [
                f for f in filings
                if f["form"] in ("10-K", "10-Q")
                and f.get("filing_date", "")[:4].isdigit()
                and start <= int(f["filing_date"][:4]) <= end
            ]

            for filing in relevant[:8]:  # Max 8 filings per company (2 years of 10-K + 10-Q)
                html = download_filing_text(cik, filing["accession"], filing["primary_document"])
                if not html:
                    continue

                # Extract sections
                risk_factors = extract_risk_factors(html)
                mda = extract_mda(html)
                combined_text = risk_factors + "\n\n" + mda

                if len(combined_text) < 500:
                    continue

                # Extract mentions
                mentions = extract_geopolitical_mentions(
                    combined_text, ticker, filing["filing_date"], filing["form"]
                )

                if mentions:
                    stored = store_mentions(conn, mentions)
                    total_mentions += stored
                    logger.info(f"    {filing['form']} ({filing['filing_date']}): {stored} geo mentions")

        logger.info(f"Total geopolitical mentions extracted: {total_mentions}")

    if step == "search":
        if not query:
            logger.error("--query is required for step=search")
            raise SystemExit(1)

        logger.info("=" * 60)
        logger.info(f"STEP: EFTS search for '{query}'")
        logger.info("=" * 60)

        params = {
            "q": f'"{query}"',
            "forms": "10-K,10-Q",
            "dateRange": "custom",
            "startdt": f"{start}-01-01",
            "enddt": f"{end}-12-31",
        }
        response = _edgar_get(EDGAR_EFTS_URL, params=params)
        data = response.json()

        total = data.get("hits", {}).get("total", {}).get("value", 0)
        logger.info(f"Found {total} filings mentioning '{query}'")

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:20]:
            src = hit.get("_source", {})
            logger.info(
                f"  {src.get('display_names', ['?'])[0][:40]} | "
                f"{src.get('form', '?')} | {src.get('file_date', '?')}"
            )

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
