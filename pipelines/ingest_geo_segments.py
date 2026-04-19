"""
Extract geographic revenue segments from SEC EDGAR 10-K filings via inline XBRL.

For each S&P 500 company, parses the latest 10-K filing to find:
1. XBRL context definitions with segment members (geographic regions)
2. Revenue values tagged with those contexts
3. Computes % of total revenue per region

Stores results in data/mappings/company_geo_exposure_xbrl.json

Rate limit: 10 req/sec max (EDGAR fair access policy).

Usage:
    python pipelines/ingest_geo_segments.py
    python pipelines/ingest_geo_segments.py --limit 10   # first 10 companies only
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import click
import requests

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_logger
from pipelines.ingest_edgar import load_sp500_tickers, build_cik_map

logger = get_logger("geo_segments")

OUTPUT_PATH = ROOT_DIR / "data" / "mappings" / "company_geo_exposure_xbrl.json"

from dotenv import load_dotenv
load_dotenv()

_session = requests.Session()
_session.headers.update({
    "User-Agent": os.getenv("EDGAR_USER_AGENT", "GeopoliticsML research@example.com"),
    "Accept-Encoding": "gzip, deflate",
})

RATE_LIMIT = 0.12  # seconds between requests


def _get(url: str) -> requests.Response:
    time.sleep(RATE_LIMIT)
    resp = _session.get(url, timeout=30)
    if resp.status_code == 429:
        logger.warning("Rate limited, waiting 10s...")
        time.sleep(10)
        resp = _session.get(url, timeout=30)
    return resp


# Common geographic segment member patterns across companies
# Maps XBRL member names to standardized region codes
GEO_MEMBER_PATTERNS = {
    # Americas
    r"america": "Americas",
    r"unitedstates|domesticsegment|ussegment": "US",
    r"canada": "Canada",
    # Europe
    r"europe|emea": "Europe",
    r"uk|unitedkingdom|greatbritain": "UK",
    # Asia Pacific
    r"china|greaterchina|prc": "China",
    r"japan": "Japan",
    r"asiapacific|apac|restofasia|otherasiapacifc": "AsiaPacific",
    r"india": "India",
    r"korea": "Korea",
    r"taiwan": "Taiwan",
    # Other
    r"middleeast|mena": "MiddleEast",
    r"africa": "Africa",
    r"latinamerica|southamerica": "LatinAmerica",
    r"international|foreignoperations|allother|other": "International",
}


def classify_segment_member(member_name: str) -> str:
    """Map an XBRL segment member name to a standardized region."""
    name_lower = member_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    for pattern, region in GEO_MEMBER_PATTERNS.items():
        if re.search(pattern, name_lower):
            return region
    return member_name  # return raw name if no match


def extract_geo_segments(cik: str, ticker: str) -> dict | None:
    """
    Extract geographic revenue segments from the latest 10-K filing.

    Returns dict mapping region → revenue (in millions) or None if no data.
    """
    # Step 1: Get latest 10-K filing accession
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = _get(submissions_url)
    if resp.status_code != 200:
        return None

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    # Find latest 10-K
    tenk_idx = None
    for i, form in enumerate(forms):
        if form == "10-K":
            tenk_idx = i
            break

    if tenk_idx is None:
        logger.debug(f"  No 10-K found for CIK {cik}")
        return None

    acc = accessions[tenk_idx].replace("-", "")
    primary_doc = primary_docs[tenk_idx]
    cik_clean = cik.lstrip("0") or "0"  # don't strip ALL zeros

    # Step 2: Download the 10-K HTML
    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc}/{primary_doc}"
    resp = _get(filing_url)
    if resp.status_code != 200:
        return None

    text = resp.text
    logger.debug(f"  Filing downloaded: {len(text)} chars")
    if len(text) < 10000:
        logger.debug(f"  Filing too short ({len(text)} chars)")
        return None

    # Step 3: Find XBRL contexts with geographic segment members
    context_pattern = re.compile(
        r'<xbrli:context[^>]*id="([^"]+)"[^>]*>(.*?)</xbrli:context>',
        re.DOTALL | re.IGNORECASE,
    )
    segment_pattern = re.compile(r'>([^<]*SegmentMember)<', re.IGNORECASE)

    # Geographic keywords — only keep segments that match these
    geo_keywords = [
        "america", "europe", "emea", "china", "japan", "asia", "pacific",
        "india", "korea", "taiwan", "uk", "kingdom", "canada", "mexico",
        "latam", "latin", "africa", "middle east", "mena", "domestic",
        "international", "foreign", "united states", "apac", "rest of",
    ]

    context_segments = {}
    for m in context_pattern.finditer(text):
        ctx_id = m.group(1)
        ctx_body = m.group(2)

        seg_match = segment_pattern.search(ctx_body)
        if seg_match:
            member = seg_match.group(1).split(":")[-1]  # remove namespace prefix
            member_lower = member.lower().replace("segmentmember", "")

            # Only keep geographic segments
            if not any(kw in member_lower for kw in geo_keywords):
                continue

            region = classify_segment_member(member)
            context_segments[ctx_id] = {
                "member": member,
                "region": region,
            }

    logger.debug(f"  Found {len(context_segments)} segment contexts")
    if not context_segments:
        return None

    # Step 4: Find revenue values for these contexts
    # Look for the first revenue concept that has segment-level data
    revenue_concepts = [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:Revenue",
    ]

    geo_revenues = {}
    # Search for ANY ix:nonFraction tag referencing our segment contexts
    # Don't filter by concept name — just find values associated with segments
    geo_ctx_ids = set(context_segments.keys())

    # Broader pattern: find all ix:nonFraction tags and check context
    nf_pattern = re.compile(
        r'<ix:nonFraction[^>]*contextRef="([^"]+)"[^>]*name="([^"]+)"[^>]*>([^<]*)</ix:nonFraction>',
        re.IGNORECASE,
    )
    # Also try reversed attribute order
    nf_pattern2 = re.compile(
        r'<ix:nonFraction[^>]*name="([^"]+)"[^>]*contextRef="([^"]+)"[^>]*>([^<]*)</ix:nonFraction>',
        re.IGNORECASE,
    )

    revenue_keywords = ["revenue", "sales", "netrevenue"]

    for pattern, ctx_group, name_group, val_group in [
        (nf_pattern, 1, 2, 3),
        (nf_pattern2, 2, 1, 3),
    ]:
        for m in pattern.finditer(text):
            ctx_ref = m.group(ctx_group)
            concept = m.group(name_group).lower()
            value_str = m.group(val_group).replace(",", "").replace("$", "").strip()

            if ctx_ref in geo_ctx_ids and any(kw in concept for kw in revenue_keywords):
                try:
                    value = int(value_str)
                    region = context_segments[ctx_ref]["region"]
                    if region not in geo_revenues or value > geo_revenues[region]:
                        geo_revenues[region] = value
                except ValueError:
                    continue

    if not geo_revenues:
        return None

    # Step 5: Compute percentages
    total = sum(geo_revenues.values())
    if total == 0:
        return None

    result = {}
    for region, revenue in sorted(geo_revenues.items(), key=lambda x: -x[1]):
        pct = round(revenue / total * 100, 1)
        result[region] = {"revenue_M": revenue, "pct": pct}

    return result


@click.command()
@click.option("--limit", default=None, type=int, help="Limit to first N tickers")
def main(limit):
    """Extract geographic revenue segments from EDGAR 10-K filings."""
    # Load EDGAR User-Agent
    from dotenv import load_dotenv
    load_dotenv()

    ua = os.getenv("EDGAR_USER_AGENT", "")
    if not ua or "example" in ua:
        logger.error("Set EDGAR_USER_AGENT in .env")
        return

    _session.headers["User-Agent"] = ua

    cik_map = build_cik_map()
    tickers = load_sp500_tickers()
    if limit:
        tickers = tickers[:limit]

    logger.info(f"Extracting geographic segments for {len(tickers)} companies...")

    results = {}
    success = 0
    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker)
        if not cik:
            continue

        logger.info(f"  [{i+1}/{len(tickers)}] {ticker}")
        geo = extract_geo_segments(cik, ticker)

        if geo:
            results[ticker] = geo
            success += 1
            regions = ", ".join(f"{r}={d['pct']}%" for r, d in geo.items())
            logger.info(f"    {regions}")
        else:
            logger.info(f"    No geographic segment data found")

    # Save results
    output = {
        "_description": "Geographic revenue segments extracted from EDGAR 10-K inline XBRL",
        "_companies_found": success,
        "_companies_attempted": len(tickers),
    }
    output.update(results)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults: {success}/{len(tickers)} companies with geographic data")
    logger.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
