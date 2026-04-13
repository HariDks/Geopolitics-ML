"""
BIS Entity List ingestion pipeline.

Fetches the US BIS Entity List from the ITA Consolidated Screening List (CSL) API,
enriches with dates from the Federal Register API, maps to taxonomy, and stores
in geopolitical_events.

Primary source: ITA CSL API (structured, no API key needed)
  https://api.trade.gov/gateway/v2/consolidated_screening_list/search?sources=Entity+List

Supplementary source: Federal Register API (for publication dates of additions)
  https://www.federalregister.gov/api/v1/documents.json

Maps to: technology_controls (primary), sanctions_financial_restrictions (secondary)

Usage:
    python pipelines/ingest_bis.py
    python pipelines/ingest_bis.py --since 2020-01-01
    python pipelines/ingest_bis.py --dry-run
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
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import (
    get_db_connection,
    get_logger,
    log_ingestion_end,
    log_ingestion_start,
    make_event_id,
    make_run_id,
)

logger = get_logger("ingest_bis")

CSL_BULK_URL = "https://data.trade.gov/downloadable_consolidated_screening_list/v1/consolidated.json"
FR_API_URL = "https://www.federalregister.gov/api/v1/documents.json"

# Countries that dominate Entity List additions (for severity scoring)
HIGH_SEVERITY_COUNTRIES = {"CN", "RU", "IR", "KP"}


# ─── CSL bulk download ────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=60))
def fetch_all_csl_entities() -> list[dict]:
    """Download the full CSL bulk JSON and filter to Entity List entries only."""
    logger.info(f"  Downloading CSL bulk JSON from {CSL_BULK_URL}")
    response = requests.get(CSL_BULK_URL, timeout=120)
    response.raise_for_status()
    data = response.json()

    all_results = data.get("results", [])
    logger.info(f"  Downloaded {len(all_results):,} total CSL entries")

    # Filter to BIS Entity List only
    entity_list = [e for e in all_results if "Entity List" in (e.get("source") or "")]
    logger.info(f"  Filtered to {len(entity_list):,} Entity List entries")
    return entity_list


# ─── Federal Register date enrichment ────────────────────────────────────────

def fetch_entity_list_fr_dates() -> dict[str, str]:
    """
    Fetch Federal Register publications for BIS Entity List rules.
    Returns mapping of FR citation → publication_date.
    E.g., {"85 FR 29853": "2020-05-18", ...}
    """
    fr_dates = {}
    page = 1

    while True:
        logger.info(f"  Fetching Federal Register page {page}")
        try:
            response = requests.get(
                FR_API_URL,
                params={
                    "conditions[agencies][]": "bureau-of-industry-and-security",
                    "conditions[term]": '"entity list"',
                    "conditions[type][]": "RULE",
                    "per_page": 100,
                    "page": page,
                    "fields[]": ["citation", "publication_date", "title"],
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"  Federal Register API error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for doc in results:
            citation = doc.get("citation", "")
            pub_date = doc.get("publication_date", "")
            if citation and pub_date:
                fr_dates[citation] = pub_date

        total_pages = data.get("total_pages", 1)
        if page >= total_pages:
            break

        page += 1
        time.sleep(0.5)

    logger.info(f"  Fetched {len(fr_dates)} Federal Register citations with dates")
    return fr_dates


# ─── Transform ───────────────────────────────────────────────────────────────

def build_event_records(
    entities: list[dict],
    fr_dates: dict[str, str],
    since_date: date | None,
) -> Iterator[dict]:
    """
    Convert CSL Entity List entries to geopolitical_events records.
    Groups entities by (FR citation, country) to create rule-level events
    rather than one event per entity.
    """
    from collections import defaultdict

    # Group by (FR citation, country)
    groups: dict[tuple, list] = defaultdict(list)
    for entity in entities:
        fr_notice = entity.get("federal_register_notice") or ""
        country = entity.get("country") or "UNKNOWN"

        # Try to get date from FR dates mapping, then start_date, then fallback
        entity_date = None
        if fr_notice and fr_notice in fr_dates:
            entity_date = fr_dates[fr_notice]
        elif entity.get("start_date"):
            entity_date = entity["start_date"]

        # Skip if we can't determine a date
        if not entity_date:
            continue

        try:
            parsed_date = date.fromisoformat(str(entity_date).split("T")[0])
        except (ValueError, AttributeError):
            continue

        if since_date and parsed_date < since_date:
            continue

        groups[(fr_notice, country, entity_date)].append(entity)

    # Convert groups to event records
    for (fr_notice, country, date_str), group_entities in groups.items():
        try:
            event_date = date.fromisoformat(str(date_str).split("T")[0])
        except (ValueError, AttributeError):
            continue

        # Determine category: tech controls for most, sanctions for some
        category = "technology_controls"
        license_text = " ".join(
            (e.get("license_requirement") or "") for e in group_entities
        ).lower()
        if "nuclear" in license_text or "chemical" in license_text or "biological" in license_text:
            subtype = "export_control_dual_use"
        elif "national security" in license_text:
            subtype = "export_control_semiconductor"
        else:
            subtype = "entity_list_addition"

        # Severity based on country and group size
        country_iso = country[:2].upper() if country else ""
        severity = 4 if country_iso in HIGH_SEVERITY_COUNTRIES else 3
        if len(group_entities) > 20:
            severity = min(5, severity + 1)

        entity_names = [e.get("name", "") for e in group_entities[:5] if e.get("name")]
        description = (
            f"BIS Entity List addition: {len(group_entities)} entities ({country}). "
            f"FR: {fr_notice}. "
        )
        if entity_names:
            description += "Includes: " + ", ".join(entity_names)

        countries = set()
        for e in group_entities:
            c = e.get("country", "")
            if c and len(c) <= 3:
                countries.add(c.upper())

        source_id = f"BIS-{fr_notice}-{country}" if fr_notice else f"BIS-{date_str}-{country}"

        yield {
            "source": "bis",
            "source_event_id": source_id,
            "event_category": category,
            "event_subtype": subtype,
            "event_date": event_date.isoformat(),
            "event_end_date": None,
            "affected_countries": json.dumps(sorted(countries)),
            "affected_sectors": json.dumps([]),
            "severity_estimate": severity,
            "onset_speed": "sudden",
            "expected_duration": "structural",
            "description_text": description[:1000],
            "source_url": f"https://www.federalregister.gov/d/{fr_notice}" if fr_notice else "",
            "goldstein_scale": None,
            "fatalities": None,
            "num_mentions": None,
            "avg_tone": None,
            "mapping_confidence": "high",
        }


# ─── Store ────────────────────────────────────────────────────────────────────

def store_events(conn, records: list[dict]) -> tuple[int, int]:
    """Bulk insert with INSERT OR IGNORE."""
    rows = []
    for record in records:
        event_id = make_event_id("bis", date.fromisoformat(record["event_date"]))
        rows.append((
            event_id, record["source"], record["source_event_id"],
            record["event_category"], record["event_subtype"],
            record["event_date"], record["event_end_date"],
            record["affected_countries"], record["affected_sectors"],
            record["severity_estimate"], record["onset_speed"],
            record["expected_duration"], record["description_text"],
            record["source_url"], record["goldstein_scale"],
            record["fatalities"], record["num_mentions"],
            record["avg_tone"], record["mapping_confidence"],
        ))

    if not rows:
        return 0, 0

    conn.executemany(
        """INSERT OR IGNORE INTO geopolitical_events (
            event_id, source, source_event_id, event_category, event_subtype,
            event_date, event_end_date, affected_countries, affected_sectors,
            severity_estimate, onset_speed, expected_duration, description_text,
            source_url, goldstein_scale, fatalities, num_mentions, avg_tone,
            mapping_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows), 0


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--since", default=None, help="Only include entities added on/after YYYY-MM-DD")
@click.option("--dry-run", is_flag=True, help="Fetch but do not write to DB")
@click.option("--skip-fr-dates", is_flag=True, help="Skip Federal Register date enrichment (faster)")
def main(since: str | None, dry_run: bool, skip_fr_dates: bool) -> None:
    """Ingest BIS Entity List into the geopolitical events database."""
    since_date = date.fromisoformat(since) if since else None
    if since_date:
        logger.info(f"BIS ingestion (entities added since {since_date})")
    else:
        logger.info("BIS ingestion (all Entity List entries)")

    conn = get_db_connection()
    run_id = make_run_id("bis")
    today = date.today()
    log_ingestion_start(conn, run_id, "bis", since_date or date(2000, 1, 1), today)

    try:
        # Step 1: Fetch all Entity List entries from CSL
        entities = fetch_all_csl_entities()
        logger.info(f"Fetched {len(entities):,} Entity List entries from CSL")

        # Step 2: Enrich with Federal Register dates
        fr_dates = {}
        if not skip_fr_dates:
            fr_dates = fetch_entity_list_fr_dates()
        else:
            logger.info("Skipping Federal Register date enrichment")

        # Step 3: Transform and store
        records = list(build_event_records(entities, fr_dates, since_date))

        cat_counts = {}
        for r in records:
            cat = r["event_category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        logger.info(f"Generated {len(records)} events from {len(entities)} entities")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {cnt}")

        if not dry_run:
            stored, skipped = store_events(conn, records)
            logger.info(f"Stored {stored}, skipped {skipped}")
            log_ingestion_end(conn, run_id, len(entities), stored, skipped, "success")
        else:
            logger.info("Dry run — not writing to DB")
            log_ingestion_end(conn, run_id, len(entities), 0, 0, "dry_run")

    except Exception as e:
        log_ingestion_end(conn, run_id, 0, 0, 0, "failed", str(e))
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
