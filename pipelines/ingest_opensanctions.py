"""
OpenSanctions ingestion pipeline.

Downloads the consolidated sanctions database (OFAC + EU + UK + UN + 200 watchlists)
and maps entries to our geopolitical events taxonomy.

Coverage: ~3x more than OFAC alone.
Data: free, updated daily.
Source: https://opensanctions.org

Usage:
    python pipelines/ingest_opensanctions.py
    python pipelines/ingest_opensanctions.py --limit 5000  # first 5000 entries
"""

import csv
import io
import sys
from datetime import date
from pathlib import Path

import click
import requests

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger, make_event_id, make_run_id, log_ingestion_start, log_ingestion_end

logger = get_logger("opensanctions")

# OpenSanctions simplified targets CSV — one row per sanctioned entity
DATA_URL = "https://data.opensanctions.org/datasets/latest/default/targets.simple.csv"


def download_targets(limit: int = None) -> list[dict]:
    """Download OpenSanctions targets CSV (streaming to handle 460MB)."""
    logger.info(f"Downloading OpenSanctions targets from {DATA_URL}...")

    session = requests.Session()
    resp = session.get(DATA_URL, stream=True, timeout=60)
    resp.raise_for_status()

    # Stream CSV — read line by line to handle 460MB file
    entries = []
    line_iter = resp.iter_lines(decode_unicode=True)
    header_line = next(line_iter)
    header_reader = csv.reader(io.StringIO(header_line))
    fieldnames = next(header_reader)

    # Process remaining lines as CSV rows
    def row_generator():
        for line in line_iter:
            try:
                parsed = next(csv.reader(io.StringIO(line)))
                if len(parsed) == len(fieldnames):
                    yield dict(zip(fieldnames, parsed))
            except Exception:
                continue

    reader = row_generator()

    count = 0
    for row in reader:
        if limit and count >= limit:
            break

        schema = row.get("schema", "")
        if schema not in ("Person", "Organization", "Company", "LegalEntity"):
            continue

        name = row.get("name", "")
        dataset = row.get("dataset", "")
        if not name or not dataset:
            continue

        entry = {
            "id": row.get("id", ""),
            "schema": schema,
            "name": name,
            "datasets": dataset,
            "first_seen": row.get("first_seen", ""),
            "last_seen": row.get("last_seen", ""),
            "countries": row.get("countries", ""),
            "sanctions": row.get("sanctions", ""),
        }

        entries.append(entry)
        count += 1

    logger.info(f"Downloaded {len(entries)} sanctioned entities")
    return entries


def map_to_taxonomy(entry: dict) -> dict:
    """Map an OpenSanctions entry to our event taxonomy."""
    datasets = entry.get("datasets", "").lower()
    countries = entry.get("countries", "")

    # Determine program/source
    if "ofac" in datasets or "us_" in datasets:
        program = "OFAC/US"
    elif "eu_" in datasets:
        program = "EU"
    elif "un_" in datasets:
        program = "UN"
    elif "uk_" in datasets:
        program = "UK"
    else:
        program = "Other"

    # Build description
    name = entry.get("name", "Unknown")
    schema = entry.get("schema", "Entity")
    country_str = countries[:50] if countries else "unknown"
    desc = f"OpenSanctions: {schema} '{name}' sanctioned by {program}. Countries: {country_str}."

    # Parse date
    first_seen = entry.get("first_seen", "")
    event_date = first_seen[:10] if first_seen and len(first_seen) >= 10 else None

    return {
        "source": "opensanctions",
        "source_event_id": entry.get("id", ""),
        "event_category": "sanctions_financial_restrictions",
        "event_subtype": f"entity_listing_{program.lower()}",
        "event_date": event_date,
        "affected_countries": countries,
        "description_text": desc[:500],
        "severity_estimate": 3,
    }


@click.command()
@click.option("--limit", default=10000, type=int, help="Max entries to ingest (default 10000)")
def main(limit):
    """Ingest OpenSanctions consolidated sanctions data."""
    entries = download_targets(limit=limit)

    if not entries:
        logger.error("No entries downloaded")
        return

    conn = get_db_connection()
    run_id = make_run_id("opensanctions")
    log_ingestion_start(conn, run_id, "opensanctions", date(2020, 1, 1), date.today())

    stored = 0
    skipped = 0
    for entry in entries:
        mapped = map_to_taxonomy(entry)
        if not mapped["event_date"]:
            skipped += 1
            continue

        event_id = make_event_id("opensanctions", date.fromisoformat(mapped["event_date"]))

        try:
            conn.execute(
                """INSERT OR IGNORE INTO geopolitical_events
                   (event_id, source, source_event_id, event_category, event_subtype,
                    event_date, affected_countries, description_text, severity_estimate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, mapped["source"], mapped["source_event_id"],
                 mapped["event_category"], mapped["event_subtype"],
                 mapped["event_date"], mapped["affected_countries"],
                 mapped["description_text"], mapped["severity_estimate"]),
            )
            stored += 1
        except Exception:
            skipped += 1

    conn.commit()
    log_ingestion_end(conn, run_id, len(entries), stored, skipped, "success")
    conn.close()

    logger.info(f"\nResults: {stored} stored, {skipped} skipped out of {len(entries)} entries")


if __name__ == "__main__":
    main()
