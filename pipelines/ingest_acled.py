"""
ACLED ingestion pipeline.

Fetches ACLED (Armed Conflict Location & Event Data) via their REST API,
maps event_type/sub_event_type to taxonomy, and stores in geopolitical_events.

ACLED API docs: https://developer.acleddata.com/rehd/cms/views/acled_api/documents/API-User-Guide.pdf
Registration: https://developer.acleddata.com/ (free for research, 1-2 day approval)

Usage:
    python pipelines/ingest_acled.py --start 2020-01-01 --end 2025-12-31
    python pipelines/ingest_acled.py --start 2022-02-01 --end 2022-04-30 --countries UA,RU

Note:
    ACLED API returns max 500 records per request. This pipeline handles pagination.
    Rate limit: 1 request per second for free tier.
"""

import json
import os
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
    acled_to_taxonomy,
    event_exists,
    get_db_connection,
    get_logger,
    log_ingestion_end,
    log_ingestion_start,
    make_event_id,
    make_run_id,
)

logger = get_logger("ingest_acled")

ACLED_API_BASE = "https://acleddata.com/api/acled/read"
ACLED_AUTH_URL = "https://acleddata.com/oauth/token"
ACLED_PAGE_SIZE = 5000         # new API supports up to 5000 rows per call
ACLED_RATE_LIMIT_DELAY = 0.0   # pagination doesn't count toward rate limits per ACLED docs

# Minimum fatalities threshold for noise filtering (0 = include all)
MIN_FATALITY_FILTER = 0

# ACLED event_types to ingest. "Protests" with sub_event_type="Peaceful protest"
# are extremely noisy — filtered below.
INCLUDED_EVENT_TYPES = {
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Protests",
    "Riots",
    "Strategic developments",
}


# ─── OAuth authentication ─────────────────────────────────────────────────────

def get_acled_token(email: str, password: str) -> str:
    """
    Authenticate with ACLED OAuth endpoint and return a Bearer access token.
    Token is valid for 24 hours.
    Docs: https://acleddata.com/api-documentation/getting-started
    """
    logger.info("Authenticating with ACLED OAuth...")
    response = requests.post(
        ACLED_AUTH_URL,
        data={
            "username": email,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise ValueError(f"No access_token in ACLED auth response: {response.text}")
    logger.info("ACLED OAuth token obtained (valid 24 hours)")
    return token


# ─── API fetch ────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def fetch_acled_page(
    token: str,
    start_date: date,
    end_date: date,
    page: int = 1,
    countries: list[str] | None = None,
) -> dict:
    """Fetch one page of ACLED events using Bearer token auth."""
    params = {
        "event_date": f"{start_date.isoformat()}|{end_date.isoformat()}",
        "event_date_where": "BETWEEN",
        "limit": ACLED_PAGE_SIZE,
        "page": page,
        "fields": "event_id_cnty|event_date|event_type|sub_event_type|actor1|actor2|"
                  "inter1|inter2|country|iso|admin1|location|latitude|longitude|"
                  "fatalities|notes|source|source_scale",
    }
    if countries:
        params["country"] = "|".join(countries)
        params["country_where"] = "IN"

    response = requests.get(
        ACLED_API_BASE,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_all_acled_events(
    token: str,
    start_date: date,
    end_date: date,
    countries: list[str] | None = None,
) -> list[dict]:
    """Paginate through all ACLED results for the given date range."""
    all_events = []
    page = 1

    while True:
        logger.info(f"  Fetching ACLED page {page} ({start_date} → {end_date})")
        try:
            response = fetch_acled_page(token, start_date, end_date, page, countries)
        except Exception as e:
            logger.error(f"  ACLED API error on page {page}: {e}")
            break

        data = response.get("data", [])
        if not data:
            break

        all_events.extend(data)
        total_count = response.get("total_count", 0)
        logger.info(f"  Page {page}: {len(data)} events (total so far: {len(all_events)}/{total_count})")

        if len(data) < ACLED_PAGE_SIZE:
            break  # last page (partial page means no more data)

        page += 1
        time.sleep(ACLED_RATE_LIMIT_DELAY)

    return all_events


# ─── Filter ───────────────────────────────────────────────────────────────────

def is_relevant_event(event: dict) -> bool:
    """
    Filter out low-signal ACLED events.
    Key exclusions:
    - Peaceful protests with 0 fatalities (too noisy)
    - Events with no country data
    """
    event_type = event.get("event_type", "")
    sub_event_type = event.get("sub_event_type", "")
    fatalities = int(event.get("fatalities", 0) or 0)
    country = event.get("country", "")

    if not country:
        return False

    if event_type == "Protests" and sub_event_type == "Peaceful protest" and fatalities == 0:
        return False

    if event_type not in INCLUDED_EVENT_TYPES:
        return False

    return True


# ─── Transform ───────────────────────────────────────────────────────────────

def build_event_records(acled_events: list[dict]) -> Iterator[dict]:
    """
    Convert ACLED events to geopolitical_events schema.
    Applies ACLED → taxonomy mapping. Skips unmapped events.
    """
    for event in acled_events:
        if not is_relevant_event(event):
            continue

        event_type = event.get("event_type", "")
        sub_event_type = event.get("sub_event_type")
        mapping = acled_to_taxonomy(event_type, sub_event_type)
        if mapping is None:
            continue

        # Parse date
        date_str = event.get("event_date", "")
        try:
            event_date = date.fromisoformat(date_str.split(" ")[0])
        except (ValueError, AttributeError):
            continue

        # Severity: take max of default subtype severity and fatality-based severity
        fatalities = int(event.get("fatalities", 0) or 0)
        fatality_severity = _fatality_to_severity(fatalities)
        default_severity = mapping.get("default_severity", 3)
        severity = max(default_severity, fatality_severity)

        # Country — ACLED uses full names; convert to ISO via iso field
        iso_code = str(event.get("iso", "")).strip()
        country = event.get("country", "")
        countries = [iso_code] if iso_code and iso_code.isdigit() is False else [country[:2].upper()]

        notes = event.get("notes", "") or ""
        source_url = event.get("source", "") or ""

        yield {
            "source": "acled",
            "source_event_id": str(event.get("event_id_cnty", "")),
            "event_category": mapping["taxonomy_category"],
            "event_subtype": _acled_subtype_key(event_type, sub_event_type),
            "event_date": event_date.isoformat(),
            "event_end_date": None,
            "affected_countries": json.dumps(countries),
            "affected_sectors": json.dumps([]),   # ACLED has no sector info
            "severity_estimate": severity,
            "onset_speed": mapping.get("onset_speed", "sudden"),
            "expected_duration": None,
            "description_text": notes[:1000] if notes else None,
            "source_url": source_url[:500],
            "goldstein_scale": None,
            "fatalities": fatalities,
            "num_mentions": None,
            "avg_tone": None,
            "mapping_confidence": mapping.get("confidence", "medium"),
        }


def _fatality_to_severity(fatalities: int) -> int:
    if fatalities >= 500:
        return 5
    elif fatalities >= 50:
        return 4
    elif fatalities >= 10:
        return 3
    elif fatalities >= 1:
        return 2
    else:
        return 1


def _acled_subtype_key(event_type: str, sub_event_type: str | None) -> str:
    """Convert ACLED strings to snake_case subtype key."""
    base = event_type.lower().replace(" ", "_").replace("/", "_")
    if sub_event_type:
        sub = sub_event_type.lower().replace(" ", "_").replace("/", "_")
        return f"{base}__{sub}"
    return base


# ─── Store ────────────────────────────────────────────────────────────────────

def store_events(conn, records: list[dict]) -> tuple[int, int]:
    """
    Bulk insert using INSERT OR IGNORE — skips duplicates at DB level,
    no per-record SELECT needed. ~100x faster than individual inserts.
    """
    rows = []
    for record in records:
        event_id = make_event_id("acled", date.fromisoformat(record["event_date"]))
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
    stored = conn.execute("SELECT changes()").fetchone()[0]
    skipped = len(rows) - stored
    return stored, skipped


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option("--countries", default=None, help="Comma-separated ISO codes to filter, e.g. UA,RU,SD")
@click.option("--dry-run", is_flag=True, help="Fetch and filter but do not write to DB")
def main(start: str, end: str, countries: str | None, dry_run: bool) -> None:
    """Ingest ACLED events for a date range into the geopolitical events database."""
    email = os.getenv("ACLED_EMAIL")
    password = os.getenv("ACLED_PASSWORD")

    if not email or not password:
        logger.error(
            "ACLED_EMAIL and ACLED_PASSWORD must be set in .env. "
            "Register at https://acleddata.com/myacled"
        )
        raise SystemExit(1)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    country_list = [c.strip() for c in countries.split(",")] if countries else None

    logger.info(
        f"ACLED ingestion: {start_date} → {end_date}"
        + (f", countries={country_list}" if country_list else ", all countries")
    )

    token = get_acled_token(email, password)

    conn = get_db_connection()
    run_id = make_run_id("acled")
    log_ingestion_start(conn, run_id, "acled", start_date, end_date)

    try:
        acled_events = fetch_all_acled_events(token, start_date, end_date, country_list)
        records = list(build_event_records(acled_events))
        logger.info(f"Fetched {len(acled_events):,} raw ACLED events → {len(records)} mapped")

        if not dry_run:
            stored, skipped = store_events(conn, records)
            logger.info(f"Stored {stored}, skipped {skipped} duplicates")
            log_ingestion_end(conn, run_id, len(acled_events), stored, skipped, "success")
        else:
            logger.info("Dry run — not writing to DB")
            log_ingestion_end(conn, run_id, len(acled_events), 0, 0, "dry_run")

    except Exception as e:
        log_ingestion_end(conn, run_id, 0, 0, 0, "failed", str(e))
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
