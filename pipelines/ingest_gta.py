"""
Global Trade Alert (GTA) ingestion pipeline.

Fetches trade-distorting policy interventions from the GTA API, maps
intervention_type to our taxonomy, and stores in geopolitical_events.

GTA is the primary source for:
- Trade Policy Actions (tariffs, quotas, bans, subsidies, trade remedies)
- Regulatory & Sovereignty Shifts (FDI screening, local content, data localization)
- Secondary source for Sanctions & Financial Restrictions

API docs: https://github.com/global-trade-alert/docs/blob/main/.api/gta-data.md
Register: https://globaltradealert.org/api-access (demo key is free)

Usage:
    python pipelines/ingest_gta.py --start 2020-01-01 --end 2025-12-31
    python pipelines/ingest_gta.py --start 2020-01-01 --end 2025-12-31 --red-only
    python pipelines/ingest_gta.py --start 2022-01-01 --end 2022-12-31 --dry-run
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
    get_db_connection,
    get_logger,
    load_gta_mapping,
    log_ingestion_end,
    log_ingestion_start,
    make_event_id,
    make_run_id,
)

logger = get_logger("ingest_gta")

GTA_API_URL = "https://api.globaltradealert.org/api/v1/data/"
GTA_PAGE_SIZE = 1000
GTA_RATE_LIMIT_DELAY = 0.5  # seconds between paginated requests (be polite)

# GTA evaluation IDs
GTA_EVAL_RED = 1      # harmful / discriminatory
GTA_EVAL_AMBER = 2    # likely discriminatory
GTA_EVAL_GREEN = 3    # liberalizing

GTA_EVAL_NAMES = {1: "Red", 2: "Amber", 3: "Green"}


# ─── Taxonomy mapping ────────────────────────────────────────────────────────

def build_intervention_type_lookup() -> dict:
    """
    Build a flat lookup: intervention_type_string → mapping dict.
    Reads from gta_to_taxonomy.json and flattens the grouped structure.
    """
    gta_mapping = load_gta_mapping()
    mappings = gta_mapping.get("gta_mappings", {})
    override_rules = gta_mapping.get("gta_severity_override_rules", [])

    lookup = {}
    for group_key, group in mappings.items():
        intervention_types = group.get("gta_intervention_types", [])
        for itype in intervention_types:
            lookup[itype.lower()] = {
                "group_key": group_key,
                "taxonomy_category": group.get("taxonomy_category"),
                "taxonomy_category_primary": group.get("taxonomy_category_primary"),
                "taxonomy_category_secondary": group.get("taxonomy_category_secondary"),
                "routing_logic": group.get("routing_logic"),
                "event_subtype": group.get("event_subtype"),
                "default_severity": group.get("default_severity", {}),
                "onset_speed": group.get("onset_speed", "phased"),
                "expected_duration": group.get("expected_duration", "structural"),
                "confidence": group.get("confidence", "medium"),
            }

    return lookup, override_rules


INTERVENTION_LOOKUP, OVERRIDE_RULES = build_intervention_type_lookup()


def map_gta_to_taxonomy(intervention: dict) -> dict | None:
    """
    Map a GTA intervention to our taxonomy.
    Returns dict with taxonomy_category, event_subtype, severity, etc.
    Returns None if the intervention type is unmapped.
    """
    itype = (intervention.get("intervention_type") or "").strip().lower()
    gta_eval = intervention.get("gta_evaluation", "")
    affected_sectors = intervention.get("affected_sectors", [])
    affected_products = intervention.get("affected_products", [])

    mapping = INTERVENTION_LOOKUP.get(itype)
    if mapping is None:
        # Try partial matching for intervention types with extra qualifiers
        for key, val in INTERVENTION_LOOKUP.items():
            if key in itype or itype in key:
                mapping = val
                break

    if mapping is None:
        return None

    # Determine taxonomy category
    category = mapping.get("taxonomy_category")

    # Handle dual-category mappings (e.g., export controls → trade_policy OR technology_controls)
    if mapping.get("routing_logic") and mapping.get("taxonomy_category_primary"):
        category = mapping["taxonomy_category_primary"]
        # Check if this is a tech/semiconductor export control
        sector_text = " ".join(str(s) for s in affected_sectors + affected_products).lower()
        tech_keywords = ["semiconductor", "chip", "gpu", "ai", "dual-use", "3344", "3812"]
        mineral_keywords = ["lithium", "cobalt", "rare earth", "nickel", "manganese", "graphite"]

        if any(kw in sector_text for kw in tech_keywords):
            category = "technology_controls"
        elif any(kw in sector_text for kw in mineral_keywords):
            category = "resource_energy_disruptions"
        elif mapping.get("taxonomy_category_secondary"):
            # Default to secondary if primary doesn't match
            pass  # keep primary

    if category is None:
        return None

    # Determine severity from evaluation color
    default_sev = mapping.get("default_severity", {})
    if isinstance(default_sev, dict):
        severity = default_sev.get(gta_eval, default_sev.get("Red", 3))
    elif isinstance(default_sev, int):
        severity = default_sev
    else:
        severity = 3

    # Apply override rules for specific intervention + sector combinations
    for rule in OVERRIDE_RULES:
        if rule.get("override_category"):
            cond = rule.get("condition", "")
            if "semiconductor" in cond.lower() or "chip" in cond.lower():
                sector_text = " ".join(str(s) for s in affected_sectors + affected_products).lower()
                if any(kw in sector_text for kw in ["semiconductor", "chip", "gpu", "3344"]):
                    category = rule["override_category"]
                    severity = rule.get("override_severity", severity)
            elif "critical mineral" in cond.lower():
                sector_text = " ".join(str(s) for s in affected_sectors + affected_products).lower()
                if any(kw in sector_text for kw in ["lithium", "cobalt", "rare earth"]):
                    category = rule["override_category"]
                    severity = rule.get("override_severity", severity)

    return {
        "taxonomy_category": category,
        "event_subtype": mapping.get("event_subtype"),
        "severity": severity,
        "onset_speed": mapping.get("onset_speed", "phased"),
        "expected_duration": mapping.get("expected_duration", "structural"),
        "confidence": mapping.get("confidence", "medium"),
    }


# ─── API fetch ────────────────────────────────────────────────────────────────

# Reuse a single TCP connection for all API calls (prevents port exhaustion)
_gta_session = requests.Session()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def fetch_gta_page(
    api_key: str,
    start_date: date,
    end_date: date,
    offset: int = 0,
    red_only: bool = False,
) -> dict:
    """Fetch one page of GTA interventions. Uses persistent session (1 TCP connection)."""
    request_data = {
        "announcement_period": [start_date.isoformat(), end_date.isoformat()],
    }

    if red_only:
        request_data["gta_evaluation"] = [GTA_EVAL_RED]

    payload = {
        "request_data": request_data,
        "limit": GTA_PAGE_SIZE,
        "offset": offset,
        "sorting": "date_announced",
    }

    _gta_session.headers.update({
        "Authorization": f"APIKey {api_key}",
        "Content-Type": "application/json",
    })
    response = _gta_session.post(GTA_API_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_all_gta_interventions(
    api_key: str,
    start_date: date,
    end_date: date,
    red_only: bool = False,
) -> list[dict]:
    """Paginate through all GTA results for the given date range."""
    all_interventions = []
    offset = 0

    while True:
        logger.info(f"  Fetching GTA page (offset={offset})")
        try:
            response = fetch_gta_page(api_key, start_date, end_date, offset, red_only)
        except Exception as e:
            logger.error(f"  GTA API error at offset {offset}: {e}")
            break

        # Response is a list of interventions (not wrapped in a data key)
        if isinstance(response, list):
            data = response
        elif isinstance(response, dict):
            data = response.get("data", response.get("interventions", []))
            if not data and isinstance(response, dict):
                # The response itself might be the list wrapped differently
                data = []
        else:
            data = []

        if not data:
            break

        all_interventions.extend(data)
        logger.info(f"  Offset {offset}: {len(data)} interventions (total: {len(all_interventions)})")

        if len(data) < GTA_PAGE_SIZE:
            break  # last page

        offset += GTA_PAGE_SIZE
        time.sleep(GTA_RATE_LIMIT_DELAY)

    return all_interventions


# ─── Transform ───────────────────────────────────────────────────────────────

def build_event_records(interventions: list[dict]) -> Iterator[dict]:
    """Convert GTA interventions to geopolitical_events schema."""
    for intervention in interventions:
        mapping = map_gta_to_taxonomy(intervention)
        if mapping is None:
            continue

        # Parse dates
        date_announced = intervention.get("date_announced") or intervention.get("date_implemented")
        if not date_announced:
            continue
        try:
            event_date = date.fromisoformat(str(date_announced).split("T")[0])
        except (ValueError, AttributeError):
            continue

        date_removed = intervention.get("date_removed")
        event_end_date = None
        if date_removed:
            try:
                event_end_date = date.fromisoformat(str(date_removed).split("T")[0]).isoformat()
            except (ValueError, AttributeError):
                pass

        # Extract country codes
        implementing = intervention.get("implementing_jurisdictions", [])
        affected = intervention.get("affected_jurisdictions", [])
        countries = set()
        for j in implementing:
            iso = j.get("iso", "") if isinstance(j, dict) else str(j)
            if iso and len(iso) <= 3:
                countries.add(iso.upper())
        for j in affected:
            iso = j.get("iso", "") if isinstance(j, dict) else str(j)
            if iso and len(iso) <= 3:
                countries.add(iso.upper())

        # Extract sector codes
        sectors = intervention.get("affected_sectors", [])
        sector_list = [str(s) for s in sectors[:20]] if sectors else []

        # Build description
        gta_eval = intervention.get("gta_evaluation", "")
        title = intervention.get("state_act_title", "") or ""
        itype = intervention.get("intervention_type", "") or ""
        description = f"[{gta_eval}] {itype}: {title}"[:1000]

        intervention_id = intervention.get("intervention_id", "")
        source_url = intervention.get("intervention_url", "") or ""

        yield {
            "source": "gta",
            "source_event_id": str(intervention_id),
            "event_category": mapping["taxonomy_category"],
            "event_subtype": mapping["event_subtype"],
            "event_date": event_date.isoformat(),
            "event_end_date": event_end_date,
            "affected_countries": json.dumps(sorted(countries)),
            "affected_sectors": json.dumps(sector_list),
            "severity_estimate": mapping["severity"],
            "onset_speed": mapping["onset_speed"],
            "expected_duration": mapping["expected_duration"],
            "description_text": description,
            "source_url": source_url[:500],
            "goldstein_scale": None,
            "fatalities": None,
            "num_mentions": None,
            "avg_tone": None,
            "mapping_confidence": mapping["confidence"],
        }


# ─── Store ────────────────────────────────────────────────────────────────────

def store_events(conn, records: list[dict]) -> tuple[int, int]:
    """Bulk insert with INSERT OR IGNORE."""
    rows = []
    for record in records:
        event_id = make_event_id("gta", date.fromisoformat(record["event_date"]))
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
    stored = len(rows)  # INSERT OR IGNORE doesn't reliably report via changes()
    return stored, 0


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD (announcement date)")
@click.option("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option("--red-only", is_flag=True, help="Only fetch Red (harmful) interventions")
@click.option("--dry-run", is_flag=True, help="Fetch and map but do not write to DB")
def main(start: str, end: str, red_only: bool, dry_run: bool) -> None:
    """Ingest GTA interventions into the geopolitical events database."""
    api_key = os.getenv("GTA_API_KEY")

    if not api_key:
        logger.error(
            "GTA_API_KEY must be set in .env. "
            "Register at https://globaltradealert.org/api-access"
        )
        raise SystemExit(1)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    logger.info(
        f"GTA ingestion: {start_date} → {end_date}"
        + (" (Red only)" if red_only else " (all evaluations)")
    )

    conn = get_db_connection()
    run_id = make_run_id("gta")
    log_ingestion_start(conn, run_id, "gta", start_date, end_date)

    try:
        interventions = fetch_all_gta_interventions(api_key, start_date, end_date, red_only)
        records = list(build_event_records(interventions))

        # Count by category for logging
        cat_counts = {}
        for r in records:
            cat = r["event_category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        logger.info(f"Fetched {len(interventions):,} GTA interventions → {len(records)} mapped")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {cnt}")

        if not dry_run:
            stored, skipped = store_events(conn, records)
            logger.info(f"Stored {stored}, skipped {skipped}")
            log_ingestion_end(conn, run_id, len(interventions), stored, skipped, "success")
        else:
            logger.info("Dry run — not writing to DB")
            log_ingestion_end(conn, run_id, len(interventions), 0, 0, "dry_run")

    except Exception as e:
        log_ingestion_end(conn, run_id, 0, 0, 0, "failed", str(e))
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
