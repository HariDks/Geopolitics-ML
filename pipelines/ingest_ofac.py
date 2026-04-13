"""
OFAC SDN (Specially Designated Nationals) list ingestion pipeline.

Downloads the OFAC SDN XML feed, extracts entity addition dates, and
maps to sanctions_financial_restrictions taxonomy category.

OFAC SDN list: https://sanctionslist.ofac.treas.gov/Home/SdnList
Feed format: XML (sdn_advanced.xml) — includes entity type, programs, dates.

Usage:
    python pipelines/ingest_ofac.py --since 2020-01-01
    python pipelines/ingest_ofac.py  # defaults to all available

Note:
    OFAC does not provide a timestamped change feed — the SDN list is a point-in-time
    snapshot. We use the "dateOfBirth" or "publishedDate" where available, but for
    many entries the addition date must be inferred from news/Federal Register.
    For high-quality sanctions timeline data, supplement with:
    - OFAC press releases (https://ofac.treasury.gov/recent-actions)
    - EU Consolidated List (has effective_from dates)
"""

import json
import os
import sys
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

import click
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import (
    event_exists,
    get_db_connection,
    get_logger,
    log_ingestion_end,
    log_ingestion_start,
    make_event_id,
    make_run_id,
)

logger = get_logger("ingest_ofac")

OFAC_SDN_URL = "https://sanctionslist.ofac.treas.gov/"
OFAC_XML_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN_ADVANCED.XML"

# OFAC sanction programs that map to our taxonomy
PROGRAM_TO_SUBTYPE = {
    "RUSSIA-EO14024": "russia_sanctions",
    "UKRAINE-EO13685": "ukraine_crimea_sanctions",
    "IRAN": "iran_sanctions",
    "NPWMD": "wmd_proliferation_sanctions",
    "SDNT": "drug_trafficking_sanctions",
    "CYBER2": "cyber_sanctions",
    "DPRK": "north_korea_sanctions",
    "SYRIA": "syria_sanctions",
    "BELARUS": "belarus_sanctions",
    "CHINA-MILITARY-COMPLEX": "china_military_sanctions",
    "TCO": "transnational_criminal_org_sanctions",
}

# SDN entity types to include (exclude individual persons unless high-profile)
INCLUDED_ENTITY_TYPES = {
    "Entity",
    "Vessel",
    "Aircraft",
    # "Individual",  # Uncomment if you want to include individual persons
}


# ─── Download ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=60))
def download_sdn_xml(url: str, raw_dir: Path) -> Path:
    """Download OFAC SDN XML to local cache. Returns path to saved file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "sdn_advanced.xml"

    logger.info(f"Downloading OFAC SDN list from {url}")
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)

    file_size_mb = out_path.stat().st_size / 1_000_000
    logger.info(f"Downloaded SDN XML: {file_size_mb:.1f} MB → {out_path}")
    return out_path


# ─── Parse ────────────────────────────────────────────────────────────────────

def parse_sdn_xml(xml_path: Path) -> list[dict]:
    """
    Parse the OFAC SDN Advanced XML file (new schema as of 2024).

    New schema structure:
    - <Sanctions> root with namespace sanctionslistservice.ofac.treas.gov/...
    - <DistinctParty> elements contain Profile > Identity > Alias > NamePartValue (name)
    - <SanctionsEntries> section maps ProfileID → EntryEvent Date + SanctionsMeasure (program)
    - PartySubTypeID encodes entity type (3=Entity, 4=Individual, 5=Vessel, 6=Aircraft)

    Returns list of entity dicts with: uid, name, sdn_type, programs, published_date.
    """
    logger.info(f"Parsing SDN XML: {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Detect namespace
    ns_uri = ""
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].lstrip("{")
    ns = {"s": ns_uri} if ns_uri else {}

    def tag(local: str) -> str:
        return f"{{{ns_uri}}}{local}" if ns_uri else local

    # PartySubTypeID → entity type label
    PARTY_SUBTYPE = {
        "3": "Entity",
        "4": "Individual",
        "5": "Vessel",
        "6": "Aircraft",
    }

    # ── Step 1: Build profile_id → name mapping from DistinctParty ──
    profile_names: dict[str, str] = {}
    profile_types: dict[str, str] = {}

    for party in root.iter(tag("DistinctParty")):
        profile = party.find(tag("Profile"))
        if profile is None:
            continue
        profile_id = profile.get("ID", "")
        subtype_id = profile.get("PartySubTypeID", "")
        sdn_type = PARTY_SUBTYPE.get(subtype_id, "Unknown")
        profile_types[profile_id] = sdn_type

        # Primary name: find Alias with Primary="true", get first NamePartValue
        for alias in profile.iter(tag("Alias")):
            if alias.get("Primary") == "true":
                parts = []
                for npv in alias.iter(tag("NamePartValue")):
                    if npv.text and npv.get("ScriptStatusID") == "1":  # Latin script
                        parts.append(npv.text.strip())
                if parts:
                    profile_names[profile_id] = " ".join(parts)
                    break

    # ── Step 2: Build profile_id → (date, programs) from SanctionsEntries ──
    profile_dates: dict[str, str] = {}
    profile_programs: dict[str, list[str]] = {}

    for entry in root.iter(tag("SanctionsEntry")):
        profile_id = entry.get("ProfileID", "")

        # Entry date from EntryEvent > Date
        entry_event = entry.find(tag("EntryEvent"))
        if entry_event is not None:
            date_el = entry_event.find(tag("Date"))
            if date_el is not None:
                year = date_el.findtext(tag("Year"))
                month = date_el.findtext(tag("Month"))
                day = date_el.findtext(tag("Day"))
                if year and month and day:
                    try:
                        profile_dates[profile_id] = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                    except ValueError:
                        pass

        # Programs from SanctionsMeasure > Comment (contains program name like "RUSSIA-EO14024")
        programs = []
        for measure in entry.findall(tag("SanctionsMeasure")):
            comment = measure.findtext(tag("Comment"))
            if comment and comment.strip():
                programs.append(comment.strip())
        if programs:
            profile_programs[profile_id] = programs

    # ── Step 3: Assemble entity list ──
    entities = []
    for profile_id, sdn_type in profile_types.items():
        if sdn_type not in INCLUDED_ENTITY_TYPES:
            continue
        name = profile_names.get(profile_id)
        if not name:
            continue
        entities.append({
            "uid": profile_id,
            "name": name,
            "sdn_type": sdn_type,
            "programs": profile_programs.get(profile_id, []),
            "published_date": profile_dates.get(profile_id),
        })

    logger.info(f"Parsed {len(entities):,} SDN entities (entity/vessel/aircraft types)")
    return entities


# ─── Transform ───────────────────────────────────────────────────────────────

def build_event_records(entities: list[dict], since_date: date | None) -> list[dict]:
    """
    Convert SDN entities to geopolitical_events records.
    Groups entities by (program, published_date) to create program-level events
    rather than one event per entity (which would inflate the count massively).
    """
    from collections import defaultdict

    # Group by (primary_program, date)
    groups = defaultdict(list)
    for entity in entities:
        primary_program = entity["programs"][0] if entity["programs"] else "UNKNOWN"
        pub_date = entity.get("published_date") or "2020-01-01"  # fallback
        groups[(primary_program, pub_date)].append(entity)

    records = []
    for (program, pub_date_str), group_entities in groups.items():
        # Parse date
        try:
            pub_date = date.fromisoformat(pub_date_str.split("T")[0])
        except (ValueError, AttributeError):
            pub_date = date(2020, 1, 1)

        if since_date and pub_date < since_date:
            continue

        subtype = PROGRAM_TO_SUBTYPE.get(program, "financial_restriction")
        entity_names = [e["name"] for e in group_entities[:5] if e["name"]]
        description = f"OFAC SDN addition: {len(group_entities)} entities under program {program}. "
        if entity_names:
            description += "Includes: " + ", ".join(entity_names)

        records.append({
            "source": "ofac",
            "source_event_id": f"OFAC-{program}-{pub_date_str}",
            "event_category": "sanctions_financial_restrictions",
            "event_subtype": subtype,
            "event_date": pub_date.isoformat(),
            "event_end_date": None,
            "affected_countries": json.dumps(_programs_to_countries(program)),
            "affected_sectors": json.dumps([]),
            "severity_estimate": _program_to_severity(program, len(group_entities)),
            "onset_speed": "sudden",
            "expected_duration": "structural",
            "description_text": description[:1000],
            "source_url": OFAC_SDN_URL,
            "goldstein_scale": None,
            "fatalities": None,
            "num_mentions": None,
            "avg_tone": None,
            "mapping_confidence": "high",
        })

    return records


def _programs_to_countries(program: str) -> list[str]:
    """Map OFAC program name to affected country ISO codes."""
    program_country_map = {
        "RUSSIA-EO14024": ["RU"],
        "UKRAINE-EO13685": ["UA", "RU"],
        "IRAN": ["IR"],
        "DPRK": ["KP"],
        "SYRIA": ["SY"],
        "BELARUS": ["BY"],
        "CHINA-MILITARY-COMPLEX": ["CN"],
        "CUBA": ["CU"],
        "VENEZUELA": ["VE"],
        "SUDAN": ["SD"],
        "MYANMAR": ["MM"],
    }
    for key, countries in program_country_map.items():
        if key in program:
            return countries
    return []


def _program_to_severity(program: str, entity_count: int) -> int:
    """
    Estimate severity from program name and entity count.
    Comprehensive sanctions (Russia, Iran) warrant higher baseline severity.
    """
    high_severity_programs = {"RUSSIA-EO14024", "IRAN", "DPRK", "CHINA-MILITARY-COMPLEX"}
    baseline = 4 if any(p in program for p in high_severity_programs) else 3
    if entity_count > 100:
        return min(5, baseline + 1)
    return baseline


# ─── Store ────────────────────────────────────────────────────────────────────

def store_events(conn, records: list[dict]) -> tuple[int, int]:
    stored, skipped = 0, 0
    for record in records:
        if event_exists(conn, "ofac", record["source_event_id"]):
            skipped += 1
            continue

        event_id = make_event_id("ofac", date.fromisoformat(record["event_date"]))
        conn.execute(
            """INSERT INTO geopolitical_events (
                event_id, source, source_event_id, event_category, event_subtype,
                event_date, event_end_date, affected_countries, affected_sectors,
                severity_estimate, onset_speed, expected_duration, description_text,
                source_url, goldstein_scale, fatalities, num_mentions, avg_tone,
                mapping_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, record["source"], record["source_event_id"],
                record["event_category"], record["event_subtype"],
                record["event_date"], record["event_end_date"],
                record["affected_countries"], record["affected_sectors"],
                record["severity_estimate"], record["onset_speed"],
                record["expected_duration"], record["description_text"],
                record["source_url"], record["goldstein_scale"],
                record["fatalities"], record["num_mentions"],
                record["avg_tone"], record["mapping_confidence"],
            ),
        )
        stored += 1

    conn.commit()
    return stored, skipped


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--since", default=None, help="Only include entities published on/after YYYY-MM-DD")
@click.option("--dry-run", is_flag=True, help="Parse but do not write to DB")
def main(since: str | None, dry_run: bool) -> None:
    """Download and ingest OFAC SDN list into the geopolitical events database."""
    from pathlib import Path
    raw_dir = Path(__file__).parent.parent / "data" / "raw" / "ofac"

    since_date = date.fromisoformat(since) if since else None
    if since_date:
        logger.info(f"OFAC ingestion (entities published since {since_date})")
    else:
        logger.info("OFAC ingestion (all SDN entities)")

    conn = get_db_connection()
    run_id = make_run_id("ofac")
    today = date.today()
    log_ingestion_start(conn, run_id, "ofac", since_date or date(2000, 1, 1), today)

    try:
        xml_path = download_sdn_xml(OFAC_XML_URL, raw_dir)
        entities = parse_sdn_xml(xml_path)
        records = build_event_records(entities, since_date)
        logger.info(f"Generated {len(records)} program-level sanction events from {len(entities)} entities")

        if not dry_run:
            stored, skipped = store_events(conn, records)
            logger.info(f"Stored {stored}, skipped {skipped} duplicates")
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
