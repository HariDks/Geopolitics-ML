"""
GDELT ingestion pipeline.

Fetches GDELT 2.0 Event records for a date range, applies filtering (CAMEO code
relevance, NumMentions threshold, GoldsteinScale threshold), maps to taxonomy,
deduplicates, and stores in the geopolitical_events table.

Usage:
    python pipelines/ingest_gdelt.py --start 2020-01-01 --end 2025-12-31
    python pipelines/ingest_gdelt.py --start 2022-02-01 --end 2022-03-31  # Ukraine invasion

Docs:
    GDELT 2.0 Event record format: https://www.gdeltproject.org/data.html#documentation
    CAMEO codebook: https://eventdata.utdallas.edu/data.html
"""

import io
import json
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import click
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.utils import (
    cameo_to_taxonomy,
    event_exists,
    get_db_connection,
    get_logger,
    log_ingestion_end,
    log_ingestion_start,
    make_event_id,
    make_run_id,
)

logger = get_logger("ingest_gdelt")

# ─── GDELT filtering thresholds ──────────────────────────────────────────────

MIN_NUM_MENTIONS = 10       # Remove noise: events with < 10 mentions are unreliable
GOLDSTEIN_CONFLICT_MAX = -5.0   # For conflict/coercion events, require scale <= -5
GOLDSTEIN_GENERAL_MAX = -3.0    # For protest/political events, softer threshold

# CAMEO root codes to include. All others are dropped before taxonomy lookup.
# Codes 01-12 are cooperation/diplomacy — irrelevant to geopolitical risk detection.
RELEVANT_CAMEO_ROOTS = {
    "13",   # THREATEN
    "14",   # PROTEST
    "15",   # EXHIBIT FORCE
    "16",   # REDUCE RELATIONS
    "17",   # COERCE
    "18",   # ASSAULT
    "19",   # FIGHT
    "20",   # USE UNCONVENTIONAL MASS VIOLENCE
}

# GDELT column names (2.0 format, tab-delimited)
GDELT_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]


# ─── Fetch ────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def fetch_gdelt_day(target_date: date) -> pd.DataFrame:
    """
    Fetch one day of GDELT 2.0 events.

    GDELT 2.0 publishes daily files at:
      http://data.gdeltproject.org/events/{YYYYMMDD}.export.CSV.zip

    Returns raw DataFrame with all GDELT columns, or empty DataFrame on failure.
    """
    date_str = target_date.strftime("%Y%m%d")
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    logger.info(f"  Fetching GDELT for {date_str}")
    try:
        # Download with timeout to avoid hanging on slow GDELT servers
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    sep="\t",
                    header=None,
                    names=GDELT_COLS,
                    dtype={"EventCode": str, "EventBaseCode": str, "EventRootCode": str},
                    low_memory=False,
                )
        logger.info(f"  Fetched {len(df):,} raw events for {date_str}")
        return df
    except Exception as e:
        logger.warning(f"  Failed to fetch GDELT for {date_str}: {e}")
        return pd.DataFrame(columns=GDELT_COLS)


# ─── Filter ───────────────────────────────────────────────────────────────────

def filter_relevant_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply GDELT filtering to retain only geopolitically relevant events.
    Steps: (1) CAMEO root code filter, (2) NumMentions threshold, (3) GoldsteinScale.
    """
    if df.empty:
        return df

    original_count = len(df)

    # Step 1: CAMEO root code filter
    df = df[df["EventRootCode"].isin(RELEVANT_CAMEO_ROOTS)].copy()
    logger.debug(f"  After CAMEO root filter: {len(df):,} / {original_count:,}")

    # Step 2: NumMentions threshold
    df = df[df["NumMentions"] >= MIN_NUM_MENTIONS].copy()
    logger.debug(f"  After NumMentions >= {MIN_NUM_MENTIONS}: {len(df):,}")

    # Step 3: GoldsteinScale — conflict events need strong negative score
    conflict_roots = {"15", "17", "18", "19", "20"}
    soft_roots = {"13", "14", "16"}

    conflict_mask = (
        df["EventRootCode"].isin(conflict_roots) &
        (df["GoldsteinScale"] <= GOLDSTEIN_CONFLICT_MAX)
    )
    soft_mask = (
        df["EventRootCode"].isin(soft_roots) &
        (df["GoldsteinScale"] <= GOLDSTEIN_GENERAL_MAX)
    )
    df = df[conflict_mask | soft_mask].copy()
    logger.debug(f"  After GoldsteinScale filter: {len(df):,}")

    return df


# ─── Deduplicate ─────────────────────────────────────────────────────────────

def deduplicate_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate events that are multiple media reports of the same geopolitical action.
    Strategy: group by (Day, EventCode, Actor1CountryCode, Actor2CountryCode, ActionGeo_CountryCode)
    and keep the row with max NumMentions (most widely reported version).
    """
    if df.empty:
        return df

    group_cols = [
        "Day", "EventCode", "Actor1CountryCode",
        "Actor2CountryCode", "ActionGeo_CountryCode",
    ]
    # Fill NaN for grouping
    for col in group_cols:
        df[col] = df[col].fillna("")

    deduped = (
        df.sort_values("NumMentions", ascending=False)
          .drop_duplicates(subset=group_cols, keep="first")
          .reset_index(drop=True)
    )
    logger.debug(f"  After deduplication: {len(deduped):,} (was {len(df):,})")
    return deduped


# ─── Transform ───────────────────────────────────────────────────────────────

def build_event_records(df: pd.DataFrame) -> Iterator[dict]:
    """
    Convert filtered GDELT rows to geopolitical_events schema dicts.
    Applies CAMEO → taxonomy mapping. Skips rows with no taxonomy match.
    """
    for _, row in df.iterrows():
        cameo_code = str(row.get("EventCode", "")).strip()
        mapping = cameo_to_taxonomy(cameo_code)
        if mapping is None:
            continue

        # Parse event date
        day_str = str(int(row.get("Day", 0)))
        try:
            event_date = date(int(day_str[:4]), int(day_str[4:6]), int(day_str[6:8]))
        except (ValueError, IndexError):
            continue

        # Collect affected countries from all actor/action geo fields
        countries = set()
        for field in ["Actor1CountryCode", "Actor2CountryCode", "ActionGeo_CountryCode",
                       "Actor1Geo_CountryCode", "Actor2Geo_CountryCode"]:
            val = str(row.get(field, "")).strip()
            if val and val != "nan" and len(val) <= 3:
                countries.add(val.upper())

        severity = _estimate_severity_from_goldstein(row.get("GoldsteinScale"))

        yield {
            "source": "gdelt",
            "source_event_id": str(row.get("GlobalEventID", "")),
            "event_category": mapping["taxonomy_category"],
            "event_subtype": _cameo_to_subtype(cameo_code),
            "event_date": event_date.isoformat(),
            "event_end_date": None,
            "affected_countries": json.dumps(sorted(countries)),
            "affected_sectors": json.dumps([]),   # GDELT has no sector info
            "severity_estimate": severity,
            "onset_speed": "sudden",              # GDELT events are point-in-time
            "expected_duration": None,
            "description_text": None,
            "source_url": str(row.get("SOURCEURL", ""))[:500],
            "goldstein_scale": float(row.get("GoldsteinScale", 0) or 0),
            "fatalities": None,
            "num_mentions": int(row.get("NumMentions", 0) or 0),
            "avg_tone": float(row.get("AvgTone", 0) or 0),
            "mapping_confidence": mapping.get("confidence", "medium"),
        }


def _estimate_severity_from_goldstein(scale_val) -> int:
    """Map GoldsteinScale (-10 to +10) to severity 1-5."""
    try:
        scale = float(scale_val)
    except (TypeError, ValueError):
        return 3
    if scale <= -8:
        return 5
    elif scale <= -6:
        return 4
    elif scale <= -4:
        return 3
    elif scale <= -2:
        return 2
    else:
        return 1


def _cameo_to_subtype(cameo_code: str) -> str | None:
    """Map CAMEO code to a human-readable subtype string."""
    subtypes = {
        "190": "armed_clash", "191": "military_strike", "192": "shelling",
        "193": "aerial_attack", "194": "interstate_war", "195": "siege",
        "180": "assault", "181": "abduction", "183": "bombing", "186": "assassination",
        "171": "sanctions_imposed", "172": "blockade", "173": "state_of_emergency",
        "145": "violent_protest", "144": "obstruction",
        "152": "military_alert", "153": "troop_mobilization",
        "161": "diplomatic_reduction", "165": "break_relations", "166": "embargo",
        "136": "threat_armed_force", "133": "ultimatum",
        "200": "mass_violence", "202": "mass_killing", "204": "wmd_use",
    }
    return subtypes.get(cameo_code)


# ─── Store ────────────────────────────────────────────────────────────────────

def store_events(conn, records: list[dict]) -> tuple[int, int]:
    """Bulk insert with INSERT OR IGNORE — handles duplicates at DB level."""
    rows = []
    for record in records:
        event_id = make_event_id("gdelt", date.fromisoformat(record["event_date"]))
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
@click.option("--dry-run", is_flag=True, help="Fetch and filter but do not write to DB")
def main(start: str, end: str, dry_run: bool) -> None:
    """Ingest GDELT events for a date range into the geopolitical events database."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    total_days = (end_date - start_date).days + 1

    logger.info(f"GDELT ingestion: {start_date} → {end_date} ({total_days} days)")

    conn = get_db_connection()
    run_id = make_run_id("gdelt")
    log_ingestion_start(conn, run_id, "gdelt", start_date, end_date)

    total_fetched = total_stored = total_skipped = 0
    current = start_date

    try:
        while current <= end_date:
            raw_df = fetch_gdelt_day(current)
            filtered_df = filter_relevant_events(raw_df)
            deduped_df = deduplicate_events(filtered_df)
            records = list(build_event_records(deduped_df))

            total_fetched += len(raw_df)

            if not dry_run:
                stored, skipped = store_events(conn, records)
                total_stored += stored
                total_skipped += skipped
                logger.info(
                    f"  {current}: {len(raw_df):,} raw → {len(records)} mapped "
                    f"→ {stored} stored, {skipped} skipped"
                )
            else:
                logger.info(
                    f"  {current}: {len(raw_df):,} raw → {len(records)} mapped "
                    f"(dry run, not stored)"
                )

            current += timedelta(days=1)

        log_ingestion_end(conn, run_id, total_fetched, total_stored, total_skipped, "success")
        logger.info(
            f"Done. Total: {total_fetched:,} fetched, {total_stored:,} stored, "
            f"{total_skipped:,} skipped (duplicates)"
        )

    except Exception as e:
        log_ingestion_end(conn, run_id, total_fetched, total_stored, total_skipped,
                          "failed", str(e))
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
