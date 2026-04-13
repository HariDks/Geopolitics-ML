"""
Shared utilities for all ingestion pipelines.
Handles: DB connection, event ID generation, taxonomy loading, logging.
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "config"
MAPPINGS_DIR = ROOT_DIR / "data" / "mappings"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─── Taxonomy loading ─────────────────────────────────────────────────────────

def load_taxonomy() -> dict:
    with open(CONFIG_DIR / "taxonomy.json") as f:
        return json.load(f)


def load_cameo_mapping() -> dict:
    with open(MAPPINGS_DIR / "cameo_to_taxonomy.json") as f:
        return json.load(f)


def load_acled_mapping() -> dict:
    with open(MAPPINGS_DIR / "acled_to_taxonomy.json") as f:
        return json.load(f)


def load_gta_mapping() -> dict:
    with open(MAPPINGS_DIR / "gta_to_taxonomy.json") as f:
        return json.load(f)


def load_lambda_rates() -> dict:
    with open(CONFIG_DIR / "lambda_rates.json") as f:
        data = json.load(f)
    return {k: v["lambda"] for k, v in data["lambda_by_category"].items()}


# ─── Database connection ──────────────────────────────────────────────────────

def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection. Schema is initialized on first call."""
    db_path = os.getenv("SQLITE_DB_PATH", "data/processed/geopolitical_events.db")
    db_path = ROOT_DIR / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # safer concurrent writes

    # Initialize schema
    schema_path = ROOT_DIR / "database" / "schema.sql"
    with open(schema_path) as f:
        schema_sql = f.read()
    # Strip PostgreSQL-specific lines before executing on SQLite
    sqlite_sql = "\n".join(
        line for line in schema_sql.splitlines()
        if not line.strip().startswith("-- PostgreSQL")
    )
    conn.executescript(sqlite_sql)
    return conn


# ─── Event ID generation ──────────────────────────────────────────────────────

def make_event_id(source: str, event_date: date, seq: int | None = None) -> str:
    date_str = event_date.strftime("%Y%m%d")
    suffix = seq if seq is not None else uuid.uuid4().hex[:6].upper()
    return f"EVT-{source.upper()}-{date_str}-{suffix}"


def make_run_id(source: str) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"RUN-{source.upper()}-{ts}"


# ─── CAMEO taxonomy lookup ────────────────────────────────────────────────────

_cameo_mapping: dict | None = None


def cameo_to_taxonomy(cameo_code: str) -> dict[str, Any] | None:
    """
    Map a CAMEO event code to taxonomy category.
    Tries exact match first, then root-code match (e.g., '190' → 'CAMEO_190').

    Returns dict with 'taxonomy_category', 'confidence', 'note' or None if no mapping.
    """
    global _cameo_mapping
    if _cameo_mapping is None:
        _cameo_mapping = load_cameo_mapping()

    mappings = _cameo_mapping.get("cameo_mappings", {})

    # Normalize: GDELT uses numeric codes like "190", "1911", etc.
    key = f"CAMEO_{cameo_code}"
    if key in mappings:
        entry = mappings[key]
        if entry.get("taxonomy_category") is not None:
            return entry
        return None

    # Try root code (first 2 digits for 3-digit codes)
    if len(cameo_code) == 3:
        root_key = f"CAMEO_{cameo_code[:2]}"
        if root_key in mappings:
            entry = mappings[root_key]
            if entry.get("taxonomy_category") is not None:
                return {"confidence": "low", **entry}

    return None


# ─── ACLED taxonomy lookup ────────────────────────────────────────────────────

_acled_mapping: dict | None = None


def acled_to_taxonomy(event_type: str, sub_event_type: str | None = None) -> dict[str, Any] | None:
    """
    Map ACLED event_type (and optionally sub_event_type) to taxonomy category.
    Sub-type takes precedence over parent type mapping.
    """
    global _acled_mapping
    if _acled_mapping is None:
        _acled_mapping = load_acled_mapping()

    mappings = _acled_mapping.get("acled_mappings", {})

    parent = mappings.get(event_type)
    if parent is None:
        return None

    if sub_event_type and "sub_event_types" in parent:
        sub = parent["sub_event_types"].get(sub_event_type)
        if sub and sub.get("taxonomy_category"):
            return sub

    if parent.get("taxonomy_category"):
        return {
            "taxonomy_category": parent["taxonomy_category"],
            "confidence": parent.get("confidence", "medium"),
            "default_severity": parent.get("default_severity", 3),
        }

    return None


# ─── Ingestion log helpers ────────────────────────────────────────────────────

def log_ingestion_start(conn: sqlite3.Connection, run_id: str, source: str,
                         start_date: date, end_date: date) -> None:
    conn.execute(
        """INSERT INTO ingestion_log (run_id, source, start_date, end_date, status)
           VALUES (?, ?, ?, ?, 'running')""",
        (run_id, source, start_date.isoformat(), end_date.isoformat()),
    )
    conn.commit()


def log_ingestion_end(conn: sqlite3.Connection, run_id: str, records_fetched: int,
                       records_stored: int, records_skipped: int, status: str,
                       error_message: str | None = None) -> None:
    conn.execute(
        """UPDATE ingestion_log
           SET records_fetched=?, records_stored=?, records_skipped=?,
               status=?, error_message=?
           WHERE run_id=?""",
        (records_fetched, records_stored, records_skipped, status, error_message, run_id),
    )
    conn.commit()


def event_exists(conn: sqlite3.Connection, source: str, source_event_id: str) -> bool:
    """Check if an event from this source with this source ID already exists."""
    row = conn.execute(
        "SELECT 1 FROM geopolitical_events WHERE source=? AND source_event_id=?",
        (source, source_event_id),
    ).fetchone()
    return row is not None
