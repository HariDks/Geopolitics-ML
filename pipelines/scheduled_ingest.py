"""
Scheduled re-ingestion pipeline for keeping the database current.

Runs incremental updates from data sources:
- GDELT: every 15 minutes (new events since last run)
- ACLED: weekly (new events since last run)
- GTA: weekly (new interventions since last run)
- OFAC: daily (SDN list changes)
- BIS: weekly (entity list updates)
- EDGAR: quarterly (new filings after earnings season)

After ingestion, re-runs data prep (specificity scoring, event linking)
and optionally re-generates auto-labels.

Usage:
    python pipelines/scheduled_ingest.py --source gdelt         # single source
    python pipelines/scheduled_ingest.py --source all            # all sources
    python pipelines/scheduled_ingest.py --source gdelt --dry-run  # preview only

Cron examples:
    */15 * * * *  cd /path/to/project && python pipelines/scheduled_ingest.py --source gdelt
    0 2 * * 1     cd /path/to/project && python pipelines/scheduled_ingest.py --source all
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.utils import get_db_connection, get_logger, make_run_id

logger = get_logger("scheduled_ingest")


def get_last_ingest_date(conn, source: str) -> date | None:
    """Get the end_date of the most recent successful ingestion for a source."""
    row = conn.execute(
        """SELECT MAX(end_date) as last_date FROM ingestion_log
           WHERE source = ? AND status = 'success'""",
        (source,),
    ).fetchone()
    if row and row["last_date"]:
        return date.fromisoformat(row["last_date"])
    return None


def run_gdelt(conn, start_date: date, end_date: date, dry_run: bool = False):
    """Incremental GDELT ingestion."""
    logger.info(f"GDELT: {start_date} to {end_date}")
    if dry_run:
        logger.info("  DRY RUN — skipping")
        return

    from pipelines.ingest_gdelt import ingest_gdelt_range
    try:
        ingest_gdelt_range(conn, start_date, end_date)
    except Exception as e:
        logger.error(f"  GDELT failed: {e}")


def run_acled(conn, start_date: date, end_date: date, dry_run: bool = False):
    """Incremental ACLED ingestion."""
    logger.info(f"ACLED: {start_date} to {end_date}")
    if dry_run:
        logger.info("  DRY RUN — skipping")
        return

    try:
        from pipelines.ingest_acled import ingest_acled_range
        ingest_acled_range(conn, start_date, end_date)
    except Exception as e:
        logger.error(f"  ACLED failed: {e}")


def run_ofac(conn, dry_run: bool = False):
    """Full OFAC SDN list refresh (always fetches the full list)."""
    logger.info("OFAC: full SDN list refresh")
    if dry_run:
        logger.info("  DRY RUN — skipping")
        return

    try:
        from pipelines.ingest_ofac import ingest_ofac
        ingest_ofac(conn)
    except Exception as e:
        logger.error(f"  OFAC failed: {e}")


def run_data_prep(conn, dry_run: bool = False):
    """Re-run data prep steps on new data."""
    logger.info("Running data prep (deltas, linking, specificity)...")
    if dry_run:
        logger.info("  DRY RUN — skipping")
        return

    try:
        from pipelines.data_prep import (
            compute_financial_deltas,
            link_mentions_to_events,
            score_mention_specificity,
        )
        compute_financial_deltas(conn)
        link_mentions_to_events(conn)
        score_mention_specificity(conn)
    except Exception as e:
        logger.error(f"  Data prep failed: {e}")


SOURCE_RUNNERS = {
    "gdelt": run_gdelt,
    "acled": run_acled,
}


@click.command()
@click.option("--source", required=True,
              type=click.Choice(["gdelt", "acled", "ofac", "all"]),
              help="Which source to ingest")
@click.option("--days-back", default=7, type=int,
              help="How many days back to look for new data (default: 7)")
@click.option("--dry-run", is_flag=True, help="Preview what would be ingested")
@click.option("--skip-prep", is_flag=True, help="Skip data prep after ingestion")
def main(source: str, days_back: int, dry_run: bool, skip_prep: bool):
    """Run scheduled data ingestion."""
    conn = get_db_connection()
    today = date.today()

    sources_to_run = ["gdelt", "acled", "ofac"] if source == "all" else [source]

    for src in sources_to_run:
        last_date = get_last_ingest_date(conn, src)
        start_date = last_date + timedelta(days=1) if last_date else today - timedelta(days=days_back)

        if start_date > today:
            logger.info(f"{src}: already up to date (last: {last_date})")
            continue

        logger.info(f"{'=' * 60}")
        logger.info(f"Source: {src} | {start_date} to {today}")
        logger.info(f"{'=' * 60}")

        if src == "ofac":
            run_ofac(conn, dry_run=dry_run)
        elif src in SOURCE_RUNNERS:
            SOURCE_RUNNERS[src](conn, start_date, today, dry_run=dry_run)
        else:
            logger.warning(f"No runner for source: {src}")

    if not skip_prep:
        run_data_prep(conn, dry_run=dry_run)

    conn.close()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Scheduled ingestion complete.")
    if dry_run:
        logger.info("DRY RUN — no changes made.")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
