"""
Ingest macro-level geopolitical risk indices for benchmarking and features.

Two indices:
1. Economic Policy Uncertainty (EPU) — Baker, Bloom, Davis 2016
   Source: policyuncertainty.com

2. Geopolitical Risk Index (GPR) — Caldara, Iacoviello 2022 (Federal Reserve)
   Source: matteoiacoviello.com/gpr.htm

Both are established academic benchmarks. Our GRI should correlate with GPR.
Both can be used as features in the impact estimator.

Usage:
    python pipelines/ingest_macro_indices.py
"""

import sys
from datetime import date
from pathlib import Path

import click
import pandas as pd
import requests

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("macro_indices")

# Data URLs
EPU_URL = "https://policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def create_table(conn):
    """Create macro_indices table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_indices (
            date TEXT NOT NULL,
            index_name TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (date, index_name)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_indices(date)")
    conn.commit()


def ingest_epu(conn) -> int:
    """Ingest Economic Policy Uncertainty index."""
    logger.info("Downloading EPU index...")
    try:
        df = pd.read_csv(EPU_URL)
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Rows: {len(df)}")

        # EPU has columns: Year, Month, and various index columns
        # The main one is usually the last column or "News_Based_Policy_Uncert_Index"
        date_cols = [c for c in df.columns if 'year' in c.lower()]
        month_cols = [c for c in df.columns if 'month' in c.lower()]

        if date_cols and month_cols:
            year_col = date_cols[0]
            month_col = month_cols[0]
            # Find the EPU value column
            value_cols = [c for c in df.columns if c not in [year_col, month_col] and df[c].dtype in ('float64', 'int64')]
            if value_cols:
                value_col = value_cols[-1]  # usually the last numeric column is the main index
                logger.info(f"  Using: year={year_col}, month={month_col}, value={value_col}")

                stored = 0
                for _, row in df.iterrows():
                    try:
                        y = int(row[year_col])
                        m = int(row[month_col])
                        v = float(row[value_col])
                        if y >= 2010 and not pd.isna(v):
                            d = f"{y}-{m:02d}-01"
                            conn.execute(
                                "INSERT OR REPLACE INTO macro_indices (date, index_name, value) VALUES (?, ?, ?)",
                                (d, "epu_us", v),
                            )
                            stored += 1
                    except (ValueError, TypeError):
                        continue
                conn.commit()
                logger.info(f"  Stored {stored} EPU monthly values")
                return stored

        logger.warning("  Could not parse EPU format")
        return 0
    except Exception as e:
        logger.error(f"  EPU download failed: {e}")
        return 0


def ingest_gpr(conn) -> int:
    """Ingest Geopolitical Risk Index."""
    logger.info("Downloading GPR index...")
    try:
        # GPR is an Excel file
        df = pd.read_excel(GPR_URL)
        logger.info(f"  Columns: {list(df.columns)[:10]}")
        logger.info(f"  Rows: {len(df)}")

        # GPR has a date column and GPR value columns
        date_col = None
        for c in df.columns:
            if 'date' in str(c).lower():
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]  # first column is usually date

        # Find GPR column
        gpr_col = None
        for c in df.columns:
            if 'gpr' in str(c).lower() and 'act' not in str(c).lower() and 'threat' not in str(c).lower():
                gpr_col = c
                break
        if gpr_col is None:
            gpr_col = df.columns[1]  # second column

        logger.info(f"  Using: date={date_col}, gpr={gpr_col}")

        stored = 0
        for _, row in df.iterrows():
            try:
                d = pd.Timestamp(row[date_col])
                v = float(row[gpr_col])
                if d.year >= 2010 and not pd.isna(v):
                    date_str = d.strftime("%Y-%m-%d")
                    conn.execute(
                        "INSERT OR REPLACE INTO macro_indices (date, index_name, value) VALUES (?, ?, ?)",
                        (date_str, "gpr_daily", v),
                    )
                    stored += 1
            except (ValueError, TypeError):
                continue
        conn.commit()
        logger.info(f"  Stored {stored} GPR daily values")
        return stored
    except Exception as e:
        logger.error(f"  GPR download failed: {e}")
        return 0


@click.command()
def main():
    """Ingest EPU and GPR macro indices."""
    conn = get_db_connection()
    create_table(conn)

    epu_count = ingest_epu(conn)
    gpr_count = ingest_gpr(conn)

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM macro_indices").fetchone()[0]
    indices = conn.execute("SELECT index_name, COUNT(*), MIN(date), MAX(date) FROM macro_indices GROUP BY index_name").fetchall()

    conn.close()

    print(f"\nMacro indices summary:")
    print(f"  Total records: {total}")
    for r in indices:
        print(f"  {r[0]}: {r[1]} records ({r[2]} to {r[3]})")


if __name__ == "__main__":
    main()
