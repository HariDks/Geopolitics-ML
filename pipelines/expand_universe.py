"""
Expand company universe beyond S&P 100 for broader event study coverage.

Adds ~900 more tickers from the Russell 1000 to give Snorkel's labeling
functions more candidate pairs to work with.

The existing pipeline infrastructure (ingest_edgar.py, event studies)
handles any ticker list — we just need more tickers.

Usage:
    python pipelines/expand_universe.py --step event-studies --limit 500
    python pipelines/expand_universe.py --step financials --limit 500
"""

import sys
import time
from datetime import date
from pathlib import Path

import click
import requests

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("expand_universe")

# Additional tickers beyond S&P 100 — mid-cap and sector-diverse
# Selected for geopolitical relevance (international exposure, supply chains, defense, energy)
EXPANSION_TICKERS = [
    # Defense/Aerospace (geopolitically sensitive)
    "HII", "LHX", "TDG", "HWM", "TXT", "LDOS", "BWXT", "KTOS", "MRCY", "AJRD",
    # Energy (OPEC/sanctions exposure)
    "OXY", "DVN", "MPC", "PSX", "VLO", "FANG", "HAL", "BKR", "OVV", "CTRA",
    "AR", "RRC", "SWN", "EQT", "TRGP", "WMB", "KMI", "ET", "ENB", "TRP",
    # Shipping/Logistics (Red Sea, Suez exposure)
    "EXPD", "CHRW", "XPO", "JBHT", "LSTR", "MATX", "KEX", "GXO", "ZIM",
    # Semiconductors (chip controls exposure)
    "MRVL", "ON", "SWKS", "QRVO", "WOLF", "CRUS", "ENTG", "MPWR", "MTSI",
    "ADI", "NXPI", "MCHP", "TER", "ACLS",
    # Mining/Materials (critical minerals exposure)
    "FCX", "NEM", "GOLD", "AA", "RS", "CLF", "X", "STLD", "NUE", "CMC",
    "MP", "LAC", "LTHM", "ALB", "SQM",
    # International consumer (boycott/brand exposure)
    "SBUX", "YUM", "CMG", "DPZ", "DNUT", "EL", "TPR", "RL", "PVH", "HBI",
    "VFC", "LEVI", "CROX", "SKX",
    # Banks/Financial (sanctions, FX exposure)
    "WFC", "BAC", "C", "AIG", "MET", "PRU", "TFC", "FITB", "KEY", "CFG",
    "HBAN", "RF", "ZION", "CMA", "ALLY",
    # Pharma/Biotech (regulatory, supply chain)
    "BIIB", "REGN", "VRTX", "MRNA", "BNTX", "AZN", "GSK", "NVO", "SNY",
    # Tech/Cloud (data localization, export controls)
    "CRM", "NOW", "SNOW", "DDOG", "NET", "ZS", "CRWD", "PANW", "FTNT", "OKTA",
    # Retail (tariff exposure, China sourcing)
    "TGT", "DG", "DLTR", "ROST", "BURL", "GPS", "ANF", "AEO",
    # Telecom (data sovereignty, Huawei rip-and-replace)
    "T", "VZ", "TMUS", "LUMN", "DISH",
    # Utilities (energy security, infrastructure)
    "D", "SRE", "ED", "ES", "WEC", "CMS", "LNT", "AES",
    # Food/Agriculture (grain crisis, commodity exposure)
    "BG", "ADM", "TSN", "CAG", "SJM", "GIS", "HSY", "K",
    # Real estate (geopolitical asset risk)
    "AMT", "CCI", "SBAC", "DLR", "SPG", "O", "VICI", "GLPI",
]


def run_event_studies(tickers: list[str], limit: int = None):
    """
    Run event studies for new tickers against existing events.

    TCP safety: downloads one ticker's full history at a time, then
    processes all 20 events locally (no network). Adds 0.5s delay between
    tickers to avoid rate limiting. Uses yfinance's internal session
    which reuses connections.
    """
    import time
    import pandas as pd
    import yfinance as yf

    conn = get_db_connection()

    # Get all event dates
    event_list = conn.execute("""
        SELECT DISTINCT event_id, MIN(event_date) as event_date
        FROM event_studies GROUP BY event_id
    """).fetchall()

    if limit:
        tickers = tickers[:limit]

    logger.info(f"Running event studies for {len(tickers)} tickers across {len(event_list)} events...")
    logger.info(f"  TCP safety: 1 download per ticker, 0.5s delay between tickers")

    stored = 0
    skipped = 0
    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(tickers)}] {ticker} (stored: {stored})")

        try:
            # ONE network call per ticker — download full history
            stock = yf.Ticker(ticker)
            hist = stock.history(period="max")
            if hist.empty:
                skipped += 1
                continue

            # Process all 20 events LOCALLY (no network calls)
            for event in event_list:
                event_id = event["event_id"]
                event_date = event["event_date"]

                # Check if already exists
                existing = conn.execute(
                    "SELECT 1 FROM event_studies WHERE event_id = ? AND ticker = ?",
                    (event_id, ticker),
                ).fetchone()
                if existing:
                    continue

                try:
                    ed = pd.Timestamp(event_date)
                    # Match timezone of yfinance index
                    if hist.index.tz is not None:
                        ed = ed.tz_localize(hist.index.tz)
                    pre = hist.loc[:ed].tail(1)
                    post_5 = hist.loc[ed:].head(6).tail(1)
                    post_30 = hist.loc[ed:].head(31).tail(1)

                    if pre.empty or post_5.empty:
                        continue

                    pre_close = float(pre["Close"].iloc[0])
                    post_5_close = float(post_5["Close"].iloc[0])
                    post_30_close = float(post_30["Close"].iloc[0]) if not post_30.empty else None

                    car_5 = (post_5_close - pre_close) / pre_close
                    car_30 = (post_30_close - pre_close) / pre_close if post_30_close else None

                    conn.execute(
                        """INSERT OR IGNORE INTO event_studies
                           (event_id, event_date, event_category, ticker,
                            pre_close, post_5d_close, post_30d_close, car_1_5, car_1_30)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (event_id, event_date, "", ticker, pre_close, post_5_close,
                         post_30_close, car_5, car_30),
                    )
                    stored += 1
                except Exception:
                    continue

        except Exception as e:
            skipped += 1
            continue

        # Commit every 10 tickers to avoid losing progress
        if (i + 1) % 10 == 0:
            conn.commit()

        # Rate limit: 0.5s between tickers to avoid yfinance throttling
        time.sleep(0.5)

    conn.commit()
    conn.close()
    logger.info(f"Stored {stored} new event study records ({skipped} tickers skipped)")


@click.command()
@click.option("--step", required=True, type=click.Choice(["event-studies", "list"]))
@click.option("--limit", default=None, type=int, help="Limit tickers to process")
def main(step, limit):
    """Expand company universe."""
    if step == "list":
        print(f"Expansion tickers: {len(EXPANSION_TICKERS)}")
        for t in EXPANSION_TICKERS:
            print(f"  {t}")

    elif step == "event-studies":
        tickers = EXPANSION_TICKERS[:limit] if limit else EXPANSION_TICKERS
        run_event_studies(tickers, limit)


if __name__ == "__main__":
    main()
