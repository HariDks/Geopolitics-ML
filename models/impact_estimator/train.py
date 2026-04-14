"""
Model 3: Impact Estimator — Quantile regression for financial impact ranges.

Given event + company exposure, predicts:
  - Low/Mid/High financial impact as percentage of revenue
  - Impact timeline (days to peak impact)
  - Confidence score

Uses quantile regression (XGBoost) to produce calibrated prediction intervals.
The 10th/50th/90th percentiles give low/mid/high estimates.

Training data:
  - 163 seed labels with revenue_delta_pct and/or car_1_5 (quantitative targets)
  - 1,973 event studies (car_1_5, car_1_30 for ~99 tickers x 20 events)
  - Financial deltas (YoY revenue change as baseline)

Usage:
    python models/impact_estimator/train.py
    python models/impact_estimator/train.py --eval-only
"""

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.utils import get_db_connection, get_logger

logger = get_logger("impact_estimator")

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = Path(__file__).parent / "saved"
SEED_LABELS_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"

EVENT_CATEGORIES = [
    "trade_policy_actions",
    "sanctions_financial_restrictions",
    "armed_conflict_instability",
    "regulatory_sovereignty_shifts",
    "technology_controls",
    "resource_energy_disruptions",
    "political_transitions_volatility",
    "institutional_alliance_realignment",
]

IMPACT_CHANNELS = [
    "procurement_supply_chain",
    "revenue_market_access",
    "capital_allocation_investment",
    "regulatory_compliance_cost",
    "logistics_operations",
    "innovation_ip",
    "workforce_talent",
    "reputation_stakeholder",
    "financial_treasury",
    "cybersecurity_it",
]

# Reuse the event→category mapping from Model 2
from models.exposure_scorer.train import EVENT_TO_CATEGORY, gics_sector


def safe_float(val, default=None):
    if val is None or val == "" or val == "nan":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


FEATURE_NAMES = (
    [f"cat_{c}" for c in EVENT_CATEGORIES]
    + [f"ch_{c}" for c in IMPACT_CHANNELS]
    + [
        "gics_sector",
        "mention_sentiment",
        "car_1_5_seed",
        "car_1_5_es",
        "car_1_30_es",
        "rev_yoy",
        "gross_margin",
        "gm_delta_pp",
        "log_revenue_M",
        "mention_count",
        "avg_specificity",
        "max_specificity",
        "cogs_delta_pct",
        "oi_delta_pct",
    ]
)


def build_dataset(conn) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build feature matrix and target vector for impact estimation.

    Target: revenue_delta_pct (percentage change in revenue attributed to the event).
    For labels without revenue_delta, use car_1_5 * 100 as a proxy (market-implied impact).

    Returns:
        X: features (n, n_features)
        y: target impact percentage (n,)
        metadata: sample info
    """
    seed_labels = []
    with open(SEED_LABELS_PATH) as f:
        seed_labels = list(csv.DictReader(f))

    # Pre-load DB lookups
    event_studies = {}
    for r in conn.execute("SELECT * FROM event_studies").fetchall():
        event_studies[(r["event_id"], r["ticker"])] = dict(r)

    fin_deltas = {}
    for r in conn.execute("SELECT * FROM financial_deltas WHERE revenue_standalone IS NOT NULL").fetchall():
        fin_deltas[(r["ticker"], r["fiscal_year"], r["fiscal_period"])] = dict(r)

    mention_signals = {}
    for r in conn.execute("""
        SELECT ticker, primary_category,
               COUNT(*) as mention_count,
               AVG(specificity_score) as avg_specificity,
               MAX(specificity_score) as max_specificity
        FROM geopolitical_mentions GROUP BY ticker, primary_category
    """).fetchall():
        mention_signals[(r["ticker"], r["primary_category"])] = dict(r)

    X_rows = []
    y_rows = []
    metadata = []

    # ── Source 1: Seed labels with quantitative targets ──
    for label in seed_labels:
        rev_delta = safe_float(label.get("revenue_delta_pct"))
        car_5 = safe_float(label.get("car_1_5"))

        # Need at least one quantitative signal
        if rev_delta is None and car_5 is None:
            continue

        # Target: prefer revenue_delta, fall back to car_1_5 * 100
        if rev_delta is not None:
            target = rev_delta
        else:
            target = car_5 * 100  # scale to percentage

        event_id = label["event_id"]
        ticker = label["company_ticker"]
        channel = label.get("impact_channel", "")

        event_cat = EVENT_TO_CATEGORY.get(event_id, "")
        cat_features = [1.0 if c == event_cat else 0.0 for c in EVENT_CATEGORIES]
        ch_features = [1.0 if c == channel else 0.0 for c in IMPACT_CHANNELS]

        sector = gics_sector(label.get("sector_gics", ""))
        sentiment = safe_float(label.get("mention_sentiment"), 0.0)
        seed_car = safe_float(label.get("car_1_5"), 0.0)

        # Event study lookup
        from models.exposure_scorer.train import _map_to_event_study_id
        es_id = _map_to_event_study_id(event_id)
        es = event_studies.get((es_id, ticker), {}) if es_id else {}
        es_car_5 = es.get("car_1_5", seed_car) or seed_car
        es_car_30 = es.get("car_1_30", 0.0) or 0.0

        # Financial context
        quarter = label.get("quarter", "")
        fy, fp = _parse_quarter(quarter)
        fd = fin_deltas.get((ticker, fy, fp), {})
        rev_yoy = fd.get("revenue_yoy_pct", 0.0) or 0.0
        gm = fd.get("gross_margin", 0.0) or 0.0
        gm_delta = fd.get("gross_margin_delta_pp", 0.0) or 0.0
        rev_s = fd.get("revenue_standalone", 0.0) or 0.0
        log_rev = np.log1p(abs(rev_s) / 1e6) if rev_s else 0.0

        ms = mention_signals.get((ticker, event_cat), {})
        mention_count = ms.get("mention_count", 0) or 0
        avg_spec = ms.get("avg_specificity", 0.0) or 0.0
        max_spec = ms.get("max_specificity", 0.0) or 0.0

        cogs_delta = safe_float(label.get("cogs_delta_pct"), 0.0)
        oi_delta = safe_float(label.get("operating_income_delta_pct"), 0.0)

        features = (
            cat_features      # 8: event category
            + ch_features     # 10: impact channel
            + [
                sector,
                sentiment,
                seed_car,
                es_car_5,
                es_car_30,
                rev_yoy,
                gm,
                gm_delta,
                log_rev,
                mention_count,
                avg_spec,
                max_spec,
                cogs_delta,
                oi_delta,
            ]
        )

        X_rows.append(features)
        y_rows.append(target)
        metadata.append({
            "event_id": event_id,
            "ticker": ticker,
            "channel": channel,
            "source": "seed_label",
            "rev_delta_original": rev_delta,
            "car_proxy": rev_delta is None,
        })

    # ── Source 2: Event studies (broader coverage, car as target) ──
    # Use event studies for companies not in seed labels
    seed_tickers = set((m["event_id"], m["ticker"]) for m in metadata)

    for (es_id, ticker), es in event_studies.items():
        car_5 = es.get("car_1_5")
        car_30 = es.get("car_1_30")
        if car_5 is None:
            continue

        # Skip if already in seed labels for a related event
        if any(es_id in _map_to_event_study_id(sid) for sid, t in seed_tickers if t == ticker
               if _map_to_event_study_id(sid)):
            continue

        target = car_5 * 100  # market-implied impact percentage

        event_cat = EVENT_TO_CATEGORY.get(es_id, "")
        if not event_cat:
            continue
        cat_features = [1.0 if c == event_cat else 0.0 for c in EVENT_CATEGORIES]
        ch_features = [0.0] * len(IMPACT_CHANNELS)  # unknown channel

        fd = _get_latest_financials(fin_deltas, ticker)
        rev_yoy = fd.get("revenue_yoy_pct", 0.0) or 0.0
        gm = fd.get("gross_margin", 0.0) or 0.0
        gm_delta = fd.get("gross_margin_delta_pp", 0.0) or 0.0
        rev_s = fd.get("revenue_standalone", 0.0) or 0.0
        log_rev = np.log1p(abs(rev_s) / 1e6) if rev_s else 0.0

        ms = mention_signals.get((ticker, event_cat), {})
        mention_count = ms.get("mention_count", 0) or 0
        avg_spec = ms.get("avg_specificity", 0.0) or 0.0
        max_spec = ms.get("max_specificity", 0.0) or 0.0

        features = (
            cat_features + ch_features + [
                0,          # sector unknown
                0.0,        # sentiment unknown
                car_5,      # stock reaction
                car_5,
                car_30 or 0.0,
                rev_yoy, gm, gm_delta, log_rev,
                mention_count, avg_spec, max_spec,
                0.0, 0.0,  # cogs/oi delta unknown
            ]
        )

        X_rows.append(features)
        y_rows.append(target)
        metadata.append({
            "event_id": es_id,
            "ticker": ticker,
            "channel": "",
            "source": "event_study",
            "rev_delta_original": None,
            "car_proxy": True,
        })

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    logger.info(f"Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"  Seed labels: {sum(1 for m in metadata if m['source'] == 'seed_label')}")
    logger.info(f"  Event studies: {sum(1 for m in metadata if m['source'] == 'event_study')}")
    logger.info(f"  Target range: [{y.min():.1f}%, {y.max():.1f}%]")
    logger.info(f"  Target mean: {y.mean():.2f}%, median: {np.median(y):.2f}%")

    return X, y, metadata


def _parse_quarter(quarter: str) -> tuple[int, str]:
    import re
    m = re.match(r"(\d{4})(Q\d|FY)", quarter)
    if m:
        return (int(m.group(1)), m.group(2))
    return (0, "")


def _get_latest_financials(fin_deltas: dict, ticker: str) -> dict:
    best = {}
    best_key = (0, "")
    for (t, fy, fp), data in fin_deltas.items():
        if t == ticker and (fy, fp) > best_key:
            best_key = (fy, fp)
            best = data
    return best


def train_quantile_models(
    X_train, y_train, X_val, y_val
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor, xgb.XGBRegressor]:
    """
    Train three XGBoost models for quantile regression:
    - q05: 5th percentile (optimistic / low impact)
    - q50: 50th percentile (median / mid impact)
    - q95: 95th percentile (pessimistic / high impact)

    Using q05/q95 instead of q10/q90 for wider prediction intervals.
    Target: 90% of actual outcomes fall within [q05, q95].
    Previous q10/q90 only achieved 60.6% coverage — too narrow.
    """
    models = {}
    for quantile, name in [(0.05, "q10"), (0.5, "q50"), (0.95, "q90")]:
        logger.info(f"Training {name} (quantile={quantile})...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[name] = model

    return models["q10"], models["q50"], models["q90"]


def evaluate_models(q10, q50, q90, X_val, y_val, metadata_val):
    """Evaluate calibration and accuracy of quantile predictions."""
    pred_low = q10.predict(X_val)
    pred_mid = q50.predict(X_val)
    pred_high = q90.predict(X_val)

    # Calibration: what fraction of actuals fall within [q05, q95]?
    in_range = np.sum((y_val >= pred_low) & (y_val <= pred_high))
    coverage = in_range / len(y_val)

    # MAE of median prediction
    mae_mid = np.mean(np.abs(y_val - pred_mid))

    # RMSE of median
    rmse_mid = np.sqrt(np.mean((y_val - pred_mid) ** 2))

    # Interval width
    avg_width = np.mean(pred_high - pred_low)

    print("\n" + "=" * 70)
    print("IMPACT ESTIMATOR — EVALUATION")
    print("=" * 70)
    print(f"  Samples:           {len(y_val)}")
    print(f"  Coverage (q10-q90): {coverage:.1%} (target: 80%)")
    print(f"  MAE (median):       {mae_mid:.2f} pp")
    print(f"  RMSE (median):      {rmse_mid:.2f} pp")
    print(f"  Avg interval width: {avg_width:.2f} pp")
    print()

    # Breakdown by source
    seed_mask = [m["source"] == "seed_label" for m in metadata_val]
    es_mask = [m["source"] == "event_study" for m in metadata_val]

    if any(seed_mask):
        seed_idx = [i for i, s in enumerate(seed_mask) if s]
        seed_y = y_val[seed_idx]
        seed_pred = pred_mid[seed_idx]
        seed_low = pred_low[seed_idx]
        seed_high = pred_high[seed_idx]
        seed_cov = np.sum((seed_y >= seed_low) & (seed_y <= seed_high)) / len(seed_y)
        seed_mae = np.mean(np.abs(seed_y - seed_pred))
        print(f"  Seed labels only:")
        print(f"    Coverage: {seed_cov:.1%}, MAE: {seed_mae:.2f} pp, n={len(seed_idx)}")

    if any(es_mask):
        es_idx = [i for i, s in enumerate(es_mask) if s]
        es_y = y_val[es_idx]
        es_pred = pred_mid[es_idx]
        es_low = pred_low[es_idx]
        es_high = pred_high[es_idx]
        es_cov = np.sum((es_y >= es_low) & (es_y <= es_high)) / len(es_y)
        es_mae = np.mean(np.abs(es_y - es_pred))
        print(f"  Event studies only:")
        print(f"    Coverage: {es_cov:.1%}, MAE: {es_mae:.2f} pp, n={len(es_idx)}")

    # Show example predictions for extreme cases
    print("\n  SAMPLE PREDICTIONS (seed labels):")
    print(f"  {'Ticker':8s} {'Event':35s} {'Actual':>8s} {'Low':>8s} {'Mid':>8s} {'High':>8s}")
    print("  " + "-" * 80)

    seed_indices = [i for i, m in enumerate(metadata_val) if m["source"] == "seed_label"]
    # Sort by actual impact (most negative first)
    seed_sorted = sorted(seed_indices, key=lambda i: y_val[i])
    show = seed_sorted[:5] + seed_sorted[-5:] if len(seed_sorted) > 10 else seed_sorted
    for i in show:
        m = metadata_val[i]
        print(f"  {m['ticker']:8s} {m['event_id'][:35]:35s} "
              f"{y_val[i]:>7.1f}% {pred_low[i]:>7.1f}% {pred_mid[i]:>7.1f}% {pred_high[i]:>7.1f}%")

    # Feature importance from q50
    print("\n  TOP 10 FEATURES (median model):")
    importances = q50.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
        print(f"    {name:35s} {importances[i]:.4f}")


@click.command()
@click.option("--eval-only", is_flag=True, help="Only evaluate saved model")
def main(eval_only):
    """Train or evaluate the impact estimator."""
    conn = get_db_connection()
    X, y, metadata = build_dataset(conn)
    conn.close()

    # Split — stratify by source to ensure seed labels in both train/val
    sources = [m["source"] for m in metadata]
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X, y, metadata, test_size=0.2, random_state=42,
        stratify=sources,
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    if eval_only:
        q10 = xgb.XGBRegressor(); q10.load_model(MODEL_DIR / "q10.json")
        q50 = xgb.XGBRegressor(); q50.load_model(MODEL_DIR / "q50.json")
        q90 = xgb.XGBRegressor(); q90.load_model(MODEL_DIR / "q90.json")
    else:
        q10, q50, q90 = train_quantile_models(X_train, y_train, X_val, y_val)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        q10.save_model(MODEL_DIR / "q10.json")
        q50.save_model(MODEL_DIR / "q50.json")
        q90.save_model(MODEL_DIR / "q90.json")

        with open(MODEL_DIR / "config.json", "w") as f:
            json.dump({
                "feature_names": FEATURE_NAMES,
                "target": "impact_pct (revenue delta or car*100 proxy)",
                "quantiles": [0.05, 0.5, 0.95],
            }, f, indent=2)

        logger.info(f"Models saved to {MODEL_DIR}")

    evaluate_models(q10, q50, q90, X_val, y_val, meta_val)
    logger.info("Done.")


if __name__ == "__main__":
    main()
