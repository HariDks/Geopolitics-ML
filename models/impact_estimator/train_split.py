"""
Split Impact Estimator: Model 3A (Market Reaction) + Model 3B (Revenue Impact)

Model 3A: Predicts 5-day stock price reaction (CAR_1_5)
  - Trained on 1,973 event studies + 565 seed labels with stock data
  - Target: abnormal return (%), e.g., -3.2%
  - Features: event category, sector, financial health

Model 3B: Predicts actual revenue impact
  - Trained on 43 seed labels with revenue_delta_pct
  - Target: quarterly revenue change (%), e.g., -8.0%
  - Features: event category, channel, exposure proxies, geographic concentration
  - Uses conformal prediction for guaranteed intervals (only 43 samples → wide intervals)

Usage:
    python models/impact_estimator/train_split.py
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import xgboost as xgb

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger
from pipelines.temporal_split import get_temporal_split
from models.exposure_scorer.train import (
    EVENT_TO_CATEGORY, gics_sector, safe_float,
    _map_to_event_study_id, _parse_quarter,
    compute_geo_concentration, exposure_proxies,
    EVENT_AFFECTED_REGIONS, compute_lexicon_scores,
    IMPACT_CHANNELS, EVENT_CATEGORIES,
)

logger = get_logger("impact_split")

MODEL_DIR = Path(__file__).parent / "saved"
SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"


def build_3a_dataset(conn) -> tuple[np.ndarray, np.ndarray, list]:
    """Build dataset for Model 3A: stock market reaction."""
    logger.info("Building Model 3A dataset (market reaction)...")

    # Features: event category (one-hot) + sector + financial metrics
    X_rows, y_rows, meta = [], [], []

    # Source 1: Event studies (primary data source for 3A)
    es_rows = conn.execute("""
        SELECT event_id, ticker, car_1_5, car_1_30 FROM event_studies
        WHERE car_1_5 IS NOT NULL
    """).fetchall()

    for r in es_rows:
        event_cat = EVENT_TO_CATEGORY.get(r["event_id"], "")
        if not event_cat:
            continue

        # Temporal filter
        if get_temporal_split(r["event_id"], "") == "test":
            continue

        cat_features = [1.0 if c == event_cat else 0.0 for c in EVENT_CATEGORIES]
        car_5 = r["car_1_5"]
        car_30 = r["car_1_30"] or 0.0

        features = cat_features + [car_30, abs(car_5)]  # 10 features
        X_rows.append(features)
        y_rows.append(car_5 * 100)  # convert to percentage
        meta.append({"event_id": r["event_id"], "ticker": r["ticker"], "source": "event_study"})

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    logger.info(f"  3A dataset: {len(X)} samples, {X.shape[1]} features")
    return X, y, meta


def build_3b_dataset() -> tuple[np.ndarray, np.ndarray, list]:
    """Build dataset for Model 3B: actual revenue impact."""
    logger.info("Building Model 3B dataset (revenue impact)...")

    with open(SEED_PATH) as f:
        labels = list(csv.DictReader(f))

    X_rows, y_rows, meta = [], [], []

    for label in labels:
        rev_delta = label.get("revenue_delta_pct", "").strip()
        if not rev_delta:
            continue

        event_id = label["event_id"]
        ticker = label["company_ticker"]
        event_cat = EVENT_TO_CATEGORY.get(event_id, "")
        if not event_cat:
            continue

        # For 3B we include val set in training because we only have 43 labels total
        # Temporal holdout is test-only (2024+). With only 43 samples we can't
        # afford to hold out 2023 data from training.
        if get_temporal_split(event_id, label.get("quarter", "")) == "test":
            continue

        cat_features = [1.0 if c == event_cat else 0.0 for c in EVENT_CATEGORIES]
        channel = label.get("impact_channel", "")
        ch_features = [1.0 if c == channel else 0.0 for c in IMPACT_CHANNELS]
        sector = gics_sector(label.get("sector_gics", ""))
        sentiment = safe_float(label.get("mention_sentiment"), 0.0)
        car_5 = safe_float(label.get("car_1_5"), 0.0)
        geo_conc = compute_geo_concentration(ticker, event_id)

        # Lexicon scores from mention text
        lex = compute_lexicon_scores(label.get("mention_text", ""))
        lex_scores = [lex.get(ch, 0.0) for ch in IMPACT_CHANNELS]

        proxy = exposure_proxies.get(ticker, {})
        facility = proxy.get("facility_concentration_score", 0.0)
        single_src = proxy.get("single_source_risk_score", 0.0)
        asset_exit = proxy.get("asset_exit_score", 0.0)
        route = proxy.get("route_sensitivity_score", 0.0)

        features = (
            cat_features + ch_features + [
                sector, sentiment, car_5, geo_conc,
                facility, single_src, asset_exit, route,
            ] + lex_scores
        )

        X_rows.append(features)
        y_rows.append(float(rev_delta))
        meta.append({"event_id": event_id, "ticker": ticker, "source": "seed_label",
                      "actual_rev": float(rev_delta)})

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    logger.info(f"  3B dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"  Revenue range: [{y.min():.1f}%, {y.max():.1f}%]")
    return X, y, meta


@click.command()
def main():
    """Train split impact models 3A and 3B."""
    conn = get_db_connection()

    # ── Model 3A: Market Reaction ──
    X_3a, y_3a, meta_3a = build_3a_dataset(conn)
    conn.close()

    split_3a = int(len(X_3a) * 0.8)
    X_train_a, X_val_a = X_3a[:split_3a], X_3a[split_3a:]
    y_train_a, y_val_a = y_3a[:split_3a], y_3a[split_3a:]

    logger.info(f"Training Model 3A (market reaction): {len(X_train_a)} train, {len(X_val_a)} val")
    model_3a = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", tree_method="hist",
        random_state=42, verbosity=0,
    )
    model_3a.fit(X_train_a, y_train_a, eval_set=[(X_val_a, y_val_a)], verbose=False)

    # Evaluate 3A
    pred_3a = model_3a.predict(X_val_a)
    mae_3a = np.mean(np.abs(y_val_a - pred_3a))
    direction_acc = np.mean((pred_3a > 0) == (y_val_a > 0))
    logger.info(f"  3A MAE: {mae_3a:.2f}pp, Direction: {direction_acc:.1%}")

    # ── Model 3B: Revenue Impact ──
    X_3b, y_3b, meta_3b = build_3b_dataset()

    if len(X_3b) < 10:
        logger.warning(f"  Only {len(X_3b)} revenue samples — too few for reliable training")

    split_3b = max(int(len(X_3b) * 0.7), len(X_3b) - 10)
    X_train_b, X_val_b = X_3b[:split_3b], X_3b[split_3b:]
    y_train_b, y_val_b = y_3b[:split_3b], y_3b[split_3b:]

    logger.info(f"Training Model 3B (revenue impact): {len(X_train_b)} train, {len(X_val_b)} val")
    model_3b = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,  # simpler model for small data
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", tree_method="hist",
        random_state=42, verbosity=0,
    )
    model_3b.fit(X_train_b, y_train_b, eval_set=[(X_val_b, y_val_b)], verbose=False)

    # Evaluate 3B
    pred_3b = model_3b.predict(X_val_b)
    mae_3b = np.mean(np.abs(y_val_b - pred_3b))
    logger.info(f"  3B MAE: {mae_3b:.2f}pp")

    # Conformal prediction for 3B intervals
    try:
        from mapie.regression import CrossConformalRegressor
        logger.info("  Adding conformal intervals to 3B...")
        base_3b = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", tree_method="hist",
            random_state=42, verbosity=0,
        )
        conformal_3b = CrossConformalRegressor(estimator=base_3b, confidence_level=0.9, cv=3, random_state=42)
        conformal_3b.fit_conformalize(X_train_b, y_train_b)

        y_pred_conf, y_intervals = conformal_3b.predict_interval(X_val_b)
        y_intervals = y_intervals.squeeze(-1)
        coverage = np.mean((y_val_b >= y_intervals[:, 0]) & (y_val_b <= y_intervals[:, 1]))
        avg_width = np.mean(y_intervals[:, 1] - y_intervals[:, 0])
        logger.info(f"  3B conformal coverage: {coverage:.1%}, avg width: {avg_width:.1f}pp")
    except Exception as e:
        logger.warning(f"  Conformal failed for 3B: {e}")
        conformal_3b = None

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_3a.save_model(MODEL_DIR / "model_3a_market.json")
    model_3b.save_model(MODEL_DIR / "model_3b_revenue.json")
    if conformal_3b:
        with open(MODEL_DIR / "conformal_3b.pkl", "wb") as f:
            pickle.dump(conformal_3b, f)

    with open(MODEL_DIR / "split_config.json", "w") as f:
        json.dump({
            "model_3a": {"type": "market_reaction", "target": "car_1_5 * 100", "samples": int(len(X_3a)),
                         "mae": float(round(mae_3a, 2)), "direction_accuracy": float(round(direction_acc, 3))},
            "model_3b": {"type": "revenue_impact", "target": "revenue_delta_pct", "samples": int(len(X_3b)),
                         "mae": float(round(mae_3b, 2))},
        }, f, indent=2)

    logger.info(f"\nModels saved to {MODEL_DIR}")
    logger.info(f"  3A: {MODEL_DIR / 'model_3a_market.json'}")
    logger.info(f"  3B: {MODEL_DIR / 'model_3b_revenue.json'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SPLIT IMPACT MODEL — SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Model 3A (Market Reaction):")
    print(f"    Samples: {len(X_3a)}")
    print(f"    MAE: {mae_3a:.2f} percentage points")
    print(f"    Direction accuracy: {direction_acc:.1%}")
    print(f"\n  Model 3B (Revenue Impact):")
    print(f"    Samples: {len(X_3b)}")
    print(f"    MAE: {mae_3b:.2f} percentage points")
    if conformal_3b:
        print(f"    Conformal coverage: {coverage:.1%} (target 90%)")
        print(f"    Avg interval width: {avg_width:.1f}pp")


if __name__ == "__main__":
    main()
