"""
Prediction logger — stores every prediction and correction in the database.

Replaces the CSV-based feedback system with a proper relational store
that supports retraining, drift detection, and accuracy tracking.

Usage:
    from pipelines.prediction_logger import log_prediction, log_correction

    pred_id = log_prediction(
        input_text="Russia invaded Ukraine...",
        input_ticker="BP",
        results=pipeline_output,
    )

    log_correction(
        prediction_id=pred_id,
        useful="No",
        correct_channel="capital_allocation_investment",
    )
"""

import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_db_connection, get_logger

logger = get_logger("prediction_logger")


def log_prediction(
    input_text: str,
    input_ticker: str = "",
    input_company: str = "",
    input_revenue: float = 0.0,
    results: dict = None,
    user_id: str = "dashboard",
) -> str:
    """
    Log a prediction to the database. Returns the prediction_id.
    """
    if results is None:
        return ""

    pred_id = f"PRED-{uuid.uuid4().hex[:12]}"

    evt = results.get("evt", {})
    exp = results.get("exp", {})
    imp = results.get("imp", {})

    probs = exp.get("channel_probabilities", {})
    ranked = sorted(probs.items(), key=lambda x: -x[1])

    try:
        conn = get_db_connection()
        conn.execute(
            """INSERT INTO predictions
               (prediction_id, user_id, input_text, input_ticker, input_company, input_revenue,
                predicted_category, predicted_category_confidence,
                predicted_channel_1, predicted_channel_2, predicted_channel_confidence,
                channel_mode, channel_reliability,
                predicted_impact_low, predicted_impact_mid, predicted_impact_high)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id, user_id, input_text[:1000], input_ticker, input_company, input_revenue,
                evt.get("category", ""), evt.get("confidence", 0.0),
                ranked[0][0] if len(ranked) > 0 else "",
                ranked[1][0] if len(ranked) > 1 else "",
                ranked[0][1] if len(ranked) > 0 else 0.0,
                exp.get("channel_mode", ""),
                exp.get("channel_reliability", ""),
                imp.get("impact_low_pct", 0.0),
                imp.get("impact_mid_pct", 0.0),
                imp.get("impact_high_pct", 0.0),
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Logged prediction {pred_id}")
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")
        return ""

    return pred_id


def log_correction(
    prediction_id: str,
    useful: str = "",
    correct_category: str = "",
    correct_channel: str = "",
    actual_impact_pct: float = None,
    notes: str = "",
    reviewer_id: str = "dashboard_user",
) -> str:
    """Log a correction linked to a prediction."""
    corr_id = f"CORR-{uuid.uuid4().hex[:12]}"

    try:
        conn = get_db_connection()
        conn.execute(
            """INSERT INTO corrections
               (correction_id, prediction_id, reviewer_id, useful,
                correct_category, correct_channel, actual_impact_pct, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                corr_id, prediction_id, reviewer_id, useful,
                correct_category or None, correct_channel or None,
                actual_impact_pct, notes or None,
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Logged correction {corr_id} for prediction {prediction_id}")
    except Exception as e:
        logger.warning(f"Failed to log correction: {e}")
        return ""

    return corr_id


def get_prediction_stats() -> dict:
    """Get summary stats on predictions and corrections."""
    try:
        conn = get_db_connection()
        total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        total_corrs = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        useful_yes = conn.execute("SELECT COUNT(*) FROM corrections WHERE useful='Yes'").fetchone()[0]
        useful_no = conn.execute("SELECT COUNT(*) FROM corrections WHERE useful='No'").fetchone()[0]
        channel_corrections = conn.execute(
            "SELECT COUNT(*) FROM corrections WHERE correct_channel IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        return {
            "total_predictions": total_preds,
            "total_corrections": total_corrs,
            "useful_yes": useful_yes,
            "useful_no": useful_no,
            "channel_corrections": channel_corrections,
        }
    except Exception:
        return {"total_predictions": 0, "total_corrections": 0}


def get_corrections_for_retraining() -> list[dict]:
    """
    Get all corrections with their original predictions for retraining.
    Returns dicts with input_text, input_ticker, correct_channel, etc.
    """
    try:
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT p.input_text, p.input_ticker, p.input_company,
                   p.predicted_category, p.predicted_channel_1,
                   c.correct_category, c.correct_channel, c.actual_impact_pct,
                   c.useful, c.notes
            FROM corrections c
            JOIN predictions p ON c.prediction_id = p.prediction_id
            WHERE c.correct_channel IS NOT NULL OR c.correct_category IS NOT NULL
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []
