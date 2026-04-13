"""
Model 2 inference: score company exposure to a geopolitical event.

Usage:
    from models.exposure_scorer.predict import ExposureScorer
    scorer = ExposureScorer()
    result = scorer.score(
        event_category="armed_conflict_instability",
        ticker="AAPL",
        mention_sentiment=-0.5,
        car_1_5=-0.03,
    )
    # → {"channel_prediction": "procurement_supply_chain",
    #    "channel_probabilities": {...},
    #    "severity_score": -0.45,
    #    "top_3_channels": [...]}

    # CLI
    python models/exposure_scorer/predict.py --event armed_conflict_instability --ticker AAPL
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.utils import get_db_connection

MODEL_DIR = Path(__file__).parent / "saved"

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


class ExposureScorer:
    """Score company exposure to geopolitical events using XGBoost."""

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else MODEL_DIR
        self.channel_clf = xgb.XGBClassifier()
        self.channel_clf.load_model(path / "channel_classifier.json")
        self.severity_reg = xgb.XGBRegressor()
        self.severity_reg.load_model(path / "severity_regressor.json")

        # Pre-load financial data for lookups
        self._fin_cache = {}
        self._mention_cache = {}

    def _load_company_data(self, ticker: str):
        """Lazily load financial and mention data for a ticker."""
        if ticker in self._fin_cache:
            return

        conn = get_db_connection()

        # Latest financial delta
        row = conn.execute("""
            SELECT * FROM financial_deltas
            WHERE ticker = ? AND revenue_standalone IS NOT NULL
            ORDER BY fiscal_year DESC, fiscal_period DESC
            LIMIT 1
        """, (ticker,)).fetchone()
        self._fin_cache[ticker] = dict(row) if row else {}

        # Mention signals by category
        rows = conn.execute("""
            SELECT primary_category,
                   COUNT(*) as mention_count,
                   AVG(specificity_score) as avg_specificity,
                   MAX(specificity_score) as max_specificity,
                   AVG(keyword_count) as avg_keywords
            FROM geopolitical_mentions
            WHERE ticker = ?
            GROUP BY primary_category
        """, (ticker,)).fetchall()
        self._mention_cache[ticker] = {r["primary_category"]: dict(r) for r in rows}
        conn.close()

    def score(
        self,
        event_category: str,
        ticker: str = "",
        gics_sector: int = 0,
        mention_sentiment: float = 0.0,
        car_1_5: float = 0.0,
        car_1_30: float = 0.0,
        rev_delta_pct: float = 0.0,
    ) -> dict:
        """
        Score a company's exposure to a geopolitical event.

        Args:
            event_category: one of 8 taxonomy categories
            ticker: company ticker (used to look up financials + mentions)
            gics_sector: GICS sector code (first 2 digits)
            mention_sentiment: sentiment of relevant mention (-1 to 1)
            car_1_5: 5-day cumulative abnormal return
            car_1_30: 30-day cumulative abnormal return
            rev_delta_pct: known revenue delta percentage

        Returns:
            dict with channel_prediction, channel_probabilities, severity_score, top_3_channels
        """
        # Load company data if ticker provided
        rev_yoy, gm, gm_delta, log_rev = 0.0, 0.0, 0.0, 0.0
        mention_count, avg_spec, max_spec, avg_kw = 0.0, 0.0, 0.0, 0.0

        if ticker:
            self._load_company_data(ticker)
            fd = self._fin_cache.get(ticker, {})
            rev_yoy = fd.get("revenue_yoy_pct", 0.0) or 0.0
            gm = fd.get("gross_margin", 0.0) or 0.0
            gm_delta = fd.get("gross_margin_delta_pp", 0.0) or 0.0
            rev_s = fd.get("revenue_standalone", 0.0) or 0.0
            log_rev = np.log1p(abs(rev_s) / 1e6) if rev_s else 0.0

            ms = self._mention_cache.get(ticker, {}).get(event_category, {})
            mention_count = ms.get("mention_count", 0) or 0
            avg_spec = ms.get("avg_specificity", 0.0) or 0.0
            max_spec = ms.get("max_specificity", 0.0) or 0.0
            avg_kw = ms.get("avg_keywords", 0.0) or 0.0

        cat_features = [1.0 if c == event_category else 0.0 for c in EVENT_CATEGORIES]

        features = np.array(
            cat_features + [
                gics_sector, mention_sentiment, car_1_5, car_1_5, car_1_30,
                rev_yoy, gm, gm_delta, log_rev,
                mention_count, avg_spec, max_spec, avg_kw,
                rev_delta_pct,
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        # Channel prediction
        channel_probs = self.channel_clf.predict_proba(features)[0]
        pred_idx = channel_probs.argmax()

        # Severity prediction
        severity = float(self.severity_reg.predict(features)[0])

        # Top 3 channels
        top3_idx = channel_probs.argsort()[::-1][:3]
        top3 = [
            {"channel": IMPACT_CHANNELS[i], "probability": float(round(channel_probs[i], 4))}
            for i in top3_idx
        ]

        return {
            "channel_prediction": IMPACT_CHANNELS[pred_idx],
            "channel_confidence": float(round(channel_probs[pred_idx], 4)),
            "channel_probabilities": {
                IMPACT_CHANNELS[i]: float(round(channel_probs[i], 4))
                for i in range(len(IMPACT_CHANNELS))
            },
            "severity_score": round(severity, 4),
            "top_3_channels": top3,
        }


@click.command()
@click.option("--event", required=True, type=click.Choice(EVENT_CATEGORIES))
@click.option("--ticker", default="", help="Company ticker")
@click.option("--sentiment", default=0.0, type=float, help="Mention sentiment (-1 to 1)")
@click.option("--car5", default=0.0, type=float, help="5-day abnormal return")
def main(event, ticker, sentiment, car5):
    """Score company exposure to a geopolitical event."""
    scorer = ExposureScorer()
    result = scorer.score(
        event_category=event,
        ticker=ticker,
        mention_sentiment=sentiment,
        car_1_5=car5,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
