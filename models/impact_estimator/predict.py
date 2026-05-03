"""
Model 3 inference: estimate financial impact ranges for company-event pairs.

Usage:
    from models.impact_estimator.predict import ImpactEstimator
    est = ImpactEstimator()
    result = est.estimate(
        event_category="armed_conflict_instability",
        impact_channel="revenue_market_access",
        ticker="MCD",
        mention_sentiment=-0.7,
        car_1_5=-0.01,
        revenue_usd=23_000_000_000,
    )
    # → {"impact_low_pct": -1.2, "impact_mid_pct": -3.5, "impact_high_pct": -8.1,
    #    "impact_low_usd": -276M, "impact_mid_usd": -805M, "impact_high_usd": -1.86B,
    #    "confidence": 0.72}

    # CLI
    python models/impact_estimator/predict.py \\
        --event armed_conflict_instability --channel revenue_market_access \\
        --ticker MCD --sentiment -0.7 --car5 -0.01 --revenue 23000000000
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


class ImpactEstimator:
    """Estimate financial impact ranges using Model 3A (market) + 3B (revenue with conformal intervals)."""

    def __init__(self, model_path: str | Path | None = None):
        import pickle
        path = Path(model_path) if model_path else MODEL_DIR

        # Try loading split models (3A market + 3B revenue with conformal)
        self._use_conformal = False
        if (path / "model_3a_market.json").exists() and (path / "conformal_3b.pkl").exists():
            self.model_3a = xgb.XGBRegressor()
            self.model_3a.load_model(path / "model_3a_market.json")
            with open(path / "conformal_3b.pkl", "rb") as f:
                self.conformal_3b = pickle.load(f)
            self._use_conformal = True

        # Fallback to quantile models
        self.q10 = xgb.XGBRegressor(); self.q10.load_model(path / "q10.json")
        self.q50 = xgb.XGBRegressor(); self.q50.load_model(path / "q50.json")
        self.q90 = xgb.XGBRegressor(); self.q90.load_model(path / "q90.json")

        self._fin_cache = {}
        self._mention_cache = {}

    def _load_company_data(self, ticker: str):
        if ticker in self._fin_cache:
            return
        try:
            conn = get_db_connection()
            row = conn.execute("""
                SELECT * FROM financial_deltas
                WHERE ticker = ? AND revenue_standalone IS NOT NULL
                ORDER BY fiscal_year DESC, fiscal_period DESC LIMIT 1
            """, (ticker,)).fetchone()
            self._fin_cache[ticker] = dict(row) if row else {}

            rows = conn.execute("""
                SELECT primary_category,
                       COUNT(*) as mention_count,
                       AVG(specificity_score) as avg_specificity,
                       MAX(specificity_score) as max_specificity
                FROM geopolitical_mentions WHERE ticker = ?
                GROUP BY primary_category
            """, (ticker,)).fetchall()
            self._mention_cache[ticker] = {r["primary_category"]: dict(r) for r in rows}
            conn.close()
        except Exception:
            self._fin_cache[ticker] = {}
            self._mention_cache[ticker] = {}

    def estimate(
        self,
        event_category: str,
        impact_channel: str = "",
        ticker: str = "",
        gics_sector: int = 0,
        mention_sentiment: float = 0.0,
        car_1_5: float = 0.0,
        car_1_30: float = 0.0,
        cogs_delta_pct: float = 0.0,
        oi_delta_pct: float = 0.0,
        revenue_usd: float = 0.0,
    ) -> dict:
        """
        Estimate financial impact range.

        Returns impact as percentage and optionally in USD if revenue_usd is provided.
        """
        rev_yoy, gm, gm_delta, log_rev = 0.0, 0.0, 0.0, 0.0
        mention_count, avg_spec, max_spec = 0.0, 0.0, 0.0

        if ticker:
            self._load_company_data(ticker)
            fd = self._fin_cache.get(ticker, {})
            rev_yoy = fd.get("revenue_yoy_pct", 0.0) or 0.0
            gm = fd.get("gross_margin", 0.0) or 0.0
            gm_delta = fd.get("gross_margin_delta_pp", 0.0) or 0.0
            rev_s = fd.get("revenue_standalone", 0.0) or 0.0
            log_rev = np.log1p(abs(rev_s) / 1e6) if rev_s else 0.0
            if revenue_usd == 0.0 and rev_s:
                revenue_usd = abs(rev_s) * 4  # annualize quarterly

            ms = self._mention_cache.get(ticker, {}).get(event_category, {})
            mention_count = ms.get("mention_count", 0) or 0
            avg_spec = ms.get("avg_specificity", 0.0) or 0.0
            max_spec = ms.get("max_specificity", 0.0) or 0.0

        cat_features = [1.0 if c == event_category else 0.0 for c in EVENT_CATEGORIES]
        ch_features = [1.0 if c == impact_channel else 0.0 for c in IMPACT_CHANNELS]

        features = np.array(
            cat_features + ch_features + [
                gics_sector, mention_sentiment, car_1_5, car_1_5, car_1_30,
                rev_yoy, gm, gm_delta, log_rev,
                mention_count, avg_spec, max_spec,
                cogs_delta_pct, oi_delta_pct,
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        # Use conformal prediction if available (guaranteed coverage intervals)
        if self._use_conformal:
            try:
                # Model 3A: market reaction (point estimate)
                cat_only = np.array(
                    [1.0 if c == event_category else 0.0 for c in EVENT_CATEGORIES]
                    + [car_1_30, abs(car_1_5)],
                    dtype=np.float32,
                ).reshape(1, -1)
                market_pct = float(self.model_3a.predict(cat_only)[0])

                # Model 3B: revenue impact with conformal intervals
                # 3B features: event cats (8) + channels (10) + [sector, sentiment,
                # car_5, geo_conc, facility, single_src, asset_exit, route] + lexicon (10) = 36
                from models.exposure_scorer.train import (
                    compute_geo_concentration, exposure_proxies as _proxies,
                    compute_lexicon_scores,
                )
                geo_conc = 0.0
                proxy = _proxies.get(ticker, {})
                facility = proxy.get("facility_concentration_score", 0.0)
                single_src = proxy.get("single_source_risk_score", 0.0)
                asset_exit = proxy.get("asset_exit_score", 0.0)
                route = proxy.get("route_sensitivity_score", 0.0)
                lex_scores = [0.0] * len(IMPACT_CHANNELS)  # no event text available here

                features_3b = np.array(
                    [1.0 if c == event_category else 0.0 for c in EVENT_CATEGORIES]
                    + [1.0 if c == impact_channel else 0.0 for c in IMPACT_CHANNELS]
                    + [gics_sector, mention_sentiment, car_1_5, geo_conc,
                       facility, single_src, asset_exit, route]
                    + lex_scores,
                    dtype=np.float32,
                ).reshape(1, -1)

                y_pred, y_intervals = self.conformal_3b.predict_interval(features_3b)
                y_intervals = y_intervals.squeeze(-1)
                mid = float(y_pred[0])
                low = float(y_intervals[0, 0])
                high = float(y_intervals[0, 1])
                low, mid, high = sorted([low, mid, high])

                width = high - low
                confidence = max(0.3, min(0.95, 1.0 - width / 50.0))

                result = {
                    "impact_low_pct": round(low, 2),
                    "impact_mid_pct": round(mid, 2),
                    "impact_high_pct": round(high, 2),
                    "market_reaction_pct": round(market_pct, 2),
                    "confidence": round(confidence, 2),
                    "interval_method": "conformal_90pct",
                }
            except Exception:
                # Fall through to quantile fallback
                self._use_conformal = False

        if not self._use_conformal:
            low = float(self.q10.predict(features)[0])
            mid = float(self.q50.predict(features)[0])
            high = float(self.q90.predict(features)[0])

            low, mid, high = sorted([low, mid, high])
            width = high - low
            confidence = max(0.3, min(0.95, 1.0 - width / 50.0))

            result = {
                "impact_low_pct": round(low, 2),
                "impact_mid_pct": round(mid, 2),
                "impact_high_pct": round(high, 2),
                "confidence": round(confidence, 2),
                "interval_method": "quantile",
            }

        if revenue_usd > 0:
            result["impact_low_usd"] = round(revenue_usd * low / 100)
            result["impact_mid_usd"] = round(revenue_usd * mid / 100)
            result["impact_high_usd"] = round(revenue_usd * high / 100)
            result["revenue_basis_usd"] = round(revenue_usd)

        return result


def _fmt_usd(val: float) -> str:
    """Format USD value for display."""
    abs_val = abs(val)
    sign = "-" if val < 0 else "+"
    if abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.0f}M"
    else:
        return f"{sign}${abs_val:,.0f}"


@click.command()
@click.option("--event", required=True, type=click.Choice(EVENT_CATEGORIES))
@click.option("--channel", default="", type=click.Choice(IMPACT_CHANNELS + [""]))
@click.option("--ticker", default="", help="Company ticker")
@click.option("--sentiment", default=0.0, type=float)
@click.option("--car5", default=0.0, type=float)
@click.option("--revenue", default=0.0, type=float, help="Annual revenue in USD")
def main(event, channel, ticker, sentiment, car5, revenue):
    """Estimate financial impact range."""
    est = ImpactEstimator()
    result = est.estimate(
        event_category=event,
        impact_channel=channel,
        ticker=ticker,
        mention_sentiment=sentiment,
        car_1_5=car5,
        revenue_usd=revenue,
    )

    print(json.dumps(result, indent=2))

    if "impact_mid_usd" in result:
        print(f"\nEstimated impact: {_fmt_usd(result['impact_low_usd'])} to "
              f"{_fmt_usd(result['impact_high_usd'])} "
              f"(mid: {_fmt_usd(result['impact_mid_usd'])})")


if __name__ == "__main__":
    main()
