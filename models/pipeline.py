"""
Full 4-model geopolitical risk pipeline.

Chains all 4 models: Event Classifier → Exposure Scorer → Impact Estimator → Strategy Recommender.

Usage:
    from models.pipeline import GeopoliticalRiskPipeline
    pipe = GeopoliticalRiskPipeline()
    result = pipe.analyze(
        text="US imposed 25% tariffs on all Chinese imports",
        ticker="AAPL",
        revenue_usd=394_000_000_000,
    )

    # CLI
    python models/pipeline.py "Houthi rebels attacked container ship in Red Sea" --ticker AAPL
"""

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

class GeopoliticalRiskPipeline:
    """End-to-end geopolitical risk analysis pipeline.

    Lazily loads each model on first use to avoid memory issues.
    """

    def __init__(self):
        self._classifier = None
        self._scorer = None
        self._estimator = None
        self._recommender = None

    @property
    def classifier(self):
        if self._classifier is None:
            from models.event_classifier.predict import EventClassifier
            self._classifier = EventClassifier()
        return self._classifier

    @property
    def scorer(self):
        if self._scorer is None:
            from models.exposure_scorer.predict import ExposureScorer
            self._scorer = ExposureScorer()
        return self._scorer

    @property
    def estimator(self):
        if self._estimator is None:
            from models.impact_estimator.predict import ImpactEstimator
            self._estimator = ImpactEstimator()
        return self._estimator

    @property
    def recommender(self):
        if self._recommender is None:
            from models.strategy_recommender.recommend import StrategyRecommender
            self._recommender = StrategyRecommender()
        return self._recommender

    def analyze(
        self,
        text: str,
        ticker: str = "",
        revenue_usd: float = 0.0,
        company_size: str = "large",
        top_strategies: int = 3,
    ) -> dict:
        """
        Full pipeline analysis.

        Args:
            text: raw event text (news headline, ACLED description, etc.)
            ticker: company ticker for company-specific analysis
            revenue_usd: annual revenue for USD impact estimation
            company_size: "small" / "medium" / "large" for strategy filtering
            top_strategies: number of strategies per channel

        Returns:
            Complete analysis dict with event classification, exposure,
            impact estimates, and strategy recommendations.
        """
        # Step 1: Classify the event
        event_result = self.classifier.predict(text)

        # Step 2: Score company exposure
        exposure_result = self.scorer.score(
            event_category=event_result["category"],
            ticker=ticker,
            mention_sentiment=-0.5 if event_result["confidence"] > 0.7 else -0.3,
        )

        # Step 3: Estimate financial impact
        impact_result = self.estimator.estimate(
            event_category=event_result["category"],
            impact_channel=exposure_result["channel_prediction"],
            ticker=ticker,
            mention_sentiment=-0.5,
            car_1_5=0.0,
            revenue_usd=revenue_usd,
        )

        # Step 4: Recommend strategies
        strategies = self.recommender.recommend_full(
            event_category=event_result["category"],
            top_channels=exposure_result["top_3_channels"],
            severity=impact_result["impact_mid_pct"] / 100,
            company_size=company_size,
            top_n_per_channel=top_strategies,
        )

        return {
            "disclaimer": "Predictions are based on historical correlations, not causal analysis. "
                          "Direction accuracy: ~90%. Channel accuracy: ~50%. "
                          "Companies with concentrated geographic exposure may see larger actual impacts.",
            "input": {
                "text": text[:200],
                "ticker": ticker,
                "revenue_usd": revenue_usd,
            },
            "event_classification": {
                "category": event_result["category"],
                "confidence": event_result["confidence"],
            },
            "exposure": {
                "primary_channel": exposure_result["channel_prediction"],
                "channel_confidence": exposure_result["channel_confidence"],
                "severity_score": exposure_result["severity_score"],
                "top_3_channels": exposure_result["top_3_channels"],
            },
            "impact_estimate": impact_result,
            "strategies": strategies,
        }


def _fmt_usd(val: float) -> str:
    abs_val = abs(val)
    sign = "-" if val < 0 else "+"
    if abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.0f}M"
    else:
        return f"{sign}${abs_val:,.0f}"


def print_analysis(result: dict):
    """Pretty-print a full pipeline analysis."""
    inp = result["input"]
    evt = result["event_classification"]
    exp = result["exposure"]
    imp = result["impact_estimate"]
    strats = result["strategies"]

    print("\n" + "=" * 80)
    print("GEOPOLITICAL RISK ANALYSIS")
    print("=" * 80)

    print(f"\nEvent: \"{inp['text']}\"")
    if inp["ticker"]:
        print(f"Company: {inp['ticker']}")

    print(f"\n--- EVENT CLASSIFICATION ---")
    print(f"  Category:   {evt['category']}")
    print(f"  Confidence: {evt['confidence']:.1%}")

    print(f"\n--- EXPOSURE ASSESSMENT ---")
    print(f"  Primary channel: {exp['primary_channel']} ({exp['channel_confidence']:.1%})")
    print(f"  Severity score:  {exp['severity_score']:+.2f}")
    print(f"  Top channels:")
    for ch in exp["top_3_channels"]:
        print(f"    {ch['channel']:35s} {ch['probability']:.1%}")

    print(f"\n--- FINANCIAL IMPACT ---")
    print(f"  Low estimate:  {imp['impact_low_pct']:+.1f}%", end="")
    if "impact_low_usd" in imp:
        print(f"  ({_fmt_usd(imp['impact_low_usd'])})", end="")
    print()
    print(f"  Mid estimate:  {imp['impact_mid_pct']:+.1f}%", end="")
    if "impact_mid_usd" in imp:
        print(f"  ({_fmt_usd(imp['impact_mid_usd'])})", end="")
    print()
    print(f"  High estimate: {imp['impact_high_pct']:+.1f}%", end="")
    if "impact_high_usd" in imp:
        print(f"  ({_fmt_usd(imp['impact_high_usd'])})", end="")
    print()
    print(f"  Confidence:    {imp['confidence']:.0%}")

    print(f"\n--- RECOMMENDED STRATEGIES ---")
    for channel, channel_strats in strats.items():
        print(f"\n  [{channel}]")
        for s in channel_strats:
            print(f"    #{s['rank']} [{s['strategy_category'].upper():8s}] {s['strategy_name']}")
            print(f"       Cost: {s['typical_cost']:12s} | Time: {s['implementation_time']}")

    print("\n" + "=" * 80)


@click.command()
@click.argument("text")
@click.option("--ticker", default="", help="Company ticker")
@click.option("--revenue", default=0.0, type=float, help="Annual revenue in USD")
@click.option("--size", default="large", type=click.Choice(["small", "medium", "large"]))
@click.option("--json-output", is_flag=True, help="Output as JSON")
def main(text, ticker, revenue, size, json_output):
    """Run full geopolitical risk analysis pipeline."""
    pipe = GeopoliticalRiskPipeline()
    result = pipe.analyze(
        text=text,
        ticker=ticker,
        revenue_usd=revenue,
        company_size=size,
    )

    if json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_analysis(result)


if __name__ == "__main__":
    main()
