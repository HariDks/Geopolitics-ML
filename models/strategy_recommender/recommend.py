"""
Model 4 inference: recommend ranked strategies for company-event-channel combinations.

Given (event_category, impact_channel, severity, company_size), returns
top-N strategies ranked by relevance and feasibility.

Ranking factors:
  1. Cell match: exact (event, channel) match gets highest weight
  2. Adjacent cell: same event OR same channel gets partial credit
  3. Strategy category fit: severity drives category preference
     - High severity → exit/hedge strategies rank higher
     - Low severity → monitor/engage strategies rank higher
     - Positive impact → capture strategies rank higher
  4. Company size: large companies can afford more expensive strategies

Usage:
    from models.strategy_recommender.recommend import StrategyRecommender
    rec = StrategyRecommender()
    results = rec.recommend(
        event_category="armed_conflict_instability",
        impact_channel="logistics_operations",
        severity=-0.7,
        company_size="large",
    )

    # CLI
    python models/strategy_recommender/recommend.py \\
        --event armed_conflict_instability --channel logistics_operations \\
        --severity -0.7
"""

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.utils import get_db_connection, get_logger

logger = get_logger("strategy_recommender")

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

# Category preference by severity level
# Negative severity (damage) → mitigate/hedge/exit
# Positive severity (opportunity) → capture/engage
# Low absolute severity → monitor
SEVERITY_CATEGORY_WEIGHTS = {
    "high_negative": {"mitigate": 1.0, "hedge": 0.9, "exit": 0.8, "monitor": 0.4, "engage": 0.5, "capture": 0.2},
    "moderate_negative": {"mitigate": 1.0, "hedge": 0.8, "exit": 0.5, "monitor": 0.6, "engage": 0.6, "capture": 0.3},
    "low": {"mitigate": 0.6, "hedge": 0.5, "exit": 0.2, "monitor": 1.0, "engage": 0.8, "capture": 0.5},
    "positive": {"mitigate": 0.3, "hedge": 0.4, "exit": 0.1, "monitor": 0.5, "engage": 0.7, "capture": 1.0},
}

# Company size → cost filter
SIZE_COST_CAPS = {
    "small": {"low", "low-medium"},
    "medium": {"low", "low-medium", "medium"},
    "large": {"low", "low-medium", "medium", "medium-high", "high"},
}


def _severity_bucket(severity: float) -> str:
    if severity >= 0.1:
        return "positive"
    elif severity >= -0.3:
        return "low"
    elif severity >= -0.6:
        return "moderate_negative"
    else:
        return "high_negative"


class StrategyRecommender:
    """Retrieval-based strategy recommender using Phase 1 strategy database."""

    def __init__(self):
        self._strategies = None

    def _load_strategies(self):
        if self._strategies is not None:
            return
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM strategies").fetchall()
        conn.close()
        self._strategies = [dict(r) for r in rows]

    def recommend(
        self,
        event_category: str,
        impact_channel: str,
        severity: float = -0.5,
        company_size: str = "large",
        top_n: int = 5,
    ) -> list[dict]:
        """
        Recommend ranked strategies for a given event-channel-severity combination.

        Args:
            event_category: one of 8 taxonomy categories
            impact_channel: one of 10 impact channels
            severity: impact severity (-1 to 1, negative = damage, positive = opportunity)
            company_size: "small" / "medium" / "large"
            top_n: number of strategies to return

        Returns:
            List of strategy dicts with added 'relevance_score' and 'rank' fields
        """
        self._load_strategies()

        sev_bucket = _severity_bucket(severity)
        cat_weights = SEVERITY_CATEGORY_WEIGHTS[sev_bucket]
        cost_cap = SIZE_COST_CAPS.get(company_size, SIZE_COST_CAPS["large"])

        scored = []
        for strat in self._strategies:
            score = 0.0

            # 1. Cell match (0-50 points)
            event_match = strat["event_category"] == event_category
            channel_match = strat["impact_channel"] == impact_channel

            if event_match and channel_match:
                score += 50  # exact cell match
            elif event_match:
                score += 25  # same event, different channel
            elif channel_match:
                score += 20  # different event, same channel

            # Skip if no match at all (irrelevant)
            if score == 0:
                continue

            # 2. Strategy category fit (0-30 points)
            cat_weight = cat_weights.get(strat["strategy_category"], 0.5)
            score += cat_weight * 30

            # 3. Priority score from Phase 1 (0-10 points)
            priority = strat.get("precedent_count", 0)
            score += min(priority * 5, 10)

            # 4. Cost feasibility filter (penalize but don't exclude)
            if strat["typical_cost_range"] not in cost_cap:
                score *= 0.5  # halve score if too expensive for company size

            scored.append({
                "strategy_id": strat["strategy_id"],
                "strategy_name": strat["strategy_name"],
                "strategy_category": strat["strategy_category"],
                "event_category": strat["event_category"],
                "impact_channel": strat["impact_channel"],
                "description": strat["description"],
                "typical_cost": strat["typical_cost_range"],
                "implementation_time": strat["implementation_time"],
                "historical_precedent": strat.get("historical_precedents", "[]"),
                "relevance_score": round(score, 1),
                "match_type": "exact" if (event_match and channel_match) else
                              "event" if event_match else "channel",
            })

        # Sort by relevance score descending
        scored.sort(key=lambda x: -x["relevance_score"])

        # Add rank
        for i, s in enumerate(scored[:top_n]):
            s["rank"] = i + 1

        return scored[:top_n]

    def recommend_full(
        self,
        event_category: str,
        top_channels: list[dict],
        severity: float = -0.5,
        company_size: str = "large",
        top_n_per_channel: int = 3,
    ) -> dict:
        """
        Full recommendation pipeline: given an event and top impacted channels
        (from Model 2), recommend strategies for each channel.

        Args:
            event_category: event type
            top_channels: list of {"channel": str, "probability": float} from Model 2
            severity: from Model 3
            company_size: company size tier

        Returns:
            Dict mapping channel → list of strategies
        """
        results = {}
        for ch_info in top_channels:
            channel = ch_info["channel"]
            strategies = self.recommend(
                event_category=event_category,
                impact_channel=channel,
                severity=severity,
                company_size=company_size,
                top_n=top_n_per_channel,
            )
            if strategies:
                results[channel] = strategies

        return results


@click.command()
@click.option("--event", required=True, type=click.Choice(EVENT_CATEGORIES))
@click.option("--channel", required=True, type=click.Choice(IMPACT_CHANNELS))
@click.option("--severity", default=-0.5, type=float)
@click.option("--size", default="large", type=click.Choice(["small", "medium", "large"]))
@click.option("--top-n", default=5, type=int)
@click.option("--json-output", is_flag=True, help="Output as JSON")
def main(event, channel, severity, size, top_n, json_output):
    """Recommend strategies for a geopolitical event-channel combination."""
    rec = StrategyRecommender()
    results = rec.recommend(
        event_category=event,
        impact_channel=channel,
        severity=severity,
        company_size=size,
        top_n=top_n,
    )

    if json_output:
        print(json.dumps(results, indent=2))
    else:
        sev_label = "DAMAGE" if severity < 0 else "OPPORTUNITY"
        print(f"\nStrategies for: {event} → {channel}")
        print(f"Severity: {severity:+.1f} ({sev_label}) | Company size: {size}")
        print("=" * 80)

        for s in results:
            print(f"\n  #{s['rank']} [{s['strategy_category'].upper():8s}] {s['strategy_name']}")
            print(f"     Match: {s['match_type']:8s} | Cost: {s['typical_cost']:12s} | "
                  f"Time: {s['implementation_time']:12s} | Score: {s['relevance_score']}")

            precedents = json.loads(s.get("historical_precedent", "[]"))
            if precedents and precedents[0]:
                print(f"     Precedent: {precedents[0][:120]}...")

        if not results:
            print("  No matching strategies found for this combination.")


if __name__ == "__main__":
    main()
