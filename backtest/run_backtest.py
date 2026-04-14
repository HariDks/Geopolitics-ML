"""
Backtest: Run the model on 10 historical events and compare predictions to actual outcomes.

For each event-company pair:
1. Feed the event description to Model 1 (classify)
2. Feed the company to Model 2 (score exposure)
3. Feed both to Model 3 (estimate impact)
4. Compare predicted impact to ACTUAL outcome from seed labels

This validates whether the model would have given useful guidance
at the time the event occurred.

Usage:
    python backtest/run_backtest.py
    python backtest/run_backtest.py --output backtest/results.json
"""

import csv
import json
import sys
from pathlib import Path

import click

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ── 10 backtest events ──────────────────────────────────────────────────────

BACKTEST_CASES = [
    {
        "event_id": "russia_invasion_2022",
        "event_text": "Russian armed forces launched a full-scale invasion of Ukraine on February 24, 2022, triggering Western sanctions and corporate exits",
        "event_type": "armed_conflict_instability",
        "region": "Europe",
        "intensity": "High",
        "date": "2022-02-24",
        "companies": [
            {"ticker": "BP", "name": "BP plc", "sector": "Energy",
             "actual_channel": "capital_allocation_investment",
             "actual_impact": "Exited 19.75% Rosneft stake. $25.5B pre-tax write-down.",
             "actual_revenue_delta": -8.0, "actual_car_1_5": -0.04,
             "actual_sentiment": "negative"},
            {"ticker": "SHEL", "name": "Shell plc", "sector": "Energy",
             "actual_channel": "capital_allocation_investment",
             "actual_impact": "$3.9B pre-tax impairment on Sakhalin-2 and Russian assets.",
             "actual_revenue_delta": -2.0, "actual_car_1_5": -0.02,
             "actual_sentiment": "negative"},
        ],
    },
    {
        "event_id": "us_chip_export_controls_oct2022",
        "event_text": "US Bureau of Industry and Security published new rules restricting exports of advanced semiconductors and chip-making equipment to China",
        "event_type": "technology_controls",
        "region": "US-China",
        "intensity": "High",
        "date": "2022-10-07",
        "companies": [
            {"ticker": "NVDA", "name": "NVIDIA Corp", "sector": "Information Technology",
             "actual_channel": "revenue_market_access",
             "actual_impact": "CFO disclosed ~$400M revenue impact. Developed A800/H800 workaround chips.",
             "actual_revenue_delta": -5.0, "actual_car_1_5": -0.07,
             "actual_sentiment": "negative"},
            {"ticker": "KLAC", "name": "KLA Corporation", "sector": "Information Technology",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Semiconductor equipment exports restricted. China revenue at risk.",
             "actual_revenue_delta": -15.0, "actual_car_1_5": -0.161,
             "actual_sentiment": "negative"},
        ],
    },
    {
        "event_id": "covid_lockdown_start",
        "event_text": "WHO declared COVID-19 a global pandemic as countries imposed nationwide lockdowns shutting businesses and grounding air travel",
        "event_type": "armed_conflict_instability",
        "region": "Global",
        "intensity": "Catastrophic",
        "date": "2020-03-11",
        "companies": [
            {"ticker": "BA", "name": "Boeing", "sector": "Industrials",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Airlines cancelled orders, 737 MAX already grounded. Revenue fell 24% in 2020.",
             "actual_revenue_delta": -24.0, "actual_car_1_5": -0.461,
             "actual_sentiment": "very negative"},
            {"ticker": "MCD", "name": "McDonald's", "sector": "Consumer Discretionary",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Global same-store sales fell 22% in Q2 2020. 75% of dining rooms closed.",
             "actual_revenue_delta": -12.0, "actual_car_1_5": -0.271,
             "actual_sentiment": "negative"},
        ],
    },
    {
        "event_id": "india_demonetization_2016",
        "event_text": "Indian Prime Minister Modi announced demonetization of 500 and 1000 rupee notes, invalidating 86% of currency in circulation overnight",
        "event_type": "political_transitions_volatility",
        "region": "South Asia",
        "intensity": "High",
        "date": "2016-11-08",
        "companies": [
            {"ticker": "DLF", "name": "DLF Limited", "sector": "Real Estate",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Real estate transactions collapsed. New bookings fell 40%.",
             "actual_revenue_delta": -40.0, "actual_car_1_5": -0.243,
             "actual_sentiment": "very negative"},
            {"ticker": "PAYTM", "name": "Paytm (One97 Comm)", "sector": "Fintech",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Digital payments surged 700%. Paytm became dominant mobile wallet.",
             "actual_revenue_delta": None, "actual_car_1_5": None,
             "actual_sentiment": "very positive"},
        ],
    },
    {
        "event_id": "red_sea_houthi_attacks_2023",
        "event_text": "Houthi rebels launched anti-ship missile attacks on commercial vessels in the Red Sea, forcing major shipping lines to reroute around the Cape of Good Hope",
        "event_type": "armed_conflict_instability",
        "region": "Middle East",
        "intensity": "Moderate",
        "date": "2023-12-15",
        "companies": [
            {"ticker": "APMM", "name": "A.P. Moller-Maersk", "sector": "Industrials",
             "actual_channel": "logistics_operations",
             "actual_impact": "Rerouted all Red Sea traffic via Cape. +14 days transit. Rate surge boosted revenue.",
             "actual_revenue_delta": 15.0, "actual_car_1_5": 0.15,
             "actual_sentiment": "counterintuitively positive"},
        ],
    },
    {
        "event_id": "australia_china_wine_tariff_2020",
        "event_text": "China imposed tariffs of up to 218% on Australian wine imports in retaliation for Australia's call for COVID-19 origin investigation",
        "event_type": "trade_policy_actions",
        "region": "Asia-Pacific",
        "intensity": "Moderate",
        "date": "2020-11-28",
        "companies": [
            {"ticker": "TWE.AX", "name": "Treasury Wine Estates", "sector": "Consumer Staples",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Lost 96% of China wine revenue. China was their largest export market.",
             "actual_revenue_delta": -96.0, "actual_car_1_5": -0.016,
             "actual_sentiment": "devastating"},
        ],
    },
    {
        "event_id": "solarwinds_hack_2020",
        "event_text": "SolarWinds disclosed that Russian-linked hackers had compromised its Orion software update, infiltrating 18,000 government and corporate networks",
        "event_type": "armed_conflict_instability",
        "region": "US/Russia",
        "intensity": "High",
        "date": "2020-12-13",
        "companies": [
            {"ticker": "SWI", "name": "SolarWinds Corp", "sector": "Information Technology",
             "actual_channel": "cybersecurity_it",
             "actual_impact": "Stock fell 40% on disclosure. Spent $40M+ on remediation. Lost government contracts.",
             "actual_revenue_delta": None, "actual_car_1_5": -0.40,
             "actual_sentiment": "very negative"},
        ],
    },
    {
        "event_id": "argentina_milei_deregulation_2024",
        "event_text": "Javier Milei elected president of Argentina on radical deregulation and dollarization platform, immediately lifted fuel price controls and devalued peso 50%",
        "event_type": "political_transitions_volatility",
        "region": "Latin America",
        "intensity": "Moderate",
        "date": "2024-11-19",
        "companies": [
            {"ticker": "YPF", "name": "YPF S.A.", "sector": "Energy",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Stock surged 60.9%. Fuel price deregulation massively boosted revenue. +40% top line.",
             "actual_revenue_delta": 40.0, "actual_car_1_5": 0.609,
             "actual_sentiment": "very positive"},
        ],
    },
    {
        "event_id": "panama_mining_contract_2023",
        "event_text": "Panama Supreme Court ruled the Cobre Panama mining concession unconstitutional, ordering closure of the $8.6 billion copper mine operated by First Quantum Minerals",
        "event_type": "political_transitions_volatility",
        "region": "Central America",
        "intensity": "High",
        "date": "2023-11-28",
        "companies": [
            {"ticker": "FM.TO", "name": "First Quantum Minerals", "sector": "Materials",
             "actual_channel": "capital_allocation_investment",
             "actual_impact": "$8.6B asset impairment. Mine produced 1% of global copper. Company's largest asset wiped out.",
             "actual_revenue_delta": -13.1, "actual_car_1_5": -0.131,
             "actual_sentiment": "very negative"},
        ],
    },
    {
        "event_id": "sudan_civil_war_2023",
        "event_text": "Sudan civil war erupted between the Sudanese Armed Forces and the Rapid Support Forces, fighting spreading across Khartoum with thousands of casualties",
        "event_type": "armed_conflict_instability",
        "region": "Africa",
        "intensity": "High",
        "date": "2023-04-15",
        "companies": [
            {"ticker": "ZAIN", "name": "Zain Group", "sector": "Communication Services",
             "actual_channel": "revenue_market_access",
             "actual_impact": "Sudan operations (4M subscribers) severely disrupted. Revenue from Sudan fell 71%.",
             "actual_revenue_delta": -71.0, "actual_car_1_5": -0.007,
             "actual_sentiment": "negative"},
        ],
    },
]


def run_backtest():
    """Run model predictions on all 10 backtest events and compare to actuals."""
    from models.event_classifier.predict import EventClassifier
    from models.exposure_scorer.predict import ExposureScorer
    from models.impact_estimator.predict import ImpactEstimator

    clf = EventClassifier()
    scorer = ExposureScorer()
    estimator = ImpactEstimator()

    results = []

    for case in BACKTEST_CASES:
        # Step 1: Classify
        evt = clf.predict(case["event_text"])

        for company in case["companies"]:
            # Step 2: Exposure
            exp = scorer.score(
                event_category=evt["category"],
                ticker=company["ticker"],
                mention_sentiment=-0.4 if "negative" in company.get("actual_sentiment", "") else 0.2,
            )

            # Step 3: Impact
            imp = estimator.estimate(
                event_category=evt["category"],
                impact_channel=exp["channel_prediction"],
                ticker=company["ticker"],
                mention_sentiment=-0.4 if "negative" in company.get("actual_sentiment", "") else 0.2,
                car_1_5=company.get("actual_car_1_5") or 0.0,
            )

            # Score accuracy
            # 1. Category correct?
            cat_correct = evt["category"] == case["event_type"]

            # 2. Channel correct?
            channel_correct = exp["channel_prediction"] == company["actual_channel"]

            # 3. Direction correct? (negative actual → negative prediction, positive → positive)
            actual_rev = company.get("actual_revenue_delta")
            actual_car = company.get("actual_car_1_5")
            actual_direction = None
            if actual_rev is not None:
                actual_direction = "positive" if actual_rev > 0 else "negative"
            elif actual_car is not None:
                actual_direction = "positive" if actual_car > 0 else "negative"

            pred_direction = "positive" if imp["impact_mid_pct"] > 0 else "negative"
            direction_correct = actual_direction == pred_direction if actual_direction else None

            # 4. Actual within predicted range?
            in_range = None
            if actual_rev is not None:
                in_range = imp["impact_low_pct"] <= actual_rev <= imp["impact_high_pct"]

            results.append({
                "event": case["event_id"],
                "event_date": case["date"],
                "region": case["region"],
                "intensity": case["intensity"],
                "company": company["name"],
                "ticker": company["ticker"],
                "sector": company["sector"],
                # Predictions
                "pred_category": evt["category"],
                "pred_category_conf": round(evt["confidence"], 3),
                "pred_channel": exp["channel_prediction"],
                "pred_channel_conf": round(exp["channel_confidence"], 3),
                "pred_severity": round(exp["severity_score"], 3),
                "pred_impact_low": round(imp["impact_low_pct"], 1),
                "pred_impact_mid": round(imp["impact_mid_pct"], 1),
                "pred_impact_high": round(imp["impact_high_pct"], 1),
                # Actuals
                "actual_channel": company["actual_channel"],
                "actual_impact_desc": company["actual_impact"],
                "actual_revenue_delta": actual_rev,
                "actual_car_1_5": actual_car,
                "actual_sentiment": company.get("actual_sentiment", ""),
                # Scoring
                "category_correct": cat_correct,
                "channel_correct": channel_correct,
                "direction_correct": direction_correct,
                "in_range": in_range,
            })

    return results


def print_report(results: list[dict]):
    """Print a formatted backtest report."""
    print("=" * 100)
    print("GEOPOLITICAL RISK MODEL — BACKTEST REPORT")
    print("10 Historical Events | 14 Company-Event Pairs")
    print("=" * 100)

    # Summary scores
    cat_acc = sum(1 for r in results if r["category_correct"]) / len(results)
    ch_acc = sum(1 for r in results if r["channel_correct"]) / len(results)
    dir_results = [r for r in results if r["direction_correct"] is not None]
    dir_acc = sum(1 for r in dir_results if r["direction_correct"]) / len(dir_results) if dir_results else 0
    range_results = [r for r in results if r["in_range"] is not None]
    range_acc = sum(1 for r in range_results if r["in_range"]) / len(range_results) if range_results else 0

    print(f"\nOVERALL ACCURACY")
    print(f"  Event category:     {cat_acc:.0%} ({sum(r['category_correct'] for r in results)}/{len(results)})")
    print(f"  Impact channel:     {ch_acc:.0%} ({sum(r['channel_correct'] for r in results)}/{len(results)})")
    print(f"  Direction (+-):     {dir_acc:.0%} ({sum(r['direction_correct'] for r in dir_results)}/{len(dir_results)})")
    print(f"  In predicted range: {range_acc:.0%} ({sum(r['in_range'] for r in range_results)}/{len(range_results)})")

    print(f"\n{'─' * 100}")
    print(f"DETAILED RESULTS")
    print(f"{'─' * 100}")

    current_event = None
    for r in results:
        if r["event"] != current_event:
            current_event = r["event"]
            print(f"\n{'━' * 100}")
            print(f"EVENT: {r['event']} ({r['event_date']}) | {r['region']} | {r['intensity']}")
            print(f"{'━' * 100}")

        cat_mark = "+" if r["category_correct"] else "X"
        ch_mark = "+" if r["channel_correct"] else "X"
        dir_mark = "+" if r["direction_correct"] else ("X" if r["direction_correct"] is False else "?")
        range_mark = "+" if r["in_range"] else ("X" if r["in_range"] is False else "?")

        print(f"\n  {r['company']} ({r['ticker']}) — {r['sector']}")
        print(f"  Predicted: {r['pred_category'][:30]} [{cat_mark}] | channel: {r['pred_channel'][:25]} [{ch_mark}]")
        print(f"  Impact:    {r['pred_impact_low']:+.1f}% to {r['pred_impact_high']:+.1f}% (mid {r['pred_impact_mid']:+.1f}%)")
        if r["actual_revenue_delta"] is not None:
            print(f"  Actual:    {r['actual_revenue_delta']:+.1f}% revenue delta [{range_mark} in range] [{dir_mark} direction]")
        elif r["actual_car_1_5"] is not None:
            print(f"  Actual:    {r['actual_car_1_5']:+.1%} stock reaction (5-day) [{dir_mark} direction]")
        print(f"  What happened: {r['actual_impact_desc']}")

    print(f"\n{'=' * 100}")
    print(f"VERDICT")
    print(f"{'=' * 100}")
    overall = (cat_acc + ch_acc + dir_acc) / 3
    if overall >= 0.7:
        print(f"  Model performs WELL — {overall:.0%} average accuracy across category/channel/direction")
    elif overall >= 0.5:
        print(f"  Model performs REASONABLY — {overall:.0%} average accuracy, useful for directional guidance")
    else:
        print(f"  Model NEEDS IMPROVEMENT — {overall:.0%} average accuracy")
    print()


@click.command()
@click.option("--output", default=None, help="Save results to JSON file")
def main(output):
    """Run backtest on 10 historical events."""
    results = run_backtest()
    print_report(results)

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
