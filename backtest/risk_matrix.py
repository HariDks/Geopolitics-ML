"""
Generate the S&P 500 Geopolitical Risk Matrix.

Runs every company through every scenario, producing a 37x10 matrix
of severity scores. Then identifies non-obvious findings.

Usage:
    python backtest/risk_matrix.py
    python backtest/risk_matrix.py --output backtest/risk_matrix.json
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import COMPANIES, EVENT_SCENARIOS


def run_matrix():
    """Run all company-scenario combinations."""
    from models.event_classifier.predict import EventClassifier
    from models.exposure_scorer.predict import ExposureScorer
    from models.impact_estimator.predict import ImpactEstimator

    clf = EventClassifier()
    scorer = ExposureScorer()
    estimator = ImpactEstimator()

    companies = {k: v for k, v in COMPANIES.items() if k != "Other (enter manually)"}
    scenarios = {k: v for k, v in EVENT_SCENARIOS.items() if k != "Custom (enter your own)" and v}

    # Pre-classify all events
    print("Classifying events...")
    event_classes = {}
    for name, text in scenarios.items():
        evt = clf.predict(text)
        event_classes[name] = evt
        print(f"  {name:40s} -> {evt['category']:35s} ({evt['confidence']:.0%})")

    # Run all combinations
    print(f"\nRunning {len(companies)} companies x {len(scenarios)} scenarios = {len(companies) * len(scenarios)} analyses...")

    results = []
    for i, (company_name, info) in enumerate(companies.items()):
        ticker = info["ticker"]
        revenue = info["revenue"]
        sector = info["sector"]

        for scenario_name, text in scenarios.items():
            evt = event_classes[scenario_name]

            exp = scorer.score(
                event_category=evt["category"],
                ticker=ticker,
                mention_sentiment=-0.4,
            )

            imp = estimator.estimate(
                event_category=evt["category"],
                impact_channel=exp["channel_prediction"],
                ticker=ticker,
                mention_sentiment=-0.4,
                revenue_usd=revenue,
            )

            results.append({
                "company": company_name.split(" (")[0],
                "ticker": ticker,
                "sector": sector,
                "revenue_B": round(revenue / 1e9),
                "scenario": scenario_name,
                "event_category": evt["category"],
                "channel": exp["channel_prediction"],
                "severity": round(exp["severity_score"], 3),
                "impact_mid_pct": round(imp["impact_mid_pct"], 2),
                "impact_mid_usd_M": round(imp.get("impact_mid_usd", 0) / 1e6) if imp.get("impact_mid_usd") else 0,
                "geo_exposure": info.get("geo_exposure", ""),
            })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(companies)} companies done")

    print(f"  {len(companies)}/{len(companies)} companies done")
    return results


def find_surprises(results: list[dict]) -> list[dict]:
    """
    Find non-obvious exposures — cases where:
    1. A company's biggest risk ISN'T the scenario you'd expect for its sector
    2. A company benefits from a scenario you'd expect to hurt it
    3. Two companies in the same sector have opposite reactions to the same event
    """
    df = pd.DataFrame(results)
    surprises = []

    # ── 1. Each company's #1 risk (most negative severity) ──
    # Compare to what you'd "expect" for their sector
    expected_risks = {
        "Energy": "OPEC production cut",
        "Information Technology": "US-China tariff escalation",
        "Financials": "Emerging market debt crisis",
        "Consumer Discretionary": "US-China tariff escalation",
        "Consumer Staples": "US-China tariff escalation",
        "Health Care": "EU regulatory crackdown",
        "Industrials": "Red Sea shipping disruption",
        "Communication Services": "EU regulatory crackdown",
    }

    for ticker in df["ticker"].unique():
        company_df = df[df["ticker"] == ticker].sort_values("severity")
        worst = company_df.iloc[0]
        best = company_df.iloc[-1]
        sector = worst["sector"]
        expected = expected_risks.get(sector, "")

        if worst["scenario"] != expected and expected:
            surprises.append({
                "type": "unexpected_top_risk",
                "company": worst["company"],
                "ticker": ticker,
                "sector": sector,
                "finding": f"{worst['company']}'s #1 geopolitical risk is '{worst['scenario']}' (severity {worst['severity']:+.2f}), not '{expected}' as you'd expect for {sector}",
                "expected": expected,
                "actual_top_risk": worst["scenario"],
                "severity": worst["severity"],
                "impact_pct": worst["impact_mid_pct"],
            })

        # Companies that BENEFIT from a typically negative event
        if best["severity"] > 0.1:
            surprises.append({
                "type": "beneficiary",
                "company": best["company"],
                "ticker": ticker,
                "sector": sector,
                "finding": f"{best['company']} BENEFITS from '{best['scenario']}' (severity {best['severity']:+.2f}, {best['impact_mid_pct']:+.1f}% impact)",
                "scenario": best["scenario"],
                "severity": best["severity"],
                "impact_pct": best["impact_mid_pct"],
            })

    # ── 2. Same-sector opposites ──
    for scenario in df["scenario"].unique():
        scenario_df = df[df["scenario"] == scenario]
        for sector in scenario_df["sector"].unique():
            sector_df = scenario_df[scenario_df["sector"] == sector].sort_values("severity")
            if len(sector_df) < 2:
                continue
            worst = sector_df.iloc[0]
            best = sector_df.iloc[-1]
            if worst["severity"] < -0.2 and best["severity"] > 0.0:
                surprises.append({
                    "type": "same_sector_split",
                    "finding": f"In '{scenario}', {worst['company']} (severity {worst['severity']:+.2f}) and {best['company']} (severity {best['severity']:+.2f}) are in the SAME sector ({sector}) but react oppositely",
                    "scenario": scenario,
                    "sector": sector,
                    "loser": worst["company"],
                    "loser_severity": worst["severity"],
                    "winner": best["company"],
                    "winner_severity": best["severity"],
                })

    return surprises


def print_matrix_summary(results, surprises):
    """Print the key findings."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("THE HIDDEN GEOPOLITICAL RISK MAP OF THE S&P 500")
    print("=" * 100)
    print(f"\n{len(df)} company-scenario analyses across {df['ticker'].nunique()} companies and {df['scenario'].nunique()} scenarios\n")

    # Pivot: company x scenario severity
    pivot = df.pivot_table(index=["company", "ticker", "sector"], columns="scenario", values="severity")

    # Most exposed companies overall (average severity across all scenarios)
    avg_severity = df.groupby(["company", "ticker", "sector"])["severity"].mean().sort_values()
    print("TOP 10 MOST GEOPOLITICALLY EXPOSED COMPANIES (avg severity across all scenarios)")
    print("-" * 80)
    for (company, ticker, sector), sev in avg_severity.head(10).items():
        worst_scenario = df[(df["ticker"] == ticker)].sort_values("severity").iloc[0]["scenario"]
        print(f"  {company:25s} ({ticker:5s}) {sector:25s} avg={sev:+.3f}  worst: {worst_scenario}")

    print(f"\nLEAST EXPOSED / MOST RESILIENT COMPANIES")
    print("-" * 80)
    for (company, ticker, sector), sev in avg_severity.tail(5).items():
        print(f"  {company:25s} ({ticker:5s}) {sector:25s} avg={sev:+.3f}")

    # Most dangerous scenario
    avg_by_scenario = df.groupby("scenario")["severity"].mean().sort_values()
    print(f"\nMOST DANGEROUS SCENARIOS (avg severity across all companies)")
    print("-" * 80)
    for scenario, sev in avg_by_scenario.items():
        bar = "█" * int(abs(sev) * 100)
        direction = "←" if sev < 0 else "→"
        print(f"  {scenario:40s} {sev:+.3f} {direction} {bar}")

    # SURPRISES
    print(f"\n{'=' * 100}")
    print(f"NON-OBVIOUS FINDINGS ({len(surprises)} surprises)")
    print(f"{'=' * 100}")

    unexpected = [s for s in surprises if s["type"] == "unexpected_top_risk"]
    beneficiaries = [s for s in surprises if s["type"] == "beneficiary"]
    splits = [s for s in surprises if s["type"] == "same_sector_split"]

    if unexpected:
        print(f"\nUNEXPECTED #1 RISKS (company's top risk isn't what you'd guess from their sector):")
        for s in sorted(unexpected, key=lambda x: x["severity"])[:10]:
            print(f"  {s['finding']}")

    if beneficiaries:
        print(f"\nCRISIS BENEFICIARIES (companies that gain from geopolitical events):")
        for s in sorted(beneficiaries, key=lambda x: -x["severity"])[:10]:
            print(f"  {s['finding']}")

    if splits:
        print(f"\nSAME-SECTOR SPLITS (companies in same industry react oppositely):")
        for s in splits[:8]:
            print(f"  {s['finding']}")


@click.command()
@click.option("--output", default=None, help="Save full matrix to JSON")
def main(output):
    results = run_matrix()
    surprises = find_surprises(results)
    print_matrix_summary(results, surprises)

    OUTPUT_DIR = Path(__file__).parent
    if output:
        with open(output, "w") as f:
            json.dump({"matrix": results, "surprises": surprises}, f, indent=2)
        print(f"\nSaved to {output}")

    # Always save the matrix CSV for the interactive viz
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "risk_matrix.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved matrix CSV to {csv_path}")

    # Save surprises
    with open(OUTPUT_DIR / "surprises.json", "w") as f:
        json.dump(surprises, f, indent=2)
    print(f"Saved {len(surprises)} surprises to {OUTPUT_DIR / 'surprises.json'}")


if __name__ == "__main__":
    main()
