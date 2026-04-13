"""End-to-end pipeline test using subprocess isolation to avoid torch+sqlite segfault."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def run_step(code: str) -> str:
    """Run Python code in a subprocess and return stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if result.returncode != 0 and "Segmentation" not in result.stderr:
        print(f"STDERR: {result.stderr[:200]}", file=sys.stderr)
    # Get last non-empty line (skip loading bar output)
    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def fmt_usd(val):
    a = abs(val)
    s = "-" if val < 0 else "+"
    if a >= 1e9:
        return f"{s}${a/1e9:.1f}B"
    elif a >= 1e6:
        return f"{s}${a/1e6:.0f}M"
    return f"{s}${a:,.0f}"


def main():
    text = "On 25 February 2022, Russian armed forces launched a full-scale invasion of Ukraine, triggering NATO sanctions"
    ticker = "AAPL"
    revenue = 394_000_000_000

    print("=" * 80)
    print("GEOPOLITICAL RISK PIPELINE — END-TO-END TEST")
    print("=" * 80)
    print(f'\nInput: "{text}"')
    print(f"Company: {ticker} | Revenue: ${revenue/1e9:.0f}B\n")

    # Step 1: Event Classification
    out = run_step("""
import json
from models.event_classifier.predict import EventClassifier
clf = EventClassifier()
r = clf.predict("On 25 February 2022, Russian armed forces launched a full-scale invasion of Ukraine, triggering NATO sanctions")
print(json.dumps(r))
""")
    evt = json.loads(out)
    print(f"STEP 1 — EVENT CLASSIFICATION")
    print(f"  Category:   {evt['category']}")
    print(f"  Confidence: {evt['confidence']:.1%}")

    # Step 2: Exposure Scoring
    out = run_step(f"""
import json
from models.exposure_scorer.predict import ExposureScorer
s = ExposureScorer()
r = s.score(event_category="{evt['category']}", ticker="{ticker}", mention_sentiment=-0.5)
print(json.dumps(r, default=str))
""")
    exp = json.loads(out)
    print(f"\nSTEP 2 — EXPOSURE ASSESSMENT")
    print(f"  Primary channel: {exp['channel_prediction']} ({exp['channel_confidence']:.1%})")
    print(f"  Severity:        {exp['severity_score']:+.2f}")
    for c in exp["top_3_channels"]:
        print(f"    {c['channel']:35s} {c['probability']:.1%}")

    # Step 3: Impact Estimation
    out = run_step(f"""
import json
from models.impact_estimator.predict import ImpactEstimator
e = ImpactEstimator()
r = e.estimate(event_category="{evt['category']}", impact_channel="{exp['channel_prediction']}", ticker="{ticker}", mention_sentiment=-0.5, revenue_usd={revenue})
print(json.dumps(r, default=str))
""")
    imp = json.loads(out)
    print(f"\nSTEP 3 — FINANCIAL IMPACT")
    print(f"  Low:  {imp['impact_low_pct']:+.1f}%", end="")
    if "impact_low_usd" in imp:
        print(f"  ({fmt_usd(imp['impact_low_usd'])})", end="")
    print()
    print(f"  Mid:  {imp['impact_mid_pct']:+.1f}%", end="")
    if "impact_mid_usd" in imp:
        print(f"  ({fmt_usd(imp['impact_mid_usd'])})", end="")
    print()
    print(f"  High: {imp['impact_high_pct']:+.1f}%", end="")
    if "impact_high_usd" in imp:
        print(f"  ({fmt_usd(imp['impact_high_usd'])})", end="")
    print()
    print(f"  Confidence: {imp['confidence']:.0%}")

    # Step 4: Strategy Recommendations
    for ch in exp["top_3_channels"][:2]:
        out = run_step(f"""
import json
from models.strategy_recommender.recommend import StrategyRecommender
rec = StrategyRecommender()
strats = rec.recommend(event_category="{evt['category']}", impact_channel="{ch['channel']}", severity={imp['impact_mid_pct']/100})
print(json.dumps(strats, default=str))
""")
        strats = json.loads(out)
        print(f"\nSTEP 4 — STRATEGIES for [{ch['channel']}]")
        for s in strats[:3]:
            print(f"  #{s['rank']} [{s['strategy_category'].upper():8s}] {s['strategy_name']}")
            print(f"     Cost: {s['typical_cost']:12s} | Time: {s['implementation_time']}")

    print("\n" + "=" * 80)
    print("Pipeline complete — 4 models executed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
