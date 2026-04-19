"""
Negative backtest: test whether the model correctly predicts LOW impact
for companies that WEREN'T materially affected by an event.

The original backtest only tested cases where companies WERE affected.
That's selection bias — a model that predicts "everything is bad" would
score well on those tests. We need to also test that the model doesn't
over-predict impact for unaffected companies.

Cases chosen:
- Companies in sectors unrelated to the event
- Companies that maintained/grew revenue despite the event
- Events in regions where the company has no operations

Usage:
    python backtest/negative_backtest.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.event_classifier.predict import EventClassifier
from models.exposure_scorer.predict import ExposureScorer
from models.impact_estimator.predict import ImpactEstimator

NEGATIVE_CASES = [
    # Company was NOT affected by this event — model should predict mild/no impact
    {
        "event_text": "Houthi rebels fired anti-ship missiles at container vessels in the Red Sea",
        "ticker": "UNH",
        "company": "UnitedHealth",
        "revenue_usd": 400e9,
        "why_not_affected": "US domestic health insurer — no shipping, no international supply chain through Red Sea",
        "actual": "Revenue grew 14% in Q1 2024. Zero mention of Red Sea in earnings call.",
    },
    {
        "event_text": "US Bureau of Industry and Security restricted exports of advanced semiconductors to China",
        "ticker": "MCD",
        "company": "McDonald's",
        "revenue_usd": 26e9,
        "why_not_affected": "Fast food chain — doesn't sell, buy, or use advanced semiconductors",
        "actual": "No impact. Chip controls not mentioned in any filing.",
    },
    {
        "event_text": "Russia launched full-scale invasion of Ukraine triggering Western sanctions",
        "ticker": "HD",
        "company": "Home Depot",
        "revenue_usd": 157e9,
        "why_not_affected": "US/Canada/Mexico only — zero operations in Russia, Ukraine, or Europe",
        "actual": "Revenue grew 4.1% in FY2022. Russia not mentioned as a risk factor.",
    },
    {
        "event_text": "OPEC announced a surprise production cut of 2 million barrels per day",
        "ticker": "MSFT",
        "company": "Microsoft",
        "revenue_usd": 245e9,
        "why_not_affected": "Software/cloud company — energy costs are <2% of operating expenses",
        "actual": "OPEC decisions have no measurable impact on Microsoft's P&L.",
    },
    {
        "event_text": "Military coup in Myanmar: army detained Aung San Suu Kyi and seized power",
        "ticker": "JPM",
        "company": "JPMorgan Chase",
        "revenue_usd": 177e9,
        "why_not_affected": "No retail banking, investment, or significant exposure to Myanmar",
        "actual": "Myanmar coup had zero impact on JPMorgan earnings.",
    },
    {
        "event_text": "India demonetized 86% of currency in circulation overnight",
        "ticker": "AAPL",
        "company": "Apple",
        "revenue_usd": 383e9,
        "why_not_affected": "India was <2% of Apple's revenue in 2016. iPhone sales are card/digital-payment friendly.",
        "actual": "No measurable impact on Apple's FY2017 results. India not mentioned as risk.",
    },
    {
        "event_text": "Sudan civil war erupted between the Sudanese Armed Forces and Rapid Support Forces",
        "ticker": "COST",
        "company": "Costco",
        "revenue_usd": 242e9,
        "why_not_affected": "No stores, suppliers, or members in Sudan. Zero operational connection.",
        "actual": "Sudan not mentioned in any Costco filing or earnings call.",
    },
    {
        "event_text": "Panama Supreme Court ruled the Cobre Panama mining concession unconstitutional",
        "ticker": "GS",
        "company": "Goldman Sachs",
        "revenue_usd": 51e9,
        "why_not_affected": "Investment bank with no mining operations. May have had minor commodity trading exposure.",
        "actual": "Panama mine closure not mentioned in Goldman earnings. No material impact.",
    },
    {
        "event_text": "China imposed tariffs of 218% on Australian wine imports",
        "ticker": "LMT",
        "company": "Lockheed Martin",
        "revenue_usd": 68e9,
        "why_not_affected": "Defense contractor — doesn't sell wine, doesn't trade with China or Australia",
        "actual": "Zero connection to wine tariffs. Not mentioned in any filing.",
    },
    {
        "event_text": "EU passed the Digital Markets Act requiring Big Tech to open their platforms",
        "ticker": "XOM",
        "company": "Exxon Mobil",
        "revenue_usd": 344e9,
        "why_not_affected": "Oil company — not a digital gatekeeper, not subject to DMA requirements",
        "actual": "DMA has no relevance to Exxon's business. Not mentioned in filings.",
    },
]


def run_negative_backtest():
    clf = EventClassifier()
    scorer = ExposureScorer()
    estimator = ImpactEstimator()

    print("=" * 90)
    print("NEGATIVE BACKTEST — Companies That WEREN'T Affected")
    print("Model should predict LOW severity and SMALL impact for these cases.")
    print("=" * 90)

    false_alarms = 0
    total = len(NEGATIVE_CASES)
    results = []

    for case in NEGATIVE_CASES:
        evt = clf.predict(case["event_text"])
        exp = scorer.score(
            event_category=evt["category"],
            ticker=case["ticker"],
            mention_sentiment=-0.3,
        )
        imp = estimator.estimate(
            event_category=evt["category"],
            impact_channel=exp["channel_prediction"],
            ticker=case["ticker"],
            mention_sentiment=-0.3,
            revenue_usd=case["revenue_usd"],
        )

        # A "false alarm" = model predicts significant impact (|mid| > 2% or |severity| > 0.4)
        is_false_alarm = abs(imp["impact_mid_pct"]) > 2.0 or abs(exp["severity_score"]) > 0.4
        if is_false_alarm:
            false_alarms += 1

        verdict = "FALSE ALARM" if is_false_alarm else "CORRECT (low)"
        results.append({
            "company": case["company"],
            "event": case["event_text"][:50],
            "severity": exp["severity_score"],
            "impact_mid": imp["impact_mid_pct"],
            "channel": exp["channel_prediction"],
            "verdict": verdict,
            "why": case["why_not_affected"],
            "actual": case["actual"],
        })

        print(f"\n  {case['company']} ({case['ticker']}) vs {case['event_text'][:50]}...")
        print(f"    Severity: {exp['severity_score']:+.2f} | Impact: {imp['impact_mid_pct']:+.1f}% | Channel: {exp['channel_prediction']}")
        print(f"    Verdict: {verdict}")
        print(f"    Reality: {case['actual']}")

    # Summary
    correct = total - false_alarms
    print(f"\n{'=' * 90}")
    print(f"NEGATIVE BACKTEST RESULTS")
    print(f"{'=' * 90}")
    print(f"  Correctly predicted LOW impact: {correct}/{total} ({correct/total:.0%})")
    print(f"  False alarms (predicted impact where none existed): {false_alarms}/{total} ({false_alarms/total:.0%})")

    if false_alarms > total * 0.3:
        print(f"\n  WARNING: Model has a high false alarm rate ({false_alarms/total:.0%}).")
        print(f"  It tends to predict impact even for unrelated companies.")
        print(f"  This means the backtest's 92% direction accuracy is partly driven by")
        print(f"  the model predicting 'negative' for everything, not genuine signal.")
    elif false_alarms > 0:
        print(f"\n  MODERATE: {false_alarms} false alarms. Model slightly over-predicts exposure")
        print(f"  for companies with no connection to the event.")
    else:
        print(f"\n  EXCELLENT: Zero false alarms. Model correctly identifies unaffected companies.")


if __name__ == "__main__":
    run_negative_backtest()
