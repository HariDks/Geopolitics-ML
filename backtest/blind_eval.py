"""
Blind evaluation of channel prediction on 70 frozen event-company pairs.

Mode A: No event text (structured features only)
Mode B: With event text (lexicon features enabled)

Usage:
    python backtest/blind_eval.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.exposure_scorer.predict import ExposureScorer
from models.exposure_scorer.train import EVENT_TO_CATEGORY

DATASET_PATH = Path(__file__).parent / "blind_eval_dataset.json"

IMPACT_CHANNELS = [
    "procurement_supply_chain", "revenue_market_access",
    "capital_allocation_investment", "regulatory_compliance_cost",
    "logistics_operations", "innovation_ip", "workforce_talent",
    "reputation_stakeholder", "financial_treasury", "cybersecurity_it",
]


def run_eval():
    with open(DATASET_PATH) as f:
        data = json.load(f)
    examples = data["examples"]

    scorer = ExposureScorer()

    # Infer event_category from the event_text using the classifier
    from models.event_classifier.predict import EventClassifier
    clf = EventClassifier()

    results_a = []  # Mode A: no text
    results_b = []  # Mode B: with text

    for ex in examples:
        event_text = ex["event_text"]
        ticker = ex["ticker"]
        primary = ex["primary"]
        secondary = ex.get("secondary")
        confidence = ex.get("confidence", "high")

        # Classify event
        evt = clf.predict(event_text)
        event_cat = evt["category"]

        # Mode A: no event text
        exp_a = scorer.score(event_category=event_cat, ticker=ticker, event_text="")
        probs_a = exp_a["channel_probabilities"]
        ranked_a = sorted(probs_a.items(), key=lambda x: -x[1])

        # Mode B: with event text
        exp_b = scorer.score(event_category=event_cat, ticker=ticker, event_text=event_text)
        probs_b = exp_b["channel_probabilities"]
        ranked_b = sorted(probs_b.items(), key=lambda x: -x[1])

        for mode, ranked, results in [("A", ranked_a, results_a), ("B", ranked_b, results_b)]:
            top1 = ranked[0][0]
            top2 = [r[0] for r in ranked[:2]]

            top1_correct = top1 == primary
            top2_correct = primary in top2
            top2_with_secondary = top2_correct or (secondary is not None and secondary in top2)

            results.append({
                "company": ex["company"],
                "ticker": ticker,
                "event_text": event_text[:50],
                "primary": primary,
                "secondary": secondary,
                "pred_1": top1,
                "pred_2": ranked[1][0],
                "top1_correct": top1_correct,
                "top2_correct": top2_correct,
                "top2_with_secondary": top2_with_secondary,
                "confidence": confidence,
                "mode": mode,
                "reliability": exp_b.get("channel_reliability", "unknown") if mode == "B" else exp_a.get("channel_reliability", "unknown"),
            })

    # ── Print results ──
    print("=" * 90)
    print("BLIND EVALUATION — 70 Event-Company Pairs (Frozen Dataset)")
    print("=" * 90)

    for mode_label, results in [("MODE A: No Event Text (structured only)", results_a),
                                 ("MODE B: With Event Text (lexicon enabled)", results_b)]:
        n = len(results)
        top1 = sum(r["top1_correct"] for r in results)
        top2 = sum(r["top2_correct"] for r in results)
        top2_sec = sum(r["top2_with_secondary"] for r in results)

        # By confidence
        high_conf = [r for r in results if r["confidence"] == "high"]
        med_conf = [r for r in results if r["confidence"] == "medium"]
        low_conf = [r for r in results if r["confidence"] == "low"]

        print(f"\n{'─' * 90}")
        print(f"{mode_label}")
        print(f"{'─' * 90}")
        print(f"  Top-1 accuracy:                    {top1}/{n} ({top1/n:.1%})")
        print(f"  Top-2 accuracy (primary only):     {top2}/{n} ({top2/n:.1%})")
        print(f"  Top-2 accuracy (primary+secondary): {top2_sec}/{n} ({top2_sec/n:.1%})")

        if high_conf:
            hc_top1 = sum(r["top1_correct"] for r in high_conf)
            hc_top2 = sum(r["top2_correct"] for r in high_conf)
            print(f"\n  High-confidence labels only:  top-1={hc_top1}/{len(high_conf)} ({hc_top1/len(high_conf):.1%})  "
                  f"top-2={hc_top2}/{len(high_conf)} ({hc_top2/len(high_conf):.1%})")

    # Delta
    a_top1 = sum(r["top1_correct"] for r in results_a)
    b_top1 = sum(r["top1_correct"] for r in results_b)
    a_top2 = sum(r["top2_correct"] for r in results_a)
    b_top2 = sum(r["top2_correct"] for r in results_b)
    n = len(results_a)

    print(f"\n{'=' * 90}")
    print(f"DELTA: Text vs No Text")
    print(f"{'=' * 90}")
    print(f"  {'Metric':35s} {'No Text':>10s} {'With Text':>12s} {'Delta':>8s}")
    print(f"  {'─' * 68}")
    print(f"  {'Top-1':35s} {a_top1/n:>9.1%} {b_top1/n:>11.1%} {(b_top1-a_top1)/n:>+7.1%}")
    print(f"  {'Top-2':35s} {a_top2/n:>9.1%} {b_top2/n:>11.1%} {(b_top2-a_top2)/n:>+7.1%}")

    # Error analysis
    print(f"\n{'=' * 90}")
    print(f"ERROR ANALYSIS (Mode B — With Text)")
    print(f"{'=' * 90}")

    errors = [r for r in results_b if not r["top1_correct"]]
    if errors:
        # Confusion pairs
        confusion = Counter()
        for e in errors:
            confusion[(e["primary"], e["pred_1"])] += 1

        print(f"\n  Top confusion pairs (actual → predicted):")
        for (actual, pred), count in confusion.most_common(10):
            print(f"    {actual[:25]:25s} → {pred[:25]:25s}  ({count}x)")

        # Failure types
        adjacent = sum(1 for e in errors if e["top2_correct"])
        completely_wrong = sum(1 for e in errors if not e["top2_correct"] and e["confidence"] == "high")
        ambiguous = sum(1 for e in errors if e["confidence"] in ("low", "medium"))

        print(f"\n  Failure breakdown ({len(errors)} total errors):")
        print(f"    Adjacent channel (correct in top-2):     {adjacent}")
        print(f"    Completely wrong (high-conf label):      {completely_wrong}")
        print(f"    Ambiguous ground truth (low/med conf):   {ambiguous}")


if __name__ == "__main__":
    run_eval()
