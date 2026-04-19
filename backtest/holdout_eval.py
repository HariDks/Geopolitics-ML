"""
Holdout evaluation: test Model 2 on ONLY the original 163 manual labels.

This is the unbiased evaluation. The auto-generated 439 labels were created
by a pipeline that uses Model 2's logic (GICS sector mapping) — so testing
on auto-labels is partially circular. The 163 manual labels were created by
a human before any model existed, making them truly independent.

This script answers: "Is Model 2's 0.825 macro F1 real, or is it inflated
by training on labels the model helped create?"

Usage:
    python backtest/holdout_eval.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.exposure_scorer.predict import ExposureScorer
from models.impact_estimator.predict import ImpactEstimator
from models.exposure_scorer.train import EVENT_TO_CATEGORY

SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"

IMPACT_CHANNELS = [
    "procurement_supply_chain", "revenue_market_access",
    "capital_allocation_investment", "regulatory_compliance_cost",
    "logistics_operations", "innovation_ip", "workforce_talent",
    "reputation_stakeholder", "financial_treasury", "cybersecurity_it",
]


def load_manual_labels():
    """Load only the 163 original hand-curated labels."""
    with open(SEED_PATH) as f:
        all_labels = list(csv.DictReader(f))
    return [r for r in all_labels if r.get("labeled_by", "") in ("claude_verified", "human_review", "")]


def evaluate():
    manual = load_manual_labels()
    scorer = ExposureScorer()
    estimator = ImpactEstimator()

    print("=" * 80)
    print("HOLDOUT EVALUATION — Original 163 Manual Labels Only")
    print("These labels were created BEFORE any model existed.")
    print("This is the unbiased test of whether Model 2 actually works.")
    print("=" * 80)
    print(f"\nManual labels: {len(manual)}")

    # ── Channel prediction ──
    correct_channel = 0
    correct_direction = 0
    direction_total = 0
    in_range = 0
    range_total = 0
    channel_results = {ch: {"tp": 0, "fp": 0, "fn": 0} for ch in IMPACT_CHANNELS}

    results = []
    for label in manual:
        event_id = label["event_id"]
        ticker = label["company_ticker"]
        actual_channel = label["impact_channel"]
        actual_rev = label.get("revenue_delta_pct", "").strip()
        actual_car = label.get("car_1_5", "").strip()
        sentiment = float(label.get("mention_sentiment", 0) or 0)

        event_cat = EVENT_TO_CATEGORY.get(event_id, "")
        if not event_cat:
            continue

        # Predict
        exp = scorer.score(
            event_category=event_cat,
            ticker=ticker,
            mention_sentiment=sentiment,
            car_1_5=float(actual_car) if actual_car else 0.0,
            event_id=event_id,
        )

        pred_channel = exp["channel_prediction"]

        # Channel accuracy
        if pred_channel == actual_channel:
            correct_channel += 1
            channel_results[actual_channel]["tp"] += 1
        else:
            channel_results[actual_channel]["fn"] += 1
            channel_results[pred_channel]["fp"] += 1

        # Direction accuracy (if we have quantitative data)
        if actual_rev:
            actual_rev_f = float(actual_rev)
            imp = estimator.estimate(
                event_category=event_cat,
                impact_channel=pred_channel,
                ticker=ticker,
                mention_sentiment=sentiment,
                car_1_5=float(actual_car) if actual_car else 0.0,
            )

            pred_direction = "positive" if imp["impact_mid_pct"] > 0 else "negative"
            actual_direction = "positive" if actual_rev_f > 0 else "negative"
            if pred_direction == actual_direction:
                correct_direction += 1
            direction_total += 1

            # In range?
            if imp["impact_low_pct"] <= actual_rev_f <= imp["impact_high_pct"]:
                in_range += 1
            range_total += 1

            results.append({
                "ticker": ticker,
                "event": event_id[:30],
                "actual_channel": actual_channel,
                "pred_channel": pred_channel,
                "channel_correct": pred_channel == actual_channel,
                "actual_rev": actual_rev_f,
                "pred_mid": imp["impact_mid_pct"],
                "pred_low": imp["impact_low_pct"],
                "pred_high": imp["impact_high_pct"],
                "direction_correct": pred_direction == actual_direction,
                "in_range": imp["impact_low_pct"] <= actual_rev_f <= imp["impact_high_pct"],
            })

    # ── Print results ──
    total = sum(1 for _ in manual if EVENT_TO_CATEGORY.get(_["event_id"]))
    print(f"\n{'METRIC':35s} {'SCORE':>10s} {'DETAILS':>20s}")
    print("-" * 70)
    print(f"{'Channel accuracy':35s} {correct_channel/total:>9.1%} {correct_channel}/{total}")
    if direction_total:
        print(f"{'Direction accuracy (+/-)':35s} {correct_direction/direction_total:>9.1%} {correct_direction}/{direction_total}")
    if range_total:
        print(f"{'Actual in predicted range':35s} {in_range/range_total:>9.1%} {in_range}/{range_total}")

    # Per-channel F1
    print(f"\n{'CHANNEL':35s} {'PREC':>7s} {'RECALL':>7s} {'F1':>7s} {'SUPPORT':>8s}")
    print("-" * 70)
    f1_scores = []
    for ch in IMPACT_CHANNELS:
        tp = channel_results[ch]["tp"]
        fp = channel_results[ch]["fp"]
        fn = channel_results[ch]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        support = tp + fn
        if support > 0:
            print(f"{ch:35s} {prec:>6.1%} {recall:>6.1%} {f1:>6.1%} {support:>7d}")
            f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    print(f"\n{'Macro F1 (holdout)':35s} {macro_f1:>9.3f}")

    # Compare to reported F1
    print(f"\n{'=' * 70}")
    print(f"COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Reported macro F1 (mixed eval):':35s} {'0.825':>10s}  ← includes auto-labels in val set")
    print(f"{'Holdout macro F1 (manual only):':35s} {macro_f1:>9.3f}  ← independent human labels only")
    gap = 0.825 - macro_f1
    if gap > 0.1:
        print(f"\n  GAP: {gap:.3f} — auto-labels are inflating reported accuracy significantly.")
        print(f"  The model's real-world accuracy is closer to {macro_f1:.1%} than 82.5%.")
    elif gap > 0.05:
        print(f"\n  GAP: {gap:.3f} — moderate inflation from auto-labels. Model is decent but not as good as reported.")
    else:
        print(f"\n  GAP: {gap:.3f} — minimal inflation. The 0.825 figure is approximately honest.")

    # Show worst predictions
    if results:
        print(f"\n{'WORST PREDICTIONS (largest gap between predicted and actual):':}")
        results.sort(key=lambda r: abs(r["actual_rev"] - r["pred_mid"]), reverse=True)
        print(f"  {'Ticker':8s} {'Event':30s} {'Actual':>8s} {'Predicted':>10s} {'Gap':>8s} {'Ch OK':>6s}")
        for r in results[:10]:
            gap_val = abs(r["actual_rev"] - r["pred_mid"])
            ch_mark = "Y" if r["channel_correct"] else "N"
            print(f"  {r['ticker']:8s} {r['event']:30s} {r['actual_rev']:>+7.1f}% {r['pred_mid']:>+9.1f}% {gap_val:>7.1f}pp {ch_mark:>5s}")


if __name__ == "__main__":
    evaluate()
