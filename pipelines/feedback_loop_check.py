"""
Feedback loop detector: measure whether auto-generated labels agree
with manual labels when both exist for the same (event, company).

The semi-supervised pipeline creates labels using Model 2 + GICS rules.
If those auto-labels systematically disagree with independent human
labels for the same cases, the feedback loop is amplifying bias.

Usage:
    python pipelines/feedback_loop_check.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"


def check_feedback_loop():
    with open(SEED_PATH) as f:
        labels = list(csv.DictReader(f))

    manual = [r for r in labels if r.get("labeled_by", "") in ("claude_verified", "human_review", "")]
    auto = [r for r in labels if r.get("labeled_by", "") not in ("claude_verified", "human_review", "")]

    # Find cases where both manual and auto labels exist for same event
    manual_events = {r["event_id"] for r in manual}
    auto_events = {r["event_id"] for r in auto}
    shared_events = manual_events & auto_events

    print("=" * 80)
    print("FEEDBACK LOOP ANALYSIS")
    print("=" * 80)
    print(f"\nManual labels: {len(manual)}")
    print(f"Auto labels: {len(auto)}")
    print(f"Shared events (both have labels): {len(shared_events)}")

    # For shared events, compare channel distributions
    print(f"\nCHANNEL DISTRIBUTION COMPARISON (shared events only):")
    print(f"{'Channel':35s} {'Manual':>8s} {'Auto':>8s} {'Delta':>8s}")
    print("-" * 65)

    manual_shared = [r for r in manual if r["event_id"] in shared_events]
    auto_shared = [r for r in auto if r["event_id"] in shared_events]

    manual_ch = Counter(r["impact_channel"] for r in manual_shared)
    auto_ch = Counter(r["impact_channel"] for r in auto_shared)

    all_channels = sorted(set(list(manual_ch.keys()) + list(auto_ch.keys())))
    total_manual = len(manual_shared)
    total_auto = len(auto_shared)

    divergences = []
    for ch in all_channels:
        m_pct = manual_ch.get(ch, 0) / total_manual * 100 if total_manual else 0
        a_pct = auto_ch.get(ch, 0) / total_auto * 100 if total_auto else 0
        delta = a_pct - m_pct
        divergences.append((ch, m_pct, a_pct, delta))
        marker = " **" if abs(delta) > 10 else ""
        print(f"{ch:35s} {m_pct:>7.1f}% {a_pct:>7.1f}% {delta:>+7.1f}%{marker}")

    # Overall divergence
    total_divergence = sum(abs(d[3]) for d in divergences) / 2  # divide by 2 because shifts balance out
    print(f"\nTotal distribution shift: {total_divergence:.1f}pp")

    if total_divergence > 20:
        print("  HIGH DIVERGENCE: Auto-labels have a significantly different channel")
        print("  distribution than manual labels for the same events.")
        print("  This confirms the feedback loop is amplifying certain patterns.")
    elif total_divergence > 10:
        print("  MODERATE DIVERGENCE: Some systematic bias in auto-labeling.")
    else:
        print("  LOW DIVERGENCE: Auto-labels roughly match manual patterns.")

    # Per-event comparison
    print(f"\n{'=' * 80}")
    print(f"PER-EVENT CHANNEL AGREEMENT")
    print(f"{'=' * 80}")

    for event_id in sorted(shared_events):
        m_labels = [r for r in manual if r["event_id"] == event_id]
        a_labels = [r for r in auto if r["event_id"] == event_id]

        m_channels = Counter(r["impact_channel"] for r in m_labels)
        a_channels = Counter(r["impact_channel"] for r in a_labels)

        m_top = m_channels.most_common(1)[0][0] if m_channels else "?"
        a_top = a_channels.most_common(1)[0][0] if a_channels else "?"

        agreement = "AGREE" if m_top == a_top else "DISAGREE"
        print(f"  {event_id[:40]:40s} manual top: {m_top[:20]:20s} auto top: {a_top[:20]:20s} {agreement}")

    # Specific bias detection
    print(f"\n{'=' * 80}")
    print(f"BIAS DETECTION: Which channels is the auto-labeler over/under-predicting?")
    print(f"{'=' * 80}")
    for ch, m_pct, a_pct, delta in sorted(divergences, key=lambda x: -abs(x[3])):
        if abs(delta) > 5:
            direction = "OVER-predicts" if delta > 0 else "UNDER-predicts"
            print(f"  Auto-labeler {direction} {ch} by {abs(delta):.1f}pp")
            if delta > 0:
                print(f"    This means the model is learning to predict {ch} MORE than humans would.")
                print(f"    Risk: feedback loop amplifying this bias in future retraining.")


if __name__ == "__main__":
    check_feedback_loop()
