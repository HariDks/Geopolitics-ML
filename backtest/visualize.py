"""
Generate publishable backtest visualizations.

Creates:
1. Predicted vs Actual scatter plot (the money chart)
2. Prediction range chart showing intervals
3. Summary scorecard

Usage:
    python backtest/visualize.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load results
with open(Path(__file__).parent / "results.json") as f:
    results = json.load(f)


def fig1_predicted_vs_actual():
    """Scatter plot: predicted mid vs actual revenue delta."""
    pairs = []
    for r in results:
        actual = r.get("actual_revenue_delta")
        if actual is not None:
            pairs.append({
                "predicted": r["pred_impact_mid"],
                "actual": actual,
                "label": f"{r['ticker']}",
                "event": r["event"].replace("_", " ")[:25],
                "direction_correct": r["direction_correct"],
            })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Perfect prediction line
    min_val = min(min(p["actual"] for p in pairs), min(p["predicted"] for p in pairs)) - 5
    max_val = max(max(p["actual"] for p in pairs), max(p["predicted"] for p in pairs)) + 5
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="Perfect prediction")

    # Scatter
    for p in pairs:
        color = "#2ecc71" if p["direction_correct"] else "#e74c3c"
        ax.scatter(p["actual"], p["predicted"], c=color, s=120, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(p["label"], (p["actual"], p["predicted"]),
                   textcoords="offset points", xytext=(8, 5), fontsize=8, alpha=0.8)

    ax.set_xlabel("Actual Revenue Impact (%)", fontsize=12)
    ax.set_ylabel("Model Predicted Impact (%)", fontsize=12)
    ax.set_title("Geopolitical Risk Model: Predicted vs Actual Impact\n10 Historical Events, 12 Company-Event Pairs", fontsize=14)

    green_patch = mpatches.Patch(color="#2ecc71", label="Direction correct")
    red_patch = mpatches.Patch(color="#e74c3c", label="Direction wrong")
    ax.legend(handles=[green_patch, red_patch], loc="upper left", fontsize=10)

    ax.axhline(y=0, color="gray", alpha=0.2)
    ax.axvline(x=0, color="gray", alpha=0.2)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    path = OUTPUT_DIR / "predicted_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig2_prediction_ranges():
    """Horizontal bar chart showing predicted ranges with actual marked."""
    pairs = []
    for r in results:
        actual = r.get("actual_revenue_delta")
        if actual is None:
            actual = (r.get("actual_car_1_5") or 0) * 100  # use stock as proxy
        pairs.append({
            "label": f"{r['ticker']} — {r['event'].replace('_', ' ')[:20]}",
            "low": r["pred_impact_low"],
            "mid": r["pred_impact_mid"],
            "high": r["pred_impact_high"],
            "actual": actual,
            "in_range": r.get("in_range"),
        })

    # Sort by actual
    pairs.sort(key=lambda x: x["actual"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    y_pos = range(len(pairs))

    for i, p in enumerate(pairs):
        # Predicted range bar
        color = "#3498db" if p.get("in_range") else "#95a5a6"
        ax.barh(i, p["high"] - p["low"], left=p["low"], height=0.5,
               color=color, alpha=0.6, edgecolor="white")
        # Predicted mid
        ax.plot(p["mid"], i, "D", color="#2c3e50", markersize=6, zorder=5)
        # Actual
        ax.plot(p["actual"], i, "X", color="#e74c3c", markersize=10, markeredgewidth=2, zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p["label"] for p in pairs], fontsize=9)
    ax.set_xlabel("Impact (%)", fontsize=12)
    ax.set_title("Prediction Ranges vs Actual Outcomes\nBlue bar = predicted range, Red X = actual, Diamond = predicted mid", fontsize=13)
    ax.axvline(x=0, color="gray", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)

    blue_patch = mpatches.Patch(color="#3498db", alpha=0.6, label="Predicted range (actual inside)")
    gray_patch = mpatches.Patch(color="#95a5a6", alpha=0.6, label="Predicted range (actual outside)")
    red_marker = plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#e74c3c",
                            markeredgewidth=2, markersize=10, label="Actual outcome")
    ax.legend(handles=[blue_patch, gray_patch, red_marker], loc="lower right", fontsize=9)

    fig.tight_layout()
    path = OUTPUT_DIR / "prediction_ranges.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig3_scorecard():
    """Summary scorecard as a figure."""
    cat_acc = sum(1 for r in results if r["category_correct"]) / len(results)
    ch_acc = sum(1 for r in results if r["channel_correct"]) / len(results)
    dir_results = [r for r in results if r["direction_correct"] is not None]
    dir_acc = sum(1 for r in dir_results if r["direction_correct"]) / len(dir_results)
    range_results = [r for r in results if r["in_range"] is not None]
    range_acc = sum(1 for r in range_results if r["in_range"]) / len(range_results)

    # Highlight cases
    highlights = [
        ("NVIDIA (chip controls)", "Predicted: -6.3%", "Actual: -5.0%", True),
        ("KLAC (chip controls)", "Predicted: -14.6%", "Actual: -15.0%", True),
        ("First Quantum (Panama)", "Predicted: -12.2%", "Actual: -13.1%", True),
        ("Maersk (Red Sea)", "Predicted: +13.8%", "Actual: +15.0%", True),
        ("Boeing (COVID)", "Predicted: -35.0%", "Actual: -24.0%", True),
        ("Treasury Wine (China tariff)", "Predicted: ~0%", "Actual: -96.0%", False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: Accuracy bars
    ax = axes[0]
    metrics = ["Direction\n(+/-)", "Category\nClassification", "In Predicted\nRange", "Impact\nChannel"]
    values = [dir_acc, cat_acc, range_acc, ch_acc]
    colors = ["#2ecc71" if v >= 0.7 else "#f39c12" if v >= 0.5 else "#e74c3c" for v in values]

    bars = ax.barh(range(len(metrics)), values, color=colors, height=0.6, edgecolor="white", linewidth=2)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Accuracy", fontsize=11)

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=12, fontweight="bold")

    ax.set_title("Model Accuracy Metrics", fontsize=13, fontweight="bold")
    ax.axvline(x=0.5, color="gray", alpha=0.2, linestyle="--")
    ax.axvline(x=0.8, color="gray", alpha=0.2, linestyle="--")

    # Right: Highlight cases
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Key Predictions", fontsize=13, fontweight="bold")

    y = 0.92
    for company, pred, actual, correct in highlights:
        icon = "+" if correct else "X"
        color = "#2ecc71" if correct else "#e74c3c"
        ax2.text(0.05, y, f"[{icon}]", fontsize=12, color=color, fontweight="bold",
                transform=ax2.transAxes, fontfamily="monospace")
        ax2.text(0.12, y, company, fontsize=11, fontweight="bold", transform=ax2.transAxes)
        ax2.text(0.12, y - 0.05, f"{pred}  |  {actual}", fontsize=10,
                color="#555", transform=ax2.transAxes)
        y -= 0.14

    fig.suptitle("Geopolitical Risk Model — Backtest Scorecard\n10 Events Across 5 Continents, 5 Event Types",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "scorecard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fig1_predicted_vs_actual()
    fig2_prediction_ranges()
    fig3_scorecard()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
