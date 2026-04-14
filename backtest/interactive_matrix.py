"""
Generate interactive heatmap of the S&P 500 Geopolitical Risk Matrix.

Creates an HTML file with a Plotly heatmap that can be published as a standalone page.

Usage:
    python backtest/interactive_matrix.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(Path(__file__).parent / "risk_matrix.csv")
with open(Path(__file__).parent / "surprises.json") as f:
    surprises = json.load(f)


def build_heatmap():
    """Build interactive heatmap: companies (rows) x scenarios (columns)."""
    # Pivot to matrix
    pivot = df.pivot_table(index="company", columns="scenario", values="severity")

    # Sort companies by average severity (most exposed at top)
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg")
    pivot = pivot.drop(columns=["avg"])

    # Shorten scenario names for display
    short_names = {
        "Red Sea shipping disruption": "Red Sea",
        "US-China tariff escalation": "US-China Tariffs",
        "Russia-Ukraine escalation": "Russia-Ukraine",
        "OPEC production cut": "OPEC Cut",
        "Taiwan Strait tensions": "Taiwan",
        "Iran nuclear crisis": "Iran Nuclear",
        "Major cyberattack": "Cyberattack",
        "EU regulatory crackdown": "EU Regulation",
        "Emerging market debt crisis": "EM Debt Crisis",
        "Global supply chain shock": "Supply Chain",
    }
    pivot.columns = [short_names.get(c, c) for c in pivot.columns]

    # Build hover text with details
    hover_df = df.pivot_table(index="company", columns="scenario", values="impact_mid_pct")
    hover_df.columns = [short_names.get(c, c) for c in hover_df.columns]
    channel_df = df.pivot_table(index="company", columns="scenario", values="channel", aggfunc="first")
    channel_df.columns = [short_names.get(c, c) for c in channel_df.columns]

    hover_text = []
    for company in pivot.index:
        row = []
        for scenario in pivot.columns:
            sev = pivot.loc[company, scenario]
            impact = hover_df.loc[company, scenario] if company in hover_df.index else 0
            channel = channel_df.loc[company, scenario] if company in channel_df.index else ""
            channel_display = channel.replace("_", " ").title() if isinstance(channel, str) else ""
            row.append(
                f"<b>{company}</b> vs {scenario}<br>"
                f"Severity: {sev:+.2f}<br>"
                f"Impact: {impact:+.1f}%<br>"
                f"Channel: {channel_display}"
            )
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[
            [0, "#c0392c"],      # dark red (most negative/exposed)
            [0.3, "#e74c3c"],    # red
            [0.45, "#f39c12"],   # orange
            [0.5, "#f1c40f"],    # yellow (neutral)
            [0.55, "#f1c40f"],
            [0.7, "#2ecc71"],    # green
            [1, "#27ae60"],      # dark green (positive/beneficiary)
        ],
        zmid=-0.33,
        text=hover_text,
        hoverinfo="text",
        colorbar=dict(
            title=dict(text="Severity", side="right"),
            tickvals=[-0.45, -0.35, -0.25],
            ticktext=["High Risk", "Moderate", "Low Risk"],
        ),
    ))

    fig.update_layout(
        title=dict(
            text="<b>The Hidden Geopolitical Risk Map of the S&P 500</b><br>"
                 "<span style='font-size:14px;color:gray'>37 Companies x 10 Scenarios — Severity Score (negative = exposed, positive = beneficiary)</span>",
            x=0.5,
            font=dict(size=20),
        ),
        xaxis=dict(title="Geopolitical Scenario", tickangle=45, side="bottom"),
        yaxis=dict(title="", autorange="reversed"),
        width=1100,
        height=1000,
        margin=dict(l=180, r=50, t=100, b=120),
        font=dict(family="Arial", size=11),
    )

    path = OUTPUT_DIR / "risk_heatmap.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved interactive heatmap to {path}")

    # Also save as static image
    try:
        fig.write_image(str(OUTPUT_DIR / "risk_heatmap.png"), width=1100, height=1000, scale=2)
        print(f"Saved static heatmap to {OUTPUT_DIR / 'risk_heatmap.png'}")
    except Exception as e:
        print(f"Static image export skipped (install kaleido: pip install kaleido): {e}")


def build_surprise_chart():
    """Bar chart of non-obvious top risks."""
    unexpected = [s for s in surprises if s["type"] == "unexpected_top_risk"]

    # Take top 15 most surprising
    unexpected = sorted(unexpected, key=lambda x: x["severity"])[:15]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[f"{s['company']} ({s['ticker']})" for s in unexpected],
        x=[s["severity"] for s in unexpected],
        orientation="h",
        marker_color=["#e74c3c" if s["severity"] < -0.35 else "#f39c12" for s in unexpected],
        text=[f"#{1}: {s['actual_top_risk']}<br>(Expected: {s['expected']})" for s in unexpected],
        hoverinfo="text+x",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Unexpected #1 Geopolitical Risks</b><br>"
                 "<span style='font-size:13px;color:gray'>Each company's actual top risk vs what you'd expect from their sector</span>",
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(title="Severity Score", range=[-0.45, 0]),
        yaxis=dict(title=""),
        width=900,
        height=600,
        margin=dict(l=200, r=50, t=100, b=50),
        font=dict(family="Arial", size=11),
    )

    path = OUTPUT_DIR / "surprise_risks.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved surprise chart to {path}")


def build_sector_comparison():
    """Grouped bar chart: average severity by sector for each scenario."""
    sector_avg = df.groupby(["sector", "scenario"])["severity"].mean().reset_index()

    short_names = {
        "Red Sea shipping disruption": "Red Sea",
        "US-China tariff escalation": "US-China",
        "Russia-Ukraine escalation": "Russia",
        "OPEC production cut": "OPEC",
        "Taiwan Strait tensions": "Taiwan",
        "Iran nuclear crisis": "Iran",
        "Major cyberattack": "Cyber",
        "EU regulatory crackdown": "EU Reg",
        "Emerging market debt crisis": "EM Debt",
        "Global supply chain shock": "Supply Chain",
    }
    sector_avg["scenario_short"] = sector_avg["scenario"].map(short_names)

    fig = px.bar(
        sector_avg,
        x="scenario_short",
        y="severity",
        color="sector",
        barmode="group",
        title="<b>Geopolitical Exposure by Sector</b><br>"
              "<span style='font-size:13px;color:gray'>Which sectors are most vulnerable to each scenario?</span>",
    )

    fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Average Severity",
        width=1100,
        height=600,
        legend_title="Sector",
        font=dict(family="Arial", size=11),
    )

    path = OUTPUT_DIR / "sector_exposure.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved sector comparison to {path}")


if __name__ == "__main__":
    build_heatmap()
    build_surprise_chart()
    build_sector_comparison()
    print(f"\nAll interactive visualizations saved to {OUTPUT_DIR}/")
    print("Open the .html files in a browser to explore.")
