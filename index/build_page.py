"""
Build standalone HTML page for the Geopolitical Risk Index.

Creates a single self-contained HTML file with:
- Main GRI line chart (90-day or full history)
- Sub-index sparklines for each event category
- "What's driving it today" section
- Key stats and methodology

Usage:
    python index/build_page.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent.parent
OUTPUT = Path(__file__).parent / "geopolitical_risk_index.html"

# Load data
gri = pd.read_csv(Path(__file__).parent / "gri.csv", parse_dates=["event_date"])

# Sub-index columns
sub_cols = [c for c in gri.columns if c.startswith("gri_")]
sub_names = {
    "gri_armed_conflict": "Armed Conflict",
    "gri_sanctions_finan": "Sanctions",
    "gri_trade_policy_ac": "Trade Policy",
    "gri_political_trans": "Political Transitions",
    "gri_regulatory_sove": "Regulatory Shifts",
}

# ── Main GRI chart ──
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.5, 0.3, 0.2],
    subplot_titles=("Geopolitical Risk Index (GRI)", "Sub-Indices", "Event Volume"),
    vertical_spacing=0.08,
)

# GRI line
fig.add_trace(go.Scatter(
    x=gri["event_date"], y=gri["gri"],
    mode="lines",
    name="GRI",
    line=dict(color="#2c3e50", width=2),
    fill="tozeroy",
    fillcolor="rgba(231, 76, 60, 0.1)",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>GRI: %{y:.1f}/100<extra></extra>",
), row=1, col=1)

# Risk zones
fig.add_hline(y=75, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1,
             annotation_text="High Risk", annotation_position="right")
fig.add_hline(y=55, line_dash="dash", line_color="orange", opacity=0.3, row=1, col=1,
             annotation_text="Elevated", annotation_position="right")

# Key event annotations
events_to_annotate = [
    ("2022-02-24", "Russia invades\nUkraine"),
    ("2022-10-07", "US chip\nexport controls"),
    ("2023-10-07", "Hamas\nattack"),
    ("2023-12-15", "Red Sea\nattacks begin"),
    ("2025-04-02", "US tariffs\n2025"),
]
for date_str, label in events_to_annotate:
    date_val = pd.Timestamp(date_str)
    gri_row = gri[gri["event_date"] == date_val]
    if not gri_row.empty:
        fig.add_annotation(
            x=date_val, y=gri_row.iloc[0]["gri"],
            text=label, showarrow=True, arrowhead=2,
            arrowsize=0.8, arrowcolor="#c0392c",
            font=dict(size=9, color="#c0392c"),
            ax=0, ay=-40,
            row=1, col=1,
        )

# Sub-indices
colors = {"gri_armed_conflict": "#e74c3c", "gri_sanctions_finan": "#e67e22",
          "gri_trade_policy_ac": "#3498db", "gri_political_trans": "#9b59b6",
          "gri_regulatory_sove": "#1abc9c"}

for col in sub_cols:
    if col in sub_names:
        fig.add_trace(go.Scatter(
            x=gri["event_date"], y=gri[col],
            mode="lines",
            name=sub_names[col],
            line=dict(width=1.5, color=colors.get(col, "#95a5a6")),
            hovertemplate=f"{sub_names[col]}: %{{y:.1f}}<extra></extra>",
        ), row=2, col=1)

# Event volume
fig.add_trace(go.Bar(
    x=gri["event_date"], y=gri["total_events"],
    name="Daily Events",
    marker_color="rgba(52, 152, 219, 0.4)",
    hovertemplate="Events: %{y:,}<extra></extra>",
), row=3, col=1)

fig.update_layout(
    title=dict(
        text="<b>Geopolitical Risk Index (GRI)</b><br>"
             "<span style='font-size:13px;color:#777'>A daily composite score of global geopolitical risk, "
             "computed from 7.76M events across GDELT and ACLED</span>",
        x=0.5,
        font=dict(size=22),
    ),
    height=900,
    width=1200,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Arial", size=11),
    hovermode="x unified",
    xaxis3=dict(title=""),
    yaxis1=dict(title="GRI (0-100)", range=[30, 90]),
    yaxis2=dict(title="Sub-Index", range=[30, 90]),
    yaxis3=dict(title="Events/day"),
)

# Add range slider
fig.update_xaxes(
    rangeslider=dict(visible=True, thickness=0.05),
    row=3, col=1,
)

# Build full HTML with methodology section
latest = gri.iloc[-1]
peak = gri.loc[gri["gri"].idxmax()]
mean_gri = gri["gri"].mean()

html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Geopolitical Risk Index</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ font-size: 32px; margin-bottom: 5px; color: #2c3e50; }}
        .header p {{ color: #777; font-size: 14px; }}
        .stats {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ background: white; padding: 20px 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; min-width: 150px; }}
        .stat .value {{ font-size: 36px; font-weight: bold; }}
        .stat .label {{ font-size: 12px; color: #777; margin-top: 5px; }}
        .stat.high .value {{ color: #e74c3c; }}
        .stat.medium .value {{ color: #f39c12; }}
        .stat.low .value {{ color: #27ae60; }}
        .methodology {{ background: white; padding: 30px; border-radius: 8px; margin-top: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .methodology h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .methodology h3 {{ color: #34495e; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Geopolitical Risk Index</h1>
        <p>A daily composite score of global geopolitical risk | Open-source | Methodology-transparent</p>
    </div>

    <div class="stats">
        <div class="stat {'high' if latest['gri'] > 70 else 'medium' if latest['gri'] > 55 else 'low'}">
            <div class="value">{latest['gri']:.1f}</div>
            <div class="label">Current GRI<br>{latest['event_date'].date()}</div>
        </div>
        <div class="stat">
            <div class="value">{mean_gri:.1f}</div>
            <div class="label">Historical Average<br>(2022-2025)</div>
        </div>
        <div class="stat high">
            <div class="value">{peak['gri']:.1f}</div>
            <div class="label">All-Time High<br>{peak['event_date'].date()}</div>
        </div>
        <div class="stat">
            <div class="value">{int(latest['total_events']):,}</div>
            <div class="label">Events Today</div>
        </div>
    </div>

    <div id="chart"></div>

    <div class="methodology">
        <h2>Methodology</h2>
        <p>The GRI is computed daily from three signals, each normalized to 0-100:</p>

        <h3>1. Event Volume (40% weight)</h3>
        <p>Daily count of geopolitical events from GDELT and ACLED, weighted by category importance
        (armed conflict weights 1.0, institutional realignment weights 0.4). Normalized against a
        90-day rolling average using z-scores. Spikes above the rolling mean indicate escalation.</p>

        <h3>2. Event Severity (40% weight)</h3>
        <p>Average Goldstein scale of daily events. The Goldstein scale ranges from -10 (interstate war)
        to +10 (formal alliance). More negative values indicate higher conflict intensity. Inverted and
        scaled to 0-100.</p>

        <h3>3. Category Concentration (20% weight)</h3>
        <p>Herfindahl-Hirschman Index across 8 event categories. When events are concentrated in one
        category (e.g., 90% armed conflict), it signals a specific crisis rather than background noise.
        Higher concentration = higher risk score.</p>

        <h3>Sub-Indices</h3>
        <p>Each of the 8 geopolitical event categories has its own sub-index, computed from category-specific
        volume and severity. This allows users to see whether a GRI spike is driven by trade policy,
        armed conflict, sanctions, or other factors.</p>

        <h3>Category Weights</h3>
        <p>Derived from our ML model trained on 602 labeled company-event impact pairs. Armed conflict
        (weight 1.0) has the highest average corporate impact. Institutional realignment (0.4) has the lowest.
        These weights reflect real-world financial materiality, not headline frequency.</p>

        <h3>Data Sources</h3>
        <ul>
            <li><b>GDELT</b> — Global Database of Events, Language, and Tone (updated every 15 minutes)</li>
            <li><b>ACLED</b> — Armed Conflict Location and Event Data (updated weekly)</li>
            <li>Total training corpus: 7.76 million events (2020-2025)</li>
        </ul>

        <h3>Limitations</h3>
        <ul>
            <li>GDELT over-counts armed conflict due to media coverage bias (addressed via category weighting)</li>
            <li>The index reflects event volume and severity, not market impact directly</li>
            <li>Sub-indices for categories with few GDELT events (technology controls, regulatory shifts) may be noisy</li>
        </ul>
    </div>

    <div class="footer">
        Built on the WEF/IMD/BCG "Building Geopolitical Muscle" framework (2026)<br>
        Open-source: DistilBERT + XGBoost + ONNX Runtime | GDELT + ACLED + GTA + OFAC + SEC EDGAR
    </div>

    <script>
    </script>
</body>
</html>
"""

# Write plotly chart to div
chart_html = fig.to_html(full_html=False, include_plotlyjs=True)

# Insert chart into template
full_html = html_template.replace('<div id="chart"></div>', f'<div id="chart">{chart_html}</div>')

with open(OUTPUT, "w") as f:
    f.write(full_html)

print(f"Saved index page to {OUTPUT}")
print(f"Open in browser: file://{OUTPUT.resolve()}")
