"""
Streamlit dashboard for exploring geopolitical risk analysis.

Features:
1. Interactive analysis: enter event text + company → get full pipeline output
2. Database explorer: browse events, mentions, event studies
3. Model performance: view per-category and per-channel metrics
4. Seed label browser: explore the 602 labeled examples

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Geopolitical Risk Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ── Cached model loading ─────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    from models.event_classifier.predict import EventClassifier
    from models.exposure_scorer.predict import ExposureScorer
    from models.impact_estimator.predict import ImpactEstimator
    from models.strategy_recommender.recommend import StrategyRecommender
    return {
        "classifier": EventClassifier(),
        "scorer": ExposureScorer(),
        "estimator": ImpactEstimator(),
        "recommender": StrategyRecommender(),
    }

@st.cache_resource
def get_db():
    from pipelines.utils import get_db_connection
    return get_db_connection()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Geopolitical Risk ML")
page = st.sidebar.radio("Navigate", [
    "Analyze Event",
    "Database Explorer",
    "Seed Labels",
    "Model Stats",
])

# ── Page 1: Analyze Event ────────────────────────────────────────────────────

if page == "Analyze Event":
    st.title("Geopolitical Risk Analysis")
    st.markdown("Enter an event description and company to get a full risk assessment.")

    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area(
            "Event description",
            value="Houthi rebels fired anti-ship missiles at container vessels in the Red Sea",
            height=100,
        )
    with col2:
        ticker = st.text_input("Company ticker", value="COST")
        revenue = st.number_input("Annual revenue (USD)", value=242_000_000_000, step=1_000_000_000, format="%d")
        company_size = st.selectbox("Company size", ["large", "medium", "small"])

    if st.button("Analyze", type="primary"):
        models = load_models()

        with st.spinner("Running pipeline..."):
            # Step 1
            evt = models["classifier"].predict(text)
            st.subheader("1. Event Classification")
            col_a, col_b = st.columns(2)
            col_a.metric("Category", evt["category"].replace("_", " ").title())
            col_b.metric("Confidence", f"{evt['confidence']:.1%}")

            scores_df = pd.DataFrame([
                {"Category": k.replace("_", " ").title(), "Score": v}
                for k, v in evt["all_scores"].items()
            ]).sort_values("Score", ascending=True)
            st.bar_chart(scores_df.set_index("Category"))

            # Step 2
            exp = models["scorer"].score(
                event_category=evt["category"], ticker=ticker,
                mention_sentiment=-0.4,
            )
            st.subheader("2. Exposure Assessment")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Primary Channel", exp["channel_prediction"].replace("_", " ").title())
            col_b.metric("Confidence", f"{exp['channel_confidence']:.1%}")
            col_c.metric("Severity", f"{exp['severity_score']:+.2f}")

            ch_df = pd.DataFrame(exp["top_3_channels"])
            ch_df["channel"] = ch_df["channel"].str.replace("_", " ").str.title()
            st.dataframe(ch_df, use_container_width=True, hide_index=True)

            # Step 3
            imp = models["estimator"].estimate(
                event_category=evt["category"],
                impact_channel=exp["channel_prediction"],
                ticker=ticker,
                mention_sentiment=-0.4,
                revenue_usd=revenue,
            )
            st.subheader("3. Financial Impact Estimate")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Low", f"{imp['impact_low_pct']:+.1f}%",
                        delta=f"${imp.get('impact_low_usd', 0)/1e9:+.1f}B" if imp.get("impact_low_usd") else None)
            col_b.metric("Mid", f"{imp['impact_mid_pct']:+.1f}%",
                        delta=f"${imp.get('impact_mid_usd', 0)/1e9:+.1f}B" if imp.get("impact_mid_usd") else None)
            col_c.metric("High", f"{imp['impact_high_pct']:+.1f}%",
                        delta=f"${imp.get('impact_high_usd', 0)/1e9:+.1f}B" if imp.get("impact_high_usd") else None)

            # Step 4
            strats = models["recommender"].recommend_full(
                event_category=evt["category"],
                top_channels=exp["top_3_channels"],
                severity=imp["impact_mid_pct"] / 100,
                company_size=company_size,
            )
            st.subheader("4. Strategy Recommendations")
            for channel, channel_strats in strats.items():
                with st.expander(f"{channel.replace('_', ' ').title()} ({len(channel_strats)} strategies)"):
                    for s in channel_strats:
                        st.markdown(f"**#{s['rank']} [{s['strategy_category'].upper()}]** {s['strategy_name']}")
                        st.caption(f"Cost: {s['typical_cost']} | Time: {s['implementation_time']} | Score: {s['relevance_score']}")

# ── Page 2: Database Explorer ────────────────────────────────────────────────

elif page == "Database Explorer":
    st.title("Database Explorer")

    conn = get_db()

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    events = conn.execute("SELECT COUNT(*) as cnt FROM geopolitical_events").fetchone()["cnt"]
    mentions = conn.execute("SELECT COUNT(*) as cnt FROM geopolitical_mentions").fetchone()["cnt"]
    studies = conn.execute("SELECT COUNT(*) as cnt FROM event_studies").fetchone()["cnt"]
    strategies = conn.execute("SELECT COUNT(*) as cnt FROM strategies").fetchone()["cnt"]

    col1.metric("Events", f"{events:,}")
    col2.metric("Mentions", f"{mentions:,}")
    col3.metric("Event Studies", f"{studies:,}")
    col4.metric("Strategies", f"{strategies:,}")

    # Events by source
    st.subheader("Events by Source")
    source_df = pd.read_sql("SELECT source, COUNT(*) as count FROM geopolitical_events GROUP BY source ORDER BY count DESC", conn)
    st.bar_chart(source_df.set_index("source"))

    # Event studies summary
    st.subheader("Event Studies — Average Stock Reaction")
    es_df = pd.read_sql("""
        SELECT event_id, COUNT(DISTINCT ticker) as companies,
               ROUND(AVG(car_1_5) * 100, 2) as avg_car_5d_pct,
               ROUND(MIN(car_1_5) * 100, 2) as min_car_pct,
               ROUND(MAX(car_1_5) * 100, 2) as max_car_pct
        FROM event_studies GROUP BY event_id ORDER BY avg_car_5d_pct
    """, conn)
    st.dataframe(es_df, use_container_width=True, hide_index=True)

    # Mentions by category
    st.subheader("EDGAR Mentions by Category (Non-Boilerplate)")
    mentions_df = pd.read_sql("""
        SELECT primary_category, COUNT(*) as total,
               SUM(CASE WHEN specificity_score > 30 THEN 1 ELSE 0 END) as non_boilerplate,
               ROUND(AVG(specificity_score), 1) as avg_specificity
        FROM geopolitical_mentions GROUP BY primary_category ORDER BY non_boilerplate DESC
    """, conn)
    st.dataframe(mentions_df, use_container_width=True, hide_index=True)

# ── Page 3: Seed Labels ─────────────────────────────────────────────────────

elif page == "Seed Labels":
    st.title("Seed Labels Browser")

    import csv
    with open(ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv") as f:
        labels = list(csv.DictReader(f))

    df = pd.DataFrame(labels)
    st.metric("Total Labels", len(df))

    # Channel distribution
    st.subheader("Labels by Impact Channel")
    channel_counts = df["impact_channel"].value_counts()
    st.bar_chart(channel_counts)

    # Event distribution
    st.subheader("Labels by Event")
    event_counts = df["event_id"].value_counts().head(20)
    st.bar_chart(event_counts)

    # Source distribution
    st.subheader("Labels by Source")
    if "labeled_by" in df.columns:
        source_counts = df["labeled_by"].value_counts()
        st.bar_chart(source_counts)

    # Browse labels
    st.subheader("Browse Labels")
    channel_filter = st.selectbox("Filter by channel", ["All"] + sorted(df["impact_channel"].unique()))
    if channel_filter != "All":
        df = df[df["impact_channel"] == channel_filter]
    st.dataframe(
        df[["event_id", "company_ticker", "impact_channel", "mention_sentiment", "car_1_5", "confidence", "labeled_by"]],
        use_container_width=True, hide_index=True,
    )

# ── Page 4: Model Stats ─────────────────────────────────────────────────────

elif page == "Model Stats":
    st.title("Model Performance")

    st.subheader("Model 1: Event Classifier")
    st.markdown("""
    | Metric | Source Format | News Text |
    |--------|:-:|:-:|
    | Accuracy | 94.6% | **95.3%** |
    | Backend | ONNX (0.8MB) | ONNX |
    """)

    st.subheader("Model 2: Exposure Scorer")
    m2_data = {
        "Channel": [
            "revenue_market_access", "procurement_supply_chain", "logistics_operations",
            "regulatory_compliance", "innovation_ip", "financial_treasury",
            "workforce_talent", "cybersecurity_it", "capital_allocation",
            "reputation_stakeholder",
        ],
        "F1 Score": [0.818, 0.789, 0.880, 0.857, 0.917, 0.842, 0.842, 0.824, 0.778, 0.706],
        "Labels": [104, 97, 60, 57, 58, 51, 45, 40, 46, 44],
    }
    m2_df = pd.DataFrame(m2_data).sort_values("F1 Score", ascending=False)
    st.dataframe(m2_df, use_container_width=True, hide_index=True)
    st.metric("Macro F1", "0.825")

    st.subheader("Model 3: Impact Estimator")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Coverage (q05-q95) | **80.7%** |
    | MAE (median) | 0.52 pp |
    | Seed label coverage | 75.7% |
    """)

    st.subheader("Model 4: Strategy Recommender")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Total strategies | 148 |
    | Event-channel cells covered | 34/34 |
    | Strategy categories | mitigate, hedge, exit, capture, engage, monitor |
    """)
