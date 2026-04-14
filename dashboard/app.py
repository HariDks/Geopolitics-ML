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

# ── Company lookup data ──────────────────────────────────────────────────────

# S&P 500 companies with name, ticker, sector, and approximate annual revenue
COMPANIES = {
    "Apple (AAPL)": {"ticker": "AAPL", "revenue": 383_000_000_000, "size": "large", "sector": "Information Technology"},
    "Microsoft (MSFT)": {"ticker": "MSFT", "revenue": 245_000_000_000, "size": "large", "sector": "Information Technology"},
    "Amazon (AMZN)": {"ticker": "AMZN", "revenue": 620_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "NVIDIA (NVDA)": {"ticker": "NVDA", "revenue": 130_000_000_000, "size": "large", "sector": "Information Technology"},
    "Alphabet/Google (GOOGL)": {"ticker": "GOOGL", "revenue": 350_000_000_000, "size": "large", "sector": "Communication Services"},
    "Meta/Facebook (META)": {"ticker": "META", "revenue": 165_000_000_000, "size": "large", "sector": "Information Technology"},
    "Tesla (TSLA)": {"ticker": "TSLA", "revenue": 97_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "Berkshire Hathaway (BRK-B)": {"ticker": "BRK-B", "revenue": 364_000_000_000, "size": "large", "sector": "Financials"},
    "JPMorgan Chase (JPM)": {"ticker": "JPM", "revenue": 177_000_000_000, "size": "large", "sector": "Financials"},
    "Visa (V)": {"ticker": "V", "revenue": 36_000_000_000, "size": "large", "sector": "Financials"},
    "UnitedHealth (UNH)": {"ticker": "UNH", "revenue": 400_000_000_000, "size": "large", "sector": "Health Care"},
    "Johnson & Johnson (JNJ)": {"ticker": "JNJ", "revenue": 85_000_000_000, "size": "large", "sector": "Health Care"},
    "Exxon Mobil (XOM)": {"ticker": "XOM", "revenue": 344_000_000_000, "size": "large", "sector": "Energy"},
    "Chevron (CVX)": {"ticker": "CVX", "revenue": 196_000_000_000, "size": "large", "sector": "Energy"},
    "Procter & Gamble (PG)": {"ticker": "PG", "revenue": 84_000_000_000, "size": "large", "sector": "Consumer Staples"},
    "Costco (COST)": {"ticker": "COST", "revenue": 242_000_000_000, "size": "large", "sector": "Consumer Staples"},
    "McDonald's (MCD)": {"ticker": "MCD", "revenue": 26_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "Boeing (BA)": {"ticker": "BA", "revenue": 78_000_000_000, "size": "large", "sector": "Industrials"},
    "Caterpillar (CAT)": {"ticker": "CAT", "revenue": 67_000_000_000, "size": "large", "sector": "Industrials"},
    "Lockheed Martin (LMT)": {"ticker": "LMT", "revenue": 68_000_000_000, "size": "large", "sector": "Industrials"},
    "Raytheon/RTX (RTX)": {"ticker": "RTX", "revenue": 69_000_000_000, "size": "large", "sector": "Industrials"},
    "Broadcom (AVGO)": {"ticker": "AVGO", "revenue": 51_000_000_000, "size": "large", "sector": "Information Technology"},
    "Cisco (CSCO)": {"ticker": "CSCO", "revenue": 54_000_000_000, "size": "large", "sector": "Information Technology"},
    "Intel (INTC)": {"ticker": "INTC", "revenue": 54_000_000_000, "size": "large", "sector": "Information Technology"},
    "Nike (NKE)": {"ticker": "NKE", "revenue": 51_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "Goldman Sachs (GS)": {"ticker": "GS", "revenue": 51_000_000_000, "size": "large", "sector": "Financials"},
    "Coca-Cola (KO)": {"ticker": "KO", "revenue": 46_000_000_000, "size": "large", "sector": "Consumer Staples"},
    "PepsiCo (PEP)": {"ticker": "PEP", "revenue": 92_000_000_000, "size": "large", "sector": "Consumer Staples"},
    "Walmart (WMT)": {"ticker": "WMT", "revenue": 648_000_000_000, "size": "large", "sector": "Consumer Staples"},
    "Home Depot (HD)": {"ticker": "HD", "revenue": 157_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "Pfizer (PFE)": {"ticker": "PFE", "revenue": 58_000_000_000, "size": "large", "sector": "Health Care"},
    "Eli Lilly (LLY)": {"ticker": "LLY", "revenue": 41_000_000_000, "size": "large", "sector": "Health Care"},
    "Mastercard (MA)": {"ticker": "MA", "revenue": 28_000_000_000, "size": "large", "sector": "Financials"},
    "Schlumberger (SLB)": {"ticker": "SLB", "revenue": 36_000_000_000, "size": "large", "sector": "Energy"},
    "FedEx (FDX)": {"ticker": "FDX", "revenue": 88_000_000_000, "size": "large", "sector": "Industrials"},
    "General Motors (GM)": {"ticker": "GM", "revenue": 172_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "Ford (F)": {"ticker": "F", "revenue": 176_000_000_000, "size": "large", "sector": "Consumer Discretionary"},
    "Other (enter manually)": {"ticker": "", "revenue": 0, "size": "large", "sector": ""},
}

# Pre-built event scenarios for quick analysis
EVENT_SCENARIOS = {
    "Custom (enter your own)": "",
    "Red Sea shipping disruption": "Houthi rebels fired anti-ship missiles at container vessels in the Red Sea, forcing major shipping lines to reroute around the Cape of Good Hope",
    "US-China tariff escalation": "US announced 25% tariffs on all Chinese imports including electronics, semiconductors, and consumer goods",
    "Russia-Ukraine escalation": "Russian forces launched a major new offensive in eastern Ukraine, triggering additional Western sanctions packages",
    "OPEC production cut": "OPEC announced a surprise production cut of 2 million barrels per day, sending oil prices surging 8%",
    "Taiwan Strait tensions": "China conducted large-scale military exercises around Taiwan including live fire drills and naval blockade rehearsals",
    "Iran nuclear crisis": "IAEA reported Iran enriching uranium to 90% purity, triggering emergency UN Security Council meeting",
    "Major cyberattack": "State-sponsored hackers launched coordinated ransomware attack on critical infrastructure across multiple countries",
    "EU regulatory crackdown": "EU passed sweeping new Digital Markets Act enforcement actions against major US tech platforms",
    "Emerging market debt crisis": "Multiple emerging market currencies collapsed as the Fed raised rates, triggering capital flight from developing economies",
    "Global supply chain shock": "Major earthquake in Taiwan disrupted semiconductor manufacturing, with TSMC reporting 3-month production delays",
}

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
    "Portfolio Scanner",
    "Database Explorer",
    "Seed Labels",
    "Model Stats",
])

# ── Page 1: Analyze Event ────────────────────────────────────────────────────

if page == "Analyze Event":
    st.title("Geopolitical Risk Analysis")
    st.markdown("Select an event scenario and company to get a full risk assessment.")

    # Event selection — dropdown of scenarios OR custom text
    col_event, col_company = st.columns([3, 2])

    with col_event:
        scenario = st.selectbox("Event scenario", list(EVENT_SCENARIOS.keys()))
        if scenario == "Custom (enter your own)":
            text = st.text_area("Describe the geopolitical event", height=100,
                               placeholder="e.g., China imposed export controls on rare earth minerals...")
        else:
            text = EVENT_SCENARIOS[scenario]
            st.info(f'"{text}"')

    with col_company:
        company_name = st.selectbox("Company", list(COMPANIES.keys()))
        company_info = COMPANIES[company_name]

        if company_name == "Other (enter manually)":
            ticker = st.text_input("Ticker symbol", value="")
            revenue = st.number_input("Annual revenue (USD)", value=0, step=1_000_000_000, format="%d")
            company_size = st.selectbox("Company size", ["large", "medium", "small"])
        else:
            ticker = company_info["ticker"]
            revenue = company_info["revenue"]
            company_size = company_info["size"]
            st.caption(f"Ticker: **{ticker}** | Revenue: **${revenue/1e9:.0f}B** | Sector: {company_info['sector']}")

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

# ── Page 2: Portfolio Scanner ────────────────────────────────────────────────

elif page == "Portfolio Scanner":
    st.title("Portfolio Risk Scanner")
    st.markdown("Select companies and an event scenario to see which holdings are most exposed.")

    # Event selection
    scenario = st.selectbox("Event scenario", [k for k in EVENT_SCENARIOS.keys() if k != "Custom (enter your own)"])
    event_text = EVENT_SCENARIOS[scenario]
    st.info(f'"{event_text}"')

    # Portfolio selection
    all_companies = [k for k in COMPANIES.keys() if k != "Other (enter manually)"]
    default_portfolio = ["Apple (AAPL)", "Exxon Mobil (XOM)", "Boeing (BA)", "JPMorgan Chase (JPM)",
                        "NVIDIA (NVDA)", "McDonald's (MCD)", "Costco (COST)", "Lockheed Martin (LMT)",
                        "Goldman Sachs (GS)", "Pfizer (PFE)"]
    selected = st.multiselect("Select portfolio companies", all_companies, default=default_portfolio)

    if st.button("Scan Portfolio", type="primary") and selected:
        models = load_models()

        with st.spinner("Classifying event..."):
            evt = models["classifier"].predict(event_text)
        st.subheader(f"Event: {evt['category'].replace('_', ' ').title()} ({evt['confidence']:.0%})")

        results = []
        progress = st.progress(0)
        for i, company_name in enumerate(selected):
            info = COMPANIES[company_name]
            exp = models["scorer"].score(
                event_category=evt["category"], ticker=info["ticker"],
                mention_sentiment=-0.4,
            )
            imp = models["estimator"].estimate(
                event_category=evt["category"],
                impact_channel=exp["channel_prediction"],
                ticker=info["ticker"],
                mention_sentiment=-0.4,
                revenue_usd=info["revenue"],
            )
            results.append({
                "Company": company_name.split(" (")[0],
                "Ticker": info["ticker"],
                "Sector": info["sector"],
                "Primary Channel": exp["channel_prediction"].replace("_", " ").title(),
                "Severity": exp["severity_score"],
                "Impact (Mid %)": imp["impact_mid_pct"],
                "Impact (Mid $)": imp.get("impact_mid_usd", 0),
                "Confidence": exp["channel_confidence"],
            })
            progress.progress((i + 1) / len(selected))

        results_df = pd.DataFrame(results).sort_values("Severity")
        progress.empty()

        # Risk heatmap
        st.subheader("Exposure Ranking")
        st.dataframe(
            results_df.style.background_gradient(subset=["Severity"], cmap="RdYlGn_r")
                .format({"Severity": "{:+.2f}", "Impact (Mid %)": "{:+.1f}%",
                         "Impact (Mid $)": "${:,.0f}", "Confidence": "{:.0%}"}),
            use_container_width=True, hide_index=True,
        )

        # Most and least exposed
        col1, col2 = st.columns(2)
        with col1:
            most = results_df.iloc[0]
            st.metric("Most Exposed", most["Company"],
                     delta=f"{most['Severity']:+.2f} severity",
                     delta_color="inverse")
        with col2:
            least = results_df.iloc[-1]
            st.metric("Least Exposed", least["Company"],
                     delta=f"{least['Severity']:+.2f} severity")

# ── Page 3: Database Explorer ────────────────────────────────────────────────

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
