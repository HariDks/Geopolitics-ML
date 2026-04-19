"""
Geopolitical Impact Tester — Test how geopolitical events affect companies.

Input: event description + company → Output: impact channels, severity range,
reliability level, explanation, and mechanism signals.

Two explicit modes:
- text_rich: uses event text → high reliability (~75% top-2 with secondary)
- text_poor: structured fallback → lower reliability (~46% top-2)

Usage:
    streamlit run dashboard/app.py
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Geopolitical Impact Tester",
    layout="wide",
)

# ── Data ─────────────────────────────────────────────────────────────────────

COMPANIES = {
    "Apple (AAPL)": {"ticker": "AAPL", "revenue": 383e9, "sector": "Information Technology"},
    "Microsoft (MSFT)": {"ticker": "MSFT", "revenue": 245e9, "sector": "Information Technology"},
    "Amazon (AMZN)": {"ticker": "AMZN", "revenue": 620e9, "sector": "Consumer Discretionary"},
    "NVIDIA (NVDA)": {"ticker": "NVDA", "revenue": 130e9, "sector": "Information Technology"},
    "Tesla (TSLA)": {"ticker": "TSLA", "revenue": 97e9, "sector": "Consumer Discretionary"},
    "JPMorgan Chase (JPM)": {"ticker": "JPM", "revenue": 177e9, "sector": "Financials"},
    "Exxon Mobil (XOM)": {"ticker": "XOM", "revenue": 344e9, "sector": "Energy"},
    "Boeing (BA)": {"ticker": "BA", "revenue": 78e9, "sector": "Industrials"},
    "Lockheed Martin (LMT)": {"ticker": "LMT", "revenue": 68e9, "sector": "Industrials"},
    "Raytheon/RTX (RTX)": {"ticker": "RTX", "revenue": 69e9, "sector": "Industrials"},
    "McDonald's (MCD)": {"ticker": "MCD", "revenue": 26e9, "sector": "Consumer Discretionary"},
    "Costco (COST)": {"ticker": "COST", "revenue": 242e9, "sector": "Consumer Staples"},
    "Walmart (WMT)": {"ticker": "WMT", "revenue": 648e9, "sector": "Consumer Staples"},
    "Meta/Facebook (META)": {"ticker": "META", "revenue": 165e9, "sector": "Information Technology"},
    "Intel (INTC)": {"ticker": "INTC", "revenue": 54e9, "sector": "Information Technology"},
    "Nike (NKE)": {"ticker": "NKE", "revenue": 51e9, "sector": "Consumer Discretionary"},
    "Pfizer (PFE)": {"ticker": "PFE", "revenue": 58e9, "sector": "Health Care"},
    "Goldman Sachs (GS)": {"ticker": "GS", "revenue": 51e9, "sector": "Financials"},
    "Visa (V)": {"ticker": "V", "revenue": 36e9, "sector": "Financials"},
    "Mastercard (MA)": {"ticker": "MA", "revenue": 28e9, "sector": "Financials"},
    "FedEx (FDX)": {"ticker": "FDX", "revenue": 88e9, "sector": "Industrials"},
    "Coca-Cola (KO)": {"ticker": "KO", "revenue": 46e9, "sector": "Consumer Staples"},
    "Chevron (CVX)": {"ticker": "CVX", "revenue": 196e9, "sector": "Energy"},
    "Johnson & Johnson (JNJ)": {"ticker": "JNJ", "revenue": 85e9, "sector": "Health Care"},
    "Home Depot (HD)": {"ticker": "HD", "revenue": 157e9, "sector": "Consumer Discretionary"},
    "Other (enter manually)": {"ticker": "", "revenue": 0, "sector": ""},
}

PREBUILT_EXAMPLES = {
    "Select an example...": {"text": "", "company": ""},
    "Russia invasion -> BP (asset write-down)": {
        "text": "Russia launched full-scale invasion of Ukraine, triggering Western sanctions and corporate exits from Russia",
        "company": "Exxon Mobil (XOM)",
    },
    "Chip export controls -> NVIDIA (revenue loss)": {
        "text": "US Bureau of Industry and Security restricted exports of advanced AI chips and semiconductor equipment to China",
        "company": "NVIDIA (NVDA)",
    },
    "Red Sea disruption -> Costco (shipping costs)": {
        "text": "Houthi rebels fired anti-ship missiles at commercial vessels in the Red Sea forcing major shipping lines to reroute around the Cape of Good Hope",
        "company": "Costco (COST)",
    },
    "Ransomware attack -> critical infrastructure": {
        "text": "DarkSide ransomware group encrypted Colonial Pipeline IT systems forcing shutdown of the largest US fuel pipeline for six days",
        "company": "Intel (INTC)",
    },
    "OPEC production cut -> Boeing (fuel costs)": {
        "text": "OPEC announced a surprise production cut of 2 million barrels per day sending oil prices surging 8 percent",
        "company": "Boeing (BA)",
    },
    "EU Digital Markets Act -> Apple (compliance)": {
        "text": "EU passed the Digital Markets Act requiring Big Tech platforms to allow sideloading third-party app stores and interoperability",
        "company": "Apple (AAPL)",
    },
    "China tariffs -> Walmart (sourcing costs)": {
        "text": "US imposed 25 percent tariffs on all Chinese imports including electronics semiconductors and consumer goods",
        "company": "Walmart (WMT)",
    },
    "Xinjiang boycott -> Nike (brand risk)": {
        "text": "Xinjiang forced labor allegations triggered Western brands boycott of Chinese cotton and consumer backlash in China against brands that spoke out",
        "company": "Nike (NKE)",
    },
}

CHANNEL_DESCRIPTIONS = {
    "procurement_supply_chain": "Input costs, supplier disruption, manufacturing delays",
    "revenue_market_access": "Lost customers, market access restrictions, demand decline",
    "capital_allocation_investment": "Asset impairments, write-downs, investment losses",
    "regulatory_compliance_cost": "New compliance requirements, fines, licensing changes",
    "logistics_operations": "Route disruptions, shipping delays, operational downtime",
    "innovation_ip": "Technology restrictions, IP access, R&D limitations",
    "workforce_talent": "Labor shortages, evacuations, visa restrictions",
    "reputation_stakeholder": "Brand damage, boycotts, public backlash",
    "financial_treasury": "Currency risk, trapped capital, asset freezes",
    "cybersecurity_it": "Cyberattacks, ransomware, IT system destruction",
}

FEEDBACK_PATH = ROOT_DIR / "data" / "feedback.csv"


@st.cache_resource
def load_models():
    from models.event_classifier.predict import EventClassifier
    from models.exposure_scorer.predict import ExposureScorer
    from models.impact_estimator.predict import ImpactEstimator
    return {
        "classifier": EventClassifier(),
        "scorer": ExposureScorer(),
        "estimator": ImpactEstimator(),
    }


def fmt_usd(val):
    if not val:
        return ""
    a, s = abs(val), "-" if val < 0 else "+"
    if a >= 1e9: return f"{s}${a/1e9:.1f}B"
    if a >= 1e6: return f"{s}${a/1e6:.0f}M"
    return f"{s}${a:,.0f}"


def save_feedback(data: dict):
    """Append feedback to CSV."""
    file_exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# ── Header ───────────────────────────────────────────────────────────────────

st.title("Geopolitical Impact Tester")
st.caption("Test how a geopolitical event affects a specific company. Powered by ML models trained on 7.76M events and 602 labeled impacts.")

# ── Input Panel ──────────────────────────────────────────────────────────────

st.subheader("Input")

input_mode = st.radio("Input mode", ["Quick examples", "Custom event"], horizontal=True, label_visibility="collapsed")

if input_mode == "Quick examples":
    example = st.selectbox("Select a scenario", list(PREBUILT_EXAMPLES.keys()))
    ex_data = PREBUILT_EXAMPLES[example]
    event_text = ex_data.get("text", "")
    default_company = ex_data.get("company", "")

    if event_text:
        st.info(f'"{event_text}"')
        company_name = st.selectbox("Company", list(COMPANIES.keys()),
                                     index=list(COMPANIES.keys()).index(default_company) if default_company in COMPANIES else 0)
    else:
        company_name = list(COMPANIES.keys())[0]
else:
    event_text = st.text_area("Describe the geopolitical event (1-3 sentences)",
                               placeholder="e.g., China restricted exports of gallium and germanium critical minerals used in semiconductor manufacturing",
                               height=100)
    company_name = st.selectbox("Company", list(COMPANIES.keys()))

company_info = COMPANIES[company_name]
if company_name == "Other (enter manually)":
    col1, col2 = st.columns(2)
    ticker = col1.text_input("Ticker")
    revenue = col2.number_input("Annual revenue (USD)", value=0, step=1_000_000_000, format="%d")
else:
    ticker = company_info["ticker"]
    revenue = company_info["revenue"]
    st.caption(f"**{ticker}** | ${revenue/1e9:.0f}B revenue | {company_info['sector']}")

# ── Run Analysis ─────────────────────────────────────────────────────────────

if st.button("Analyze Impact", type="primary", disabled=not event_text):
    models = load_models()

    with st.spinner("Running analysis..."):
        # Step 1: Classify
        evt = models["classifier"].predict(event_text)

        # Step 2: Exposure (with text for lexicon)
        exp = models["scorer"].score(
            event_category=evt["category"],
            ticker=ticker,
            mention_sentiment=-0.4,
            event_text=event_text,
        )

        # Step 3: Impact
        imp = models["estimator"].estimate(
            event_category=evt["category"],
            impact_channel=exp["channel_prediction"],
            ticker=ticker,
            mention_sentiment=-0.4,
            revenue_usd=revenue,
        )

    # ── Output Panel ─────────────────────────────────────────────────────

    st.divider()
    st.subheader("Results")

    # Row 1: Classification + Reliability badge
    col1, col2, col3 = st.columns(3)
    col1.metric("Event Type", evt["category"].replace("_", " ").title())

    reliability = exp.get("channel_reliability", "unknown")
    mode = exp.get("channel_mode", "unknown")
    if reliability == "high":
        col2.success(f"Reliability: HIGH")
    elif reliability == "moderate":
        col2.warning(f"Reliability: MODERATE")
    else:
        col2.error(f"Reliability: LOW")

    col3.metric("Classification Confidence", f"{evt['confidence']:.0%}")

    # Row 2: Channel prediction (top-2)
    st.subheader("Impact Channels (Top 2)")
    probs = exp["channel_probabilities"]
    ranked = sorted(probs.items(), key=lambda x: -x[1])

    col1, col2 = st.columns(2)
    ch1_name = ranked[0][0]
    ch2_name = ranked[1][0]

    with col1:
        st.markdown(f"### Primary: {ch1_name.replace('_', ' ').title()}")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch1_name, ""))
        st.metric("Probability", f"{ranked[0][1]:.0%}")

    with col2:
        st.markdown(f"### Secondary: {ch2_name.replace('_', ' ').title()}")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch2_name, ""))
        st.metric("Probability", f"{ranked[1][1]:.0%}")

    # Row 3: Impact range
    st.subheader("Financial Impact Estimate")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimistic", f"{imp['impact_low_pct']:+.1f}%",
                delta=fmt_usd(imp.get("impact_low_usd")) or None)
    col2.metric("Base Case", f"{imp['impact_mid_pct']:+.1f}%",
                delta=fmt_usd(imp.get("impact_mid_usd")) or None)
    col3.metric("Pessimistic", f"{imp['impact_high_pct']:+.1f}%",
                delta=fmt_usd(imp.get("impact_high_usd")) or None)

    # Row 4: Explanation (mechanism signals)
    st.subheader("Explanation")

    from models.exposure_scorer.train import compute_lexicon_scores, CHANNEL_LEXICONS
    lex = compute_lexicon_scores(event_text)

    # Find which lexicon keywords matched
    active_signals = []
    event_lower = event_text.lower()
    for channel, keywords in CHANNEL_LEXICONS.items():
        matched = [kw for kw in keywords if kw in event_lower]
        if matched:
            ch_display = channel.replace("_", " ").title()
            active_signals.append(f"Detected **{', '.join(matched[:3])}** -> {ch_display}")

    if active_signals:
        for signal in active_signals:
            st.markdown(f"- {signal}")
        st.caption(f"Mode: **{mode}** | Reliability is **{reliability}** because descriptive event text was provided.")
    else:
        st.warning("No channel-specific keywords detected in event text. Prediction based on structured features only.")
        st.caption(f"Mode: **{mode}** | Reliability is **{reliability}**. Provide more descriptive text to improve accuracy.")

    # Row 5: Diagnostics (collapsed)
    with st.expander("Diagnostics"):
        st.markdown("**All channel probabilities:**")
        ch_df = pd.DataFrame([
            {"Channel": ch.replace("_", " ").title(), "Probability": f"{p:.1%}", "Raw": p}
            for ch, p in ranked
        ])
        st.dataframe(ch_df[["Channel", "Probability"]], use_container_width=True, hide_index=True)

        st.markdown("**Lexicon scores:**")
        lex_df = pd.DataFrame([
            {"Channel": ch.replace("_", " ").title(), "Score": f"{score:.2f}"}
            for ch, score in sorted(lex.items(), key=lambda x: -x[1]) if score > 0
        ])
        if not lex_df.empty:
            st.dataframe(lex_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No lexicon signals detected.")

        st.markdown(f"**Event classification scores:**")
        evt_df = pd.DataFrame([
            {"Category": k.replace("_", " ").title(), "Score": f"{v:.1%}"}
            for k, v in sorted(evt["all_scores"].items(), key=lambda x: -x[1])
        ])
        st.dataframe(evt_df, use_container_width=True, hide_index=True)

    # ── Feedback Section ─────────────────────────────────────────────────

    st.divider()
    st.subheader("Feedback")
    st.caption("Help improve the model by telling us if this prediction was useful.")

    col1, col2 = st.columns(2)
    with col1:
        useful = st.radio("Was this useful?", ["Select...", "Yes", "No"], horizontal=True)
    with col2:
        channel_correct = st.radio("Was the primary channel correct?", ["Select...", "Yes", "No"], horizontal=True)

    correct_channel = None
    comment = ""
    if channel_correct == "No":
        correct_channel = st.selectbox("What should the primary channel be?",
                                        [""] + [ch.replace("_", " ").title() for ch in CHANNEL_DESCRIPTIONS.keys()])
        comment = st.text_input("Optional comment")

    if st.button("Submit Feedback"):
        if useful != "Select..." or channel_correct != "Select...":
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "event_text": event_text[:500],
                "company": company_name,
                "ticker": ticker,
                "predicted_channel_1": ch1_name,
                "predicted_channel_2": ch2_name,
                "reliability_mode": mode,
                "useful_vote": useful if useful != "Select..." else "",
                "channel_correct": channel_correct if channel_correct != "Select..." else "",
                "suggested_channel": correct_channel or "",
                "comment": comment,
            }
            save_feedback(feedback)
            st.success("Feedback saved. Thank you!")
        else:
            st.warning("Please select at least one option.")

# ── Sidebar: Evaluation summary ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model Performance")
    st.markdown("""
    Evaluated on **69 real-world event-company pairs**:

    | Mode | Top-2 Accuracy |
    |------|:-:|
    | Without text | 46.4% |
    | **With text** | **62.3%** |
    | With secondary | **75.4%** |

    Event text improves accuracy by **+15.9pp**.

    Direction accuracy (up/down): **90%**
    """)

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Classify** event into 8 categories
    2. **Score** company exposure using sector, financials, and text signals
    3. **Estimate** financial impact range
    4. **Explain** which mechanism keywords drove the prediction
    """)

    st.markdown("---")
    st.caption("Built on WEF/IMD/BCG framework | 7.76M events | Open source")
    st.caption("[GitHub](https://github.com/HariDks/Geopolitics-ML)")
