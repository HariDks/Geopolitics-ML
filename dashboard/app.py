"""
Geopolitical Impact Tester — Test how geopolitical events affect companies.
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

st.set_page_config(page_title="Geopolitical Impact Tester", layout="wide")

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
    "FedEx (FDX)": {"ticker": "FDX", "revenue": 88e9, "sector": "Industrials"},
    "Coca-Cola (KO)": {"ticker": "KO", "revenue": 46e9, "sector": "Consumer Staples"},
    "Chevron (CVX)": {"ticker": "CVX", "revenue": 196e9, "sector": "Energy"},
    "Johnson & Johnson (JNJ)": {"ticker": "JNJ", "revenue": 85e9, "sector": "Health Care"},
    "Home Depot (HD)": {"ticker": "HD", "revenue": 157e9, "sector": "Consumer Discretionary"},
    "Other (enter manually)": {"ticker": "", "revenue": 0, "sector": ""},
}

QUICK_SCENARIOS = [
    {"label": "Russia invasion -> Exxon", "text": "Russia launched full-scale invasion of Ukraine, triggering Western sanctions and corporate exits from Russia", "company": "Exxon Mobil (XOM)"},
    {"label": "Chip export controls -> NVIDIA", "text": "US Bureau of Industry and Security restricted exports of advanced AI chips and semiconductor equipment to China", "company": "NVIDIA (NVDA)"},
    {"label": "Red Sea disruption -> Costco", "text": "Houthi rebels fired anti-ship missiles at commercial vessels in the Red Sea forcing major shipping lines to reroute around the Cape of Good Hope", "company": "Costco (COST)"},
    {"label": "Ransomware -> Intel", "text": "DarkSide ransomware group encrypted Colonial Pipeline IT systems forcing shutdown of the largest US fuel pipeline for six days", "company": "Intel (INTC)"},
    {"label": "OPEC cut -> Boeing", "text": "OPEC announced a surprise production cut of 2 million barrels per day sending oil prices surging 8 percent", "company": "Boeing (BA)"},
    {"label": "EU regulation -> Apple", "text": "EU passed the Digital Markets Act requiring Big Tech platforms to allow sideloading third-party app stores and interoperability", "company": "Apple (AAPL)"},
    {"label": "China tariffs -> Walmart", "text": "US imposed 25 percent tariffs on all Chinese imports including electronics semiconductors and consumer goods", "company": "Walmart (WMT)"},
    {"label": "Xinjiang boycott -> Nike", "text": "Xinjiang forced labor allegations triggered Western brands boycott of Chinese cotton and consumer backlash in China against brands that spoke out", "company": "Nike (NKE)"},
]

CHANNEL_DESCRIPTIONS = {
    "procurement_supply_chain": "Input costs rise, suppliers disrupted, manufacturing delayed",
    "revenue_market_access": "Lost customers, market access blocked, demand declined",
    "capital_allocation_investment": "Assets impaired, investments written down, deals blocked",
    "regulatory_compliance_cost": "New compliance requirements, fines, licensing changes",
    "logistics_operations": "Routes disrupted, shipping delayed, operations halted",
    "innovation_ip": "Technology restricted, IP access limited, R&D constrained",
    "workforce_talent": "Labor disrupted, employees relocated, visa restrictions",
    "reputation_stakeholder": "Brand damaged, boycotts triggered, public backlash",
    "financial_treasury": "Currency devalued, capital trapped, assets frozen",
    "cybersecurity_it": "Systems attacked, ransomware deployed, IT destroyed",
}

FEEDBACK_PATH = ROOT_DIR / "data" / "feedback.csv"


@st.cache_resource
def load_models():
    from models.event_classifier.predict import EventClassifier
    from models.exposure_scorer.predict import ExposureScorer
    from models.impact_estimator.predict import ImpactEstimator
    return EventClassifier(), ExposureScorer(), ImpactEstimator()


def fmt_usd(val):
    if not val: return ""
    a, s = abs(val), "-" if val < 0 else "+"
    if a >= 1e9: return f"{s}${a/1e9:.1f}B"
    if a >= 1e6: return f"{s}${a/1e6:.0f}M"
    return f"{s}${a:,.0f}"


def save_feedback(data: dict):
    exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not exists: w.writeheader()
        w.writerow(data)


# ── Header ───────────────────────────────────────────────────────────────────

st.title("Geopolitical Impact Tester")
st.caption("Test how a geopolitical event affects a specific company's business.")

# ── Quick Scenarios (hero section) ───────────────────────────────────────────

st.markdown("#### Try these scenarios")
cols = st.columns(4)
selected_scenario = None
for i, scenario in enumerate(QUICK_SCENARIOS):
    if cols[i % 4].button(scenario["label"], use_container_width=True):
        selected_scenario = scenario

st.divider()

# ── Input Panel ──────────────────────────────────────────────────────────────

col_text, col_company = st.columns([3, 2])

with col_text:
    if selected_scenario:
        event_text = selected_scenario["text"]
        st.text_area("Event description", value=event_text, height=80, disabled=True, key="event_display")
    else:
        event_text = st.text_area("Event description", height=80,
                                   placeholder="Describe a geopolitical event in 1-3 sentences...",
                                   key="event_input")

with col_company:
    if selected_scenario:
        default_idx = list(COMPANIES.keys()).index(selected_scenario["company"])
    else:
        default_idx = 0
    company_name = st.selectbox("Company", list(COMPANIES.keys()), index=default_idx)
    info = COMPANIES[company_name]

    if company_name == "Other (enter manually)":
        ticker = st.text_input("Ticker")
        revenue = st.number_input("Annual revenue (USD)", value=0, step=1_000_000_000, format="%d")
    else:
        ticker = info["ticker"]
        revenue = info["revenue"]
        st.caption(f"**{ticker}** | ${revenue/1e9:.0f}B revenue | {info['sector']}")

# ── Run ──────────────────────────────────────────────────────────────────────

run = st.button("Analyze Impact", type="primary", disabled=not event_text, use_container_width=True)

if run or selected_scenario:
    if selected_scenario and not run:
        event_text = selected_scenario["text"]

    if not event_text:
        st.stop()

    clf, scorer, estimator = load_models()

    with st.spinner("Analyzing..."):
        evt = clf.predict(event_text)
        exp = scorer.score(event_category=evt["category"], ticker=ticker, mention_sentiment=-0.4, event_text=event_text)
        imp = estimator.estimate(event_category=evt["category"], impact_channel=exp["channel_prediction"],
                                  ticker=ticker, mention_sentiment=-0.4, revenue_usd=revenue)

    probs = exp["channel_probabilities"]
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    ch1, ch2 = ranked[0][0], ranked[1][0]
    reliability = exp.get("channel_reliability", "unknown")
    mode = exp.get("channel_mode", "unknown")

    # Determine impact severity label
    mid = imp["impact_mid_pct"]
    if abs(mid) < 0.5:
        severity_label = "Minimal"
        severity_color = "gray"
    elif mid < -3:
        severity_label = "Significant negative"
        severity_color = "red"
    elif mid < 0:
        severity_label = "Moderate negative"
        severity_color = "orange"
    elif mid > 3:
        severity_label = "Significant positive"
        severity_color = "green"
    else:
        severity_label = "Moderate positive"
        severity_color = "blue"

    st.divider()

    # ── Top-line summary card ────────────────────────────────────────────

    company_short = company_name.split(" (")[0]
    ch1_short = ch1.replace("_", " ").title()
    ch1_desc = CHANNEL_DESCRIPTIONS.get(ch1, "")

    if reliability == "high":
        conf_badge = "High (text-rich)"
    elif reliability == "moderate":
        conf_badge = "Moderate"
    else:
        conf_badge = "Low (limited text signals)"

    st.markdown(f"""
    ### What this means for {company_short}

    | | |
    |---|---|
    | **Likely impact** | {severity_label} ({imp['impact_low_pct']:+.1f}% to {imp['impact_high_pct']:+.1f}%) |
    | **Main driver** | {ch1_short} — {ch1_desc.lower()} |
    | **Dollar range** | {fmt_usd(imp.get('impact_low_usd', 0))} to {fmt_usd(imp.get('impact_high_usd', 0))} (on ${revenue/1e9:.0f}B revenue) |
    | **Confidence** | {conf_badge} |
    """)

    # ── Why this prediction ──────────────────────────────────────────────

    st.markdown("#### Why this prediction?")

    from models.exposure_scorer.train import compute_lexicon_scores, CHANNEL_LEXICONS
    lex = compute_lexicon_scores(event_text)
    event_lower = event_text.lower()

    signals = []
    for channel, keywords in CHANNEL_LEXICONS.items():
        matched = [kw for kw in keywords if kw in event_lower]
        if matched:
            ch_display = channel.replace("_", " ").title()
            signals.append((channel, matched, ch_display))

    if signals:
        for channel, matched, ch_display in signals:
            kw_str = ", ".join(f'"{m}"' for m in matched[:3])
            icon = "+" if channel in (ch1, ch2) else "-"
            st.markdown(f"- [{icon}] Detected {kw_str} -> **{ch_display}**")
    else:
        st.markdown("- No channel-specific keywords detected in event text")

    # Show what's missing
    st.caption(f"Mode: **{mode}** | Confidence is **{reliability}**"
               + (" because descriptive event text was provided." if reliability == "high"
                  else ". Provide more descriptive text to improve accuracy." if reliability == "low"
                  else "."))

    # ── Impact channels (top 2) ──────────────────────────────────────────

    st.markdown("#### Impact Channels")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(ch1.replace("_", " ").title(), f"{ranked[0][1]:.0%} probability",
                  delta="Primary channel")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch1, ""))

    with col2:
        st.metric(ch2.replace("_", " ").title(), f"{ranked[1][1]:.0%} probability",
                  delta="Secondary channel")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch2, ""))

    # ── System insight ───────────────────────────────────────────────────

    st.info(
        f"**System insight:** This prediction uses event text signals but does not include "
        f"company-specific exposure data (geographic revenue concentration, supplier network, "
        f"asset locations). Actual impact may differ significantly if {company_short} has "
        f"concentrated operations in the affected region."
    )

    # ── Diagnostics (collapsed) ──────────────────────────────────────────

    with st.expander("Full diagnostics"):
        st.markdown(f"**Event classification:** {evt['category'].replace('_', ' ').title()} ({evt['confidence']:.0%})")
        st.markdown(f"**Backend:** {clf._backend}")

        st.markdown("**All channel probabilities:**")
        st.dataframe(pd.DataFrame([
            {"Channel": ch.replace("_", " ").title(), "Probability": f"{p:.1%}"}
            for ch, p in ranked
        ]), use_container_width=True, hide_index=True)

        st.markdown("**Lexicon scores:**")
        lex_data = [{"Channel": ch.replace("_", " ").title(), "Score": f"{s:.2f}"}
                    for ch, s in sorted(lex.items(), key=lambda x: -x[1]) if s > 0]
        if lex_data:
            st.dataframe(pd.DataFrame(lex_data), use_container_width=True, hide_index=True)
        else:
            st.caption("No lexicon signals detected.")

    # ── Feedback ─────────────────────────────────────────────────────────

    st.divider()
    st.markdown("#### Was this correct?")

    col1, col2 = st.columns(2)
    useful = col1.radio("Useful?", ["--", "Yes", "No"], horizontal=True, key="useful")
    ch_correct = col2.radio("Main channel correct?", ["--", "Yes", "No"], horizontal=True, key="ch_correct")

    correct_channel = ""
    comment = ""
    if ch_correct == "No":
        correct_channel = st.selectbox("What should the primary channel be?",
            [""] + [ch.replace("_", " ").title() for ch in CHANNEL_DESCRIPTIONS.keys()], key="corr_ch")
        comment = st.text_input("Optional comment", key="comment")

    if st.button("Submit Feedback", key="submit_fb"):
        if useful != "--" or ch_correct != "--":
            save_feedback({
                "timestamp": datetime.now().isoformat(),
                "event_text": event_text[:500],
                "company": company_name, "ticker": ticker,
                "predicted_channel_1": ch1, "predicted_channel_2": ch2,
                "reliability": reliability,
                "useful": useful if useful != "--" else "",
                "channel_correct": ch_correct if ch_correct != "--" else "",
                "suggested_channel": correct_channel, "comment": comment,
            })
            st.success("Feedback saved. Thank you!")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
    1. **Classify** the event into 8 geopolitical categories
    2. **Score** company exposure using text signals + structured features
    3. **Estimate** financial impact range (low / base / high)
    4. **Explain** which mechanism keywords drove the prediction
    """)

    st.markdown("---")

    st.markdown("### Accuracy (blind eval)")
    st.markdown("""
    69 real-world event-company pairs:

    | | Top-2 |
    |---|:---:|
    | Without text | 46% |
    | **With text** | **62%** |
    | + secondary | **75%** |

    Direction accuracy: **90%**
    """)

    st.markdown("---")

    st.markdown("### Two reliability modes")
    st.markdown("""
    | Mode | When | Accuracy |
    |------|------|:---:|
    | **Text-rich** | Event description provided | ~75% |
    | **Text-poor** | Only category + sector | ~46% |

    Every prediction labels which mode it's using.
    """)

    st.markdown("---")
    st.caption("Built on WEF/IMD/BCG framework")
    st.caption("7.76M events | 602 labeled impacts | Open source")
    st.caption("[GitHub](https://github.com/HariDks/Geopolitics-ML)")
