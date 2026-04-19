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

# ── Session state init ───────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None
if "event_text" not in st.session_state:
    st.session_state.event_text = ""
if "company_name" not in st.session_state:
    st.session_state.company_name = "Apple (AAPL)"

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
    {"label": "Chip controls -> NVIDIA", "text": "US Bureau of Industry and Security restricted exports of advanced AI chips and semiconductor equipment to China", "company": "NVIDIA (NVDA)"},
    {"label": "Red Sea -> Costco", "text": "Houthi rebels fired anti-ship missiles at commercial vessels in the Red Sea forcing major shipping lines to reroute around the Cape of Good Hope", "company": "Costco (COST)"},
    {"label": "Ransomware -> Intel", "text": "DarkSide ransomware group encrypted Colonial Pipeline IT systems forcing shutdown of the largest US fuel pipeline for six days", "company": "Intel (INTC)"},
    {"label": "OPEC cut -> Boeing", "text": "OPEC announced a surprise production cut of 2 million barrels per day sending oil prices surging 8 percent", "company": "Boeing (BA)"},
    {"label": "EU DMA -> Apple", "text": "EU passed the Digital Markets Act requiring Big Tech platforms to allow sideloading third-party app stores and interoperability", "company": "Apple (AAPL)"},
    {"label": "China tariffs -> Walmart", "text": "US imposed 25 percent tariffs on all Chinese imports including electronics semiconductors and consumer goods", "company": "Walmart (WMT)"},
    {"label": "Xinjiang boycott -> Nike", "text": "Xinjiang forced labor allegations triggered Western brands boycott of Chinese cotton and consumer backlash in China", "company": "Nike (NKE)"},
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


def run_analysis(event_text, ticker, revenue):
    """Run the full pipeline and store results in session state."""
    clf, scorer, estimator = load_models()
    evt = clf.predict(event_text)
    exp = scorer.score(event_category=evt["category"], ticker=ticker,
                       mention_sentiment=-0.4, event_text=event_text)
    imp = estimator.estimate(event_category=evt["category"],
                              impact_channel=exp["channel_prediction"],
                              ticker=ticker, mention_sentiment=-0.4, revenue_usd=revenue)
    return {"evt": evt, "exp": exp, "imp": imp}


# ── Header ───────────────────────────────────────────────────────────────────

st.title("Geopolitical Impact Tester")
st.caption("Test how a geopolitical event affects a specific company's business.")

# ── Quick Scenarios ──────────────────────────────────────────────────────────

st.markdown("**Try an example:**")
cols = st.columns(4)
for i, scenario in enumerate(QUICK_SCENARIOS):
    if cols[i % 4].button(scenario["label"], use_container_width=True, key=f"scenario_{i}"):
        st.session_state.event_text = scenario["text"]
        st.session_state.company_name = scenario["company"]
        # Auto-run
        info = COMPANIES[scenario["company"]]
        st.session_state.results = run_analysis(scenario["text"], info["ticker"], info["revenue"])

st.divider()

# ── Input Panel ──────────────────────────────────────────────────────────────

col_text, col_company = st.columns([3, 2])

with col_text:
    event_text = st.text_area("Event description", value=st.session_state.event_text, height=80,
                               placeholder="Describe a geopolitical event in 1-3 sentences...",
                               key="event_input_main")

with col_company:
    company_keys = list(COMPANIES.keys())
    default_idx = company_keys.index(st.session_state.company_name) if st.session_state.company_name in company_keys else 0
    company_name = st.selectbox("Company", company_keys, index=default_idx, key="company_select")
    info = COMPANIES[company_name]

    if company_name == "Other (enter manually)":
        ticker = st.text_input("Ticker")
        revenue = st.number_input("Annual revenue (USD)", value=0, step=1_000_000_000, format="%d")
    else:
        ticker = info["ticker"]
        revenue = info["revenue"]
        st.caption(f"**{ticker}** | ${revenue/1e9:.0f}B revenue | {info['sector']}")

# ── Analyze button ───────────────────────────────────────────────────────────

if st.button("Analyze Impact", type="primary", disabled=not event_text, use_container_width=True):
    st.session_state.event_text = event_text
    st.session_state.company_name = company_name
    st.session_state.results = run_analysis(event_text, ticker, revenue)

# ── Results ──────────────────────────────────────────────────────────────────

if st.session_state.results:
    r = st.session_state.results
    evt, exp, imp = r["evt"], r["exp"], r["imp"]

    probs = exp["channel_probabilities"]
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    ch1, ch2 = ranked[0][0], ranked[1][0]
    reliability = exp.get("channel_reliability", "unknown")
    mode = exp.get("channel_mode", "unknown")

    mid = imp["impact_mid_pct"]
    if abs(mid) < 0.5: severity_label = "Minimal"
    elif mid < -3: severity_label = "Significant negative"
    elif mid < 0: severity_label = "Moderate negative"
    elif mid > 3: severity_label = "Significant positive"
    else: severity_label = "Moderate positive"

    company_short = st.session_state.company_name.split(" (")[0]
    ch1_short = ch1.replace("_", " ").title()
    ch1_desc = CHANNEL_DESCRIPTIONS.get(ch1, "")

    if reliability == "high": conf_badge = "High (text-rich)"
    elif reliability == "moderate": conf_badge = "Moderate"
    else: conf_badge = "Low (limited text signals)"

    st.divider()

    # ── Summary card ─────────────────────────────────────────────────────

    st.markdown(f"""
    ### What this means for {company_short}

    | | |
    |---|---|
    | **Likely impact** | {severity_label} ({imp['impact_low_pct']:+.1f}% to {imp['impact_high_pct']:+.1f}%) |
    | **Main driver** | {ch1_short} — {ch1_desc.lower()} |
    | **Dollar range** | {fmt_usd(imp.get('impact_low_usd', 0))} to {fmt_usd(imp.get('impact_high_usd', 0))} (on ${revenue/1e9:.0f}B revenue) |
    | **Confidence** | {conf_badge} |
    """)

    # ── Explanation ──────────────────────────────────────────────────────

    st.markdown("#### Why this prediction?")

    from models.exposure_scorer.train import compute_lexicon_scores, CHANNEL_LEXICONS
    lex = compute_lexicon_scores(st.session_state.event_text)
    event_lower = st.session_state.event_text.lower()

    signals = []
    for channel, keywords in CHANNEL_LEXICONS.items():
        matched = [kw for kw in keywords if kw in event_lower]
        if matched:
            ch_display = channel.replace("_", " ").title()
            is_predicted = channel in (ch1, ch2)
            signals.append((channel, matched, ch_display, is_predicted))

    if signals:
        for channel, matched, ch_display, is_predicted in signals:
            kw_str = ", ".join(f'"{m}"' for m in matched[:3])
            icon = "+" if is_predicted else " "
            st.markdown(f"- **[{icon}]** Detected {kw_str} -> {ch_display}")
    else:
        st.markdown("- No channel-specific keywords detected in event text")

    st.caption(f"Mode: **{mode}** | Confidence: **{reliability}**"
               + (" — descriptive event text provided." if reliability == "high"
                  else " — provide more descriptive text to improve accuracy." if reliability == "low"
                  else "."))

    # ── Channels ─────────────────────────────────────────────────────────

    st.markdown("#### Impact Channels")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(ch1.replace("_", " ").title(), f"{ranked[0][1]:.0%}", delta="Primary")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch1, ""))
    with col2:
        st.metric(ch2.replace("_", " ").title(), f"{ranked[1][1]:.0%}", delta="Secondary")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch2, ""))

    # ── System insight ───────────────────────────────────────────────────

    st.info(
        f"**Limitation:** This prediction does not include company-specific exposure data "
        f"(geographic revenue, supplier network, asset locations). "
        f"Actual impact may differ if {company_short} has concentrated operations in the affected region."
    )

    # ── Diagnostics ──────────────────────────────────────────────────────

    with st.expander("Full diagnostics"):
        st.markdown(f"**Event:** {evt['category'].replace('_', ' ').title()} ({evt['confidence']:.0%})")
        st.dataframe(pd.DataFrame([
            {"Channel": ch.replace("_", " ").title(), "Probability": f"{p:.1%}"}
            for ch, p in ranked
        ]), use_container_width=True, hide_index=True)

    # ── Feedback (persisted in session state) ────────────────────────────

    st.divider()
    st.markdown("#### Was this useful?")

    fb_col1, fb_col2 = st.columns(2)
    useful = fb_col1.radio("Helpful?", ["--", "Yes", "No"], horizontal=True, key="fb_useful")
    ch_correct = fb_col2.radio("Primary channel correct?", ["--", "Yes", "No"], horizontal=True, key="fb_ch")

    if ch_correct == "No":
        st.selectbox("What should the primary channel be?",
                     ["(select)"] + [ch.replace("_", " ").title() for ch in CHANNEL_DESCRIPTIONS.keys()],
                     key="fb_correction")
        st.text_input("Optional comment", key="fb_comment")

    if st.button("Submit Feedback"):
        if useful != "--" or ch_correct != "--":
            save_feedback({
                "timestamp": datetime.now().isoformat(),
                "event_text": st.session_state.event_text[:500],
                "company": st.session_state.company_name,
                "ticker": ticker,
                "predicted_channel_1": ch1,
                "predicted_channel_2": ch2,
                "reliability": reliability,
                "useful": useful if useful != "--" else "",
                "channel_correct": ch_correct if ch_correct != "--" else "",
                "suggested_channel": st.session_state.get("fb_correction", ""),
                "comment": st.session_state.get("fb_comment", ""),
            })
            st.success("Feedback saved!")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
    1. **Classify** the event (8 categories)
    2. **Score** exposure using text + structured signals
    3. **Estimate** financial impact (low / base / high)
    4. **Explain** which keywords drove the prediction
    """)
    st.markdown("---")
    st.markdown("### Accuracy")
    st.markdown("""
    69 blind eval pairs:

    | | Top-2 |
    |---|:---:|
    | Without text | 46% |
    | **With text** | **62%** |
    | + secondary | **75%** |

    Direction: **90%**
    """)
    st.markdown("---")
    st.caption("[GitHub](https://github.com/HariDks/Geopolitics-ML) | Built on WEF framework | 7.76M events")
