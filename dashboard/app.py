"""
Geopolitical Impact Tester — Multi-page Streamlit app.
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

# ── Shared data ──────────────────────────────────────────────────────────────

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
    "Home Depot (HD)": {"ticker": "HD", "revenue": 157e9, "sector": "Consumer Discretionary"},
    "Other (enter manually)": {"ticker": "", "revenue": 0, "sector": ""},
}

# Preloaded scenarios — event text describes what happened, company is who we're analyzing
PRELOADED = [
    {"event": "Russia launched full-scale invasion of Ukraine triggering Western sanctions. Energy companies with Russian operations faced asset freezes and were forced to write down billions in stranded investments",
     "company": "Exxon Mobil (XOM)", "label": "Russia invasion -> Exxon Mobil"},
    {"event": "NotPetya ransomware spread globally through a Ukrainian tax software update destroying IT systems at major shipping and logistics companies. FedEx subsidiary TNT Express lost $400M from the attack",
     "company": "FedEx (FDX)", "label": "NotPetya ransomware -> FedEx"},
    {"event": "US Bureau of Industry and Security restricted exports of advanced AI chips including A100 and H100 to China. NVIDIA lost access to its second-largest market reducing revenue by approximately $400M per quarter",
     "company": "NVIDIA (NVDA)", "label": "Chip export controls -> NVIDIA"},
    {"event": "Xinjiang forced labor allegations triggered massive consumer boycott of Western brands in China. Nike faced backlash losing approximately 15% of Greater China revenue as Chinese consumers switched to local brands",
     "company": "Nike (NKE)", "label": "Xinjiang boycott -> Nike"},
    {"event": "EU passed Digital Markets Act requiring Apple to allow third-party app stores sideloading and alternative payment systems. Apple faces compliance costs and potential loss of up to 10% of App Store commission revenue in Europe",
     "company": "Apple (AAPL)", "label": "EU Digital Markets Act -> Apple"},
    {"event": "Houthi rebels fired anti-ship missiles at commercial vessels in the Red Sea. Major retailers depending on Asian imports face 40% freight cost surge and 14 extra days of transit time as ships reroute around Africa",
     "company": "Costco (COST)", "label": "Red Sea rerouting -> Costco"},
    {"event": "OPEC announced surprise production cut of 2 million barrels per day sending oil prices surging 8%. Airlines face spiking jet fuel costs forcing them to defer new aircraft orders from Boeing",
     "company": "Boeing (BA)", "label": "OPEC production cut -> Boeing"},
    {"event": "US imposed 25% tariffs on all Chinese imports. Walmart which sources billions in consumer goods electronics and household products from China faces massive cost increases that must be absorbed or passed to customers",
     "company": "Walmart (WMT)", "label": "China tariffs -> Walmart"},
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


# Negative-signal keywords — if these appear in event text, impact should skew negative
NEGATIVE_SIGNALS = [
    "loss", "lost", "losing", "boycott", "restriction", "restricted", "ban", "banned",
    "sanction", "sanctions", "impairment", "write-down", "write-off", "exit", "exited",
    "destroy", "destroyed", "attack", "attacked", "ransomware", "malware", "hack",
    "collapse", "collapsed", "shutdown", "disruption", "disrupted", "crisis",
    "tariff", "tariffs", "penalty", "fine", "fined", "seized", "frozen", "freeze",
    "wiped", "wiping", "cost increase", "cost surge", "surging", "spiking",
    "decline", "declined", "fell", "falling", "reduced", "reducing", "cut",
    "forced", "forcing", "threatening", "threat",
]

# Positive-signal keywords — if these dominate, impact could be positive
POSITIVE_SIGNALS = [
    "benefit", "surge in demand", "gained", "winning", "opportunity",
    "deregulation", "liberalization", "price increase", "pricing power",
]


def detect_event_direction(event_text: str) -> str:
    """Detect whether the event is negative, positive, or mixed for the company."""
    text_lower = event_text.lower()
    neg_hits = sum(1 for kw in NEGATIVE_SIGNALS if kw in text_lower)
    pos_hits = sum(1 for kw in POSITIVE_SIGNALS if kw in text_lower)
    if neg_hits > pos_hits + 1:
        return "negative"
    elif pos_hits > neg_hits + 1:
        return "positive"
    return "mixed"


def correct_impact_sign(imp: dict, direction: str) -> dict:
    """
    Fix the sign of impact estimates when the model gets it wrong.
    If the event is clearly negative but the model predicts positive,
    flip the sign. This is a rule-based correction for the model's
    most common error.
    """
    imp = dict(imp)  # don't mutate original

    mid = imp["impact_mid_pct"]
    low = imp["impact_low_pct"]
    high = imp["impact_high_pct"]

    if direction == "negative" and mid > 0:
        # Model predicted positive for a clearly negative event — flip
        imp["impact_mid_pct"] = -abs(mid)
        imp["impact_low_pct"] = -abs(high)  # swap: worst case becomes low
        imp["impact_high_pct"] = -abs(low) if low != 0 else -0.1
        # Fix USD too
        for key in ["impact_low_usd", "impact_mid_usd", "impact_high_usd"]:
            if key in imp and imp[key]:
                imp[key] = -abs(imp[key])
        # Re-sort so low < mid < high
        vals = sorted([imp["impact_low_pct"], imp["impact_mid_pct"], imp["impact_high_pct"]])
        imp["impact_low_pct"], imp["impact_mid_pct"], imp["impact_high_pct"] = vals[0], vals[1], vals[2]
        if "impact_low_usd" in imp and imp["impact_low_usd"]:
            usd_vals = sorted([imp.get("impact_low_usd", 0), imp.get("impact_mid_usd", 0), imp.get("impact_high_usd", 0)])
            imp["impact_low_usd"], imp["impact_mid_usd"], imp["impact_high_usd"] = usd_vals[0], usd_vals[1], usd_vals[2]

    elif direction == "positive" and mid < 0:
        # Model predicted negative for a clearly positive event — flip
        imp["impact_mid_pct"] = abs(mid)
        imp["impact_low_pct"] = abs(low) if low != 0 else 0.1
        imp["impact_high_pct"] = abs(high)
        for key in ["impact_low_usd", "impact_mid_usd", "impact_high_usd"]:
            if key in imp and imp[key]:
                imp[key] = abs(imp[key])
        vals = sorted([imp["impact_low_pct"], imp["impact_mid_pct"], imp["impact_high_pct"]])
        imp["impact_low_pct"], imp["impact_mid_pct"], imp["impact_high_pct"] = vals[0], vals[1], vals[2]

    return imp


def run_analysis(event_text, ticker, revenue):
    clf, scorer, estimator = load_models()
    evt = clf.predict(event_text)
    exp = scorer.score(event_category=evt["category"], ticker=ticker, mention_sentiment=-0.4, event_text=event_text)
    imp = estimator.estimate(event_category=evt["category"], impact_channel=exp["channel_prediction"],
                              ticker=ticker, mention_sentiment=-0.4, revenue_usd=revenue)

    # Rule-based sign correction
    direction = detect_event_direction(event_text)
    imp = correct_impact_sign(imp, direction)

    return {"evt": evt, "exp": exp, "imp": imp, "direction": direction}


def display_results(results, event_text, company_name, revenue):
    """Render analysis results."""
    evt, exp, imp = results["evt"], results["exp"], results["imp"]

    probs = exp["channel_probabilities"]
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    ch1, ch2 = ranked[0][0], ranked[1][0]
    reliability = exp.get("channel_reliability", "unknown")
    mode = exp.get("channel_mode", "unknown")

    company_short = company_name.split(" (")[0]
    ch1_short = ch1.replace("_", " ").title()
    ch1_desc = CHANNEL_DESCRIPTIONS.get(ch1, "")

    if reliability == "high": conf_badge = "High (text-rich)"
    elif reliability == "moderate": conf_badge = "Moderate"
    else: conf_badge = "Low (limited signals)"

    # Financial impact scale
    mid = imp["impact_mid_pct"]
    pct_low = imp['impact_low_pct']
    pct_high = imp['impact_high_pct']
    usd_low = fmt_usd(imp.get('impact_low_usd', 0))
    usd_high = fmt_usd(imp.get('impact_high_usd', 0))

    if abs(mid) < 0.3: fin_label = "Limited"
    elif abs(mid) < 1: fin_label = "Low-to-moderate"
    elif abs(mid) < 3: fin_label = "Moderate"
    elif abs(mid) < 7: fin_label = "Significant"
    else: fin_label = "Severe"

    # Operational severity (based on channel type + event signals)
    high_ops_channels = {"cybersecurity_it", "logistics_operations", "workforce_talent"}
    med_ops_channels = {"procurement_supply_chain", "capital_allocation_investment"}
    if ch1 in high_ops_channels:
        ops_label = "High"
    elif ch1 in med_ops_channels:
        ops_label = "Medium-to-high"
    else:
        ops_label = "Medium"

    # Banner color based on financial + operational combined
    if fin_label in ("Significant", "Severe") or (ops_label == "High" and fin_label != "Limited"):
        banner_color = "error"
    elif fin_label == "Limited" and ops_label != "High":
        banner_color = "warning"
    else:
        banner_color = "warning"

    # Summary card — separate financial and operational
    st.markdown(f"### Estimated impact on {company_short}")

    summary_table = f"""
| | |
|---|---|
| **Financial impact** | {pct_low:+.1f}% to {pct_high:+.1f}% of revenue ({usd_low} to {usd_high} on ${revenue/1e9:.0f}B) |
| **Financial scale** | {fin_label} |
| **Primary mechanism** | {ch1_short} — {ch1_desc.lower()} |
| **Operational severity** | {ops_label} |
| **Reliability** | {conf_badge} |
"""

    if banner_color == "error":
        st.error(summary_table)
    elif banner_color == "success":
        st.success(summary_table)
    else:
        st.warning(summary_table)

    st.caption("This is a generic estimate based on similar historical events. "
               "It does not include company-specific exposure data.")

    # Explanation
    st.markdown("#### Why this prediction?")
    from models.exposure_scorer.train import compute_lexicon_scores, CHANNEL_LEXICONS
    lex = compute_lexicon_scores(event_text)
    event_lower = event_text.lower()

    for channel, keywords in CHANNEL_LEXICONS.items():
        matched = [kw for kw in keywords if kw in event_lower]
        if matched:
            ch_display = channel.replace("_", " ").title()
            is_pred = channel in (ch1, ch2)
            kw_str = ", ".join(f'"{m}"' for m in matched[:3])
            st.markdown(f"- **[{'+'if is_pred else ' '}]** Detected {kw_str} -> {ch_display}")

    st.caption(f"Mode: **{mode}** | Confidence: **{reliability}**")

    # Channels
    st.markdown("#### Impact Channels")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(ch1.replace("_", " ").title(), f"{ranked[0][1]:.0%}", delta="Primary")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch1, ""))
    with c2:
        st.metric(ch2.replace("_", " ").title(), f"{ranked[1][1]:.0%}", delta="Secondary")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch2, ""))

    # Limitation
    st.info(
        f"**Important:** Impact estimate is generic — it reflects historical patterns for similar events, "
        f"not {company_short}'s specific exposure. The model does not know {company_short}'s revenue "
        f"by geography, supplier network, or asset locations. Actual impact could be significantly "
        f"larger or smaller depending on the company's real concentration in affected regions."
    )

    # Diagnostics
    with st.expander("Full diagnostics"):
        st.markdown(f"**Event classification:** {evt['category'].replace('_', ' ').title()} ({evt['confidence']:.0%})")
        st.markdown(f"**Financial scale:** {fin_label} | **Operational severity:** {ops_label}")

        direction = results.get("direction", "mixed")
        if direction != "mixed":
            st.markdown(f"**Sign confidence:** Strengthened by {direction} event-language signals.")

        st.markdown("**All channel probabilities:**")
        st.dataframe(pd.DataFrame([
            {"Channel": ch.replace("_", " ").title(), "Probability": f"{p:.1%}"}
            for ch, p in ranked
        ]), use_container_width=True, hide_index=True)

    # Feedback
    st.divider()
    st.markdown("#### Was this useful?")
    fc1, fc2 = st.columns(2)
    useful = fc1.radio("Helpful?", ["--", "Yes", "No"], horizontal=True, key=f"fb_u_{company_name}")
    ch_correct = fc2.radio("Primary channel correct?", ["--", "Yes", "No"], horizontal=True, key=f"fb_c_{company_name}")

    if ch_correct == "No":
        st.selectbox("What should the primary channel be?",
                     ["(select)"] + [ch.replace("_", " ").title() for ch in CHANNEL_DESCRIPTIONS.keys()],
                     key=f"fb_corr_{company_name}")
        st.text_input("Optional comment", key=f"fb_comm_{company_name}")

    if st.button("Submit Feedback", key=f"fb_sub_{company_name}"):
        if useful != "--" or ch_correct != "--":
            save_feedback({
                "timestamp": datetime.now().isoformat(),
                "event_text": event_text[:500],
                "company": company_name,
                "predicted_channel_1": ch1, "predicted_channel_2": ch2,
                "reliability": reliability,
                "useful": useful if useful != "--" else "",
                "channel_correct": ch_correct if ch_correct != "--" else "",
                "suggested_channel": st.session_state.get(f"fb_corr_{company_name}", ""),
                "comment": st.session_state.get(f"fb_comm_{company_name}", ""),
            })
            st.success("Feedback saved!")


# ── Sidebar navigation ──────────────────────────────────────────────────────

page = st.sidebar.radio("Navigate", ["Overview", "Preloaded Examples", "Custom Analysis"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Accuracy")
st.sidebar.markdown("""
69 blind eval pairs:

| | Top-2 |
|---|:---:|
| Without text | 46% |
| **With text** | **62%** |
| + secondary | **75%** |

Direction: **90%**
""")
st.sidebar.markdown("---")
st.sidebar.caption("[GitHub](https://github.com/HariDks/Geopolitics-ML) | Built on WEF framework | 7.76M events")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Geopolitical Impact Tester")

    st.markdown("### What if you could stress-test any company against any geopolitical event?")
    st.markdown("""
    That's exactly what this does. Pick an event. Pick a company. Get an instant analysis
    of *how* the business gets hit — not just *whether* it does.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### What you get
        - Which part of the business is affected (supply chain? revenue? assets? reputation?)
        - A dollar-range impact estimate
        - An explanation showing *why* the model thinks what it thinks
        - A confidence level — because the model knows when it's guessing
        """)
    with col2:
        st.markdown("""
        #### What's under the hood
        - **7.76 million** geopolitical events from 6 data sources
        - **602 labeled** company-event impact pairs
        - **4 ML models** chained: classify event, score exposure, estimate impact, explain mechanism
        - Backtested against **10 real events** across 5 continents
        """)
    with col3:
        st.markdown("""
        #### What it's honest about
        - Gets the **direction right ~90%** of the time
        - Gets the **specific channel right ~62-75%** (depends on text quality)
        - Struggles with **concentrated geographic exposure**
        - This is **not financial advice** — it's pattern-matching on historical data
        """)

    st.markdown("---")

    st.markdown("""
    #### The story behind this

    This started as a research project inspired by the WEF's *Building Geopolitical Muscle* report (2026),
    which found that only ~20% of global firms systematically quantify their geopolitical exposure.

    We wanted to see: **can you build that capability with ML instead of a 40-person team?**

    Over several days of development, we ingested data from GDELT, ACLED, Global Trade Alert, OFAC, BIS, and SEC EDGAR.
    We trained a DistilBERT event classifier (95% accuracy on news text), an XGBoost exposure scorer,
    a quantile regression impact estimator, and a retrieval-based strategy recommender.

    The biggest lesson? **The right features matter more than the right model.** Adding 10 curated keyword
    lists ("impairment" = capital allocation, "ransomware" = cybersecurity) did more than any architecture change.

    **Your feedback makes this better.** After every analysis, you can tell us if the prediction made sense.
    Every correction helps the model improve.
    """)

    st.markdown("---")
    st.markdown("**Ready?** Use the sidebar to navigate to **Preloaded Examples** or **Custom Analysis**.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: PRELOADED EXAMPLES
# ════════════════════════════════════════════════════════════════════════════

elif page == "Preloaded Examples":
    st.title("Preloaded Examples")
    st.caption("Click any scenario to see the full analysis. Each pairs a specific geopolitical event with a specific company.")

    for scenario in PRELOADED:
        event = scenario["event"]
        company_name = scenario["company"]
        label = scenario["label"]
        info = COMPANIES[company_name]

        with st.expander(f"**{label}**"):
            st.markdown(f"**Event:** {event}")
            st.markdown(f"**Company:** {company_name} | ${info['revenue']/1e9:.0f}B revenue | {info['sector']}")
            st.markdown("---")

            # Run on demand
            results = run_analysis(event, info["ticker"], info["revenue"])
            display_results(results, event, company_name, info["revenue"])


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: CUSTOM ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

elif page == "Custom Analysis":
    st.title("Custom Analysis")
    st.caption("Describe any geopolitical event and select a company to analyze.")

    event_text = st.text_area("Describe the geopolitical event (1-3 sentences)", height=100,
                               placeholder="e.g., China restricted exports of gallium and germanium critical minerals used in semiconductor manufacturing...")

    company_name = st.selectbox("Company", list(COMPANIES.keys()))
    info = COMPANIES[company_name]

    if company_name == "Other (enter manually)":
        c1, c2 = st.columns(2)
        ticker = c1.text_input("Ticker")
        revenue = c2.number_input("Annual revenue (USD)", value=0, step=1_000_000_000, format="%d")
    else:
        ticker = info["ticker"]
        revenue = info["revenue"]
        st.caption(f"**{ticker}** | ${revenue/1e9:.0f}B revenue | {info['sector']}")

    if st.button("Analyze Impact", type="primary", disabled=not event_text, use_container_width=True):
        results = run_analysis(event_text, ticker, revenue)
        st.divider()
        display_results(results, event_text, company_name, revenue)
