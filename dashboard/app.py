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
    """Save feedback to both CSV (legacy) and database (new)."""
    # Legacy CSV
    exists = FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not exists: w.writeheader()
        w.writerow(data)

    # Database correction (if prediction was logged)
    pred_id = data.get("prediction_id", "")
    if pred_id:
        try:
            from pipelines.prediction_logger import log_correction
            log_correction(
                prediction_id=pred_id,
                useful=data.get("useful", ""),
                correct_channel=data.get("suggested_channel", ""),
                notes=data.get("comment", ""),
            )
        except Exception:
            pass  # DB not available (Streamlit Cloud)


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


def run_analysis(event_text, ticker, revenue, company_name=""):
    clf, scorer, estimator = load_models()
    evt = clf.predict(event_text)
    exp = scorer.score(event_category=evt["category"], ticker=ticker, mention_sentiment=-0.4, event_text=event_text)
    imp = estimator.estimate(event_category=evt["category"], impact_channel=exp["channel_prediction"],
                              ticker=ticker, mention_sentiment=-0.4, revenue_usd=revenue)

    # Rule-based sign correction
    direction = detect_event_direction(event_text)
    imp = correct_impact_sign(imp, direction)

    results = {"evt": evt, "exp": exp, "imp": imp, "direction": direction}

    # Log prediction to database
    try:
        from pipelines.prediction_logger import log_prediction
        pred_id = log_prediction(
            input_text=event_text,
            input_ticker=ticker,
            input_company=company_name,
            input_revenue=revenue,
            results=results,
        )
        results["prediction_id"] = pred_id
    except Exception:
        results["prediction_id"] = ""

    return results


def display_results(results, event_text, company_name, revenue):
    """Render analysis results — structured for decision-making."""
    evt, exp, imp = results["evt"], results["exp"], results["imp"]

    probs = exp["channel_probabilities"]
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    ch1, ch2 = ranked[0][0], ranked[1][0]
    reliability = exp.get("channel_reliability", "unknown")
    mode = exp.get("channel_mode", "unknown")

    company_short = company_name.split(" (")[0]
    ch1_short = ch1.replace("_", " ").title()
    ch2_short = ch2.replace("_", " ").title()
    ch1_desc = CHANNEL_DESCRIPTIONS.get(ch1, "")

    if reliability == "high": conf_badge = "High"
    elif reliability == "moderate": conf_badge = "Moderate"
    else: conf_badge = "Low"

    mid = imp["impact_mid_pct"]
    pct_low = imp["impact_low_pct"]
    pct_high = imp["impact_high_pct"]
    usd_low = fmt_usd(imp.get("impact_low_usd", 0))
    usd_high = fmt_usd(imp.get("impact_high_usd", 0))

    if abs(mid) < 0.3: fin_label = "Limited"
    elif abs(mid) < 1: fin_label = "Low-to-moderate"
    elif abs(mid) < 3: fin_label = "Moderate"
    elif abs(mid) < 7: fin_label = "Significant"
    else: fin_label = "Severe"

    high_ops = {"cybersecurity_it", "logistics_operations", "workforce_talent"}
    med_ops = {"procurement_supply_chain", "capital_allocation_investment"}
    if ch1 in high_ops: ops_label = "High"
    elif ch1 in med_ops: ops_label = "Medium-to-high"
    else: ops_label = "Medium"

    direction = "negative" if mid < -0.1 else "positive" if mid > 0.1 else "neutral"
    dir_word = "mild negative" if abs(mid) < 1 else f"{fin_label.lower()} {'negative' if mid < 0 else 'positive'}"

    # ── SECTION 1: Decision takeaway ──
    st.markdown(f"### Assessment")
    takeaway = (
        f"This event is likely to have a **{dir_word}** financial impact on {company_short}, "
        f"driven primarily by **{ch1_short.lower()}** ({ch1_desc.lower()})."
    )
    if fin_label in ("Significant", "Severe") or ops_label == "High":
        st.error(f"{takeaway}\n\nConfidence: **{conf_badge}**")
    else:
        st.warning(f"{takeaway}\n\nConfidence: **{conf_badge}**")

    # ── SECTION 2: Core output ──
    st.markdown("#### Core output")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Key driver:** {ch1_short} ({ch1_desc.lower()})\n
        **Secondary driver:** {ch2_short}\n
        **Estimated impact:** {pct_low:+.1f}% to {pct_high:+.1f}% of revenue\n
        ({usd_low} to {usd_high} on ${revenue/1e9:.0f}B)
        """)
    with col2:
        st.markdown(f"""
        **Financial scale:** {fin_label}\n
        **Operational severity:** {ops_label}\n
        **Reliability:** {conf_badge}
        """)

    # ── SECTION 3: Event interpretation ──
    event_cat_display = evt["category"].replace("_", " ").title()
    st.caption(f"Event interpretation: {event_cat_display} ({evt['confidence']:.0%} confidence)")

    # ── SECTION 4: Expandable insight blocks ──
    with st.expander("What could change this"):
        st.markdown(f"""
        - If {company_short} has significant regional exposure to the affected area, losses could be larger
        - If the event escalates or extends, secondary channels may become primary
        - If {company_short} has mitigation strategies in place, actual impact could be lower
        - Market sentiment and investor reaction may amplify or dampen the financial effect
        """)

    with st.expander("Why this might be wrong"):
        st.markdown(f"""
        - Does not include {company_short}'s specific revenue by geography or supplier network
        - Does not fully capture commodity price dynamics or market reactions
        - Assumes historical patterns from similar events hold in this case
        - Channel prediction accuracy is ~62-75% on novel event-company pairs
        """)

    with st.expander("Historical context"):
        st.markdown(f"Similar events in the **{event_cat_display.lower()}** category have historically "
                    f"affected companies through {ch1_short.lower()} and {ch2_short.lower()} channels.")

    # ── SECTION 5: Channel breakdown ──
    st.markdown("#### Channel breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(ch1_short, f"{ranked[0][1]:.0%}", delta="Primary")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch1, ""))
    with c2:
        st.metric(ch2_short, f"{ranked[1][1]:.0%}", delta="Secondary")
        st.caption(CHANNEL_DESCRIPTIONS.get(ch2, ""))

    # ── SECTION 6: Why this prediction ──
    st.markdown("#### Why this prediction")
    from models.exposure_scorer.train import compute_lexicon_scores, CHANNEL_LEXICONS
    lex = compute_lexicon_scores(event_text)
    event_lower = event_text.lower()

    has_signals = False
    for channel, keywords in CHANNEL_LEXICONS.items():
        matched = [kw for kw in keywords if kw in event_lower]
        if matched:
            has_signals = True
            ch_display = channel.replace("_", " ").title()
            kw_str = ", ".join(f"\"{m}\"" for m in matched[:3])
            st.markdown(f"The model picks up signals like {kw_str}, which are typically associated with **{ch_display.lower()}** impacts.")
            break

    if not has_signals:
        st.markdown("No strong channel-specific signals detected in the event text. Prediction based on structured features.")

    st.caption(f"Mode: **{mode}** | Reliability: **{conf_badge}**")

    # ── SECTION 7: Full diagnostics ──
    with st.expander("Full diagnostics"):
        st.markdown(f"**Event classification:** {event_cat_display} ({evt['confidence']:.0%})")
        st.markdown(f"**Financial scale:** {fin_label} | **Operational severity:** {ops_label}")
        direction_det = results.get("direction", "mixed")
        if direction_det != "mixed":
            st.markdown(f"**Sign confidence:** Strengthened by {direction_det} event-language signals.")
        st.markdown("**All channel probabilities:**")
        st.dataframe(pd.DataFrame([
            {"Channel": ch.replace("_", " ").title(), "Probability": f"{p:.1%}"}
            for ch, p in ranked
        ]), use_container_width=True, hide_index=True)

    # ── SECTION 8: Feedback ──
    st.divider()
    st.markdown("#### Was this useful?")
    fc1, fc2 = st.columns(2)
    useful = fc1.radio("Helpful?", ["--", "Yes", "No"], horizontal=True, key=f"fb_u_{company_name}")
    ch_correct = fc2.radio("Main driver correct?", ["--", "Yes", "No"], horizontal=True, key=f"fb_c_{company_name}")

    if ch_correct == "No":
        st.selectbox("What should the primary driver be?",
                     ["(select)"] + [ch.replace("_", " ").title() for ch in CHANNEL_DESCRIPTIONS.keys()],
                     key=f"fb_corr_{company_name}")
        st.text_input("Optional comment", key=f"fb_comm_{company_name}")

    if st.button("Submit Feedback", key=f"fb_sub_{company_name}"):
        if useful != "--" or ch_correct != "--":
            save_feedback({
                "timestamp": datetime.now().isoformat(),
                "prediction_id": results.get("prediction_id", ""),
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
with st.sidebar.expander("Model performance"):
    st.markdown("""
    69 blind eval pairs:
    - Without text: 46% top-2
    - **With text: 62% top-2**
    - With secondary: **75%**
    - Direction: **90%**
    """)
st.sidebar.caption("[GitHub](https://github.com/HariDks/Geopolitics-ML)")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

if page == "Overview":

    st.markdown("""
    <style>
    .block-container { max-width: 900px; padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("<h1 style='text-align: center;'>Geopolitical Impact Tester</h1>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: center; font-size: 1.15em;'>
    Understand how geopolitical events impact real companies — in seconds.<br>
    Not just <em>whether</em> it hurts, but <em>how</em> it hurts:<br>
    through supply chains, revenue, assets, or reputation.
    </p>""", unsafe_allow_html=True)

    # CTAs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#f0f2f6; padding:20px; border-radius:10px; text-align:center; min-height:110px;">
        <h4>Explore Real Events</h4>
        <p style="font-size:0.9em;">Test the system on curated geopolitical scenarios</p>
        <p style="font-size:0.85em; color:#555;">Go to <b>Preloaded Examples</b> (sidebar)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#f0f2f6; padding:20px; border-radius:10px; text-align:center; min-height:110px;">
        <h4>Run Custom Analysis</h4>
        <p style="font-size:0.9em;">Describe any event and analyze its impact on a company</p>
        <p style="font-size:0.85em; color:#555;">Go to <b>Custom Analysis</b> (sidebar)</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Why this exists
    st.markdown("#### Why this exists")
    st.markdown("""
    Most companies don't systematically quantify geopolitical risk.
    They rely on judgment, intuition, and static frameworks.

    This project explores whether that process can be **partially automated**
    using machine learning and structured reasoning.

    The key insight: **understanding the *mechanism* of impact is easier
    than estimating its *magnitude*** — especially without company-specific exposure data.
    """)

    st.divider()

    # What you get
    st.markdown("#### What you get from each analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Affected area**")
        st.caption("Supply chain, revenue, assets, or reputation")
    with c2:
        st.markdown("**Financial impact**")
        st.caption("Range as % of revenue + dollar estimate")
    with c3:
        st.markdown("**Mechanism**")
        st.caption("Why and how the event affects operations")
    with c4:
        st.markdown("**Confidence**")
        st.caption("How reliable this prediction is")

    st.divider()

    # System insight
    st.markdown("""
    <div style="background:#1a3a5c; color:white; padding:22px; border-radius:10px; margin:10px 0;">
    <b>System Insight</b><br><br>
    <b>This system is better at explaining impact than estimating magnitude.</b><br><br>
    It is strongest at identifying <em>how</em> events affect companies
    when the mechanism is clearly described in text.<br><br>
    It is less reliable when company-specific exposure is unknown —
    for example, where revenue comes from or where operations are concentrated.
    In those cases, it provides a <b>generic estimate</b> based on similar events.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model performance
    st.markdown("#### Model performance")
    st.markdown("""
    - Directional accuracy: **~90%** (will this help or hurt?)
    - Mechanism accuracy: **~62-75%** (depends on text quality)
    """)
    st.markdown("**Limitations**")
    st.markdown("""
    - Less reliable when company exposure is unclear (e.g., concentrated geographic operations)
    - Does not fully capture second-order effects (e.g., commodity prices, market sentiment)
    """)

    st.divider()

    # What powers this (collapsed)
    with st.expander("What powers this"):
        st.markdown("""
        - **7.7M** geopolitical events across multiple global data sources
        - **600+** labeled company-event impact examples
        - A multi-stage system that:
          - classifies events
          - estimates exposure patterns
          - predicts financial impact
          - explains the mechanism
        """)

    # How to interpret (collapsed)
    with st.expander("How to interpret the output"):
        st.markdown("""
        A **financial impact estimate** based on patterns from similar historical events.
        A **primary mechanism** — the main way the business is affected.
        An **operational severity** rating — how disruptive the event is operationally.
        A **reliability level** — higher when event descriptions are detailed.

        Think of this as a **structured first-pass analysis** — not a final answer.
        """)

    st.divider()

    # Feedback
    st.markdown("#### Help improve this system")
    st.markdown("""
    **This system improves through real-world feedback.**

    Your corrections highlight where the model gets things wrong — and that's how it gets better.
    After running an analysis, you can mark whether the result was useful,
    correct the primary mechanism, or share what was missed.
    """)

    st.markdown("")
    st.caption("This tool is best used as a starting point for thinking through geopolitical risk — not a final answer. "
               "It is not financial advice.")

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
            results = run_analysis(event, info["ticker"], info["revenue"], company_name)
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
        results = run_analysis(event_text, ticker, revenue, company_name)
        st.divider()
        display_results(results, event_text, company_name, revenue)
