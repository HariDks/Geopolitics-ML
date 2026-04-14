"""
Streamlit dashboard for geopolitical risk analysis.

Pages:
1. Analyze Event — event + company → full risk assessment with tailored strategies
2. Portfolio Scanner — scan multiple holdings against an event
3. Company Deep Dive — one company across all major event scenarios
4. Scenario Comparison — one company, multiple events side by side

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

COMPANIES = {
    "Apple (AAPL)": {"ticker": "AAPL", "revenue": 383_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "Heavy China manufacturing (90%+ iPhones assembled there), global sales in 40+ countries"},
    "Microsoft (MSFT)": {"ticker": "MSFT", "revenue": 245_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "Cloud infrastructure worldwide, enterprise contracts with governments, Azure in 60+ regions"},
    "Amazon (AMZN)": {"ticker": "AMZN", "revenue": 620_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "Global logistics network, AWS in 30+ regions, China marketplace presence, last-mile delivery in 20+ countries"},
    "NVIDIA (NVDA)": {"ticker": "NVDA", "revenue": 130_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "China was 25% of revenue before export controls, TSMC dependency for manufacturing, data center buildout globally"},
    "Alphabet/Google (GOOGL)": {"ticker": "GOOGL", "revenue": 350_000_000_000, "size": "large", "sector": "Communication Services", "geo_exposure": "Ad revenue from 200+ countries, data centers globally, regulatory exposure in EU/India/Australia"},
    "Meta/Facebook (META)": {"ticker": "META", "revenue": 165_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "3B users globally, banned in China/Russia, GDPR compliance costs, content moderation in conflict zones"},
    "Tesla (TSLA)": {"ticker": "TSLA", "revenue": 97_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "Shanghai Gigafactory (50%+ of production), Berlin factory, China sales ~25% of revenue, battery supply chain in DRC/Chile/Australia"},
    "Berkshire Hathaway (BRK-B)": {"ticker": "BRK-B", "revenue": 364_000_000_000, "size": "large", "sector": "Financials", "geo_exposure": "Diversified across insurance, rail, energy, manufacturing; limited direct international operations"},
    "JPMorgan Chase (JPM)": {"ticker": "JPM", "revenue": 177_000_000_000, "size": "large", "sector": "Financials", "geo_exposure": "Global investment bank, correspondent banking in 100+ countries, sanctions compliance, emerging market lending"},
    "Visa (V)": {"ticker": "V", "revenue": 36_000_000_000, "size": "large", "sector": "Financials", "geo_exposure": "Payment network in 200+ countries, exited Russia 2022, currency conversion exposure, financial sanctions compliance"},
    "UnitedHealth (UNH)": {"ticker": "UNH", "revenue": 400_000_000_000, "size": "large", "sector": "Health Care", "geo_exposure": "Primarily US operations, Optum global services, pharmaceutical supply chain from India/China"},
    "Johnson & Johnson (JNJ)": {"ticker": "JNJ", "revenue": 85_000_000_000, "size": "large", "sector": "Health Care", "geo_exposure": "Global pharmaceutical and medical device sales, API sourcing from India/China, manufacturing in 60+ countries"},
    "Exxon Mobil (XOM)": {"ticker": "XOM", "revenue": 344_000_000_000, "size": "large", "sector": "Energy", "geo_exposure": "Oil production in 40+ countries, Guyana deepwater, Middle East JVs, LNG in Qatar/Australia, refining globally"},
    "Chevron (CVX)": {"ticker": "CVX", "revenue": 196_000_000_000, "size": "large", "sector": "Energy", "geo_exposure": "Major operations in Kazakhstan, Australia LNG, Gulf of Mexico, Venezuela (sanctions exemption), shipping through Hormuz/Red Sea"},
    "Procter & Gamble (PG)": {"ticker": "PG", "revenue": 84_000_000_000, "size": "large", "sector": "Consumer Staples", "geo_exposure": "Sells in 180+ countries, significant Russia/China/India revenue, consumer brand reputation sensitive to boycotts"},
    "Costco (COST)": {"ticker": "COST", "revenue": 242_000_000_000, "size": "large", "sector": "Consumer Staples", "geo_exposure": "Global sourcing, ~15% international revenue, shipping/import dependent, China sourcing for consumer goods"},
    "McDonald's (MCD)": {"ticker": "MCD", "revenue": 26_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "40,000 restaurants in 100+ countries, exited Russia 2022, MENA boycott exposure, franchise model limits direct risk"},
    "Boeing (BA)": {"ticker": "BA", "revenue": 78_000_000_000, "size": "large", "sector": "Industrials", "geo_exposure": "40% of deliveries to non-US airlines, China market uncertainty, defense contracts with NATO allies, titanium from Russia (pre-2022)"},
    "Caterpillar (CAT)": {"ticker": "CAT", "revenue": 67_000_000_000, "size": "large", "sector": "Industrials", "geo_exposure": "Infrastructure equipment sold globally, mining exposure in Latin America/Africa/Australia, Middle East construction"},
    "Lockheed Martin (LMT)": {"ticker": "LMT", "revenue": 68_000_000_000, "size": "large", "sector": "Industrials", "geo_exposure": "US defense prime, F-35 sold to NATO allies, benefits from conflict escalation, export controls on advanced weapons"},
    "Raytheon/RTX (RTX)": {"ticker": "RTX", "revenue": 69_000_000_000, "size": "large", "sector": "Industrials", "geo_exposure": "Missiles and defense systems, Patriot batteries deployed in Middle East, benefits from Red Sea/Iran tensions"},
    "Broadcom (AVGO)": {"ticker": "AVGO", "revenue": 51_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "Semiconductor supply chain, China revenue exposure, VMware enterprise software globally"},
    "Cisco (CSCO)": {"ticker": "CSCO", "revenue": 54_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "Network equipment sold globally, Huawei rip-and-replace beneficiary, enterprise IT in regulated industries"},
    "Intel (INTC)": {"ticker": "INTC", "revenue": 54_000_000_000, "size": "large", "sector": "Information Technology", "geo_exposure": "CHIPS Act beneficiary, Israel fabrication facility, China revenue under export controls, competing with TSMC"},
    "Nike (NKE)": {"ticker": "NKE", "revenue": 51_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "Greater China ~15% revenue, Vietnam/Indonesia manufacturing, Xinjiang cotton controversy, brand-sensitive to boycotts"},
    "Goldman Sachs (GS)": {"ticker": "GS", "revenue": 51_000_000_000, "size": "large", "sector": "Financials", "geo_exposure": "Global investment banking, commodity trading, emerging market exposure, sanctions compliance for trading desks"},
    "Coca-Cola (KO)": {"ticker": "KO", "revenue": 46_000_000_000, "size": "large", "sector": "Consumer Staples", "geo_exposure": "Sells in 200+ countries, bottling partners worldwide, brand boycott risk, FX exposure to 80+ currencies"},
    "PepsiCo (PEP)": {"ticker": "PEP", "revenue": 92_000_000_000, "size": "large", "sector": "Consumer Staples", "geo_exposure": "Frito-Lay global, maintained Russia operations longer than peers, India/Middle East snack business, agricultural sourcing"},
    "Walmart (WMT)": {"ticker": "WMT", "revenue": 648_000_000_000, "size": "large", "sector": "Consumer Staples", "geo_exposure": "Massive China sourcing, international stores in 19 countries, tariff pass-through to consumers, India (Flipkart)"},
    "Home Depot (HD)": {"ticker": "HD", "revenue": 157_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "Primarily US/Canada/Mexico, heavy import dependency for building materials, tariff exposure on Chinese goods"},
    "Pfizer (PFE)": {"ticker": "PFE", "revenue": 58_000_000_000, "size": "large", "sector": "Health Care", "geo_exposure": "Global pharma distribution, API manufacturing in Ireland/Puerto Rico, vaccine geopolitics, IP licensing globally"},
    "Eli Lilly (LLY)": {"ticker": "LLY", "revenue": 41_000_000_000, "size": "large", "sector": "Health Care", "geo_exposure": "US-centric revenue but global manufacturing, insulin pricing politics, China pharmaceutical market entry"},
    "Mastercard (MA)": {"ticker": "MA", "revenue": 28_000_000_000, "size": "large", "sector": "Financials", "geo_exposure": "Similar to Visa — global payment network, Russia exit, India data localization mandate, cross-border transaction volume"},
    "Schlumberger (SLB)": {"ticker": "SLB", "revenue": 36_000_000_000, "size": "large", "sector": "Energy", "geo_exposure": "Oilfield services in 120+ countries, Middle East operations critical, benefits from energy price spikes"},
    "FedEx (FDX)": {"ticker": "FDX", "revenue": 88_000_000_000, "size": "large", "sector": "Industrials", "geo_exposure": "Global logistics, air freight routes over conflict zones, Red Sea rerouting costs, China e-commerce volume"},
    "General Motors (GM)": {"ticker": "GM", "revenue": 172_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "China JV (Buick/Wuling), Mexico manufacturing (USMCA), EV battery minerals from DRC/Chile, tariff exposure"},
    "Ford (F)": {"ticker": "F", "revenue": 176_000_000_000, "size": "large", "sector": "Consumer Discretionary", "geo_exposure": "Mexico/Canada manufacturing, European operations, China JV, EV battery supply chain, tariff exposure on imports"},
    "Other (enter manually)": {"ticker": "", "revenue": 0, "size": "large", "sector": "", "geo_exposure": ""},
}

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

# Sector-specific strategy context for generating tailored recommendations
SECTOR_STRATEGY_CONTEXT = {
    "Energy": "Your company operates oil/gas infrastructure globally. Consider hedging commodity exposure, reviewing maritime insurance for tanker routes, and assessing production assets in conflict-adjacent regions.",
    "Information Technology": "Your company depends on semiconductor supply chains and may face export controls. Consider qualifying alternative chip foundries, diversifying manufacturing geography, and reviewing data localization compliance.",
    "Financials": "Your company faces sanctions compliance costs, FX volatility, and potential asset freezes. Consider stress-testing correspondent banking relationships, reviewing trapped capital exposure, and updating sanctions screening.",
    "Consumer Discretionary": "Your company's brand is exposed to consumer boycotts and market access restrictions. Consider scenario planning for market exits, local brand strategy for sensitive markets, and supply chain diversification.",
    "Consumer Staples": "Your company sources and sells globally with thin margins. Consider pre-positioning inventory for tariff-exposed goods, hedging agricultural commodity inputs, and reviewing distributor relationships in affected regions.",
    "Health Care": "Your company faces regulatory divergence and API supply chain risks. Consider dual-sourcing active pharmaceutical ingredients, reviewing clinical trial locations in affected countries, and assessing data privacy compliance.",
    "Industrials": "Your company's manufacturing and logistics are directly exposed. Consider alternative shipping routes, reviewing supply contracts for force majeure terms, and assessing worker safety in conflict-adjacent facilities.",
    "Communication Services": "Your company faces content moderation requirements in conflict zones and data sovereignty mandates. Consider reviewing government access requests, data localization costs, and platform liability in sanctioned jurisdictions.",
    "Utilities": "Your company's infrastructure assets are long-lived and location-bound. Consider political risk insurance for international assets, reviewing power purchase agreements for force majeure, and assessing fuel source diversification.",
    "Real Estate": "Your company holds location-specific assets sensitive to political stability. Consider reviewing tenant concentration in affected regions, political risk insurance, and asset valuation scenarios under prolonged instability.",
}


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


def get_db():
    from pipelines.utils import get_db_connection
    return get_db_connection()


def fmt_usd(val):
    if val is None or val == 0:
        return "N/A"
    a = abs(val)
    s = "-" if val < 0 else "+"
    if a >= 1e9:
        return f"{s}${a/1e9:.1f}B"
    elif a >= 1e6:
        return f"{s}${a/1e6:.0f}M"
    return f"{s}${a:,.0f}"


def run_analysis(models, text, ticker, revenue, sector=""):
    """Run the full pipeline and return structured results."""
    evt = models["classifier"].predict(text)
    exp = models["scorer"].score(
        event_category=evt["category"], ticker=ticker, mention_sentiment=-0.4,
    )
    imp = models["estimator"].estimate(
        event_category=evt["category"], impact_channel=exp["channel_prediction"],
        ticker=ticker, mention_sentiment=-0.4, revenue_usd=revenue,
    )
    strats = models["recommender"].recommend_full(
        event_category=evt["category"], top_channels=exp["top_3_channels"],
        severity=imp["impact_mid_pct"] / 100, company_size="large",
    )
    return evt, exp, imp, strats


def display_strategies(strats, sector, company_name, event_desc):
    """Display strategies with company-specific context."""
    # Sector-specific context
    sector_context = SECTOR_STRATEGY_CONTEXT.get(sector, "")
    if sector_context:
        st.info(f"**{sector} sector context:** {sector_context}")

    for channel, channel_strats in strats.items():
        channel_display = channel.replace("_", " ").title()
        with st.expander(f"{channel_display} ({len(channel_strats)} actions)", expanded=True):
            for s in channel_strats[:3]:
                col_action, col_detail = st.columns([2, 3])
                with col_action:
                    cat = s["strategy_category"].upper()
                    if cat == "MITIGATE":
                        st.markdown(f"**{s['strategy_name']}**")
                    elif cat == "CAPTURE":
                        st.markdown(f"**{s['strategy_name']}**")
                    elif cat == "EXIT":
                        st.markdown(f"**{s['strategy_name']}**")
                    else:
                        st.markdown(f"**{s['strategy_name']}**")
                with col_detail:
                    st.caption(f"Type: {cat} | Cost: {s['typical_cost']} | Timeline: {s['implementation_time']}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Geopolitical Risk ML")
page = st.sidebar.radio("Navigate", [
    "Analyze Event",
    "Portfolio Scanner",
    "Company Deep Dive",
    "Scenario Comparison",
])

# ── Page 1: Analyze Event ────────────────────────────────────────────────────

if page == "Analyze Event":
    st.title("Geopolitical Risk Analysis")

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
        else:
            ticker = company_info["ticker"]
            revenue = company_info["revenue"]
            st.caption(f"**{ticker}** | ${revenue/1e9:.0f}B revenue | {company_info['sector']}")
            if company_info.get("geo_exposure"):
                st.caption(f"*{company_info['geo_exposure']}*")

    if st.button("Analyze", type="primary") and text:
        models = load_models()
        sector = company_info.get("sector", "")

        with st.spinner("Running analysis..."):
            evt, exp, imp, strats = run_analysis(models, text, ticker, revenue, sector)

        # Results in clean layout
        st.divider()

        # Event + Exposure in one row
        col1, col2, col3 = st.columns(3)
        col1.metric("Event Type", evt["category"].replace("_", " ").title(),
                    delta=f"{evt['confidence']:.0%} confidence")
        col2.metric("Primary Exposure", exp["channel_prediction"].replace("_", " ").title(),
                    delta=f"{exp['channel_confidence']:.0%} confidence")
        col3.metric("Severity", f"{exp['severity_score']:+.2f}",
                    delta="damaging" if exp['severity_score'] < -0.3 else "moderate" if exp['severity_score'] < 0 else "opportunity",
                    delta_color="inverse" if exp['severity_score'] < -0.3 else "normal")

        # Impact
        st.subheader("Financial Impact Estimate")
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimistic", f"{imp['impact_low_pct']:+.1f}%",
                    delta=fmt_usd(imp.get("impact_low_usd")))
        col2.metric("Expected", f"{imp['impact_mid_pct']:+.1f}%",
                    delta=fmt_usd(imp.get("impact_mid_usd")))
        col3.metric("Pessimistic", f"{imp['impact_high_pct']:+.1f}%",
                    delta=fmt_usd(imp.get("impact_high_usd")))

        # Other exposed channels
        st.subheader("Exposure Across Channels")
        ch_df = pd.DataFrame(exp["top_3_channels"])
        ch_df["channel"] = ch_df["channel"].str.replace("_", " ").str.title()
        ch_df["probability"] = (ch_df["probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(ch_df.rename(columns={"channel": "Channel", "probability": "Exposure Probability"}),
                    use_container_width=True, hide_index=True)

        # Strategies with sector context
        st.subheader("Recommended Actions")
        display_strategies(strats, sector, company_name, text)

# ── Page 2: Portfolio Scanner ────────────────────────────────────────────────

elif page == "Portfolio Scanner":
    st.title("Portfolio Risk Scanner")
    st.markdown("How exposed is your portfolio to a specific geopolitical event?")

    scenario = st.selectbox("Event scenario", [k for k in EVENT_SCENARIOS.keys() if k != "Custom (enter your own)"])
    event_text = EVENT_SCENARIOS[scenario]
    st.info(f'"{event_text}"')

    all_companies = [k for k in COMPANIES.keys() if k != "Other (enter manually)"]
    default_portfolio = ["Apple (AAPL)", "Exxon Mobil (XOM)", "Boeing (BA)", "JPMorgan Chase (JPM)",
                        "NVIDIA (NVDA)", "McDonald's (MCD)", "Costco (COST)", "Lockheed Martin (LMT)",
                        "Goldman Sachs (GS)", "Pfizer (PFE)"]
    selected = st.multiselect("Select portfolio companies", all_companies, default=default_portfolio)

    if st.button("Scan Portfolio", type="primary") and selected:
        models = load_models()

        with st.spinner("Classifying event..."):
            evt = models["classifier"].predict(event_text)

        st.divider()
        st.subheader(f"Event: {evt['category'].replace('_', ' ').title()} ({evt['confidence']:.0%})")

        results = []
        progress = st.progress(0)
        for i, cn in enumerate(selected):
            info = COMPANIES[cn]
            exp = models["scorer"].score(event_category=evt["category"], ticker=info["ticker"], mention_sentiment=-0.4)
            imp = models["estimator"].estimate(
                event_category=evt["category"], impact_channel=exp["channel_prediction"],
                ticker=info["ticker"], mention_sentiment=-0.4, revenue_usd=info["revenue"],
            )
            results.append({
                "Company": cn.split(" (")[0],
                "Ticker": info["ticker"],
                "Sector": info["sector"],
                "How Exposed": exp["channel_prediction"].replace("_", " ").title(),
                "Severity": exp["severity_score"],
                "Impact": f"{imp['impact_mid_pct']:+.1f}%",
                "Impact ($)": fmt_usd(imp.get("impact_mid_usd", 0)),
                "Why": info.get("geo_exposure", "")[:80],
            })
            progress.progress((i + 1) / len(selected))

        progress.empty()
        results_df = pd.DataFrame(results).sort_values("Severity")

        st.dataframe(
            results_df.style.background_gradient(subset=["Severity"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

        col1, col2 = st.columns(2)
        most = results_df.iloc[0]
        least = results_df.iloc[-1]
        col1.metric("Most Exposed", f"{most['Company']} ({most['Ticker']})",
                    delta=f"Severity {most['Severity']:+.2f} | {most['Impact']}", delta_color="inverse")
        col2.metric("Least Exposed / Beneficiary", f"{least['Company']} ({least['Ticker']})",
                    delta=f"Severity {least['Severity']:+.2f} | {least['Impact']}")

# ── Page 3: Company Deep Dive ────────────────────────────────────────────────

elif page == "Company Deep Dive":
    st.title("Company Deep Dive")
    st.markdown("How does one company fare across all major geopolitical scenarios?")

    company_name = st.selectbox("Company", [k for k in COMPANIES.keys() if k != "Other (enter manually)"])
    company_info = COMPANIES[company_name]

    st.markdown(f"**{company_info['ticker']}** | ${company_info['revenue']/1e9:.0f}B revenue | {company_info['sector']}")
    if company_info.get("geo_exposure"):
        st.info(f"**Geopolitical profile:** {company_info['geo_exposure']}")

    if st.button("Run Deep Dive", type="primary"):
        models = load_models()
        scenarios = {k: v for k, v in EVENT_SCENARIOS.items() if k != "Custom (enter your own)" and v}

        results = []
        progress = st.progress(0)
        for i, (name, text) in enumerate(scenarios.items()):
            evt, exp, imp, _ = run_analysis(
                models, text, company_info["ticker"], company_info["revenue"], company_info["sector"]
            )
            results.append({
                "Scenario": name,
                "Event Type": evt["category"].replace("_", " ").title(),
                "Channel": exp["channel_prediction"].replace("_", " ").title(),
                "Severity": exp["severity_score"],
                "Impact (Mid)": f"{imp['impact_mid_pct']:+.1f}%",
                "Impact ($)": fmt_usd(imp.get("impact_mid_usd", 0)),
                "_severity": exp["severity_score"],
            })
            progress.progress((i + 1) / len(scenarios))

        progress.empty()
        results_df = pd.DataFrame(results).sort_values("_severity")

        st.subheader("Risk Profile Across Scenarios")
        display_df = results_df.drop(columns=["_severity"])
        st.dataframe(
            display_df.style.background_gradient(subset=["Severity"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

        # Highlight biggest risk and opportunity
        worst = results_df.iloc[0]
        best = results_df.iloc[-1]

        col1, col2 = st.columns(2)
        col1.error(f"**Biggest risk:** {worst['Scenario']}\n\n"
                   f"Severity {worst['Severity']:+.2f} | {worst['Impact (Mid)']} | via {worst['Channel']}")
        col2.success(f"**Least exposed / opportunity:** {best['Scenario']}\n\n"
                     f"Severity {best['Severity']:+.2f} | {best['Impact (Mid)']} | via {best['Channel']}")

# ── Page 4: Scenario Comparison ──────────────────────────────────────────────

elif page == "Scenario Comparison":
    st.title("Scenario Comparison")
    st.markdown("Compare how the same company is affected by different events, or how different companies react to the same event.")

    mode = st.radio("Compare", ["Multiple events, one company", "Multiple companies, one event"], horizontal=True)

    if mode == "Multiple events, one company":
        company_name = st.selectbox("Company", [k for k in COMPANIES.keys() if k != "Other (enter manually)"])
        company_info = COMPANIES[company_name]
        all_scenarios = [k for k in EVENT_SCENARIOS.keys() if k != "Custom (enter your own)"]
        selected_scenarios = st.multiselect("Select scenarios to compare", all_scenarios,
                                            default=all_scenarios[:4])

        if st.button("Compare", type="primary") and selected_scenarios:
            models = load_models()
            results = []
            for name in selected_scenarios:
                text = EVENT_SCENARIOS[name]
                evt, exp, imp, _ = run_analysis(
                    models, text, company_info["ticker"], company_info["revenue"]
                )
                results.append({
                    "Scenario": name,
                    "Severity": exp["severity_score"],
                    "Impact %": imp["impact_mid_pct"],
                    "Channel": exp["channel_prediction"].replace("_", " ").title(),
                })

            df = pd.DataFrame(results)
            st.subheader(f"{company_name.split(' (')[0]} — Scenario Comparison")

            chart_df = df.set_index("Scenario")[["Severity", "Impact %"]]
            st.bar_chart(chart_df)
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        scenario = st.selectbox("Event scenario", [k for k in EVENT_SCENARIOS.keys() if k != "Custom (enter your own)"])
        all_companies = [k for k in COMPANIES.keys() if k != "Other (enter manually)"]
        selected_companies = st.multiselect("Select companies to compare", all_companies,
                                             default=all_companies[:6])

        if st.button("Compare", type="primary") and selected_companies:
            models = load_models()
            text = EVENT_SCENARIOS[scenario]
            evt = models["classifier"].predict(text)

            results = []
            for cn in selected_companies:
                info = COMPANIES[cn]
                exp = models["scorer"].score(event_category=evt["category"], ticker=info["ticker"], mention_sentiment=-0.4)
                imp = models["estimator"].estimate(
                    event_category=evt["category"], impact_channel=exp["channel_prediction"],
                    ticker=info["ticker"], mention_sentiment=-0.4, revenue_usd=info["revenue"],
                )
                results.append({
                    "Company": cn.split(" (")[0],
                    "Sector": info["sector"],
                    "Severity": exp["severity_score"],
                    "Impact %": imp["impact_mid_pct"],
                    "Channel": exp["channel_prediction"].replace("_", " ").title(),
                })

            df = pd.DataFrame(results).sort_values("Severity")
            st.subheader(f'"{scenario}" — Company Comparison')

            chart_df = df.set_index("Company")[["Severity", "Impact %"]]
            st.bar_chart(chart_df)
            st.dataframe(df, use_container_width=True, hide_index=True)
