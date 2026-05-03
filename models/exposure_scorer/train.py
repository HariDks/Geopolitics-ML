"""
Model 2: Exposure Scorer — XGBoost model predicting company exposure to geopolitical events.

Given an (event, company) pair, predicts:
  1. Which of 10 impact channels are affected (multi-label classification)
  2. Exposure severity score per channel (regression, -1 to 1)

Training data combines:
  - 163 seed labels (gold standard, human-reviewed)
  - 1,973 event studies (stock reactions for ~99 tickers x 20 events)
  - Financial deltas from EDGAR XBRL
  - EDGAR geopolitical mention signals

Features:
  - Event: category (one-hot), severity proxy
  - Company: GICS sector (encoded), recent financial performance
  - Event-Company interaction: stock reaction, mention count, mention specificity
  - Financial context: YoY revenue delta, gross margin, margin change

Usage:
    python models/exposure_scorer/train.py
    python models/exposure_scorer/train.py --eval-only
"""

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np
import xgboost as xgb

# Load geographic exposure data
GEO_EXPOSURE_PATH = Path(__file__).parent.parent.parent / "data" / "mappings" / "company_geo_exposure.json"
_geo_exposure = {}
if GEO_EXPOSURE_PATH.exists():
    with open(GEO_EXPOSURE_PATH) as f:
        _geo_exposure = json.load(f)
        _geo_exposure.pop("_description", None)
        _geo_exposure.pop("_source", None)

# Map event → affected regions (ISO2 codes or region tags)
EVENT_AFFECTED_REGIONS = {
    "russia_invasion_2022": ["RU", "UA", "EU"],
    "russia_corporate_exit_2022": ["RU"],
    "russia_sanctions_2022": ["RU"],
    "us_chip_export_controls_oct2022": ["CN", "TW"],
    "us_chip_export_oct2022": ["CN", "TW"],
    "us_chip_export_oct2023": ["CN", "TW"],
    "covid_lockdown_start": ["CN", "US", "EU", "other_asia"],
    "red_sea_houthi_2023": ["MENA", "other_asia"],
    "red_sea_houthi_attacks_2023": ["MENA", "other_asia"],
    "us_tariffs_2025": ["CN", "other_asia", "EU"],
    "us_tariffs_2025_april": ["CN", "other_asia", "EU"],
    "us_china_trade_war_start": ["CN"],
    "us_china_trade_war_2018": ["CN"],
    "xinjiang_boycott_2021": ["CN"],
    "xinjiang_cotton_ban_2021": ["CN"],
    "israel_hamas_2023": ["IL", "MENA"],
    "iran_sanctions_2018": ["IR", "MENA"],
    "us_iran_war_2026": ["IR", "MENA"],
    "india_demonetization_2016": ["IN"],
    "india_demonetization_nov2016": ["IN"],
    "india_pakistan_sindoor": ["IN", "PK"],
    "india_pakistan_sindoor_2025": ["IN", "PK"],
    "india_pakistan_operation_sindoor_2025": ["IN", "PK"],
    "brexit_referendum": ["EU", "UK"],
    "brexit_financial_relocation_2021": ["EU", "UK"],
    "eu_energy_crisis_peak": ["EU", "RU"],
    "notpetya_2017": ["UA", "RU", "EU"],
    "panama_mine_closure_2023": ["PA"],
    "panama_mining_contract_2023": ["PA"],
    "chile_lithium_nationalization": ["CL"],
    "sudan_civil_war_2023": ["SD"],
    "australia_china_wine_tariff_2020": ["CN", "AU"],
    "australia_china_coal_ban_2020": ["CN", "AU"],
    "australia_china_lobster_ban_2020": ["CN", "AU"],
    "argentina_milei_deregulation_2024": ["AR"],
    "solarwinds_hack_2020": ["US"],
    "myanmar_coup_2021": ["MM"],
    "indonesia_nickel_ban_2020": ["ID"],
    "opec_price_war_2014": ["MENA"],
    "suez_blockage_2021": ["MENA"],
}


def compute_geo_concentration(ticker: str, event_id: str) -> float:
    """
    Compute % of revenue/operations in regions affected by this event.
    Returns 0-100 (percentage).
    Higher = more concentrated exposure to the affected region.
    """
    geo = _geo_exposure.get(ticker, {})
    if not geo:
        return 0.0

    affected = EVENT_AFFECTED_REGIONS.get(event_id, [])
    if not affected:
        return 0.0

    total_exposed = 0.0
    for region in affected:
        total_exposed += geo.get(region, 0.0)

    return min(total_exposed, 100.0)
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipelines.utils import get_db_connection, get_logger

logger = get_logger("exposure_scorer")

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = Path(__file__).parent / "saved"
SEED_LABELS_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"

IMPACT_CHANNELS = [
    "procurement_supply_chain",
    "revenue_market_access",
    "capital_allocation_investment",
    "regulatory_compliance_cost",
    "logistics_operations",
    "innovation_ip",
    "workforce_talent",
    "reputation_stakeholder",
    "financial_treasury",
    "cybersecurity_it",
]
CHANNEL2IDX = {c: i for i, c in enumerate(IMPACT_CHANNELS)}

EVENT_CATEGORIES = [
    "trade_policy_actions",
    "sanctions_financial_restrictions",
    "armed_conflict_instability",
    "regulatory_sovereignty_shifts",
    "technology_controls",
    "resource_energy_disruptions",
    "political_transitions_volatility",
    "institutional_alliance_realignment",
]

# Map seed label event_ids to taxonomy categories
EVENT_TO_CATEGORY = {
    "russia_invasion_2022": "armed_conflict_instability",
    "us_chip_export_controls_oct2022": "technology_controls",
    "us_chip_export_oct2022": "technology_controls",
    "us_chip_export_oct2023": "technology_controls",
    "red_sea_houthi_2023": "armed_conflict_instability",
    "us_tariffs_2025": "trade_policy_actions",
    "covid_lockdown_start": "armed_conflict_instability",
    "israel_hamas_2023": "armed_conflict_instability",
    "eu_energy_crisis_peak": "resource_energy_disruptions",
    "us_china_trade_war_start": "trade_policy_actions",
    "us_china_trade_war_2018": "trade_policy_actions",
    "xinjiang_boycott_2021": "trade_policy_actions",
    "brexit_referendum": "institutional_alliance_realignment",
    "india_pakistan_sindoor": "armed_conflict_instability",
    "india_pakistan_sindoor_2025": "armed_conflict_instability",
    "us_iran_war_2026": "armed_conflict_instability",
    "notpetya_2017": "armed_conflict_instability",
    "iran_sanctions_2018": "sanctions_financial_restrictions",
    "panama_mine_closure_2023": "political_transitions_volatility",
    "chile_lithium_nationalization": "political_transitions_volatility",
    "india_demonetization_2016": "political_transitions_volatility",
    "opec_price_war_2014": "resource_energy_disruptions",
    "suez_blockage_2021": "armed_conflict_instability",
    # Additional mappings for seed label event_ids
    "hong_kong_protests_2019": "political_transitions_volatility",
    "myanmar_coup_2021": "political_transitions_volatility",
    "argentina_milei_deregulation_2024": "political_transitions_volatility",
    "argentina_peso_devaluation_2018": "political_transitions_volatility",
    "argentina_peso_devaluation_2023": "political_transitions_volatility",
    "australia_china_coal_ban_2020": "trade_policy_actions",
    "australia_china_wine_tariff_2020": "trade_policy_actions",
    "australia_china_lobster_ban_2020": "trade_policy_actions",
    "boeing_737max_global_grounding_2019": "regulatory_sovereignty_shifts",
    "boeing_china_737max_grounding": "regulatory_sovereignty_shifts",
    "brexit_financial_relocation_2021": "institutional_alliance_realignment",
    "brexit_labor_restrictions_2021": "institutional_alliance_realignment",
    "cfius_qualcomm_broadcom_block_2018": "regulatory_sovereignty_shifts",
    "china_ev_tariff_retaliation_2024": "trade_policy_actions",
    "china_gallium_germanium_export_controls_2023": "resource_energy_disruptions",
    "china_rare_earth_controls_2023": "resource_energy_disruptions",
    "china_tech_crackdown_2021": "regulatory_sovereignty_shifts",
    "colonial_pipeline_ransomware_2021": "armed_conflict_instability",
    "colombia_coal_transition_2023": "resource_energy_disruptions",
    "covid_zhengzhou_lockdown_2022": "political_transitions_volatility",
    "drc_cobalt_royalty_dispute_2023": "resource_energy_disruptions",
    "eu_brandy_retaliation_2024": "trade_policy_actions",
    "eu_china_ev_tariff_2024": "trade_policy_actions",
    "eu_data_privacy_gdpr_2018": "regulatory_sovereignty_shifts",
    "eu_uk_vaccine_export_ban_2021": "trade_policy_actions",
    "fukushima_2011": "resource_energy_disruptions",
    "guinea_bauxite_coup_2021": "political_transitions_volatility",
    "iberdrola_mexico_forced_sale_2023": "regulatory_sovereignty_shifts",
    "india_bytedance_ban_2020": "regulatory_sovereignty_shifts",
    "india_data_localization_2018": "regulatory_sovereignty_shifts",
    "india_demonetization_nov2016": "political_transitions_volatility",
    "india_fdi_ban_chinese_apps_2020": "regulatory_sovereignty_shifts",
    "india_mastercard_ban_2021": "regulatory_sovereignty_shifts",
    "indonesia_nickel_ban_2020": "resource_energy_disruptions",
    "iran_israel_conflict_2024": "armed_conflict_instability",
    "iran_israel_missiles_2024": "armed_conflict_instability",
    "japan_korea_trade_dispute_2019": "trade_policy_actions",
    "libya_civil_war_2011": "armed_conflict_instability",
    "mexico_energy_reform_2021": "regulatory_sovereignty_shifts",
    "mexico_judicial_reform_2024": "political_transitions_volatility",
    "moveit_hack_2023": "armed_conflict_instability",
    "mozambique_lng_insurgency_2021": "armed_conflict_instability",
    "niger_coup_uranium_2023": "political_transitions_volatility",
    "nigeria_oil_theft_2022": "resource_energy_disruptions",
    "norsk_hydro_ransomware_2019": "armed_conflict_instability",
    "opec_price_war_2014_2016": "resource_energy_disruptions",
    "operation_absolute_resolve_2026": "political_transitions_volatility",
    "operation_epic_fury_2026": "armed_conflict_instability",
    "russia_china_reroute_2022": "armed_conflict_instability",
    "russia_corporate_exit_2022": "sanctions_financial_restrictions",
    "russia_sanctions_2022": "sanctions_financial_restrictions",
    "russia_ukraine_invasion_2022": "armed_conflict_instability",
    "solarwinds_hack_2020": "armed_conflict_instability",
    "south_africa_loadshedding_2023": "resource_energy_disruptions",
    "strait_of_hormuz_closure_2026": "armed_conflict_instability",
    "sudan_civil_war_2023": "armed_conflict_instability",
    "tesla_brand_damage_2025": "reputation_stakeholder",
    "tiktok_project_texas_2022": "regulatory_sovereignty_shifts",
    "turkey_lira_crisis_2021": "political_transitions_volatility",
    "ukraine_grain_crisis_2022": "resource_energy_disruptions",
    "us_iran_sanctions_reimpose_2018": "sanctions_financial_restrictions",
    "us_tariffs_2025_april": "trade_policy_actions",
    "us_tariffs_2025_universal": "trade_policy_actions",
    "venezuela_maduro_capture_2026": "political_transitions_volatility",
    "viasat_cyberattack_2022": "armed_conflict_instability",
    "wannacry_2017": "armed_conflict_instability",
    "xinjiang_cotton_ban_2021": "trade_policy_actions",
    "change_healthcare_ransomware_2024": "armed_conflict_instability",
    "jbs_ransomware_2021": "armed_conflict_instability",
}

# GICS sector code → sector group (first 2 digits)
def gics_sector(code: str) -> int:
    """Extract GICS sector (first 2 digits) as integer."""
    try:
        return int(str(code)[:2])
    except (ValueError, TypeError):
        return 0


WEAK_LABELS_PROBS_PATH = ROOT_DIR / "data" / "seed_labels" / "weak_labels_probs.csv"


def load_seed_labels() -> list[dict]:
    """Load seed labels from CSV."""
    with open(SEED_LABELS_PATH) as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_weak_labels() -> list[dict]:
    """Load weak labels with Snorkel probability distributions."""
    if not WEAK_LABELS_PROBS_PATH.exists():
        logger.warning(f"No weak labels found at {WEAK_LABELS_PROBS_PATH}")
        return []
    with open(WEAK_LABELS_PROBS_PATH) as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(val, default=0.0):
    """Safely convert to float."""
    if val is None or val == "" or val == "nan":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# Load exposure proxies
PROXY_PATH = ROOT_DIR / "data" / "mappings" / "company_exposure_proxies.json"
exposure_proxies = {}
if PROXY_PATH.exists():
    with open(PROXY_PATH) as f:
        exposure_proxies = json.load(f)
        exposure_proxies.pop("_description", None)

# ── Channel-specific lexicon features ──────────────────────────────────────
# Curated keyword lists per channel, derived from distinctive terms in manual labels.
# Each score = count of matching keywords / total keywords in lexicon.

CHANNEL_LEXICONS = {
    "revenue_market_access": [
        "revenue", "sales", "market access", "demand", "customer", "orders",
        "banned", "boycott", "export", "import ban", "market share", "lost",
        "decline in sales", "represented", "accounted for", "fell",
        "tariff", "restricted", "cannot sell", "lost access",
    ],
    "procurement_supply_chain": [
        "supply", "supply chain", "supplier", "procurement", "input cost",
        "raw material", "component", "shortage", "tariff", "duty", "production",
        "manufacturing", "assembly", "energy cost", "freight", "steel",
    ],
    "capital_allocation_investment": [
        "impairment", "write-down", "write-off", "stake", "asset disposal",
        "divest", "exit", "pre-tax charge", "billion", "goodwill",
        "investment", "capex", "suspended", "mine closure", "concession",
    ],
    "regulatory_compliance_cost": [
        "compliance", "regulation", "regulatory", "license", "screening",
        "data localization", "privacy", "gdpr", "penalty", "fine",
        "staffing", "approval", "certification", "requirement",
    ],
    "logistics_operations": [
        "rerouting", "route", "shipping", "freight", "transit", "vessel",
        "red sea", "suez", "cape of good hope", "port", "logistics",
        "disruption", "delay", "reroute", "stranded",
    ],
    "innovation_ip": [
        "chip", "semiconductor", "export control", "technology", "r&d",
        "patent", "intellectual property", "joint venture", "jv",
        "research", "development", "dual-use", "euv", "foundry",
    ],
    "workforce_talent": [
        "employee", "worker", "labor", "workforce", "evacuation", "relocated",
        "visa", "reserve duty", "talent", "hiring", "headcount",
        "remote work", "staffing", "immigration",
    ],
    "reputation_stakeholder": [
        "boycott", "reputation", "brand", "consumer backlash", "protest",
        "public pressure", "esg", "stakeholder", "sponsors of war",
        "social media", "backlash", "controversy",
    ],
    "financial_treasury": [
        "currency", "fx", "foreign exchange", "repatriation", "devaluation",
        "lira", "peso", "ruble", "trapped", "frozen", "asset freeze",
        "swift", "counterparty", "treasury",
    ],
    "cybersecurity_it": [
        "ransomware", "cyberattack", "hack", "malware", "cyber", "breach",
        "encrypted", "ransom", "incident response", "it systems",
        "notpetya", "solarwinds", "phishing",
    ],
}


def compute_lexicon_scores(text: str) -> dict[str, float]:
    """Compute channel-specific lexicon scores for a text."""
    text_lower = text.lower() if text else ""
    scores = {}
    for channel, keywords in CHANNEL_LEXICONS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[channel] = hits / len(keywords)
    return scores


# Text channel model for generating text-based probabilities
_text_model = None
_text_vectorizer = None


def _get_text_channel_probs(mention_text: str) -> np.ndarray:
    """Get channel probabilities from the text model for a mention."""
    global _text_model, _text_vectorizer
    if _text_model is None:
        import pickle
        text_model_path = MODEL_DIR / "text_channel_model.pkl"
        if text_model_path.exists():
            with open(text_model_path, "rb") as f:
                data = pickle.load(f)
                _text_model = data["model"]
                _text_vectorizer = data["vectorizer"]
        else:
            return np.zeros(len(IMPACT_CHANNELS))

    if not mention_text or len(mention_text) < 10:
        return np.zeros(len(IMPACT_CHANNELS))

    X = _text_vectorizer.transform([mention_text])
    return _text_model.predict_proba(X)[0]


def build_feature_matrix(conn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Build feature matrix from seed labels + Snorkel weak labels.

    Returns:
        X: feature matrix (n_samples, n_features)
        y_channel: channel labels (n_samples,) — integer encoded
        y_severity: severity scores (n_samples,) — mention_sentiment as proxy
        sample_weights: confidence weights (n_samples,) — 1.0 for seed, Snorkel confidence for weak
        metadata: list of dicts with sample info for debugging
    """
    seed_labels = load_seed_labels()

    # Pre-load event studies and financial deltas into lookup dicts
    event_studies = {}
    rows = conn.execute("SELECT * FROM event_studies").fetchall()
    for r in rows:
        key = (r["event_id"], r["ticker"])
        event_studies[key] = dict(r)

    fin_deltas = {}
    rows = conn.execute("""
        SELECT * FROM financial_deltas
        WHERE revenue_standalone IS NOT NULL
    """).fetchall()
    for r in rows:
        key = (r["ticker"], r["fiscal_year"], r["fiscal_period"])
        fin_deltas[key] = dict(r)

    # Pre-load mention signals
    mention_signals = {}
    rows = conn.execute("""
        SELECT ticker, primary_category,
               COUNT(*) as mention_count,
               AVG(specificity_score) as avg_specificity,
               MAX(specificity_score) as max_specificity,
               AVG(keyword_count) as avg_keywords
        FROM geopolitical_mentions
        GROUP BY ticker, primary_category
    """).fetchall()
    for r in rows:
        key = (r["ticker"], r["primary_category"])
        mention_signals[key] = dict(r)

    X_rows = []
    y_channel = []
    y_severity = []
    metadata = []

    # ── Source 1: Seed labels (gold standard) ──
    for label in seed_labels:
        event_id = label["event_id"]
        ticker = label["company_ticker"]
        channel = label["impact_channel"]

        if channel not in CHANNEL2IDX:
            continue

        # Event category (one-hot)
        event_cat = EVENT_TO_CATEGORY.get(event_id, "")
        cat_features = [1.0 if c == event_cat else 0.0 for c in EVENT_CATEGORIES]

        # Company sector
        sector = gics_sector(label.get("sector_gics", ""))

        # Sentiment as severity proxy
        sentiment = safe_float(label.get("mention_sentiment"), 0.0)

        # Stock reaction from seed label
        car_5 = safe_float(label.get("car_1_5"), 0.0)

        # Try to find matching event study for broader context
        # Map seed event_ids to event_study event_ids
        es_event_id = _map_to_event_study_id(event_id)
        es = event_studies.get((es_event_id, ticker), {}) if es_event_id else {}
        es_car_5 = es.get("car_1_5", car_5) or car_5
        es_car_30 = es.get("car_1_30", 0.0) or 0.0

        # Financial context — find closest quarter
        quarter = label.get("quarter", "")
        fy, fp = _parse_quarter(quarter)
        fd = fin_deltas.get((ticker, fy, fp), {})
        rev_yoy = fd.get("revenue_yoy_pct", 0.0) or 0.0
        gm = fd.get("gross_margin", 0.0) or 0.0
        gm_delta = fd.get("gross_margin_delta_pp", 0.0) or 0.0
        rev_standalone = fd.get("revenue_standalone", 0.0) or 0.0
        # Log-scale revenue for feature
        log_rev = np.log1p(abs(rev_standalone) / 1e6) if rev_standalone else 0.0

        # Mention signals for this company + event category
        ms = mention_signals.get((ticker, event_cat), {})
        mention_count = ms.get("mention_count", 0) or 0
        avg_specificity = ms.get("avg_specificity", 0.0) or 0.0
        max_specificity = ms.get("max_specificity", 0.0) or 0.0
        avg_keywords = ms.get("avg_keywords", 0.0) or 0.0

        # Revenue delta from seed label
        rev_delta = safe_float(label.get("revenue_delta_pct"), 0.0)

        # Geographic concentration — % of revenue in affected region
        geo_conc = compute_geo_concentration(ticker, event_id)

        # Exposure proxies from 10-K text
        proxy = exposure_proxies.get(ticker, {})
        facility_score = proxy.get("facility_concentration_score", 0.0)
        single_source = proxy.get("single_source_risk_score", 0.0)
        asset_exit = proxy.get("asset_exit_score", 0.0)
        route_sensitivity = proxy.get("route_sensitivity_score", 0.0)

        # Region-specific mention density for affected region
        geo_density = proxy.get("geo_mention_density", {})
        affected_regions = EVENT_AFFECTED_REGIONS.get(event_id, [])
        affected_geo_density = sum(geo_density.get(r, 0.0) for r in affected_regions)

        # Channel-specific lexicon scores from mention_text
        lex = compute_lexicon_scores(label.get("mention_text", ""))
        lex_scores = [lex.get(ch, 0.0) for ch in IMPACT_CHANNELS]

        # Feature vector
        features = (
            cat_features  # 8 features: event category one-hot
            + [
                sector,               # GICS sector code
                sentiment,            # mention sentiment (-1 to 1)
                car_5,                # seed label car_1_5
                es_car_5,             # event study car_1_5
                es_car_30,            # event study car_1_30
                rev_yoy,              # revenue YoY from XBRL
                gm,                   # gross margin
                gm_delta,             # gross margin change
                log_rev,              # log revenue (company size proxy)
                mention_count,        # EDGAR mention count
                avg_specificity,      # avg mention specificity
                max_specificity,      # max mention specificity
                avg_keywords,         # avg keyword count per mention
                rev_delta,            # revenue delta from seed label
                geo_conc,             # % revenue in affected region
                facility_score,       # facility/plant mention concentration
                single_source,        # single-source supplier risk
                asset_exit,           # impairment/exit history
                route_sensitivity,    # shipping route mentions
                affected_geo_density, # mention density for affected regions
            ]
            + lex_scores              # 10 features: channel-specific lexicon scores
        )

        X_rows.append(features)
        y_channel.append(CHANNEL2IDX[channel])
        y_severity.append(sentiment)
        metadata.append({
            "event_id": event_id,
            "ticker": ticker,
            "channel": channel,
            "source": "seed_label",
        })

    # ── Source 2: Snorkel weak labels (confidence-weighted) ──
    weak_labels = load_weak_labels()
    weak_count = 0
    for label in weak_labels:
        event_id = label["event_id"]
        ticker = label["company_ticker"]
        channel = label["predicted_channel"]

        if channel not in CHANNEL2IDX:
            continue

        # Skip if already in seed labels
        if any(m["event_id"] == event_id and m["ticker"] == ticker and m["source"] == "seed_label"
               for m in metadata):
            continue

        # Event category (one-hot)
        event_cat = label.get("event_category", EVENT_TO_CATEGORY.get(event_id, ""))
        cat_features = [1.0 if c == event_cat else 0.0 for c in EVENT_CATEGORIES]

        # Company sector
        sector = gics_sector(label.get("sector_gics", ""))

        # Sentiment proxy from stock direction
        car_5 = safe_float(label.get("car_1_5"), 0.0)
        sentiment = -0.3 if car_5 < 0 else 0.1

        # Event study data
        es_event_id = _map_to_event_study_id(event_id)
        es = event_studies.get((es_event_id or event_id, ticker), {})
        es_car_5 = es.get("car_1_5", car_5) or car_5
        es_car_30 = es.get("car_1_30", 0.0) or 0.0

        # Financial context — use latest available
        fd = {}
        for key, val in fin_deltas.items():
            if key[0] == ticker:
                fd = val
                break
        rev_yoy = fd.get("revenue_yoy_pct", 0.0) or 0.0
        gm = fd.get("gross_margin", 0.0) or 0.0
        gm_delta = fd.get("gross_margin_delta_pp", 0.0) or 0.0
        rev_standalone = fd.get("revenue_standalone", 0.0) or 0.0
        log_rev = np.log1p(abs(rev_standalone) / 1e6) if rev_standalone else 0.0

        # Mention signals
        ms = mention_signals.get((ticker, event_cat), {})
        mention_count = ms.get("mention_count", 0) or 0
        avg_specificity = ms.get("avg_specificity", 0.0) or 0.0
        max_specificity = ms.get("max_specificity", 0.0) or 0.0
        avg_keywords = ms.get("avg_keywords", 0.0) or 0.0

        # Geographic concentration
        geo_conc = compute_geo_concentration(ticker, event_id)

        # Exposure proxies
        proxy = exposure_proxies.get(ticker, {})
        facility_score = proxy.get("facility_concentration_score", 0.0)
        single_source = proxy.get("single_source_risk_score", 0.0)
        asset_exit = proxy.get("asset_exit_score", 0.0)
        route_sensitivity = proxy.get("route_sensitivity_score", 0.0)

        geo_density = proxy.get("geo_mention_density", {})
        affected_regions = EVENT_AFFECTED_REGIONS.get(event_id, [])
        affected_geo_density = sum(geo_density.get(r, 0.0) for r in affected_regions)

        # Lexicon scores from mention text
        lex = compute_lexicon_scores(label.get("mention_text", ""))
        lex_scores = [lex.get(ch, 0.0) for ch in IMPACT_CHANNELS]

        features = (
            cat_features
            + [
                sector, sentiment, car_5, es_car_5, es_car_30,
                rev_yoy, gm, gm_delta, log_rev,
                mention_count, avg_specificity, max_specificity, avg_keywords,
                0.0,  # rev_delta (not available for weak labels)
                geo_conc, facility_score, single_source, asset_exit,
                route_sensitivity, affected_geo_density,
            ]
            + lex_scores
        )

        # Snorkel confidence as sample weight
        snorkel_conf = safe_float(label.get("snorkel_confidence"), 0.5)

        X_rows.append(features)
        y_channel.append(CHANNEL2IDX[channel])
        y_severity.append(sentiment)
        metadata.append({
            "event_id": event_id,
            "ticker": ticker,
            "channel": channel,
            "source": "weak_label",
            "sample_weight": snorkel_conf,
        })
        weak_count += 1

    logger.info(f"  Weak labels added: {weak_count}")

    # ── Source 3: Oversample underrepresented channels ──
    # With only 163 seed labels, some channels have 1-5 examples.
    # Duplicate minority channel examples with small noise to help the model.
    channel_counts = {}
    for ch_idx in y_channel:
        channel_counts[ch_idx] = channel_counts.get(ch_idx, 0) + 1

    target_min = 15  # ensure at least 15 samples per channel
    np.random.seed(42)
    augmented_rows = []
    for ch_idx, count in channel_counts.items():
        if count >= target_min:
            continue
        # Find indices of this channel
        ch_indices = [i for i, y in enumerate(y_channel) if y == ch_idx]
        n_needed = target_min - count
        for _ in range(n_needed):
            # Pick a random existing example and add small noise
            src_idx = ch_indices[np.random.randint(len(ch_indices))]
            noisy_features = list(X_rows[src_idx])
            # Add noise to continuous features only (indices 8-21)
            for fi in range(8, len(noisy_features)):
                val = noisy_features[fi]
                if val != 0.0:
                    noisy_features[fi] = val * (1.0 + np.random.normal(0, 0.05))
            augmented_rows.append((noisy_features, ch_idx, y_severity[len(y_channel) - 1], {
                "event_id": metadata[src_idx]["event_id"],
                "ticker": metadata[src_idx]["ticker"],
                "channel": metadata[src_idx]["channel"],
                "source": "augmented",
            }))

    for feat, ch, sev, meta in augmented_rows:
        X_rows.append(feat)
        y_channel.append(ch)
        y_severity.append(sev)
        metadata.append(meta)

    X = np.array(X_rows, dtype=np.float32)
    y_ch = np.array(y_channel, dtype=np.int32)
    y_sev = np.array(y_severity, dtype=np.float32)

    # Build sample weights: 1.0 for seed/augmented, Snorkel confidence for weak
    weights = np.ones(len(metadata), dtype=np.float32)
    for i, m in enumerate(metadata):
        if m["source"] == "weak_label":
            weights[i] = m.get("sample_weight", 0.5)

    logger.info(f"Built feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"  Seed labels: {sum(1 for m in metadata if m['source'] == 'seed_label')}")
    logger.info(f"  Weak labels: {sum(1 for m in metadata if m['source'] == 'weak_label')}")
    logger.info(f"  Augmented: {sum(1 for m in metadata if m['source'] == 'augmented')}")

    return X, y_ch, y_sev, weights, metadata


def _map_to_event_study_id(event_id: str) -> str | None:
    """Map seed label event_id to event_studies event_id."""
    direct_map = {
        "us_chip_export_controls_oct2022": "us_chip_export_oct2022",
        "russia_ukraine_invasion_2022": "russia_invasion_2022",
        "russia_corporate_exit_2022": "russia_invasion_2022",
        "russia_sanctions_2022": "russia_invasion_2022",
        "india_pakistan_sindoor_2025": "india_pakistan_sindoor",
        "operation_epic_fury_2026": "us_iran_war_2026",
        "strait_of_hormuz_closure_2026": "us_iran_war_2026",
        "us_tariffs_2025_april": "us_tariffs_2025",
        "us_tariffs_2025_universal": "us_tariffs_2025",
        "xinjiang_cotton_ban_2021": "xinjiang_boycott_2021",
        "colonial_pipeline_ransomware_2021": "notpetya_2017",
        "iran_israel_conflict_2024": "israel_hamas_2023",
        "iran_israel_missiles_2024": "israel_hamas_2023",
        "us_iran_sanctions_reimpose_2018": "iran_sanctions_2018",
        "eu_energy_crisis_2022": "eu_energy_crisis_peak",
    }
    if event_id in direct_map:
        return direct_map[event_id]
    # Try direct match
    from sqlite3 import connect
    es_ids = {
        "brexit_referendum", "chile_lithium_nationalization", "covid_lockdown_start",
        "eu_energy_crisis_peak", "india_demonetization_2016", "india_pakistan_sindoor",
        "iran_sanctions_2018", "israel_hamas_2023", "notpetya_2017",
        "opec_price_war_2014", "panama_mine_closure_2023", "red_sea_houthi_2023",
        "russia_invasion_2022", "suez_blockage_2021", "us_china_trade_war_start",
        "us_chip_export_oct2022", "us_chip_export_oct2023", "us_iran_war_2026",
        "us_tariffs_2025", "xinjiang_boycott_2021",
    }
    if event_id in es_ids:
        return event_id
    return None


def _parse_quarter(quarter: str) -> tuple[int, str]:
    """Parse quarter string like '2022Q1' or '2023Q3FY' into (year, period)."""
    if not quarter:
        return (0, "")
    # Handle formats: 2022Q1, 2023FY, 2023Q3FY
    import re
    m = re.match(r"(\d{4})(Q\d|FY)", quarter)
    if m:
        return (int(m.group(1)), m.group(2))
    return (0, "")


def _get_latest_financials(fin_deltas: dict, ticker: str) -> dict:
    """Get the most recent financial data for a ticker."""
    best = {}
    best_key = (0, "")
    for (t, fy, fp), data in fin_deltas.items():
        if t == ticker and (fy, fp) > best_key:
            best_key = (fy, fp)
            best = data
    return best


FEATURE_NAMES = (
    [f"cat_{c}" for c in EVENT_CATEGORIES]
    + [
        "gics_sector", "mention_sentiment", "car_1_5_seed", "car_1_5_es",
        "car_1_30_es", "rev_yoy", "gross_margin", "gm_delta_pp",
        "log_revenue_M", "mention_count", "avg_specificity", "max_specificity",
        "avg_keywords", "rev_delta_pct", "geo_concentration_pct",
        "facility_score", "single_source_risk", "asset_exit_score",
        "route_sensitivity", "affected_geo_density",
    ]
    + [f"lex_{c[:15]}" for c in IMPACT_CHANNELS]
)


def train_channel_classifier(X_train, y_train, X_val, y_val, sample_weight=None) -> xgb.XGBClassifier:
    """Train XGBoost classifier for impact channel prediction.

    sample_weight: per-sample confidence weights. Seed labels get 1.0,
    weak labels get their Snorkel confidence (0.5-1.0). This lets the
    model learn more from high-confidence labels and less from uncertain ones.
    """
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(IMPACT_CHANNELS),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_severity_regressor(X_train, y_train, X_val, y_val, sample_weight=None) -> xgb.XGBRegressor:
    """Train XGBoost regressor for severity score prediction."""
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate_channel_classifier(model, X_val, y_val, metadata_val):
    """Evaluate channel classifier with detailed report."""
    y_pred = model.predict(X_val)

    # Only evaluate on seed labels (not negative examples)
    seed_mask = [m["source"] == "seed_label" for m in metadata_val]
    if any(seed_mask):
        seed_idx = [i for i, s in enumerate(seed_mask) if s]
        y_true_seed = y_val[seed_idx]
        y_pred_seed = y_pred[seed_idx]

        present_labels = sorted(set(y_true_seed) | set(y_pred_seed))
        present_names = [IMPACT_CHANNELS[i] for i in present_labels]

        print("\n" + "=" * 70)
        print("CHANNEL CLASSIFIER — SEED LABELS ONLY")
        print("=" * 70)
        print(classification_report(
            y_true_seed, y_pred_seed,
            labels=present_labels,
            target_names=present_names,
            digits=3,
            zero_division=0,
        ))

    # Full evaluation
    present_labels = sorted(set(y_val) | set(y_pred))
    present_names = [IMPACT_CHANNELS[i] for i in present_labels]

    print("=" * 70)
    print("CHANNEL CLASSIFIER — ALL DATA")
    print("=" * 70)
    print(classification_report(
        y_val, y_pred,
        labels=present_labels,
        target_names=present_names,
        digits=3,
        zero_division=0,
    ))

    # Feature importance
    print("TOP 10 FEATURE IMPORTANCES")
    print("-" * 40)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
        print(f"  {name:30s} {importances[i]:.4f}")


def evaluate_severity_regressor(model, X_val, y_val):
    """Evaluate severity regressor."""
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print(f"\nSEVERITY REGRESSOR")
    print(f"-" * 40)
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")


@click.command()
@click.option("--eval-only", is_flag=True, help="Only evaluate saved model")
def main(eval_only):
    """Train or evaluate the exposure scorer."""
    conn = get_db_connection()
    X, y_channel, y_severity, sample_weights, metadata = build_feature_matrix(conn)
    conn.close()

    # Temporal split (no future leakage)
    from pipelines.temporal_split import get_temporal_split
    train_idx = [i for i, m in enumerate(metadata)
                 if get_temporal_split(m.get("event_id", ""), "") in ("train",)]
    val_idx = [i for i, m in enumerate(metadata)
               if get_temporal_split(m.get("event_id", ""), "") in ("val", "test")]

    # Fallback to random split if temporal produces too few val samples
    if len(val_idx) < 20:
        logger.warning(f"Temporal split produced only {len(val_idx)} val samples. Using random split.")
        from sklearn.model_selection import train_test_split as _split
        train_idx, val_idx = [], []
        _X_train, _X_val, _y_train, _y_val, _m_train, _m_val = _split(
            list(range(len(metadata))), y_channel, metadata,
            test_size=0.2, stratify=y_channel, random_state=42,
        )
        train_idx, val_idx = list(_X_train), list(_X_val)

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_ch_train = y_channel[train_idx]
    y_ch_val = y_channel[val_idx]
    y_sev_train = y_severity[train_idx]
    y_sev_val = y_severity[val_idx]
    w_train = sample_weights[train_idx]
    meta_train = [metadata[i] for i in train_idx]
    meta_val = [metadata[i] for i in val_idx]

    # Report weight distribution
    seed_w = sum(1 for i in train_idx if metadata[i]["source"] == "seed_label")
    weak_w = sum(1 for i in train_idx if metadata[i]["source"] == "weak_label")
    logger.info(f"Train: {len(X_train)} ({seed_w} seed + {weak_w} weak + {len(X_train)-seed_w-weak_w} augmented), Val: {len(X_val)}")

    if eval_only:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        clf = xgb.XGBClassifier()
        clf.load_model(MODEL_DIR / "channel_classifier.json")
        reg = xgb.XGBRegressor()
        reg.load_model(MODEL_DIR / "severity_regressor.json")
    else:
        # Train channel classifier with confidence-weighted samples
        logger.info("Training channel classifier (with sample weights)...")
        clf = train_channel_classifier(X_train, y_ch_train, X_val, y_ch_val, sample_weight=w_train)

        # Train severity regressor with confidence-weighted samples
        logger.info("Training severity regressor (with sample weights)...")
        reg = train_severity_regressor(X_train, y_sev_train, X_val, y_sev_val, sample_weight=w_train)

        # Save models
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        clf.save_model(MODEL_DIR / "channel_classifier.json")
        reg.save_model(MODEL_DIR / "severity_regressor.json")

        # Save feature names and label maps
        with open(MODEL_DIR / "config.json", "w") as f:
            json.dump({
                "feature_names": FEATURE_NAMES,
                "impact_channels": IMPACT_CHANNELS,
                "event_categories": EVENT_CATEGORIES,
                "channel2idx": CHANNEL2IDX,
            }, f, indent=2)

        logger.info(f"Models saved to {MODEL_DIR}")

    # Evaluate
    evaluate_channel_classifier(clf, X_val, y_ch_val, meta_val)
    evaluate_severity_regressor(reg, X_val, y_sev_val)

    logger.info("Done.")


if __name__ == "__main__":
    main()
