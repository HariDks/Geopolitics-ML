"""
Bayesian impact estimator — replaces Model 3B regression with
principled uncertainty quantification.

Instead of predicting "revenue will drop X%", this model says:
"Given a sanctions event hitting a financial company with 30% exposure
to the affected region, the revenue impact is likely -2% to -8%
(80% credible interval)."

The reasoning chain:
1. Event category → base prior (what typically happens in this type of event)
2. Company exposure → scaling factor (how exposed is this company?)
3. Stock reaction → evidence update (did the market think this was bad?)
4. Posterior → credible interval with honest uncertainty

Uses conjugate Normal-Normal updating (closed-form, no MCMC needed):
- Prior: N(μ_prior, σ_prior²) from historical data per event category
- Likelihood: stock reaction as noisy evidence of impact
- Posterior: N(μ_post, σ_post²) via Bayesian updating

Usage:
    from models.impact_estimator.bayesian_impact import BayesianImpactEstimator

    est = BayesianImpactEstimator()
    result = est.estimate(
        event_category="sanctions_financial_restrictions",
        impact_channel="financial_treasury",
        ticker="JPM",
        car_1_5=-0.08,
        exposure_pct=30.0,
    )
    # → {"impact_low_pct": -8.2, "impact_mid_pct": -4.1, "impact_high_pct": -1.3,
    #    "credible_interval": 80, "prior_source": "category+channel",
    #    "exposure_description": "30% revenue from affected region",
    #    "reasoning": "..."}
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipelines.utils import get_logger

logger = get_logger("bayesian_impact")

SEED_PATH = ROOT_DIR / "data" / "seed_labels" / "seed_labels.csv"
GICS_PATH = ROOT_DIR / "data" / "mappings" / "ticker_gics.json"
GEO_PATH = ROOT_DIR / "data" / "mappings" / "company_geo_exposure.json"
PROXY_PATH = ROOT_DIR / "data" / "mappings" / "company_exposure_proxies.json"

EVENT_CATEGORIES = [
    "trade_policy_actions", "sanctions_financial_restrictions",
    "armed_conflict_instability", "regulatory_sovereignty_shifts",
    "technology_controls", "resource_energy_disruptions",
    "political_transitions_volatility", "institutional_alliance_realignment",
]

# Event category → affected regions (for exposure lookup)
EVENT_REGIONS = {
    "trade_policy_actions": ["CN", "US"],
    "sanctions_financial_restrictions": ["RU", "IR"],
    "armed_conflict_instability": ["RU", "UA", "IL", "PS"],
    "regulatory_sovereignty_shifts": ["CN", "EU", "IN"],
    "technology_controls": ["CN", "TW"],
    "resource_energy_disruptions": ["RU", "SA", "NG"],
    "political_transitions_volatility": ["AR", "TR", "MM"],
    "institutional_alliance_realignment": ["GB", "EU"],
}

# GICS sector → typical exposure channel
SECTOR_CHANNEL_MAP = {
    10: "procurement_supply_chain",      # Energy
    15: "procurement_supply_chain",      # Materials
    20: "logistics_operations",          # Industrials
    25: "revenue_market_access",         # Consumer Discretionary
    30: "revenue_market_access",         # Consumer Staples
    35: "regulatory_compliance_cost",    # Health Care
    40: "financial_treasury",            # Financials
    45: "innovation_ip",                 # Information Technology
    50: "regulatory_compliance_cost",    # Communication Services
    55: "capital_allocation_investment", # Utilities
    60: "capital_allocation_investment", # Real Estate
}


class BayesianImpactEstimator:
    """Bayesian revenue impact estimation with exposure-weighted priors."""

    def __init__(self):
        self._priors = None
        self._gics = {}
        self._geo = {}
        self._proxies = {}
        self._load_data()

    def _load_data(self):
        """Load priors from gold labels and exposure data."""
        # Load GICS sectors
        if GICS_PATH.exists():
            with open(GICS_PATH) as f:
                self._gics = json.load(f)

        # Load geographic exposure
        if GEO_PATH.exists():
            with open(GEO_PATH) as f:
                self._geo = json.load(f)
                self._geo.pop("_description", None)
                self._geo.pop("_source", None)

        # Load exposure proxies
        if PROXY_PATH.exists():
            with open(PROXY_PATH) as f:
                self._proxies = json.load(f)
                self._proxies.pop("_description", None)

        # Build priors from gold revenue labels
        self._build_priors()

    # Channel → impact group mapping
    CHANNEL_GROUP = {
        "revenue_market_access": "market_loss",
        "reputation_stakeholder": "market_loss",
        "procurement_supply_chain": "cost_increase",
        "logistics_operations": "cost_increase",
        "regulatory_compliance_cost": "cost_increase",
        "capital_allocation_investment": "asset_loss",
        "financial_treasury": "asset_loss",
        "cybersecurity_it": "operational",
        "innovation_ip": "operational",
        "workforce_talent": "operational",
    }

    def _build_priors(self):
        """Build (channel_group × exposure_level) priors from 128 tagged labels.

        The key insight: the same channel produces wildly different impacts
        depending on exposure level. A bank with 1% Russia exposure loses ~2%.
        A bank that violated sanctions for years loses 18%. Same channel,
        completely different outcome — because exposure is different.

        Priors are built at two levels:
        1. (group × exposure): e.g., "market_loss + high exposure" → median -60%
        2. (channel × exposure): more specific but thinner data
        Falls back from channel → group → overall as data gets thin.
        """
        self._priors = {}

        # Load exposure-tagged labels
        tagged_path = ROOT_DIR / "data" / "seed_labels" / "revenue_labels_with_exposure.json"
        if not tagged_path.exists():
            logger.warning("No exposure-tagged labels found. Using flat priors.")
            self._priors["fallback"] = {"mu": -10.0, "sigma": 15.0, "n": 0}
            return

        with open(tagged_path) as f:
            tagged = json.load(f)

        from collections import defaultdict

        # Build (group × exposure) priors
        group_exp = defaultdict(list)
        channel_exp = defaultdict(list)

        for l in tagged:
            group = self.CHANNEL_GROUP.get(l["channel"], "unknown")
            group_exp[(group, l["exposure_level"])].append(l["rev_pct"])
            channel_exp[(l["channel"], l["exposure_level"])].append(l["rev_pct"])

        # Store group × exposure priors (more data per bin)
        for (grp, exp), vals in group_exp.items():
            if len(vals) >= 3:
                self._priors[f"group_{grp}_{exp}"] = {
                    "mu": float(np.median(vals)),  # median is more robust with outliers
                    "sigma": max(float(np.std(vals)), 3.0),
                    "n": len(vals),
                }

        # Store channel × exposure priors (thin but specific)
        for (ch, exp), vals in channel_exp.items():
            if len(vals) >= 3:
                self._priors[f"channel_{ch}_{exp}"] = {
                    "mu": float(np.median(vals)),
                    "sigma": max(float(np.std(vals)), 3.0),
                    "n": len(vals),
                }

        # Overall fallback
        all_revs = [l["rev_pct"] for l in tagged]
        self._priors["fallback"] = {
            "mu": float(np.median(all_revs)),
            "sigma": max(float(np.std(all_revs)), 5.0),
            "n": len(all_revs),
        }

        logger.info(f"Built granular priors from {len(tagged)} exposure-tagged labels")
        for key in sorted(self._priors):
            p = self._priors[key]
            logger.info(f"  {key}: μ={p['mu']:+.1f}%, σ={p['sigma']:.1f}% (n={p['n']})")

    def _get_exposure(self, ticker: str, event_category: str) -> dict:
        """Estimate company exposure through the affected channel.

        Returns:
            {
                "exposure_pct": estimated % of business through affected channel,
                "exposure_level": "low" / "medium" / "high",
                "exposure_sources": list of evidence supporting the estimate,
            }
        """
        sources = []
        exposure_signals = []

        # 1. Geographic revenue concentration
        affected_regions = EVENT_REGIONS.get(event_category, [])
        geo = self._geo.get(ticker, {})
        if geo and affected_regions:
            region_pct = sum(geo.get(r, 0.0) for r in affected_regions)
            if region_pct > 0:
                exposure_signals.append(region_pct)
                sources.append(f"{region_pct:.0f}% revenue from affected region(s)")

        # 2. Exposure proxies (facility concentration, supplier risk, etc.)
        proxy = self._proxies.get(ticker, {})
        if proxy:
            # Average the relevant proxy scores (0-1 scale → 0-100%)
            proxy_scores = []
            if proxy.get("facility_concentration_score", 0) > 0:
                ps = proxy["facility_concentration_score"] * 100
                proxy_scores.append(ps)
                sources.append(f"Facility concentration: {ps:.0f}%")
            if proxy.get("single_source_risk_score", 0) > 0:
                ps = proxy["single_source_risk_score"] * 100
                proxy_scores.append(ps)
                sources.append(f"Single-source supplier risk: {ps:.0f}%")
            if proxy.get("route_sensitivity_score", 0) > 0:
                ps = proxy["route_sensitivity_score"] * 100
                proxy_scores.append(ps)
                sources.append(f"Shipping route sensitivity: {ps:.0f}%")
            if proxy_scores:
                exposure_signals.append(np.mean(proxy_scores))

        # 3. GICS sector as rough exposure indicator
        gics = int(self._gics.get(ticker, 0))
        sector_channel = SECTOR_CHANNEL_MAP.get(gics, "")
        if sector_channel:
            # Sector gives a very rough 10-30% baseline
            sources.append(f"Sector ({gics}) typically exposed through {sector_channel.replace('_', ' ')}")

        # Combine signals
        if exposure_signals:
            exposure_pct = np.mean(exposure_signals)
        elif gics:
            exposure_pct = 15.0  # sector-based default
        else:
            exposure_pct = 10.0  # minimal default

        # Classify level
        if exposure_pct > 30:
            level = "high"
        elif exposure_pct > 10:
            level = "medium"
        else:
            level = "low"

        return {
            "exposure_pct": round(float(exposure_pct), 1),
            "exposure_level": level,
            "exposure_sources": sources,
        }

    def estimate(
        self,
        event_category: str,
        impact_channel: str = "",
        ticker: str = "",
        exposure_pct: float = None,
        credible_interval: int = 80,
    ) -> dict:
        """
        Estimate revenue impact using Bayesian reasoning.

        NOTE: This does NOT use stock reaction (car_1_5). That belongs in
        Model 3A (market reaction). This model estimates BUSINESS impact
        through the identified channel, based on:
        1. Historical revenue impacts for this channel (prior)
        2. Company's specific exposure level (scaling)
        3. Number of historical cases (confidence)

        Args:
            event_category: type of geopolitical event
            impact_channel: predicted impact channel (from Model 2)
            ticker: company ticker (for exposure lookup)
            exposure_pct: override exposure % (if None, auto-estimated)
            credible_interval: width of credible interval (default 80%)

        Returns:
            dict with impact estimates, exposure info, and reasoning chain
        """
        reasoning = []

        # Step 1: Get exposure
        if exposure_pct is not None:
            exposure = {
                "exposure_pct": exposure_pct,
                "exposure_level": "high" if exposure_pct > 30 else "medium" if exposure_pct > 10 else "low",
                "exposure_sources": ["User-provided exposure estimate"],
            }
        elif ticker:
            exposure = self._get_exposure(ticker, event_category)
        else:
            exposure = {"exposure_pct": 10.0, "exposure_level": "low",
                       "exposure_sources": ["No company data — using conservative default"]}

        exp_level = exposure["exposure_level"]

        reasoning.append(
            f"Exposure: {exp_level} ({exposure['exposure_pct']:.0f}%)"
        )
        for src in exposure["exposure_sources"]:
            reasoning.append(f"  - {src}")

        # Step 2: Select the most specific prior available
        # Priority: (channel × exposure) → (group × exposure) → fallback
        group = self.CHANNEL_GROUP.get(impact_channel, "unknown")
        ch_display = impact_channel.replace("_", " ")

        # Try channel-specific prior for this exposure level
        prior_key = f"channel_{impact_channel}_{exp_level}"
        if prior_key in self._priors:
            prior = self._priors[prior_key].copy()
            reasoning.append(
                f"Prior: companies with {exp_level} exposure hit through {ch_display} "
                f"typically see {prior['mu']:+.0f}% revenue impact "
                f"(from {prior['n']} historical cases)"
            )
        # Fall back to group-level prior
        elif f"group_{group}_{exp_level}" in self._priors:
            prior_key = f"group_{group}_{exp_level}"
            prior = self._priors[prior_key].copy()
            reasoning.append(
                f"Prior: companies with {exp_level} exposure facing {group.replace('_', ' ')} impacts "
                f"typically see {prior['mu']:+.0f}% revenue impact "
                f"(from {prior['n']} historical cases)"
            )
        # Last resort: overall fallback
        else:
            prior = self._priors.get("fallback", {"mu": -5.0, "sigma": 15.0, "n": 0}).copy()
            reasoning.append(
                f"Prior: limited data for this channel + exposure combination. "
                f"Using overall median: {prior['mu']:+.0f}%"
            )

        direction = "negative" if prior["mu"] < 0 else "positive"

        # Step 3: Adjust confidence by sample size
        n_cases = prior["n"]
        posterior_sigma = prior["sigma"]

        if n_cases >= 15:
            posterior_sigma = posterior_sigma / np.sqrt(n_cases / 5)
            reasoning.append(f"Confidence: {n_cases} historical cases → tighter interval")
        elif n_cases >= 5:
            # Moderate data
            reasoning.append(f"Confidence: {n_cases} historical cases — moderate certainty")
        else:
            posterior_sigma = posterior_sigma * 1.5
            reasoning.append(f"Caution: only {n_cases} historical cases → wider interval")

        posterior_mu = prior["mu"]

        # Step 4: Compute credible interval
        from scipy.stats import norm
        alpha = (100 - credible_interval) / 100
        z = norm.ppf(1 - alpha / 2)

        impact_mid = posterior_mu
        impact_low = posterior_mu - z * posterior_sigma
        impact_high = posterior_mu + z * posterior_sigma

        # Ensure ordering makes sense for direction
        if direction == "negative":
            impact_low, impact_high = min(impact_low, impact_high), max(impact_low, impact_high)
            # Cap at 0 for the optimistic end if impact is negative
            impact_high = min(impact_high, 5.0)
        elif direction == "positive":
            impact_low = max(impact_low, -5.0)

        reasoning.append(f"Posterior: {impact_mid:+.1f}% ({credible_interval}% CI: {impact_low:+.1f}% to {impact_high:+.1f}%)")

        return {
            "impact_low_pct": round(float(impact_low), 1),
            "impact_mid_pct": round(float(impact_mid), 1),
            "impact_high_pct": round(float(impact_high), 1),
            "credible_interval": credible_interval,
            "interval_method": f"bayesian_{credible_interval}pct",
            "direction": direction,
            "exposure": exposure,
            "prior": {
                "mu": round(float(prior["mu"]), 1),
                "sigma": round(float(prior["sigma"]), 1),
                "n_gold_labels": prior["n"],
            },
            "posterior": {
                "mu": round(float(posterior_mu), 1),
                "sigma": round(float(posterior_sigma), 1),
            },
            "reasoning": reasoning,
        }


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--event", required=True, type=click.Choice(EVENT_CATEGORIES))
    @click.option("--channel", default="", help="Impact channel")
    @click.option("--ticker", default="", help="Company ticker")
    @click.option("--exposure", default=None, type=float, help="Override exposure %")
    def main(event, channel, ticker, exposure):
        est = BayesianImpactEstimator()
        result = est.estimate(
            event_category=event,
            impact_channel=channel,
            ticker=ticker,
            exposure_pct=exposure,
        )

        print(f"\n{'='*60}")
        print(f"BAYESIAN IMPACT ESTIMATE")
        print(f"{'='*60}")
        print(f"  Event: {event}")
        print(f"  Company: {ticker or '(generic)'}")
        print(f"  Channel: {channel or '(auto)'}")
        print(f"\n  Exposure: {result['exposure']['exposure_level']} ({result['exposure']['exposure_pct']:.0f}%)")
        for src in result["exposure"]["exposure_sources"]:
            print(f"    → {src}")
        print(f"\n  Revenue impact: {result['impact_mid_pct']:+.1f}%")
        print(f"  {result['credible_interval']}% credible interval: {result['impact_low_pct']:+.1f}% to {result['impact_high_pct']:+.1f}%")
        print(f"\n  Reasoning chain:")
        for r in result["reasoning"]:
            print(f"    {r}")

    main()
