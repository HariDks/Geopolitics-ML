# Geopolitical Muscle ML Model

## Project Overview

An ML-powered geopolitical risk and strategy engine that helps companies detect geopolitical events, quantify their business exposure, estimate financial impact, and generate ranked strategic responses. The model acts as an automated "geopolitical muscle" — the capability that the WEF/IMD/BCG white paper (January 2026) identifies as essential but only ~20% of global firms have built internally.

**Core value proposition:** Democratize what companies like Siemens ("Value at Stake" methodology), Allianz ("Political Stability Grid"), and Rio Tinto (cross-functional task forces) have built internally with 9-40 person teams — make it accessible via a model that any company can plug their data into.

**Problem being solved:** Companies can read geopolitical headlines, but they can't systematically connect those signals to their specific financial exposure and generate actionable responses. The WEF report found that most firms rely on qualitative narratives and heat maps. Only a small minority quantify exposure. This model bridges that gap.

---

## Phase 1: Intellectual Framework (COMPLETED)

### What was built

Phase 1 produced the foundational taxonomy and prioritization framework that defines what the model detects, how it classifies events, and how it maps them to business impact. All deliverables are in `geopolitical_muscle_matrix_v2.xlsx`.

### 1.1 Geopolitical Event Taxonomy (8 categories)

Every geopolitical event the model processes is classified into one of these 8 categories:

| # | Category | Sub-types | Speed of Onset | Typical Duration |
|---|----------|-----------|----------------|-----------------|
| 1 | **Trade Policy Actions** | Tariffs, quotas, export/import bans, trade agreement changes, customs reclassifications, rules of origin shifts | Days to months | Transient to structural |
| 2 | **Sanctions & Financial Restrictions** | Asset freezes, entity listings, SWIFT exclusions, secondary sanctions, sectoral sanctions | Hours to days | Structural |
| 3 | **Armed Conflict & Instability** | Interstate war, civil conflict, terrorism, piracy, coups, mass protests, insurgency | Minutes to hours | Variable (months to decades) |
| 4 | **Regulatory & Sovereignty Shifts** | Data localization, FDI screening, local content rules, nationalization, sector-specific regulations | Months to years | Structural |
| 5 | **Technology Controls** | Export controls (chips/AI/dual-use), forced tech transfer, IP restrictions, digital sovereignty mandates, standard fragmentation | Weeks to months | Structural (accelerating) |
| 6 | **Resource & Energy Disruptions** | Commodity supply shocks, energy weaponization, critical mineral access restrictions, OPEC-style coordination | Hours to months | Transient to structural |
| 7 | **Political Transitions & Volatility** | Elections, regime changes, policy reversals, populist governance shifts, constitutional crises | Days to months | Variable |
| 8 | **Institutional & Alliance Realignment** | WTO dysfunction, new trade blocs, military alliance shifts, multilateral treaty withdrawals | Months to years | Structural |

Each event also gets tagged with metadata:
- `affected_geographies`: ISO country codes
- `affected_sectors`: NAICS/GICS codes
- `onset_speed`: sudden / gradual / phased
- `expected_duration`: transient / short_term / medium_term / structural
- `escalation_probability`: float 0-1
- `severity_score`: int 1-5

### 1.2 Business Impact Channel Taxonomy (10 channels)

Every business impact is classified into one of these 10 channels:

| # | Impact Channel | Primary KPIs | Company-Specific Inputs Required |
|---|---------------|-------------|--------------------------------|
| 1 | **Procurement & Supply Chain** | Supplier concentration HHI, COGS variance %, lead time deviation, inventory cost delta | Tier 1-3 supplier list with locations, BOM origin data, inventory by SKU, logistics routes |
| 2 | **Revenue & Market Access** | Revenue-at-risk by geography, market share delta %, pricing elasticity, customer concentration | Revenue by country/region, customer list by geography, competitive market share |
| 3 | **Capital Allocation & Investment** | Capex reallocation cost, asset impairment probability, M&A risk premium, ROI sensitivity | Asset register with locations, capex pipeline, M&A pipeline, capacity utilization |
| 4 | **Regulatory Compliance Cost** | Compliance spend increase, license delay (months), penalty exposure, FTE increase | Current compliance costs, license inventory, regulatory filing calendar |
| 5 | **Logistics & Operations** | Route disruption cost, transit time increase, downtime hours, customs delay | Shipping routes/volumes, warehouse locations, production schedules |
| 6 | **Innovation & IP** | R&D location risk score, tech restriction probability, patent variance, collaboration impact | R&D facility locations, tech dependencies, patent portfolio, research partnerships |
| 7 | **Workforce & Talent** | Visa impact (headcount), hiring cost premium %, evacuation readiness, talent pipeline | Employee distribution, visa inventory, critical role mapping, local labor data |
| 8 | **Reputation & Stakeholder Mgmt** | Boycott probability, ESG rating impact, gov relationship score, media sentiment | Brand perception data, ESG ratings, gov relationship map, media monitoring |
| 9 | **Financial & Treasury** | Currency VaR, repatriation risk %, counterparty default prob, insurance cost change | FX exposure, intercompany flows, counterparty list, insurance policies |
| 10 | **Cybersecurity & IT Infrastructure** | Threat level score, sovereignty compliance cost, migration cost, incident probability | IT architecture map, data flows, vendor dependencies, cyber insurance |

### 1.3 Priority Matrix (8x10 = 80 cells)

Each cell in the Event Type x Impact Channel matrix has a **Frequency (1-5) x Severity (1-5) = Priority Score (1-25)**.

Scores were expert-assessed and revised based on recent geopolitical patterns (v2 revisions specifically upgraded Armed Conflict and Technology Controls categories based on Russia-Ukraine, Red Sea, semiconductor controls evidence).

**Current score distribution:**
- 8 cells at score 25 (critical)
- 26 cells at score 20+ (high critical)
- 34 cells at score 15+ (high priority — all fully documented)
- Remaining 46 cells at score <15 (medium/lower priority)

**Full frequency and severity scores (row = event, col = impact channel):**

```
Frequency scores [event_idx][impact_idx]:
Trade Policy:       [5, 5, 4, 4, 5, 3, 2, 3, 4, 2]
Sanctions:          [4, 4, 3, 5, 3, 3, 2, 4, 5, 3]
Armed Conflict:     [4, 4, 3, 3, 5, 2, 4, 4, 4, 4]
Regulatory:         [4, 3, 4, 5, 3, 4, 3, 3, 3, 4]
Tech Controls:      [5, 4, 4, 4, 2, 5, 3, 3, 2, 5]
Resource/Energy:    [5, 3, 3, 2, 4, 2, 2, 3, 4, 1]
Political:          [3, 4, 4, 3, 2, 2, 3, 4, 3, 2]
Institutional:      [2, 3, 3, 3, 2, 2, 2, 3, 3, 2]

Severity scores [event_idx][impact_idx]:
Trade Policy:       [5, 5, 4, 3, 4, 3, 2, 3, 4, 2]
Sanctions:          [5, 5, 4, 4, 3, 3, 3, 5, 5, 3]
Armed Conflict:     [5, 5, 5, 3, 5, 2, 5, 5, 5, 5]
Regulatory:         [3, 3, 4, 4, 2, 4, 3, 2, 3, 4]
Tech Controls:      [5, 5, 4, 4, 3, 5, 4, 3, 2, 5]
Resource/Energy:    [5, 4, 4, 2, 5, 2, 2, 3, 5, 1]
Political:          [3, 5, 5, 3, 2, 2, 3, 5, 4, 2]
Institutional:      [3, 4, 4, 3, 3, 3, 2, 3, 4, 2]
```

Impact channel order: [Procurement, Revenue, Capital, Regulatory, Logistics, Innovation, Workforce, Reputation, Financial, Cybersecurity]

### 1.4 Top 34 Cell Documentation

Each of the 34 cells scoring 15+ has been documented with:
- **Transmission mechanism**: How the event type affects the business channel (causal pathway)
- **Leading indicators**: What signals predict this impact before it materializes
- **Response archetypes**: Categories of strategic action that mitigate or capitalize on it
- **Historical example**: Real case with company names and outcomes

Full documentation is in Sheet 2 ("Top Priority Cells") of the Excel deliverable.

### 1.5 Time-Weighted Frequency Methodology

Static frequency scores are replaced with exponential decay-weighted historical event counts to make the matrix adaptive.

**Core formula:** `Weight = e^(-λ * t)` where `t` = years since event, `λ` = category-specific decay rate.

**Recommended decay rates (λ) by event category:**

| Category | λ | Weight at 1yr | Weight at 3yr | Rationale |
|----------|---|---------------|---------------|-----------|
| Trade Policy Actions | 0.5 | 0.61 (61%) | 0.22 (22%) | Recurring, accelerating tariff waves |
| Sanctions & Financial Restrictions | 0.4 | 0.67 (67%) | 0.30 (30%) | Rarely reversed; historical context stays relevant |
| Armed Conflict & Instability | 0.7 | 0.50 (50%) | 0.12 (12%) | Episodic, location-specific; recent acceleration |
| Regulatory & Sovereignty Shifts | 0.3 | 0.74 (74%) | 0.41 (41%) | Structural and cumulative |
| Technology Controls | 0.6 | 0.55 (55%) | 0.17 (17%) | Accelerating rapidly; new controls every 6-12 months |
| Resource & Energy Disruptions | 0.6 | 0.55 (55%) | 0.17 (17%) | Cyclical but energy transition adds structural layer |
| Political Transitions & Volatility | 0.8 | 0.45 (45%) | 0.09 (9%) | Highly time-bound; past elections less predictive |
| Institutional & Alliance Realignment | 0.2 | 0.82 (82%) | 0.55 (55%) | Glacial and cumulative |

**Implementation:**
1. For each matrix cell, sum decay-weighted counts of all mapped historical events
2. Normalize across all 80 cells using percentile bins → 1-5 scale
3. Recalculate monthly/quarterly
4. Flag cells with >20% score change for review
5. After 6-12 months, backtest λ values against actual corporate impact data and tune

### 1.6 ML Data Schema

Defines the exact field structure for model input/output. Three sections:

**Event Identification Fields:**
```
event_id: string                    # "EVT-2025-TARIFF-US-CN-042"
event_category: enum(8)             # one of the 8 event types
event_subtype: string               # specific action within category
event_date: datetime                # ISO format
affected_geographies: list[string]  # ISO country codes
affected_sectors: list[string]      # NAICS/GICS codes
onset_speed: enum(3)                # sudden / gradual / phased
expected_duration: enum(4)          # transient / short / medium / structural
escalation_probability: float(0-1)
severity_score: int(1-5)
time_weighted_frequency: float      # decay-weighted score
lambda_decay_rate: float            # category-specific λ used
```

**Company Exposure Input Fields:**
```
company_id: string
impact_channel: enum(10)            # one of the 10 impact channels
revenue_exposure_pct: float(0-1)    # % revenue from affected geographies
supplier_concentration_hhi: float(0-1)
cogs_at_risk_usd: float
asset_value_exposed_usd: float
employee_count_exposed: int
alternative_supplier_readiness: float(0-1)
historical_disruption_count: int
mitigation_maturity_score: float(0-1)
```

**Model Output Fields (Impact Assessment):**
```
financial_impact_low_usd: float
financial_impact_mid_usd: float
financial_impact_high_usd: float
impact_timeline_days: int
impact_duration_months: int
confidence_score: float(0-1)
```

**Model Output Fields (Strategy Recommendations):**
```
strategy_id: string
strategy_name: string
strategy_category: enum(6)         # mitigate / hedge / exit / capture / engage / monitor
implementation_cost_usd: float
implementation_time_months: int
risk_reduction_pct: float(0-1)
feasibility_score: float(0-1)
precedent_count: int
precedent_success_rate: float(0-1)
```

---

## Phase 2: Data Pipeline & Base Model (TO BE BUILT)

Phase 2 has three parallel workstreams that converge into a 4-model architecture.

### 2.1 Workstream 1: Historical Geopolitical Event Database

**Objective:** Build a structured, timestamped database of geopolitical events tagged with the Phase 1 taxonomy, covering at minimum 2015-2025.

#### Data Sources to Ingest

**Source 1: GDELT (Global Database of Events, Language, and Tone)**
- URL: https://www.gdeltproject.org/
- Python library: `gdeltPyR` (pip install gdeltPyR)
- Coverage: Global, real-time, back to 1979
- Update frequency: Every 15 minutes
- Key fields to extract: SQLDATE, Actor1CountryCode, Actor2CountryCode, EventCode (CAMEO taxonomy), GoldsteinScale (severity proxy), NumMentions, AvgTone, ActionGeo_CountryCode, SOURCEURL
- **Critical:** GDELT is extremely noisy. Must apply aggressive filtering:
  - Filter to CAMEO root codes relevant to geopolitical events (see mapping table below)
  - Filter to minimum NumMentions threshold (suggest >= 10) to remove noise
  - Filter to GoldsteinScale <= -5 for conflict/tension events
  - Deduplicate by event cluster (same event reported by multiple sources)

**CAMEO to Event Taxonomy mapping (build this):**
```
CAMEO 13x (THREATEN) → multiple categories depending on subcode
CAMEO 14x (PROTEST) → Political Transitions & Volatility
CAMEO 15x (EXHIBIT FORCE) → Armed Conflict & Instability
CAMEO 17x (COERCE) → Sanctions & Financial Restrictions / Trade Policy Actions
CAMEO 18x (ASSAULT) → Armed Conflict & Instability
CAMEO 19x (FIGHT) → Armed Conflict & Instability
CAMEO 20x (USE UNCONVENTIONAL MASS VIOLENCE) → Armed Conflict & Instability

Additional mappings needed for:
- Trade policy events (tariff announcements, trade agreement changes)
- Technology control events (export control announcements)
- Resource/energy events (OPEC decisions, commodity disruptions)
- Regulatory shifts (FDI screening, data localization)
- Institutional realignment (treaty changes, org reforms)

These categories are poorly covered by CAMEO and need supplementary sources.
```

**Source 2: ACLED (Armed Conflict Location & Event Data)**
- URL: https://acleddata.com/
- API access: Free for research (requires registration)
- Coverage: Global, back to 1997
- Key fields: event_date, event_type, sub_event_type, actor1, actor2, country, admin1, location, fatalities, notes
- Maps directly to: Armed Conflict & Instability category
- ACLED event_type mapping:
  ```
  Battles → Armed Conflict (severity based on fatalities)
  Explosions/Remote violence → Armed Conflict
  Violence against civilians → Armed Conflict
  Protests → Political Transitions & Volatility
  Riots → Political Transitions & Volatility / Armed Conflict
  Strategic developments → multiple categories
  ```

**Source 3: Global Trade Alert (GTA)**
- URL: https://www.globaltradealert.org/
- Coverage: Trade-distorting policy interventions since November 2008
- Key fields: intervention_type, implementing_jurisdiction, affected_jurisdictions, affected_sectors, date_announced, date_implemented, gta_evaluation (harmful/liberalizing)
- Maps to: Trade Policy Actions (primary), Sanctions & Financial Restrictions (secondary)
- **This is the single best source for trade policy events.** Already categorized and timestamped.

**Source 4: OFAC Sanctions Data**
- URL: https://sanctionslist.ofac.treas.gov/ (SDN list)
- Also: EU Consolidated Financial Sanctions List
- Key fields: entity_name, sanction_type, effective_date, program (country/issue), SDN_type
- Maps to: Sanctions & Financial Restrictions
- Parse the XML/CSV feeds and extract addition dates

**Source 5: BIS Entity List & Export Controls**
- URL: https://www.bis.doc.gov/index.php/the-denied-persons-list
- Coverage: US export control entity list additions and rule changes
- Maps to: Technology Controls
- Supplement with Federal Register scraping for rule change announcements

**Source 6: UN Comtrade (for trade flow baseline)**
- URL: https://comtrade.un.org/
- Use for: Establishing baseline trade flows by country x product category
- This isn't an event source — it's used to compute supply chain exposure metrics

#### Event Database Schema

```sql
CREATE TABLE geopolitical_events (
    event_id VARCHAR PRIMARY KEY,        -- "EVT-{source}-{date}-{seq}"
    source VARCHAR NOT NULL,             -- "gdelt" / "acled" / "gta" / "ofac" / "bis" / "manual"
    source_event_id VARCHAR,             -- original ID from source
    event_category VARCHAR NOT NULL,     -- one of 8 taxonomy categories
    event_subtype VARCHAR,               -- specific action type
    event_date DATE NOT NULL,
    event_end_date DATE,                 -- NULL if ongoing or point event
    affected_countries TEXT[],           -- ISO 3166-1 alpha-2 codes
    affected_sectors TEXT[],             -- NAICS or GICS codes
    severity_estimate INT CHECK (1-5),
    onset_speed VARCHAR,                 -- "sudden" / "gradual" / "phased"
    expected_duration VARCHAR,           -- "transient" / "short" / "medium" / "structural"
    description_text TEXT,
    source_url VARCHAR,
    goldstein_scale FLOAT,              -- GDELT-specific, NULL for other sources
    fatalities INT,                      -- ACLED-specific, NULL for other sources
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_events_date ON geopolitical_events(event_date);
CREATE INDEX idx_events_category ON geopolitical_events(event_category);
CREATE INDEX idx_events_countries ON geopolitical_events USING GIN(affected_countries);
```

#### Time-Weighted Frequency Implementation

```python
import numpy as np
from datetime import datetime

LAMBDA_BY_CATEGORY = {
    "trade_policy_actions": 0.5,
    "sanctions_financial_restrictions": 0.4,
    "armed_conflict_instability": 0.7,
    "regulatory_sovereignty_shifts": 0.3,
    "technology_controls": 0.6,
    "resource_energy_disruptions": 0.6,
    "political_transitions_volatility": 0.8,
    "institutional_alliance_realignment": 0.2,
}

def compute_time_weighted_frequency(events, category, reference_date=None):
    """
    Compute decay-weighted frequency score for a set of events.
    
    Args:
        events: list of event dicts with 'event_date' field
        category: event category string (for lambda lookup)
        reference_date: date to compute weights from (default: today)
    
    Returns:
        raw_weighted_score: float (sum of decay weights)
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    lam = LAMBDA_BY_CATEGORY[category]
    total_weight = 0.0
    
    for event in events:
        t_years = (reference_date - event['event_date']).days / 365.25
        weight = np.exp(-lam * t_years)
        total_weight += weight
    
    return total_weight

def normalize_to_scale(raw_scores, scale_min=1, scale_max=5):
    """
    Normalize raw weighted scores to 1-5 scale using percentile bins.
    P80+ = 5, P60-80 = 4, P40-60 = 3, P20-40 = 2, P0-20 = 1
    """
    percentiles = np.percentile(raw_scores, [20, 40, 60, 80])
    normalized = []
    for score in raw_scores:
        if score >= percentiles[3]:
            normalized.append(5)
        elif score >= percentiles[2]:
            normalized.append(4)
        elif score >= percentiles[1]:
            normalized.append(3)
        elif score >= percentiles[0]:
            normalized.append(2)
        else:
            normalized.append(1)
    return normalized
```

### 2.2 Workstream 2: Corporate Outcome Dataset

**Objective:** Build labeled training data linking geopolitical events to measurable corporate financial impact.

#### Source 1: SEC EDGAR Earnings Call Transcripts

- All US public companies file 8-K with earnings materials
- Use EDGAR full-text search API or bulk download
- **NLP Pipeline:**
  1. Download transcripts for S&P 500 companies, 2020-2025 (richest period for geopolitical mentions)
  2. Extract paragraphs mentioning geopolitical terms using keyword matching + NER:
     - Trade/tariff terms: "tariff", "trade war", "customs duty", "import ban", "export restriction", "trade policy"
     - Sanctions terms: "sanction", "OFAC", "entity list", "embargo", "SWIFT"
     - Conflict terms: "conflict", "war", "Red Sea", "shipping disruption", "military"
     - Regulatory terms: "regulatory change", "data localization", "foreign investment screening", "local content"
     - Tech control terms: "export control", "chip ban", "technology restriction", "CHIPS Act", "dual-use"
     - Energy terms: "energy crisis", "commodity price", "OPEC", "oil price", "energy security"
  3. For each mention, extract:
     - Company ticker and name
     - Quarter and date
     - Event category (classify the mention)
     - Impact channel (which business function is discussed)
     - Quantitative impact if mentioned (dollar amounts, percentage changes)
     - Sentiment/tone of the mention
  4. Link to actual financial performance:
     - Pull quarterly financials from EDGAR XBRL (revenue, COGS, operating income, capex)
     - Compute quarter-over-quarter and year-over-year deltas
     - This gives you labeled pairs: (geopolitical_mention, financial_outcome)

#### Source 2: Stock Price Event Studies

- For each major event in the event database, compute abnormal returns:
  ```
  Abnormal Return = Actual Return - Expected Return (from market model)
  CAR[-1, +5] = Cumulative Abnormal Return from day -1 to day +5
  ```
- Segment by sector (GICS) and geography exposure
- Use as a market-implied severity score
- Data source: Yahoo Finance API (free) or Alpha Vantage

#### Source 3: 10-K Risk Factor Analysis

- Parse "Risk Factors" section from annual filings
- Track which geopolitical risks companies explicitly identify
- Year-over-year comparison shows emerging vs receding concerns
- Use SEC EDGAR XBRL structured data where available

#### Corporate Outcome Schema

```sql
CREATE TABLE corporate_impacts (
    impact_id VARCHAR PRIMARY KEY,
    event_id VARCHAR REFERENCES geopolitical_events(event_id),
    company_ticker VARCHAR NOT NULL,
    company_name VARCHAR,
    sector_gics VARCHAR,
    impact_channel VARCHAR NOT NULL,     -- one of 10 channels
    quarter VARCHAR,                     -- "2025Q2"
    
    -- Qualitative
    mention_text TEXT,                   -- extracted paragraph from earnings call
    mention_sentiment FLOAT,            -- -1 to 1
    management_action_described TEXT,    -- what the company said they're doing
    
    -- Quantitative (from financial data)
    revenue_delta_pct FLOAT,
    cogs_delta_pct FLOAT,
    operating_income_delta_pct FLOAT,
    capex_delta_pct FLOAT,
    
    -- Market reaction
    car_1_5 FLOAT,                      -- cumulative abnormal return [-1, +5]
    car_1_30 FLOAT,                     -- cumulative abnormal return [-1, +30]
    
    -- Metadata
    source VARCHAR,                     -- "earnings_call" / "10k" / "event_study" / "manual"
    confidence VARCHAR,                 -- "high" / "medium" / "low"
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Manual Seed Labels

Start with ~200-300 high-quality event-impact pairs from well-documented cases. Sources for manual labeling:

1. **WEF report case studies** (~20 pairs): Teva oncology drug stockpiling, automotive $4B tariff impact, EDF M&A navigation, Rio Tinto tariff task force, etc.
2. **Major documented events** (~100 pairs): Russia-Ukraine corporate exits (compiled by Yale CELI tracker), Red Sea shipping rerouting impacts (Maersk, MSC earnings calls), US-China semiconductor impacts (NVIDIA, AMD, TSMC earnings calls)
3. **Sector-specific case studies** (~100 pairs): From industry publications, consulting firm reports, academic papers

These seed labels train the semi-supervised classifier that labels the remaining data at scale.

### 2.3 Workstream 3: Model Architecture

Build 4 models in sequence, each feeding the next.

#### Model 1: Event Classifier

**Purpose:** Classify raw text (news, government announcements) into the 8 event categories with severity and metadata extraction.

**Architecture:** Fine-tuned transformer (DistilBERT or similar) for classification + structured extraction.

**Alternative approach:** Use an LLM (Claude API or open-source like Llama) with structured output prompting for initial pass, then distill into a smaller model for production speed and cost.

**Input:** Raw text string (news headline + first 2 paragraphs)

**Output:**
```json
{
  "event_category": "trade_policy_actions",
  "event_subtype": "tariff_increase",
  "severity_score": 4,
  "affected_countries": ["US", "CN"],
  "affected_sectors": ["3361", "3344"],
  "onset_speed": "sudden",
  "confidence": 0.87
}
```

**Training data:** Use the labeled event database from Workstream 1. The GDELT/ACLED/GTA events already have source text and your mapped categories.

**Evaluation:** Precision/recall by category. Target >0.85 F1 on held-out test set.

#### Model 2: Exposure Scorer

**Purpose:** Given event characteristics + company-specific data, score how exposed the company is across each impact channel.

**Architecture:** Gradient boosted trees (XGBoost or LightGBM) — structured tabular data, not NLP.

**Features:**
```
- Geographic overlap: % of company revenue/suppliers/assets in affected countries
- Sector match: binary/weighted match between event affected sectors and company sector
- Supplier concentration: HHI of supplier geography (higher = more exposed)
- Revenue concentration: HHI of revenue geography
- Asset exposure: % of total assets in affected regions
- Historical disruption count: times company was hit by similar events before
- Alternative readiness: pre-qualified alternative supplier coverage ratio
- Event severity: from Model 1
- Event category: one-hot encoded
- Impact channel: one-hot encoded
```

**Target variable:** Exposure score (0-1), derived from corporate outcome dataset (financial impact magnitude normalized by company size).

**Training data:** Corporate outcome dataset from Workstream 2.

#### Model 3: Impact Estimator

**Purpose:** Estimate financial impact range (low/mid/high) given event + exposure.

**Architecture:** Quantile regression (to produce the low/mid/high range) on top of exposure features.

**Input:** Event characteristics + exposure scores from Model 2 + company financials (revenue, COGS, operating income as baseline).

**Output:**
```json
{
  "financial_impact_low_usd": 45000000,
  "financial_impact_mid_usd": 120000000,
  "financial_impact_high_usd": 280000000,
  "impact_timeline_days": 30,
  "impact_duration_months": 18,
  "confidence_score": 0.72
}
```

**Training data:** Corporate outcome dataset — specifically the quantitative financial deltas linked to events.

**Evaluation:** Calibration of prediction intervals (do 80% of actual outcomes fall within the low-high range?).

#### Model 4: Strategy Recommender

**Purpose:** Given event type + impact channel + company exposure, recommend ranked strategic responses.

**Architecture:** Initially retrieval-based (not ML), evolving to learned ranking.

**Phase 2A (retrieval-based):**
1. Start with the response archetypes documented in Phase 1's top 34 cells as the candidate strategy set
2. Build a structured strategy database:
   ```sql
   CREATE TABLE strategies (
       strategy_id VARCHAR PRIMARY KEY,
       event_category VARCHAR,
       impact_channel VARCHAR,
       strategy_name VARCHAR,
       strategy_category VARCHAR,  -- mitigate/hedge/exit/capture/engage/monitor
       description TEXT,
       typical_cost_range VARCHAR, -- "low/medium/high" or dollar range
       implementation_time VARCHAR,
       prerequisites TEXT[],       -- capabilities company needs
       historical_precedents TEXT[],
       success_conditions TEXT
   );
   ```
3. Match strategies to the company's event-impact cell
4. Rank by feasibility (based on company size, sector, existing capabilities)
5. Return top 3-5 with implementation details

**Phase 2B (learned ranking — after sufficient outcome data):**
- Train a learning-to-rank model on which strategies companies actually implemented and their outcomes
- Features: company profile, event characteristics, strategy characteristics
- Target: strategy effectiveness (risk reduction achieved / cost)

### 2.4 Implementation Priority and Timeline

#### Weeks 1-2: Data Infrastructure

```
Tasks:
- [ ] Set up PostgreSQL database with schemas above (or use SQLite for prototyping)
- [ ] Write GDELT ingestion pipeline using gdeltPyR
  - Filter to relevant CAMEO codes
  - Apply NumMentions >= 10 filter
  - Deduplicate
  - Store in event database
- [ ] Pull ACLED data via API for 2020-2025
- [ ] Download Global Trade Alert dataset
- [ ] Download OFAC SDN list (XML feed)
- [ ] Build CAMEO → event taxonomy mapping table
- [ ] Build ACLED event_type → event taxonomy mapping table
- [ ] Build GTA intervention_type → event taxonomy mapping table

Deliverable: Event database populated with 2020-2025 historical events, properly tagged
```

#### Weeks 3-4: Taxonomy Validation

```
Tasks:
- [ ] Run time-weighted frequency calculation on real event data
- [ ] Compare computed scores to expert-assigned scores from Phase 1
- [ ] Identify discrepancies and investigate (are expert scores wrong, or is the data noisy?)
- [ ] Tune λ parameters if needed
- [ ] Generate first "live" priority matrix from real data
- [ ] Validate that top cells match intuition (trade policy x supply chain should be high)

Deliverable: Validated time-weighted frequency scores; confidence that taxonomy mapping is correct
```

#### Weeks 5-7: Corporate Outcome Data

```
Tasks:
- [ ] Build SEC EDGAR earnings call scraper for S&P 500 (2022-2025)
- [ ] Build NLP pipeline for geopolitical mention extraction
  - Keyword matching + context window extraction
  - Classify mentions into event categories and impact channels
  - Extract any quantitative impact figures
- [ ] Pull quarterly financial data (revenue, COGS, operating income) from EDGAR XBRL
- [ ] Compute financial deltas and link to geopolitical mentions
- [ ] Run event studies for top 20 geopolitical events (stock price abnormal returns)
- [ ] Manually label 200-300 seed event-impact pairs from WEF report and documented cases

Deliverable: Corporate outcome dataset with 1000+ event-impact pairs (mix of automated + manual labels)
```

#### Weeks 8-10: Model Training

```
Tasks:
- [ ] Train Model 1 (Event Classifier) — fine-tune DistilBERT or use LLM with structured output
- [ ] Train Model 2 (Exposure Scorer) — XGBoost on company exposure features
- [ ] Train Model 3 (Impact Estimator) — quantile regression
- [ ] Build Model 4 (Strategy Recommender) — retrieval-based from Phase 1 strategy database
- [ ] Build end-to-end pipeline: raw text → event classification → exposure scoring → impact estimate → strategy recommendation
- [ ] Evaluate each model independently and in pipeline

Deliverable: Working end-to-end pipeline that takes a geopolitical event description and company profile and outputs impact estimates + ranked strategies
```

### 2.5 Tech Stack

```
Language: Python 3.10+
Database: PostgreSQL (production) or SQLite (prototyping)
Data processing: pandas, numpy
NLP: transformers (HuggingFace), spaCy for NER, sentence-transformers for embeddings
ML: scikit-learn, XGBoost/LightGBM, PyTorch (for transformer fine-tuning)
Data sources: gdeltPyR, requests (for APIs), beautifulsoup4 (for scraping)
Financial data: sec-edgar-downloader, yfinance
Visualization: matplotlib, plotly
API (if serving): FastAPI
```

### 2.6 Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GDELT noise makes event classification unreliable | Low precision on event detection | Aggressive filtering (NumMentions, GoldsteinScale); supplement with curated sources (GTA, OFAC) |
| Insufficient labeled corporate outcome data | Model 2 and 3 undertrained | Start with manual seed labels; use semi-supervised learning; augment with stock price event studies |
| Company-specific inputs hard to obtain for training | Can't train exposure scorer without company data | Use public company data (10-K disclosures) as proxy; design model to work with partial inputs |
| Lambda values poorly calibrated | Time-weighted scores don't match reality | Backtest against known high-impact events; expert review of outliers; tune after 6 months |
| Strategy recommender lacks outcome data | Can't validate which strategies work | Start retrieval-based (no outcome data needed); evolve to learned ranking as outcome data accumulates |

---

## File Structure

```
geopolitical-muscle/
├── README.md                          # This file
├── data/
│   ├── raw/                           # Raw data from sources
│   │   ├── gdelt/
│   │   ├── acled/
│   │   ├── gta/
│   │   ├── ofac/
│   │   └── edgar/
│   ├── processed/                     # Cleaned, mapped data
│   │   ├── event_database.parquet
│   │   └── corporate_outcomes.parquet
│   └── mappings/                      # Taxonomy mapping tables
│       ├── cameo_to_taxonomy.json
│       ├── acled_to_taxonomy.json
│       └── gta_to_taxonomy.json
├── models/
│   ├── event_classifier/
│   ├── exposure_scorer/
│   ├── impact_estimator/
│   └── strategy_recommender/
├── pipelines/
│   ├── ingest_gdelt.py
│   ├── ingest_acled.py
│   ├── ingest_gta.py
│   ├── ingest_ofac.py
│   ├── ingest_edgar.py
│   ├── compute_frequency_scores.py
│   └── run_event_studies.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_taxonomy_validation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_pipeline_evaluation.ipynb
├── config/
│   ├── taxonomy.json                  # Event categories + impact channels
│   ├── lambda_rates.json              # Decay rates by category
│   └── priority_matrix.json           # Current frequency x severity scores
├── phase1_deliverables/
│   └── geopolitical_muscle_matrix_v2.xlsx
├── requirements.txt
└── docker-compose.yml                 # PostgreSQL + app
```

---

## Reference: WEF Report Key Findings Used

Source: "Building Geopolitical Muscle: How Companies Turn Insights into Strategic Advantage" — WEF, IMD Business School, BCG (January 2026)

- 56 executive interviews across multiple industries and geographies
- Global uncertainty in 2025 reached 4x the 2008 financial crisis level (World Uncertainty Index)
- 82% of chief economists rated uncertainty as "very high" (April 2025)
- Most disruptive areas: trade/global value chains and international economic institutions
- <20% of companies have a dedicated geopolitics unit
- >50% house geopolitical capability within government/corporate affairs
- Median core geopolitical team size: ~9 people
- 5 building blocks: Mandate, Radar/Sonar, Operating Model, Talent, Decision Integration
- 4 operating model archetypes: Watch Tower, Influence Network, Command Cell(s), Nerve Centre
- Key case studies: Siemens, Allianz, EDF, Rio Tinto, Teva Pharmaceutical, Philips, Nissan, Airbus, LATC
