# Learning Day 6 — Semi-Supervised Labeling, Model Improvements, and Production Readiness

**Project:** Geopolitical Muscle ML Model  
**Date:** April 14, 2026  
**Written for:** Understanding the semi-supervised labeling pipeline, the 163→602 label growth strategy, and the cascade of model improvements it triggered

---

## Where We Left Off

At the end of Day 5 we had:
- **4 working models** with an end-to-end pipeline
- **163 seed labels** (hand-curated, human-reviewed)
- Event Classifier at **95.3%** on news text (fixed from 43.8%)
- Exposure Scorer at **0.601 macro F1** (limited by small dataset)
- Impact Estimator at **80.2% coverage** (widened quantiles)
- Strategy Recommender with **148 strategies** across 34 priority cells
- Everything git-versioned with 6 commits

The biggest remaining gap was Model 2 (Exposure Scorer) — only 163 labels across 10 channels meant some channels had just 5 examples.

---

## Part 1: The Semi-Supervised Labeling Pipeline (`auto_label.py`)

### The Idea

We have 163 gold-standard labels but need 500+. Manual labeling is slow. But we have **two large, untapped data sources** that can generate labels automatically:

1. **EDGAR mentions** (17,372 total, 1,133 non-boilerplate) — companies explicitly discussing geopolitical events in SEC filings
2. **Event studies** (1,973 records) — stock price reactions for ~99 companies across 20 events

If a company's stock dropped 30% during COVID AND its 10-K discusses "pandemic disruption," we can confidently label it as impacted.

### How It Works

The pipeline (`pipelines/auto_label.py`) generates candidate labels from two sources:

**Source 1: EDGAR mentions with high specificity + event links.**  
These are companies that explicitly discuss a geopolitical event in their SEC filings. Requirements:
- Mention specificity score ≥ 40 (not boilerplate)
- Successfully linked to a specific event_id
- Confidence = 0.4×specificity + 0.35×stock_reaction + 0.25×mention_count

**Source 2: Event studies with large stock reactions.**  
These are companies whose stock moved significantly during a geopolitical event. Requirements:
- |CAR_1_5| > 5% (stock moved at least 5% in 5 days)
- Not already captured from EDGAR mentions
- Confidence boosted by having EDGAR mentions and larger magnitude

### The Channel Assignment Problem

#### Roadblock 1: Model 2 Assigns Wrong Channels

The first version used Model 2 to predict the impact channel for each candidate. But Model 2 only had 0.601 macro F1 — it defaulted to `revenue_market_access` for everything. Result: 171 of 330 labels were `revenue_market_access`, making the imbalance worse.

**The fix: GICS sector-based channel assignment.** Instead of asking Model 2 "what channel?", we use economic logic:

| GICS Sector | Default Channel | Why |
|-------------|----------------|-----|
| Energy (10) | procurement_supply_chain | Energy companies are affected through input costs |
| Industrials (20) | logistics_operations | Manufacturing/transport firms face route/operational disruption |
| Consumer Disc. (25) | revenue_market_access | Consumer companies lose customers/markets |
| Consumer Staples (30) | revenue_market_access | Same as discretionary |
| Health Care (35) | regulatory_compliance_cost | Healthcare is regulation-driven |
| Financials (40) | financial_treasury | Banks/insurers face trapped capital, FX risk |
| Info Tech (45) | innovation_ip | Tech firms face export controls, IP restrictions |
| Comm Services (50) | regulatory_compliance_cost | Telecom/media face data localization |
| Utilities (55) | capital_allocation_investment | Utilities face asset impairments |
| Real Estate (60) | capital_allocation_investment | Real estate faces write-downs |

The mapping also considers the event category — e.g., tech companies during `armed_conflict_instability` get `procurement_supply_chain` (supply chain disruption), not `innovation_ip`.

This single change transformed the channel distribution from 171/330 revenue_market_access to a balanced spread across all 10 channels.

---

## Part 2: Growing from 163 to 602 Labels

### Phase 1: Broad Auto-Labeling (163 → 382)

First run generated 219 candidates:
- 52 from EDGAR mentions (companies that explicitly discuss the event)
- 167 from event studies (companies with >5% stock reactions)

Channels distributed using sector-aware logic. Every channel got new labels.

### Phase 2: Targeted Generation for Weak Channels (382 → 478)

Five channels were still underrepresented. We generated targeted labels by matching specific event-sector combinations:

| Target Channel | Event-Sector Strategy | Labels Added |
|---------------|----------------------|-------------|
| cybersecurity_it | Tech + financials during NotPetya | 8 |
| workforce_talent | Industrials + healthcare during India-Pakistan/Russia/COVID | 31 |
| reputation_stakeholder | Consumer-facing during Xinjiang boycott/Israel-Hamas/Russia | 28 |
| innovation_ip | Tech + healthcare during chip export controls | 20 |
| capital_allocation | Utilities + energy during Russia/COVID/energy crisis | 9 |

### Phase 3: Cybersecurity Boost + Underrepresented Events (478 → 602)

Cybersecurity was still at 18 labels. Added 22 more from:
- Financial companies during NotPetya (banks were heavily hit)
- Tech companies during Russia invasion / US-Iran war (cyber warfare component)

Then added 102 labels from 8 underrepresented events (Brexit, OPEC, Suez, Chile lithium, Panama mine, Xinjiang, Iran sanctions, energy crisis) to improve event diversity.

### Final Distribution

```
revenue_market_access:       104
procurement_supply_chain:     97
logistics_operations:         60
innovation_ip:                58
regulatory_compliance_cost:   57
financial_treasury:           51
capital_allocation:           46
workforce_talent:             45
reputation_stakeholder:       44
cybersecurity_it:             40
TOTAL:                       602
```

All channels at 40+. The weakest channel (cybersecurity_it at 40) has 8x more labels than before (was 5 in the original 163).

---

## Part 3: Model 2 Improvement Cascade

Each label expansion triggered a Model 2 retrain:

| Labels | Macro F1 | Worst Channel | Accuracy |
|--------|----------|---------------|----------|
| 163 (original) | 0.494 | financial_treasury 0.000 | 61.0% |
| 330 (first auto-label) | 0.539 | capital_allocation 0.250 | 69.4% |
| 382 (sector-aware) | 0.698 | capital_allocation 0.462 | 76.2% |
| 602 (full expansion) | **0.825** | reputation_stakeholder 0.706 | **82.6%** |

The biggest single-step improvement was the sector-aware channel assignment (0.539 → 0.698, +30%). The overall journey from 0.494 to 0.825 represents a **67% improvement** in macro F1.

### Per-Channel F1 Progression

| Channel | 163 labels | 602 labels |
|---------|:-:|:-:|
| revenue_market_access | 0.600 | **0.818** |
| procurement_supply_chain | 0.615 | **0.789** |
| logistics_operations | 0.500 | **0.880** |
| financial_treasury | 0.000 | **0.842** |
| regulatory_compliance | 0.400 | **0.857** |
| innovation_ip | 0.000 | **0.917** |
| workforce_talent | 0.000 | **0.842** |
| reputation_stakeholder | 0.000 | **0.706** |
| capital_allocation | 0.571 | **0.778** |
| cybersecurity_it | 0.800 | **0.824** |

Four channels went from literally 0.000 to 0.70+ F1. This is the power of targeted data generation.

---

## Part 4: Model 3 Retrained

The Impact Estimator was retrained with 573 seed labels (up from 134). Results:

| Metric | 134 labels | 573 labels |
|--------|:-:|:-:|
| Coverage (all) | 80.2% | **80.7%** |
| Coverage (seed labels) | 66.7% | **75.7%** |
| Event study MAE | 0.17 pp | **0.08 pp** |

Coverage held steady while seed label coverage improved 9 percentage points.

---

## Part 5: End-to-End Pipeline — Before and After

### Test: "Russian armed forces launched invasion of Ukraine" + Apple ($394B revenue)

**Day 5 output (broken):**
```
Step 1: political_transitions_volatility (85.7%)    ← WRONG category
Step 2: capital_allocation_investment (32.4%)        ← generic/wrong channel
Step 3: +0.1% to +1.0% (+$333M to +$3.9B)
Step 4: Investment staging, political risk insurance  ← irrelevant strategies
```

**Day 6 output (working):**
```
Step 1: armed_conflict_instability (96.2%)           ← CORRECT
Step 2: procurement_supply_chain (55.4%)             ← sensible for Apple
Step 3: -0.6% to +0.1% (-$2.2B to +$514M)           ← realistic range
Step 4: Supplier risk monitoring, strategic stockpiling ← directly relevant
```

Every step improved. The classifier correctly identifies the event, the scorer assigns a channel that makes economic sense for Apple (supply chain in Asia), the impact range is realistic (Apple has minimal direct Russia exposure), and the strategies are actionable.

---

## Part 6: Roadblock Summary for Day 6

| # | Roadblock | Solution |
|---|-----------|----------|
| 1 | Auto-labeler assigned all candidates to revenue_market_access | GICS sector-based channel assignment instead of Model 2 prediction |
| 2 | Cybersecurity channel had only 5→10 labels after first pass | Expanded to NotPetya + cyber-warfare events for tech/finance companies |
| 3 | 8 events had 0-2 auto-generated labels | Lowered stock reaction threshold, targeted generation for underrepresented events |

---

## Summary: Day 6 in One Paragraph

Day 6 built the semi-supervised labeling pipeline (`auto_label.py`) that grew seed labels from 163 to 602 using EDGAR filing mentions and event study stock reactions, with GICS sector-aware channel assignment ensuring balanced distribution across all 10 impact channels (minimum 40 per channel, up from 5). This triggered a cascade of model improvements: Model 2 (Exposure Scorer) macro F1 went from 0.494 to 0.825 (+67%), with four channels going from 0.000 to 0.70+ F1; Model 3 (Impact Estimator) seed label coverage improved from 66.7% to 75.7% while maintaining 80.7% overall coverage. The end-to-end pipeline now produces correct, economically sensible output — classifying "Russia invaded Ukraine" correctly as armed_conflict (96.2% confidence), assigning procurement_supply_chain as Apple's primary exposure channel, estimating a -$2.2B to +$514M impact range, and recommending supplier risk monitoring and strategic stockpiling as top strategies.
