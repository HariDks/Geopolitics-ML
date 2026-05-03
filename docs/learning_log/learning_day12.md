# Learning Day 12 — Fixing the Snorkel Pipeline and Wiring Everything Together

**Project:** Geopolitical Muscle ML Model  
**Date:** May 3, 2026  
**Written for:** Understanding how we diagnosed and fixed the weak supervision pipeline, connected disconnected components, and expanded the strategy knowledge base

---

## Where We Left Off

At the end of Day 11, we had completed the 4-week improvement sprint on paper:
- Temporal splits, conformal prediction, feedback loop, classifier calibration
- Model 3A/3B split, OpenSanctions, macro indices
- Embedding backup classifier, RAG recommender
- Snorkel weak supervision (307 labels), expanded company universe

But the honest assessment was: many of these were "bare minimum" implementations. The Snorkel pipeline had 2,345 labels heavily skewed toward one channel (78% logistics_operations), several components were built but not connected to the dashboard, and the case study database had only 21 entries.

Day 12 was about making it all actually work together.

---

## Part 1: The Snorkel Channel Skew Problem

### What We Started With

The Snorkel weak supervision pipeline had 15 labeling functions generating 2,345 labels. But the distribution was:

| Channel | Count | % |
|---------|-------|---|
| logistics_operations | 1,756 | 78% |
| revenue_market_access | 451 | 20% |
| procurement_supply_chain | 33 | 1.5% |
| innovation_ip | 13 | 0.6% |

Four channels out of ten, with one channel dominating. A model trained on this would basically just predict "logistics_operations" for everything.

### Attempt 1: Add More EDGAR Text Data

**The theory:** The text-based labeling functions (which look for keywords like "supply chain", "ransomware", "sanctions" in company filings) had very low coverage because we'd only ingested EDGAR filings for 99 S&P 100 companies. The other 1,200+ tickers had event study data (stock reactions) but no filing text.

**What we did:** Ran the EDGAR ingestion pipeline for 140 geopolitically-relevant tickers (defense, energy, shipping, semiconductors, mining). Extracted 20,617 new mentions, then linked them to events and scored specificity.

**The result:** Procurement_supply_chain tripled from 33 to 100. But the overall improvement was modest — only 88 new labels.

**Why it didn't work well:** 97% of the new EDGAR mentions scored as "boilerplate" by the specificity scorer. Most 10-K filings mention "geopolitical risk" generically in Item 1A without describing specific actions. The text-based LFs need specific keywords ("supply chain disruption", "OFAC sanctions") AND high specificity scores, and most filings don't have both.

### Attempt 2: Add More Events in Underrepresented Categories

**The theory:** The event database was dominated by armed conflict events (6 out of 20). Sanctions, regulatory, and trade categories had 1-2 events each. More diverse events would produce more diverse labels.

**What we did:** Added 15 new events across four underrepresented categories:
- 3 sanctions events (Russia sanctions, Iran reimposition)
- 5 regulatory events (GDPR, China tech crackdown, CFIUS, app bans, 737 MAX)
- 4 trade events (Japan-Korea, Australia-China, EU-China EV tariffs)
- 3 energy events (gallium controls, nickel ban, grain crisis)

Ran event studies for all 1,456 tickers across these 15 new events: 19,149 new records.

**The result:** The candidate pool grew from 25K to 44K, but Snorkel labels barely moved — from 2,341 to 2,345. The new categories produced thousands of LF votes, but Snorkel still wasn't confident about them.

### The Root Cause: LF Design Flaw

This is where we stopped and diagnosed the actual problem.

We ran an analysis of every confident Snorkel label, breaking down which labeling functions voted and whether they agreed or disagreed:

- **16,699 candidates** had only 1 LF voting — Snorkel can't be confident with a single vote
- **2,567 candidates** had 2+ LFs that **disagreed** — Snorkel was uncertain
- Only **472** had 2+ LFs that **agreed** — these became confident labels

**The systematic conflict:** The stock-reaction LFs (`lf_large_negative_car`, `lf_large_positive_car`) always voted `revenue_market_access` regardless of event type. So when a sanctions event caused a big stock drop:
- `lf_sanctions_event` → `financial_treasury` (correct)
- `lf_large_negative_car` → `revenue_market_access` (blind guess)

Two LFs, two different answers, every single time. Snorkel saw disagreement and threw the label out.

**The key insight:** The stock-reaction LFs were answering two questions at once:
1. "Was this company affected?" (yes — stock dropped) — **they had real signal for this**
2. "Through which channel?" (revenue_market_access) — **they had zero signal for this, just a hardcoded guess**

By forcing an answer to question 2, they systematically conflicted with every event-category LF that gave a different channel answer.

### Attempt 3: Make Stock LFs Agree with Event LFs (Rubber-Stamping)

**The idea:** If the stock-reaction LF copied the event-category LF's channel answer instead of always guessing `revenue_market_access`, they'd agree and Snorkel would be confident.

**The result:** Labels jumped from 2,345 to 3,073 with much better distribution. But when we audited:

```
event_plus_stock (rubber-stamp pattern): 2,949 (96.0%)
Genuine independent signals:              124  (4.0%)
```

**96% of the labels were just one signal counted twice.** The stock LF wasn't providing new information — it was echoing the event LF's answer. We had seen the same problem before when trying to expand to 26 LFs, and the user correctly rejected it.

**We reverted this change.**

### The Fix: Two-Stage Pipeline

Instead of fixing the LFs within Snorkel, we redesigned the pipeline architecture:

**Stage 1 — Filter (no Snorkel):** "Did this company's stock move meaningfully after this event?" If `abs(car_1_5) > 2.5%`, keep the candidate. Stock data does its job here — it tells us the company was affected. Then it exits the pipeline.

**Stage 2 — Classify channel (Snorkel):** Among filtered candidates, three types of LFs vote on which channel:

1. **Event-category LFs (7):** "This is a sanctions event → financial_treasury"
2. **GICS sector LF (1):** "This is a financial company → financial_treasury" (NEW — we built a sector cache for 1,441 tickers via yfinance)
3. **Text-based LFs (8):** "Filing mentions 'OFAC' and 'asset freeze' → financial_treasury"

All three answer the same question ("which channel?") with genuinely different information. No stock-reaction LFs muddying the water.

**The result:**

| Channel | v1 (old) | v2 (two-stage) |
|---------|----------|----------------|
| procurement_supply_chain | 76 (3.2%) | 1,752 (33.9%) |
| logistics_operations | 1,796 (76.7%) | 1,051 (20.3%) |
| regulatory_compliance_cost | 0 | 849 (16.4%) |
| revenue_market_access | 454 (19.4%) | 779 (15.1%) |
| financial_treasury | 0 | 439 (8.5%) |
| innovation_ip | 19 (0.8%) | 301 (5.8%) |
| **Total** | **2,345** | **5,171** |

Six channels covered instead of four. No single channel above 34%. Two channels that had zero labels (`financial_treasury`, `regulatory_compliance_cost`) now have hundreds.

### The Remaining Concern: 96% Event + Sector Agreement

We ran the same audit on v2 and found 97% of labels come from event-category + sector LF agreeing. Is that rubber-stamping again?

**The difference from the stock LF rubber-stamp:** Event category and company sector ARE genuinely different facts about the world. "This is a sanctions event" and "this is a financial company" come from different data sources. A financial company hit by sanctions going through `financial_treasury` is a reasonable inference.

**But the concern is valid:** Two static lookup tables agreeing is more informative than one, but less than having text evidence corroborate. The 3% of labels with all three signals (event + sector + text) are the strongest.

**The solution:** Soft labels.

---

## Part 2: Soft-Label Training

### The Concept

Instead of treating every Snorkel label as equally true (hard label: "this IS financial_treasury"), we pass Snorkel's probability distribution to the downstream model.

Snorkel outputs a probability for every channel:
- High confidence example: `innovation_ip = 98.8%, everything else < 1%`
- Low confidence example: `logistics_operations = 59.6%, revenue_market_access = 5.3%, ...`

For training, instead of saying "this is definitely logistics_operations," we give the model the full distribution. The model learns "this is PROBABLY logistics but I shouldn't be too sure."

### Implementation

XGBoost doesn't natively support probability targets for classification. So we used the next best thing: **sample weights**.

- Seed labels (human-reviewed): weight = 1.0 (fully trusted)
- Weak labels: weight = Snorkel confidence score (0.5 to 0.7 typically)

We saved Snorkel's probability distributions to a new file (`weak_labels_probs.csv`) and modified `train.py` to:
1. Load both seed labels and weak labels
2. Build features for weak labels (same feature engineering pipeline)
3. Pass `sample_weight` to XGBoost's `fit()` call

**Result:**
- Training data: 163 seed labels + 5,171 weak labels = 5,334 total
- Seed-label-only evaluation (the honest metric): macro F1 = 0.711
- `financial_treasury` F1 = 0.60 (was impossible before — zero training labels)
- `regulatory_compliance_cost` F1 = 0.83 (also previously impossible)

---

## Part 3: Wiring Disconnected Components into the Dashboard

We found three components that were built but not connected to anything users see:

### 3a. Conformal Prediction (Model 3A + 3B)

**The problem:** The dashboard showed impact estimates using old quantile regression (q10/q50/q90) which claimed 80% coverage but actually delivered 43% on holdout.

**What we did:** Updated `predict.py` to load the conformal model (`conformal_3b.pkl`) and Model 3A (`model_3a_market.json`). The estimator now:
1. Tries conformal prediction first (90% guaranteed coverage intervals)
2. Falls back to quantile if conformal model not available
3. Also returns market reaction estimate from Model 3A

The dashboard now shows `Interval method: 90% conformal` and the market reaction.

### 3b. RAG Strategy Recommender

**The problem:** The RAG recommender was built with 21 case studies and ChromaDB retrieval, but the dashboard never called it.

**What we did:** Added a new "Recommended strategies (based on historical precedents)" expander in the analysis results section. After every analysis, it:
1. Calls `RAGRecommender.recommend()` with the event text, company, sector, and predicted channel
2. Retrieves the top 3 most similar historical cases
3. Displays each with: strategy summary, precedent company and year, outcome, relevance score

### 3c. Macro Indices (GPR + EPU)

**The problem:** The Economic Policy Uncertainty (EPU) and Geopolitical Risk Index (GPR) were ingested into the database but never shown to users.

**What we did:** Added a "Risk context" expander to the sidebar that queries the latest GPR and EPU values and shows them with risk level indicators (Normal / Elevated / High).

**Small bug caught:** The database stored indices as `gpr_daily` and `epu_us`, not `gpr` and `epu`. Had to fix the query.

---

## Part 4: Temperature Scaling Calibration

### The Problem

The classifier had 3-tier confidence thresholds (low/moderate/high) but no actual calibration — we didn't know if "90% confident" predictions were actually correct 90% of the time.

### What Temperature Scaling Does

It's one number (T) that adjusts all probabilities. Before softmax, divide logits by T:
- T > 1: Model was overconfident → soften probabilities
- T < 1: Model was underconfident → sharpen probabilities
- T = 1: No change

We optimized T on the validation set by minimizing negative log-likelihood: T = 0.836.

### The Result

| Confidence Bin | Predicted Confidence | Actual Accuracy |
|---------------|---------------------|-----------------|
| 90-100% | 99.1% | 99.0% |
| 70-90% | 78.6% | 63.6% |
| 50-70% | 58.4% | 33.3% |

The top bin (where 95% of predictions land) is nearly perfectly calibrated. The middle bins are overconfident but have very few samples (9-11 each), so the statistics are noisy.

The temperature is saved to `calibration.json` and automatically loaded by `predict.py`.

---

## Part 5: Expanding Case Studies from 21 to 201

### The Problem

The RAG recommender had 21 case studies. For some queries, there simply weren't relevant precedents. The roadmap target was 200-500.

### The Approach

We expanded in three waves, always keeping to well-known, verifiable events:

1. **Wave 1 (21 → 61):** Filled channel gaps — added `financial_treasury` cases (BNP Paribas, HSBC, Standard Chartered), `regulatory_compliance_cost` (Meta GDPR, Amazon GDPR), `innovation_ip` (Huawei, TSMC, Samsung)

2. **Wave 2 (61 → 149):** Same major events, more company perspectives. Russia-Ukraine alone had documented responses from 20+ major companies. We added Equinor, Nestlé, Adidas, VW, Spotify exits. Same for COVID (Zoom, Airbnb, Walmart, Delta, Pfizer) and chip controls (AMD, Arm, MediaTek, KLA, Synopsys).

3. **Wave 3 (149 → 201):** Filled remaining thin spots — cybersecurity (Change Healthcare, MGM, Caesars, MOVEit), reputation (Boeing 737 MAX, VW Dieselgate, NBA Hong Kong), workforce (Cognizant, Grammarly, GlobalLogic relocations).

### Why We Stopped at 201

We tried mining EDGAR filings for automated case study extraction, but 97% of mentions were generic risk disclosures ("we are exposed to geopolitical risk"), not specific action descriptions ("we exited our Russian operations"). Only 17 mentions out of 37,989 contained usable action language.

Getting to 200+ with quality required writing cases from well-known events. Every case in the database is a household-name company (Apple, Google, Shell, Tesla, Nike, Boeing) responding to a major event that was widely covered in the press. No obscure companies or events.

### Final Distribution

| Channel | Cases |
|---------|-------|
| revenue_market_access | 45 |
| procurement_supply_chain | 43 |
| regulatory_compliance_cost | 20 |
| capital_allocation_investment | 20 |
| logistics_operations | 15 |
| financial_treasury | 15 |
| cybersecurity_it | 11 |
| innovation_ip | 12 |
| reputation_stakeholder | 10 |
| workforce_talent | 10 |

All 10 channels covered with 10+ cases each. All 8 event categories represented.

---

## Part 6: The Model 3B Revenue Question

### What We Tried

We attempted to auto-extract revenue delta labels by cross-referencing:
- Event studies (which companies had stock reactions)
- Company financials (quarterly revenue from EDGAR XBRL)

The idea: if a company's stock dropped 8% after sanctions, and their quarterly revenue also dropped, that's a training label for Model 3B.

### Why It Didn't Work

We extracted 539 labels, but:
- **Correlation between stock reaction and revenue change: 0.114** (nearly zero)
- 81% of labels showed positive revenue growth — because 2021-2023 was a growth period regardless of geopolitical events
- Individual examples were misleading: ADM's -75% revenue drop was attributed to Red Sea attacks, but it was actually commodity price normalization

Quarterly revenue changes reflect everything that happened to a company in that quarter, not just the geopolitical event. This is a **causal inference problem** — "what would revenue have been WITHOUT this event?" — and simple regression can't answer it.

### The Reframing

Rather than predicting a revenue number, Model 3B's real purpose should be:

> "This event hits your procurement_supply_chain channel. Your China-sourced components are 30% of COGS. Here are companies that faced similar exposure and what they did."

That's a structured reasoning chain (exposure identification → quantification → strategies), not a regression. The components for this already exist:
- Model 2 identifies the channel
- Exposure proxies quantify concentration
- RAG recommender retrieves precedents

A Bayesian approach would add principled uncertainty quantification: "Given 30% exposure through procurement and a trade policy event, the revenue impact is likely -2% to -6% (80% credible interval)" — using priors from historical data, updated by company-specific evidence.

This is the planned next step.

---

## Roadblocks and How We Tackled Them

### Roadblock 1: Snorkel LFs Systematically Fighting Each Other
- **Hit:** Labels stuck at ~2,300 despite adding more events and EDGAR data
- **Diagnosis:** Stock-reaction LFs conflicted with event-category LFs on channel prediction
- **Root cause:** Stock LFs answered "was the company affected?" but guessed at "which channel?" — always guessing `revenue_market_access`
- **First attempt (rubber-stamping):** Made stock LFs copy event LFs' answer → 96% artificial agreement
- **We audited it honestly, caught the problem, and reverted**
- **Actual fix:** Two-stage pipeline — stock data filters candidates, only channel-informed LFs vote in Snorkel

### Roadblock 2: EDGAR Mentions Were Mostly Boilerplate
- **Hit:** Ingested 20,617 new mentions for 140 tickers, expected big improvement in text-based LF coverage
- **Result:** Only 88 more Snorkel labels
- **Diagnosis:** 97% of new mentions scored < 30 on specificity — they were generic risk disclosures, not specific event responses
- **Lesson:** More data doesn't help when the data quality is low. The specificity scorer was doing its job correctly.

### Roadblock 3: Revenue Deltas Don't Isolate Event Impact
- **Hit:** 539 auto-extracted revenue labels had 0.114 correlation with stock reactions
- **Diagnosis:** Quarterly revenue = event impact + market conditions + company strategy + seasonality + 96 other factors
- **Lesson:** This is a causal inference problem, not a regression problem. Need DiD or Synthetic Control methods, or accept that 3B needs a different approach (Bayesian with priors).

### Roadblock 4: Case Studies Can't Be Automated from EDGAR
- **Hit:** Only 17 out of 37,989 EDGAR mentions contained specific action language
- **Diagnosis:** 10-K filings describe risks, not responses. Actual corporate responses are in earnings calls, press releases, and news
- **Fix:** Wrote 180 cases from well-known events manually. Every case verifiable from public sources.

---

## Current State After Day 12

### What's Working
- **Snorkel v2:** 5,171 labels across 6 channels, two-stage pipeline, soft-label training
- **Dashboard:** Conformal intervals, RAG strategies, macro indices all wired in
- **Classifier:** Temperature-calibrated (T=0.836), top bin 99.1% calibrated
- **Case studies:** 201 across all 10 channels, 130+ unique companies
- **Training:** Soft-label XGBoost with seed (weight=1.0) + weak (weight=Snorkel confidence)

### What's Left
- **Model 3B:** Needs Bayesian rewrite (exposure-weighted priors, not regression)
- **Dashboard rewire:** Present the reasoning chain (channel → exposure → impact estimate → strategies) instead of just numbers
- **Company universe:** 1,456 tickers vs 3,000+ target (diminishing returns)
- **Causal validation:** DiD and Synthetic Control for publishability (Month 2)

### Key Numbers

| Metric | Day 11 | Day 12 |
|--------|--------|--------|
| Snorkel labels | 2,345 | 5,171 |
| Channels covered | 4/10 | 6/10 |
| Largest channel share | 78% | 34% |
| Case studies | 21 | 201 |
| Event studies | 45K | 63K |
| Events in DB | 20 | 49 |
| Components wired to dashboard | 4 | 7 |
| Classifier calibration | Thresholds only | Temperature scaling (T=0.836) |

---

## What I Learned

1. **More data doesn't fix a design flaw.** We added 19,000 event studies and 20,000 EDGAR mentions, but Snorkel labels barely moved. The real problem was that two groups of labeling functions were systematically disagreeing — a structural issue, not a data volume issue.

2. **Audit everything quantitatively.** Every time we tried a fix, we ran the same audit: "What percentage of labels come from which signal sources?" This caught the rubber-stamping pattern twice — once with the modified stock LFs, once with the event+sector agreement. Without the audit, we'd have shipped 3,000 artificial labels.

3. **Stock reactions and revenue changes measure different things.** Stock price reacts in days to the event itself. Quarterly revenue reflects everything that happened over 3 months. Conflating them (as 3B's regression does) produces noise, not signal.

4. **Components that aren't connected don't count.** We had conformal prediction, RAG recommendations, and macro indices all built — but none visible to users. Building something is half the work; wiring it into the product is the other half.

5. **Case studies need human curation.** Automated extraction from EDGAR produced 17 usable cases out of 37,989 mentions. Companies describe risks in filings, not responses. The 201 hand-curated cases, each verifiable from public sources, are far more valuable than any number of automated extractions.
