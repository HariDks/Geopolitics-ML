# Response to Technical Review

## The Meta-Problem: Correlation vs Causation

**The reviewer is right.** Our system learns "when X happens, Y tends to follow" — not "X causes Y." The NVIDIA chip controls prediction (-6.3% predicted, -5.0% actual) looks causal but the model is really learning "when BIS publishes export rules AND the company is in GICS 45 AND has high EDGAR mention specificity, stock tends to drop ~5%." That's a useful correlation, but it's not the same as understanding that NVIDIA loses revenue because it can't ship A100s to Chinese data centers.

**What this means practically:** The model works well in environments similar to training data (post-2020 geopolitics). It will fail when the transmission mechanism changes — e.g., if NVIDIA found a compliant chip design that maintains China revenue, the correlation would break but our model would still predict a drop.

**What we can do:** Be explicit in all published materials that this is a statistical model, not a causal one. Label predictions as "historically correlated impact" not "predicted impact."

---

## 1. DATA LAYER — Verdict: Mostly Right

### 1.1 GDELT Dominance Bias — VALID, PARTIALLY FIXABLE

**The problem is real.** 6.88M of 7.76M events (89%) are GDELT. GDELT is a media-derived dataset — it measures "what journalists write about," not "what materially affects companies."

**Concrete consequence:** Our finding that "EU regulation scores highest in danger" (Finding 5 in FINDINGS.md) might be an artifact of EU regulation generating lots of media coverage, not necessarily because it's the most financially damaging.

**What we should do:**
1. Weight events by source quality: ACLED events (verified by researchers) should count more than GDELT events (scraped from news)
2. Add financial-impact weighting: events linked to large stock movements in our event_studies table should count more than events with no measurable market reaction
3. Explicitly state in methodology: "The GRI measures media-visible geopolitical risk, not necessarily financial-material geopolitical risk"

**What we can't fix:** We don't have a comprehensive dataset of "events that matter to companies but don't make the news." That data doesn't exist in structured form.

### 1.2 NumMentions Filter — VALID, ACKNOWLEDGED

**The reviewer is right** that this creates a bias toward well-covered events. Our chip export controls example proves it — early signals in October 2022 BIS rulemaking had low coverage weeks before publication.

**What we should do:** Add a "signal strength" dimension that tracks emerging events (rising NumMentions over 3-5 days) rather than only counting events above a static threshold. A rising trend from 2→5→15 mentions is more important than a steady 20 mentions.

### 1.3 Event Granularity — VALID, FOUNDATIONAL PROBLEM

**This is the most important data criticism.** We never formally defined "what is one event." Russia's invasion is a single event, but our database has:
- `russia_invasion_2022` (GDELT: thousands of daily entries)
- `russia_sanctions_2022` (OFAC: individual SDN additions)  
- `russia_corporate_exit_2022` (seed label)

These are treated sometimes as one event, sometimes as separate events. The seed labels compound this — some events have 5 labels, some have 1, with no principled reason for the difference.

**What we should do:** Define an event hierarchy:
- **Crisis** = the overarching situation (Russia-Ukraine conflict)
- **Episode** = a discrete action within the crisis (initial invasion, first sanctions round, Rosneft divestiture)
- **Observation** = a data source's record of an episode (one GDELT row)

Currently we conflate all three levels. This is fixable but requires re-engineering the event schema.

### 1.4 Cross-Source Duplication — PARTIALLY VALID

**Less severe than stated** because our sources cover different event types:
- GDELT: general events (from news)
- ACLED: armed conflicts (researcher-verified)
- GTA: trade policy (structured government data)
- OFAC: sanctions (official government lists)

Overlap is real for armed conflicts (GDELT + ACLED both capture them), but trade policy events are primarily in GTA with minimal GDELT overlap. Our `source` field tracks provenance.

**What we should do:** Add deduplication logic for events that appear in multiple sources on the same date for the same country.

---

## 2. LABELING — Verdict: Right, and This Is the Hardest Problem

### 2.1 Seed Label Bias — VALID, UNAVOIDABLE AT THIS STAGE

**The reviewer is correct** that our 602 labels reflect our worldview. The labels were created by:
1. Manual curation of 163 well-documented cases (human judgment)
2. Auto-generation of 439 from stock reactions + GICS sector rules (our rules)

**The honest limitation:** Every supervised ML system has this problem. The question is whether the bias is systematic. Ours is — we over-selected dramatic events (Russia, COVID, chip controls) and under-selected slow structural shifts (GDPR enforcement, data localization mandates, quiet trade policy changes).

**What we should do:** Explicitly document which event types are underrepresented in training data. Add a "confidence by event type" table to model documentation.

### 2.2 Semi-Supervised Feedback Loop — VALID, CRITICAL

**This is the subtlest and most dangerous criticism.** Our pipeline:

```
163 manual labels → train Model 2 → auto-generate 439 labels → retrain Model 2
```

If Model 2 has a systematic bias (e.g., over-predicting `procurement_supply_chain` for energy companies), the auto-labeler amplifies it because it uses GICS-sector rules that were partially informed by Model 2's early predictions.

**Concrete risk:** When we went from 0.494 to 0.825 macro F1, how much of that improvement was real learning vs. the model getting better at predicting its own biased labels?

**What we should do:**
1. Hold out the original 163 manual labels as a fixed validation set (never retrain on them)
2. Only evaluate on those 163 — they're human-verified and independent of the auto-labeling pipeline
3. Track whether auto-generated labels agree with manual labels when they exist for the same (event, company) pair

### 2.3 GICS Channel Assignment — VALID, ACKNOWLEDGED SHORTCUT

**We explicitly noted this was a shortcut** in the code:

```python
# Cap GTA at 500 per category so news-style augmentation has enough weight
# This is a shortcut — eventually need ML-based channel assignment
```

The reviewer is right that sector ≠ channel. Apple during a tariff war might face `procurement_supply_chain` (China manufacturing disruption) OR `revenue_market_access` (China retaliation blocks iPhone sales) depending on the specific tariff. Our GICS mapping always assigns `innovation_ip` for tech companies, ignoring context.

**What we should do:** Train a separate multi-label classifier that takes (event_description, company_description, sector) as input and predicts multiple possible channels with probabilities. This requires more labeled data with explicit channel reasoning.

---

## 3. TEXT + LINKING — Verdict: Partially Right

### 3.1 Specificity Filter Over-Pruning — VALID CONCERN

We classified 96% of EDGAR mentions as boilerplate. The reviewer asks: what if some of that 96% contains weak but meaningful signals?

**Example:** "We expect regulatory tightening in several of our key markets" — our filter scores this ~15 (boilerplate) because it's vague. But a human analyst would note this as a forward-looking risk indicator.

**The tradeoff we made:** We preferred high precision (only use mentions we're confident about) over high recall (use everything and accept noise). For training data, this is the right call — noisy training data degrades models. For a monitoring/alerting system, we should keep weak signals.

**What we should do:** Create a separate "weak signal" pipeline that tracks vague-but-relevant mentions without using them as training labels.

### 3.2 Linking Approximation — VALID, INHERENT LIMITATION

Our linking is keyword + category + date overlap. It will produce false positives (RTX mentions "conflict" in every filing because it's a defense company — not because of a specific event) and false negatives (a company discusses "supply chain challenges" without naming the Red Sea).

**We can't fully fix this** without NLP coreference resolution and entity linking — which requires much more sophisticated models than keyword matching. But we can improve it by incorporating company-specific context (a defense company mentioning "conflict" is less signal than a retail company mentioning "conflict").

### 3.3 Survivorship Bias in EDGAR — VALID, STRUCTURAL

Companies disclose what regulations require them to disclose. They DON'T disclose:
- Risks they haven't identified yet
- Strategic vulnerabilities (competitive intelligence)
- Early-stage exposures before they become material

**This is not fixable from public data.** It's why the project's original vision included "companies input their own data" as a future feature — proprietary exposure data (supplier lists, revenue by geography) would solve this.

---

## 4. MODEL LAYER — Verdict: Right on Limitations, Overstated on Consequences

### 4.1 Exposure Model Lacks Firm-Specific Data — VALID, PRIMARY LIMITATION

**This is the #1 technical limitation of the entire system** and the reviewer correctly identifies it. Our model has:
- Event features (category, severity)
- Company features (GICS sector, financial metrics, EDGAR mention signals)

It does NOT have:
- Revenue by geography (what % of Apple's revenue is China?)
- Supplier network (who supplies NVIDIA's packaging?)
- Asset concentration (what % of First Quantum's assets is Cobre Panama?)

This is exactly why Treasury Wine (-96% actual, ~0% predicted) and Zain Sudan (-71% actual, -0.5% predicted) fail — those companies have extreme geographic concentration that our features can't capture.

**What we should do:** Add a "geographic concentration" feature. Even a simple binary "does this company derive >30% revenue from the affected geography?" would dramatically improve predictions for concentrated companies. This data is available from 10-K geographic revenue disclosures.

### 4.2 Impact Model = Correlation Engine — VALID, BUT USEFUL

The reviewer is right that stock movements reflect many factors beyond the geopolitical event. When Boeing dropped 46% during COVID, how much was COVID lockdowns vs. 737 MAX grounding vs. debt concerns?

**But:** The model's 92% directional accuracy is genuinely useful even if it's correlational. A risk manager doesn't need to know the exact causal mechanism — they need to know "when X happens, should I worry about company Y?" The model answers that correctly most of the time.

**What we should do:** Frame predictions as "historically associated impact" rather than "predicted causal impact." Add confidence intervals that widen for multi-causal events.

### 4.3 Strategy Recommender Has No Feedback — ACKNOWLEDGED

We noted this explicitly:

> "ML-based ranking will come later when we have data on which strategies companies actually implemented and their outcomes"

This is a cold-start problem. We can't learn what works until someone uses our recommendations and reports results. The retrieval-based approach is the correct Phase 1 design.

---

## 5. SYSTEM-LEVEL — Verdict: These Are the Real Constraints

### 5.1 Missing Company Exposure Model — AGREE, MOST IMPORTANT GAP

This is the single biggest improvement the system needs. Adding three features would transform accuracy:
1. **Revenue concentration:** % from affected geography (from 10-K geographic segments)
2. **Supplier concentration:** HHI of supplier geography (from supply chain databases)
3. **Asset concentration:** % of total assets in affected geography (from 10-K)

These are available for public companies. Adding them would likely fix the Treasury Wine and Zain failures, and would make the difference between "good on large diversified companies" and "good on everything."

### 5.2 Temporal Causality — PARTIALLY VALID

The reviewer is right that:
- Markets price in events before they're officially announced (chip controls were priced in weeks before BIS published)
- Effects can lag months (Russia corporate exits were announced Feb 2022, completed over 6-18 months)

But our event studies do capture some of this — `car_1_5` (5-day) and `car_1_30` (30-day) windows give two time horizons. The real issue is that we don't model anticipatory pricing (the pre-announcement run-up) or multi-year effects.

### 5.3 Selection Bias in Backtest — VALID, ACKNOWLEDGED

**The reviewer is exactly right.** We chose 10 events with:
- Well-documented outcomes (easy to verify)
- Specific company impacts (clear attribution)
- Large effect sizes (easy to measure)

We did NOT test:
- Events where the model should predict "no impact" (negative cases)
- Events with ambiguous or multi-causal outcomes
- Events where the company was affected through unexpected channels

**What we should do:** Add "negative backtest cases" — events where companies WEREN'T materially affected despite media coverage suggesting they would be. The model's false positive rate matters as much as its true positive rate.

---

## 6. GRI LOOPHOLES — Verdict: Valid, Addressable

### 6.1 Circular Dependency — VALID

Category weights in GRI come from Model 2, which was trained on our labels, which reflect our assumptions. So:

```
our assumptions → labels → model → weights → GRI → published as "objective"
```

**Fix:** Use external validation. Compare our category weights to: academic studies of corporate impact by event type (e.g., Caldara & Iacoviello geopolitical risk index), insurance loss data, or actual S&P 500 earnings impact by event category. If our weights roughly match external sources, the circular dependency isn't distorting the output.

### 6.2 Volume Bias — VALID, PARTIALLY MITIGATED

GRI's volume component (40% weight) does inherit GDELT's media bias. We partially addressed this by using category-weighted event counts (not raw counts), which reduces the impact of over-reported armed conflicts. But the reviewer is right that media coverage ≠ corporate importance.

**Fix:** Add a "market-validated" volume signal — instead of counting events, count events that are followed by above-normal stock market volatility in the next 5 days. This filters out media noise that doesn't actually move markets.

### 6.3 No Corporate Weighting — VALID

A US-China tariff affects S&P 500 companies worth $30T in market cap. A local protest in Niger affects maybe $500M in exposed corporate assets. GRI treats both as "1 event."

**Fix:** Weight events by estimated corporate exposure — use our 37×10 risk matrix to estimate how many major companies each event type affects, and weight accordingly.

---

## What to Prioritize

| Fix | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Add geographic revenue concentration features | Very High | Moderate (10-K parsing) | **1** |
| Hold out original 163 labels for unbiased eval | High | Low (one-time split) | **2** |
| Add negative backtest cases | High | Low (run model + verify) | **3** |
| Frame predictions as correlational, not causal | High | Low (documentation) | **4** |
| Define event hierarchy (crisis/episode/observation) | High | Moderate (schema change) | **5** |
| Add "market-validated" GRI signal | Medium | Moderate | 6 |
| Track feedback loop amplification | Medium | Moderate | 7 |
| Weak signal pipeline | Medium | High | 8 |
| Multi-label channel classifier | Medium | High | 9 |
| Source quality weighting | Low-Medium | Low | 10 |

The honest summary: the reviewer identified the system's real limitations correctly. The most important ones (#5.1, #2.2, #4.1) are all aspects of the same fundamental problem — **the model doesn't know enough about individual companies.** It knows about event types and sector patterns, but it doesn't know about Apple's China manufacturing concentration or Zain's Sudan subscriber base. That's the wall we'll hit until we add firm-specific exposure data.
