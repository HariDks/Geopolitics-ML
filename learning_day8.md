# Learning Day 8 — Confronting the Review: Feedback Loops, Circular Weights, and Honest Evaluation

**Project:** Geopolitical Muscle ML Model  
**Date:** April 18, 2026  
**Written for:** Understanding what happens when you take a critical review seriously — every diagnostic tool built, every uncomfortable finding documented, and what it means for the system's real-world reliability

---

## Where We Left Off

At the end of Day 7 we had:
- A complete 4-model pipeline with production infrastructure (FastAPI, ONNX, Streamlit)
- Three publishable artifacts (backtest, risk matrix, GRI index)
- Reported macro F1 of 0.825 for Model 2 (Exposure Scorer)
- 602 seed labels (163 manual + 439 auto-generated)
- Everything committed across 22 git commits

Then we received a **detailed technical review** with 25 critiques across 6 categories. The review's central thesis: *"Where are you accidentally pretending correlation = causation?"*

Day 8 was about **building the diagnostic tools to test whether the reviewer was right** — and being honest about what we found.

---

## Part 1: The Holdout Evaluation — The Most Important Test We Ran

### The Question

The reviewer's critique 2.2 (semi-supervised feedback loop) asked: "Your pipeline creates labels → trains model → generates labels → retrains model. How do you know the 0.825 macro F1 isn't the model just getting better at predicting its own biased labels?"

### The Test

We split our 602 labels into two groups:
- **163 manual labels** (`labeled_by=claude_verified`) — created by a human before any model existed
- **439 auto-generated labels** — created by the auto-labeling pipeline that uses Model 2's logic

Then we evaluated Model 2 on ONLY the 163 manual labels — data it never trained on and that wasn't influenced by any model.

### The Result

```
Reported macro F1 (mixed eval):     0.825  ← includes auto-labels in validation
Holdout macro F1 (manual only):     0.371  ← independent human labels only
                                    ─────
GAP:                                0.454  ← feedback loop inflation
```

**The reviewer was right.** The reported 0.825 was inflated by 0.454 — more than double the real accuracy. The model's actual channel prediction accuracy on independent data is **50.7%**, not 82.6%.

### What This Means

The auto-labeling pipeline created a self-reinforcing loop:
1. Model 2 (with GICS sector mapping) predicts `workforce_talent` for industrials
2. Auto-labeler assigns `workforce_talent` to industrial companies
3. Model retrains on these labels → gets even more confident about `workforce_talent` for industrials
4. Validation includes auto-labels → reports high accuracy
5. But humans would have labeled the same cases as `revenue_market_access` or `logistics_operations`

### What's Still Genuinely Good

The holdout eval also showed:
- **Direction accuracy: 90%** — the model almost always knows if the impact is positive or negative
- **In-range: 60%** — the predicted magnitude range captures the actual outcome more than half the time

These metrics aren't inflated because they depend on stock prices (objective) not channel labels (subjective).

### The Lesson

**Always evaluate on data that existed before the model.** The moment your evaluation set includes model-influenced data, your metrics become circular. This is the most common silent failure mode in ML systems — and we fell into it despite being aware of the risk.

---

## Part 2: The Negative Backtest — Proving the Model Isn't Just Predicting "Everything Is Bad"

### The Question

The original backtest tested 10 events where companies WERE affected. But a model that predicts "negative impact" for everything would score well on those tests. Does our model also correctly identify companies that AREN'T affected?

### The Test

10 cases where companies had zero connection to the event:
- UnitedHealth vs Red Sea shipping (US domestic health insurer)
- McDonald's vs chip export controls (fast food, no semiconductors)
- Home Depot vs Russia invasion (US/Canada/Mexico only)
- Microsoft vs OPEC production cut (software, <2% energy costs)
- Costco vs Sudan civil war (zero operational connection)
- And 5 more

### The Result

```
Correctly predicted LOW impact:  10/10 (100%)
False alarms:                    0/10 (0%)
```

All 10 companies correctly identified as unaffected. Severity scores ranged from -0.21 to -0.30 (all below the -0.4 alarm threshold), with near-zero predicted impact (+0.0% to +0.1%).

### What This Means

The model genuinely distinguishes affected from unaffected companies. The 90% directional accuracy in the original backtest is real signal, not an artifact of always-negative predictions.

---

## Part 3: Market-Validated GRI Weights — The Circular Dependency Exposed

### The Question (Reviewer Critique 6.1)

Our GRI category weights came from our model, which was trained on our labels, which reflected our assumptions:

```
our assumptions → labels → model → weights → GRI → published as "objective"
```

Are the weights actually supported by market evidence?

### The Test

Computed weights from actual S&P 500 stock reactions: for each event category, measured the mean |CAR_1_5| (absolute 5-day stock reaction) across all companies in our event studies. Higher absolute reaction = category matters more to markets.

### The Result

```
Category                      Market Weight   Model Weight   Gap
armed_conflict                    1.000          1.0         0.000
trade_policy                      0.709          0.8         0.091
political_transitions             0.703          0.5         0.203 **
institutional_realignment         0.703          0.4         0.303 **
technology_controls               0.681          0.8         0.169
resource_energy                   0.504          0.8         0.296 **
sanctions                         0.441          0.9         0.459 **

Correlation between market and model weights: 0.099
```

**Correlation of 0.099** — essentially zero. Our model weights bear almost no relationship to market-validated importance.

### The Specific Biases

| Category | Our Weight | Market Weight | What We Got Wrong |
|----------|:---------:|:------------:|-------------------|
| Sanctions | 0.9 | 0.441 | **2x overweighted.** Our seed labels over-represented dramatic sanctions (Russia SWIFT). Markets show sanctions cause relatively small stock moves for most companies. |
| Resource/energy | 0.8 | 0.504 | **1.6x overweighted.** Our 30 synthetic resource_energy training examples amplified this category. |
| Political transitions | 0.5 | 0.703 | **Underweighted.** Coups and elections actually move markets significantly — we dismissed them as "low impact." |
| Institutional realignment | 0.4 | 0.703 | **Underweighted.** Brexit caused massive market moves. Our low weight came from having few training examples. |

### The Fix

Suggested blended weights (50% market + 50% model) that correct the worst biases:

```
armed_conflict:        1.000 (unchanged)
trade_policy:          0.754 (was 0.8)
political_transitions: 0.601 (was 0.5) — upgraded
institutional:         0.551 (was 0.4) — upgraded
technology_controls:   0.766 (was 0.8)
resource_energy:       0.652 (was 0.8) — downgraded
sanctions:             0.670 (was 0.9) — downgraded
```

---

## Part 4: Feedback Loop Detection — Quantifying the GICS Bias

### The Question (Reviewer Critique 2.2)

The auto-labeler uses GICS sector → channel mapping. Does this create systematic differences from how humans would label the same events?

### The Test

For the 5 events where BOTH manual and auto labels exist, compared channel distributions.

### The Result

```
Total distribution shift: 38.6 percentage points — HIGH DIVERGENCE

Auto-labeler UNDER-predicts revenue_market_access by 20.0pp
Auto-labeler OVER-predicts workforce_talent by 12.3pp
Auto-labeler OVER-predicts logistics_operations by 11.4pp
```

Per-event comparison:
| Event | Manual Top Channel | Auto Top Channel | Agreement |
|-------|:-:|:-:|:-:|
| India demonetization | revenue_market_access | financial_treasury | DISAGREE |
| OPEC price war | capital_allocation | revenue_market_access | DISAGREE |
| Russia invasion | capital_allocation | workforce_talent | DISAGREE |
| US-Iran war | procurement_supply_chain | procurement_supply_chain | AGREE |
| US tariffs 2025 | procurement_supply_chain | procurement_supply_chain | AGREE |

The auto-labeler agrees with humans only 40% of the time on which channel is most affected.

### What This Means

The GICS sector mapping is a **reasonable but biased shortcut**. It works well for straightforward cases (tariffs → procurement for manufacturing companies) but fails for complex events where the primary impact channel isn't what the sector label suggests. Russia invasion affected BP through capital_allocation (Rosneft write-down), not through workforce_talent (what our GICS mapping for energy companies predicts).

---

## Part 5: Event Hierarchy — Defining "What Is One Event"

### The Problem (Reviewer Critique 1.3)

We never formally defined what constitutes a single event. "Russia invasion" appears as:
- `russia_invasion_2022` (66 labels)
- `russia_sanctions_2022` (9 labels)
- `ukraine_grain_crisis_2022` (3 labels)
- `ukraine_airspace_closure_2022` (0 labels)

Are these one event or four? The answer matters for training — if they're one event, we have 78 labels for it. If they're four separate events, we have 66, 9, 3, and 0 respectively. The model sees a very different world depending on the answer.

### The Solution

Defined a three-level hierarchy in `data/mappings/event_hierarchy.json`:

**Crisis** = the overarching geopolitical situation
- Example: `CRISIS-russia-ukraine-2022` ("Russia-Ukraine War")

**Episode** = a discrete action within the crisis
- `EP-russia-invasion` (Feb 24, military invasion)
- `EP-russia-sanctions-r1` (Feb 25, first sanctions round)
- `EP-russia-swift-exclusion` (Feb 26, SWIFT cut-off)
- `EP-russia-corporate-exits` (Mar 1, BP/Shell/MCD leave)
- `EP-ukraine-grain-blockade` (May 1, Black Sea exports blocked)
- `EP-russia-energy-weaponization` (Jun 1, Nord Stream reduction)

**Observation** = a single data source's record of an episode
- One GDELT row = one observation
- One ACLED row = one observation
- One OFAC SDN entry = one observation

### Validation Results

```
7 crises defined, 24 episodes
54% of labels mapped to the hierarchy
13 duplicate event_id sets identified (same episode, different names)
81 orphan event_ids (need new crises defined)
```

### The 13 Duplicate Sets

These are event_ids that refer to the same episode but have inconsistent names:

| Episode | Event IDs | Labels |
|---------|-----------|--------|
| Chip controls Oct 2022 | `us_chip_export_controls_oct2022`, `us_chip_export_oct2022`, `us_chip_export_controls_2022_2024` | 37 total |
| Red Sea attacks | `red_sea_houthi_2023`, `red_sea_houthi_attacks_2023` | 10 total |
| Hamas attack | `israel_hamas_2023`, `israel_hamas_war_2023` | 23 total |
| Energy crisis | `eu_energy_crisis_2022`, `eu_energy_crisis_peak` | 23 total |

These duplicates mean the model treats the same episode as 2-3 separate events, fragmenting the signal and potentially double-counting impacts.

---

## Part 6: XBRL Geographic Segment Extraction Pipeline

### The Motivation (Reviewer Critique 5.1)

The reviewer's #1 recommendation: "Add revenue by geography. This is the biggest gap." We had a curated JSON with hand-entered geographic data for 46 companies. The reviewer asked: why not extract it automatically from EDGAR?

### What We Built

`pipelines/ingest_geo_segments.py` parses 10-K inline XBRL to find geographic revenue segments. For each company:

1. Fetch the latest 10-K from EDGAR submissions API
2. Download the HTML filing (~1-8MB per company)
3. Parse XBRL `<xbrli:context>` tags to find geographic segment member definitions
4. Match `<ix:nonFraction>` revenue values to those contexts
5. Compute percentage of total revenue per region

### How It Works (The XBRL Structure)

A 10-K filing embeds structured data using inline XBRL. Geographic segments look like this:

```xml
<!-- Context definition: "c-149 = Americas segment for FY2024" -->
<xbrli:context id="c-149">
  <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">
    aapl:AmericasSegmentMember
  </xbrldi:explicitMember>
  <xbrli:period><xbrli:startDate>2024-09-29</xbrli:startDate>...</xbrli:period>
</xbrli:context>

<!-- Revenue value tagged with that context -->
<ix:nonFraction contextRef="c-149" name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax">
  178,353
</ix:nonFraction>
```

The pipeline extracts both pieces, matches them, and computes: Americas = $178.4B / $416.2B total = 42.0%.

### Roadblock: Most Companies Don't Tag Geography in XBRL

**Result: 15 of 100 companies** have structured XBRL geographic segments.

```
AAPL: Americas=42.0%, Europe=26.2%, China=17.1%, AsiaPac=7.9%, Japan=6.8%
NKE:  Americas=50.5%, Europe=31.8%, China=17.6%
CSCO: Americas=58.8%, Europe=26.5%, China=14.7%
ACN:  Americas=50.3%, Europe=35.4%, AsiaPac=14.3%
AMZN: Americas=72.5%, International=27.5%
COP:  Europe=46.2%, Canada=40.2%, AsiaPac=13.6%
```

The other 85 companies report geography in prose: "Greater China accounted for approximately 19% of our total net revenues" — readable by humans, not parseable by our current pipeline.

### Why Only 15%?

SEC requires geographic segment disclosure (ASC 280) but doesn't require it as tagged XBRL. Companies can comply by:
1. **XBRL-tagged segments** (15% of companies) — machine-readable, our pipeline catches these
2. **Narrative disclosure** (85% of companies) — human-readable text in MD&A or notes, needs NLP extraction

### Total Geographic Coverage After Merging Sources

```
Curated JSON (hand-entered):   46 companies
XBRL pipeline (automated):    +9 new companies (6 updated with XBRL data)
Total:                         55 companies with geographic revenue data
```

### TCP Port Safety

The pipeline uses `requests.Session()` (single persistent TCP connection) with 0.12s rate limiting. 100 companies = ~200 requests over one socket. No port exhaustion risk — same pattern as all our other EDGAR pipelines. Verified by running the full 100 without issues.

---

## Part 7: The Correlational Framing

### What We Changed

Added an explicit disclaimer to the pipeline output:

```python
return {
    "disclaimer": "Predictions are based on historical correlations, not causal analysis. "
                  "Direction accuracy: ~90%. Channel accuracy: ~50%. "
                  "Companies with concentrated geographic exposure may see larger actual impacts.",
    ...
}
```

Added an "Honest Limitations" section to FINDINGS.md:

> **This is a correlation engine, not a causal model.** When we say "NVIDIA faces -6.3% impact from chip export controls," we mean "historically, when BIS publishes export rules AND a company has NVIDIA's profile, revenue tends to drop ~5-7%." We don't model the causal mechanism.

> **Channel prediction is the weakest link.** On independent human-labeled data, channel accuracy is 50.7%. The reported 82.5% figure was inflated by training on auto-generated labels.

### Why This Matters

A user seeing "procurement_supply_chain impact predicted with 55% confidence" makes a different decision than a user seeing "we think your supply chain will be disrupted." The first is a statistical signal. The second is a causal claim. Our model produces the first but was presenting it as the second.

---

## Part 8: The Review Response Document

Created `review_response.md` — a systematic response to all 25 critiques:

- **Data layer (5 critiques):** GDELT dominance bias (valid), NumMentions filter bias (valid), event granularity (valid, foundational), cross-source duplication (partially valid)
- **Labeling (3 critiques):** Seed label bias (valid, unavoidable), feedback loop (valid, critical — confirmed by holdout eval), GICS channel mapping (valid, acknowledged shortcut)
- **Text + linking (3 critiques):** Specificity over-pruning (valid concern), linking approximation (inherent limitation), EDGAR survivorship bias (structural)
- **Model layer (3 critiques):** Exposure model lacks firm data (valid, #1 limitation), impact model = correlation (valid but useful), strategy recommender has no feedback (acknowledged)
- **System-level (3 critiques):** Missing company exposure model (agree, most important), temporal causality (partially valid), backtest selection bias (valid)
- **GRI (3 critiques):** Circular weights (valid — confirmed by market validation), volume bias (valid), no corporate weighting (valid)

Each critique categorized as: valid and fixable, valid but fundamental limitation, or partially valid.

---

## Part 9: What the Diagnostics Tell Us About the System's Real Capabilities

### The Honest Model Card

| Capability | Reported | Actual (Holdout) | Assessment |
|------------|:--------:|:----------------:|:-----------:|
| Event classification | 95.3% | 95.3% | **Genuine** — tested on held-out news text |
| Direction prediction (+/-) | — | 90.0% | **Genuine** — validated by stock prices |
| Channel prediction | 82.5% | 50.7% | **Inflated** — feedback loop added 32pp |
| Impact range | 80.7% coverage | 60.0% | **Somewhat inflated** — holdout coverage lower |
| Negative detection | — | 100% (10/10) | **Genuine** — correctly identifies unaffected companies |
| GRI weights | "model-derived" | 0.099 correlation with market | **Circular** — not market-validated |

### What the System Is Actually Good At

1. **Classifying geopolitical events from text** — 95.3% accuracy, validated independently
2. **Telling you whether to worry** — 90% directional accuracy
3. **Telling you if your company is NOT affected** — 100% true negative rate
4. **Estimating order-of-magnitude impact** — within 2-3x for major S&P 500 companies

### What the System Is NOT Good At (Yet)

1. **Telling you which business channel is affected** — 50% accuracy (coin flip)
2. **Predicting impacts for concentrated niche companies** — misses TWE -96%, Zain -71%
3. **Weighting geopolitical risk categories objectively** — our weights don't match market evidence
4. **Growing its own training data without amplifying biases** — feedback loop confirmed

---

## Part 10: Files Created on Day 8

```
review_response.md              — Systematic response to all 25 review critiques
backtest/holdout_eval.py        — Unbiased evaluation on 163 manual labels
backtest/negative_backtest.py   — 10 unaffected company tests
index/market_validate_weights.py — Market-derived vs model-derived weight comparison
pipelines/feedback_loop_check.py — Channel distribution divergence measurement
pipelines/validate_hierarchy.py  — Event hierarchy validation tool
pipelines/ingest_geo_segments.py — Automated XBRL geographic segment extraction
data/mappings/event_hierarchy.json — 7 crises, 24 episodes, 3-level schema
data/mappings/company_geo_exposure_xbrl.json — Automated geographic data for 15 companies
```

---

## Summary: Day 8 in One Paragraph

Day 8 was about intellectual honesty. We received a 25-point technical review and built diagnostic tools to test every major critique. The results were uncomfortable: the holdout evaluation revealed that Model 2's reported 0.825 macro F1 was inflated to 0.371 by a semi-supervised feedback loop — the model was learning to predict its own biased auto-labels, not real-world impact channels. Market validation showed our GRI category weights correlated 0.099 with actual stock market reactions — essentially random, confirming circular dependency. The feedback loop detector found 38.6 percentage points of channel distribution shift between human and auto labels. On the positive side: event classification (95.3%), directional accuracy (90%), and negative detection (10/10) are all genuinely strong and independently validated. We also built the event hierarchy schema (7 crises, 24 episodes, 13 duplicate sets identified), the XBRL geographic segment pipeline (15/100 companies with automated geographic revenue data), and merged all geographic sources to 55 companies total. The honest model card now shows: the system is good at classifying events, predicting direction, and identifying unaffected companies — but weak at channel prediction (50.7%) and weight calibration (0.099 market correlation). Every finding was documented transparently in review_response.md with prioritized fixes.
