# Learning Day 13 — The 3B Problem, Bayesian Revenue Estimation, and Strategic Pivot

**Project:** Geopolitical Muscle ML Model  
**Date:** May 6-7, 2026  
**Written for:** Understanding why revenue prediction failed, what Bayesian estimation actually does, how we built granular priors from 128 exposure-tagged labels, and why we're pivoting from "predict a number" to "match by exposure and show precedents"

---

## Where We Left Off

Day 12 ended with:
- Snorkel v2: 5,171 labels across 6 channels with two-stage pipeline
- Dashboard fully wired: conformal prediction, RAG strategies (201 cases), macro indices
- Temperature-calibrated classifier (T=0.836)
- Soft-label training with confidence-weighted weak labels
- Model 3B still broken: 43 gold labels, regression MAE of 46.96pp

The open question was: can we fix Model 3B (revenue impact prediction)?

---

## Part 1: Attempting to Grow 3B's Training Data

### The Auto-Extraction Attempt

We had 201 case studies with rich narrative descriptions of company responses. Many contained revenue figures: "Russia was ~9% of systemwide revenue", "$25.5B write-down", "revenue dropped 12%." Could we automatically extract structured revenue labels?

**Regex extraction (percentages):** Searched for patterns like "X% of revenue", "revenue fell X%", "X% revenue dip" across all 201 case studies.
- Result: 23 extractable labels. Most case studies describe actions and outcomes in prose, not standardized percentages.

**Dollar-to-percentage conversion:** For companies in our `company_financials` table, we converted dollar amounts to revenue percentages. "$25.5B write-down" on BP's $157B revenue → ~16%.
- Result: 13 more labels, but some were wrong. The regex caught "$117B" for Qualcomm — that was the Broadcom acquisition price, not a revenue loss.

**Combined:** 23 + 13 = 36 new labels, bringing total from 43 to ~66. Better, but still thin.

### Manual Tagging: The Human Pass

Since automation only got 36 labels, we did a manual pass through all 201 case studies. For each one where the case study contained enough information to estimate a revenue percentage, we tagged it:

- "Russia was ~9% of systemwide revenue" → McDonald's, -9%, revenue_market_access
- "Lost ~€1.5B in annual Russian revenue (~3.5% of global)" → IKEA, -3.5%, revenue_market_access
- "$870M direct losses" on Merck's $48B revenue → Merck, -1.8%, cybersecurity_it
- "Revenue dropped 72% in April 2020" → Airbnb, -72%, revenue_market_access (COVID)

**Result:** 90 manually tagged labels, combined to 96 unique after deduplication with existing gold labels.

**Coverage improvement:**

| Channel | Before | After |
|---------|--------|-------|
| revenue_market_access | 33 | 36 |
| reputation_stakeholder | 4 | 10 |
| capital_allocation_investment | 1 | 9 |
| procurement_supply_chain | 1 | 8 |
| financial_treasury | 1 | 7 |
| cybersecurity_it | 0 | 5 |
| regulatory_compliance_cost | 0 | 5 |
| innovation_ip | 2 | 7 |
| logistics_operations | 1 | 6 |
| workforce_talent | 1 | 3 |

---

## Part 2: Building the Bayesian Model (First Attempt)

### Why Not Regression?

The old Model 3B used XGBoost regression: give it features (event type, company sector, stock reaction), it predicts a number: "-5.2% revenue impact."

The problems:
1. **43 training examples** (now 96, but still small). Regression needs hundreds to learn reliable patterns.
2. **Huge variance** within the data. Revenue impacts range from -96% (Australian wine to China) to +99% (Moderna's vaccine revenue). No regression can fit that spread meaningfully.
3. **MAE of 46.96 percentage points** — worse than just predicting zero every time.

### What Bayesian Does Instead

Instead of learning a function from data, Bayesian reasoning starts with what we already know and updates with evidence.

**Analogy:** Imagine you're a doctor seeing a new patient.

The regression approach says: "I've trained on 43 patients. My formula says your blood pressure will be 142."

The Bayesian approach says: "Patients with your risk profile usually have blood pressure between 120-160 (that's my prior). But you're a smoker (update upward), you exercise daily (update downward). My updated estimate: 130-150."

The doctor isn't running a formula. They're starting with a base rate and adjusting for this specific patient's characteristics.

### The First Bayesian Model

We built a model with three steps:

**Step 1 — Prior:** "What typically happens to companies hit through this channel?" From 96 labels, the average revenue impact through `revenue_market_access` is -23%.

**Step 2 — Exposure scaling:** "How exposed is this company?" If Apple has 62% China revenue, they get the full prior. If Spotify has <1% Russia exposure, they get a fraction of the prior.

**Step 3 — Stock reaction update:** "Did the market think this was bad?" An 8% stock drop implies roughly 8% revenue impact (calibrated from 35 paired observations where we have both CAR and revenue delta).

### The Flaw the User Caught

The user identified a critical problem with Step 3: **using stock reaction as evidence for revenue impact defeats the purpose of separating 3A and 3B.**

The entire reason we split Model 3 into:
- **3A = short-term stock reaction** (what the market thinks in 5 days)
- **3B = actual revenue impact** (what happens to the business over a quarter)

...was because stock reactions and revenue changes measure different things (correlation = 0.114). By feeding the stock reaction back into 3B as "evidence," we were recombining what we'd carefully separated.

The user's exact challenge: "You are saying since the model evidence is less uncertain the posterior leans towards it — does it mean a higher weight is given to the noisy stock movement? Wasn't the whole point of dividing 3A and 3B to separate short term stock movements from long term impact?"

**They were right.** We removed stock reaction from 3B entirely. 3B's evidence should come only from:
- Company exposure (geographic revenue, supplier concentration)
- Channel type (procurement vs revenue vs financial)
- Historical precedents for similar exposure profiles

### Testing Against Reality

We tested the first Bayesian model (before removing stock reaction) against four known outcomes:

| Scenario | Model said | Actual | Assessment |
|----------|-----------|--------|------------|
| Sanctions + JPMorgan | -15.6% | -0.4% | Way too pessimistic |
| Chip controls + NVIDIA | -16.0% | -5% to -8% | 2x overestimate |
| Ransomware + generic | -1.0% | -0.1% to -7% | Reasonable |
| Trade war + Apple | -4.6% | -6% to -8% | Close but slightly under |

The JPMorgan case revealed the core issue: the `financial_treasury` prior was built from extreme cases (BNP Paribas's $8.9B sanctions violation penalty, Uniper's nationalization). JPMorgan's routine Russia wind-down is a completely different scenario, but the model treated them the same because they were in the same channel.

---

## Part 3: Granular Priors (Channel × Exposure Level)

### The Key Insight

The same channel produces wildly different impacts depending on exposure level:

- BNP Paribas: financial_treasury channel, HIGH exposure (was actively violating sanctions) → -18% revenue
- JPMorgan: financial_treasury channel, LOW exposure (routine Russia business) → -0.4% revenue

Same channel, 45x difference in impact. The exposure level is the critical variable, not the channel alone.

### Exposure Tagging

We manually tagged each of the 96 revenue labels (plus the gold labels) with an exposure percentage, derived from the case study text itself:
- "Russia was ~9% of revenue" → McDonald's, 9% exposure, medium
- "$25.5B write-down on $157B revenue" → BP, 16% exposure, medium
- "China was 25% of sales" → NVIDIA, 25% exposure, medium
- "Revenue dropped 72%" → Airbnb, 90% exposure, high (COVID hit all bookings)

**Result:** 128 labels tagged with exposure level (low/medium/high).

### The Granular Prior Table

We then grouped channels into four impact types (because individual channel × exposure bins were too thin):

| Impact Group | Low exposure (<10%) | Medium (10-30%) | High (>30%) |
|-------------|--------------------|--------------------|-----------------|
| **Market loss** (revenue, reputation) | -4.0% (n=25) | -12.3% (n=23) | -47.9% (n=17) |
| **Cost increase** (procurement, logistics, regulatory) | -1.7% (n=17) | -4.0% (n=5) | +53.0% (n=2)* |
| **Asset loss** (capital allocation, financial treasury) | -3.6% (n=15) | -18.7% (n=3) | -60.0% (n=3) |
| **Operational** (cyber, innovation, workforce) | +4.3% (n=7) | -9.5% (n=4) | +13.7% (n=7)* |

*Positive values in "cost increase high" and "operational high" reflect beneficiaries — shipping companies that profited from Red Sea disruption, Moderna/Pfizer from COVID.

**The pattern that matters most:** For market loss events, the escalation from low → medium → high exposure is dramatic: -4% → -12% → -48%. This is the pattern that separates JPMorgan (-0.4%, low exposure) from BNP Paribas (-18%, high exposure).

### Testing Granular Priors Against Reality

| Scenario | Flat prior | Granular prior | Actual |
|----------|-----------|---------------|--------|
| Sanctions + JPM | -15.6% | -18.0%* | -0.4% |
| Chip controls + NVDA | -16.0% | **-10.5%** | -5% to -8% |
| Ransomware + generic | -1.0% | **-0.3%** | -0.1% to -7% |
| Trade war + Apple | -4.6% | **-5.0%** | -6% to -8% |

*JPMorgan still overestimated because the model classified it as "medium" exposure (based on GICS sector = Financials). In reality JPM's Russia exposure was ~1% — that's "low." The model lacks company-specific geographic data for JPM, so it falls back to the sector-level guess.

NVIDIA improved significantly: -16% → -10.5%, with the actual -5% to -8% now inside the 80% credible interval. The granular prior for "medium exposure × revenue_market_access" is -10.5% (from 18 cases), much more accurate than the flat channel prior of -23%.

---

## Part 4: The Fundamental Question — Is 3B Worth It?

### The Honest Assessment

After three approaches (regression, flat Bayesian, granular Bayesian), the user asked the right question: "If the output isn't useful, what's the point in saying I can try predicting a number which could be wrong significantly?"

The core problem isn't the model — it's the data. What determines whether a company loses 0.4% or 18% from the same event type through the same channel are factors we can't observe:
- Was the company violating sanctions, or just doing routine business?
- Can the company create workaround products (like NVIDIA's A800)?
- How fast did management respond?
- What are the specific contract terms?

These factors live in the CEO's head and internal documents, not in SEC filings or stock prices.

### What 3B CAN Reliably Do

1. **Identify WHICH part of the business is at risk** — the channel prediction (Model 2) at 0.711 macro F1
2. **Quantify HOW EXPOSED the company is** — from SEC filings and exposure proxies
3. **Show WHAT COMPANIES IN SIMILAR SITUATIONS DID** — from 201 case studies matched by exposure similarity
4. **Provide a ROUGH RANGE** — "companies with your exposure level typically see -4% to -12%" (from granular priors)

What it CANNOT do: predict "-7.3% revenue impact" with any precision.

### The Decision

**Drop the specific revenue number. Keep the exposure analysis and precedent matching.** The value isn't "-7.3%". It's:

> "This sanctions event hits your financial_treasury channel. Your exposure is low (~2% Russia-linked business). Companies with low exposure through this channel historically saw -2% to -5% impact. JPMorgan wound down Russia exposure at minimal cost. BNP Paribas, which had high exposure from sanctions violations, lost 18%. Your profile is closer to JPMorgan."

That's useful. A fake-precise number isn't.

---

## Part 5: The Strategic Pivot — Audience and Value Proposition

### The Original Design

The model was built for **companies**: classify event → predict channel → estimate revenue impact → recommend strategies.

### The Problem

Companies already have Bloomberg, McKinsey, and internal risk teams. Our 201 case studies vs their decades of proprietary data. Our exposure estimates from public SEC filings vs their internal knowledge of their own supply chains.

### The Insight

The user identified the real gap: **we're guessing at exposure when the company already knows their own numbers.** If a company could enter:
- "40% of our components come from Taiwan"
- "We have 3 factories in the EU, 1 in China"
- "Our top customer is 25% of revenue and they're government-linked"

Then the model becomes dramatically more accurate because we're not guessing from GICS sectors — we have the actual numbers. And the exposure-matched precedent search becomes much more precise.

### Two Audience Options Considered

**Option A: Regular investors (portfolio focus)**
- "How does this event affect my stock portfolio?"
- Easier to build, but it's essentially event study methodology — published hundreds of times
- Not a novel research contribution

**Option B: Companies with custom input (firm-level assessment)**
- "Given OUR specific exposure data, how does this event affect US?"
- Novel research contribution: nobody has built a firm-level geopolitical exposure framework
- Publishable paper: "From Macro Risk to Micro Impact"
- The custom input is what makes it unique — the model stops guessing and starts using real data

**Decision: Option B.** Better research contribution, more defensible, and the technical components we've already built (channel taxonomy, exposure matching, case studies) are exactly what this needs.

---

## Part 6: Expanding Case Studies to 201

### The Expansion Journey

| Wave | Count | Focus |
|------|-------|-------|
| Original | 21 | Core events (Russia-Ukraine, chip controls, ransomware) |
| Wave 1 (→61) | +40 | Fill channel gaps: financial_treasury, regulatory, innovation |
| Wave 2 (→149) | +88 | Same major events, more company perspectives (Russia exits, COVID responses) |
| Wave 3 (→201) | +52 | Final gaps: cybersecurity (MGM, Caesars, Change Healthcare), reputation (Boeing, VW Dieselgate, NBA) |

### Why Automation Failed for Case Studies

We tried extracting case studies from EDGAR filings (37,989 mentions across 231 tickers). Only 17 mentions out of 37,989 contained specific action language ("we closed our operations", "we recorded a charge").

The reason: 10-K filings describe **risks** ("we are exposed to geopolitical disruption"), not **responses** ("we exited our Russian operations"). The specific action narratives come from earnings calls, press releases, and news coverage — not SEC filings.

### Coverage Check

All 10 channels covered with 10+ cases each. All 8 event categories represented. 130+ unique companies, all household names with verifiable public information.

41 out of 80 possible (category × channel) combinations have zero exact precedents. But the RAG retrieval uses embedding similarity, not exact matching — it finds adjacent cases (relevance 0.48-0.53) even for gap combinations. The relevance score communicates how close the match is.

---

## Roadblocks Hit and How We Tackled Them

### Roadblock 1: Revenue Auto-Extraction Produced Noise
- **Hit:** 539 auto-extracted labels from financial data had 0.114 correlation with stock reactions
- **Diagnosis:** Quarterly revenue changes reflect everything that happened in a quarter, not just the geopolitical event. ADM's -75% was commodity price normalization, not Red Sea attacks.
- **Solution:** Manual tagging from case study narratives instead. 90 hand-tagged labels from known outcomes.

### Roadblock 2: Bayesian Model Used Stock Reaction as Revenue Evidence
- **Hit:** Model 3B used car_1_5 as evidence for revenue prediction, contradicting the 3A/3B split rationale
- **Diagnosis:** User caught it: "Wasn't the whole point of dividing 3A and 3B to separate short term stock movements from long term impact?"
- **Solution:** Removed stock reaction from 3B entirely. Evidence comes only from exposure data and channel priors.

### Roadblock 3: Flat Channel Priors Treated All Companies the Same
- **Hit:** JPMorgan (0.4% actual loss) got the same -15.6% estimate as BNP Paribas (18% actual loss) because both were "financial_treasury"
- **Diagnosis:** Exposure level is the critical variable, not channel alone. Same channel, different exposure = completely different outcome.
- **Solution:** Built (channel_group × exposure_level) granular priors from 128 exposure-tagged labels. Low exposure through asset_loss = -3.6% vs high exposure = -60%.

### Roadblock 4: No Precedent for Many Category × Channel Combinations
- **Hit:** 41 out of 80 possible (event_category × channel) combinations have zero case studies
- **Diagnosis:** Most gaps are implausible combinations (technology_controls × workforce_talent). But some are real blind spots.
- **Solution:** RAG embedding retrieval naturally degrades gracefully — finds adjacent cases at lower relevance. Relevance score communicates confidence to user.

### Roadblock 5: The Revenue Number Itself Isn't Useful
- **Hit:** Even with granular priors, estimates were off by 15+ percentage points for individual companies
- **Diagnosis:** The factors that determine whether a company loses 0.4% or 18% (management response speed, contract specifics, workaround ability) aren't observable from any external data
- **Solution:** Stop predicting a number. Instead, show the range from the granular prior, match to similar-exposure precedents, and let the user (who knows their own situation) make the judgment.

---

## What's Planned Ahead

### Immediate Next Steps (Day 14)

**1. Custom Exposure Input Form**
Build a dashboard form where companies enter their own data:
- Geographic revenue breakdown (% from each region)
- Supplier concentration (% of inputs from affected areas)
- Facility locations (number and size in affected regions)
- Key customer concentration (% of revenue from government-linked or region-specific customers)

This replaces our GICS-based guessing with actual company-specific numbers.

**2. Exposure-Similarity Matcher**
Rebuild the precedent matching to use exposure similarity, not just event text similarity. When a company enters "30% China revenue through semiconductor sales," find the historical cases where a company with ~30% China revenue through technology channels faced a similar event — and show what happened to them.

This is different from the current RAG (which matches by event description) because it matches by **exposure profile**. A company with 30% China tech exposure should see NVIDIA and Applied Materials, not random companies that happened to be mentioned alongside "chip controls."

**3. Dashboard Rewire**
Replace the current output format:
- ~~Revenue impact: -4.6%~~ → "Companies with your exposure profile typically saw -4% to -12%"
- Add "Your exposure vs the precedent" comparison
- Show action vs inaction outcomes from paired case studies

### Paper Structure (Emerging)

> **Title:** "From Macro Risk to Micro Impact: A Framework for Firm-Level Geopolitical Exposure Assessment"
>
> **Contribution 1:** A 10-channel taxonomy for how geopolitical events transmit to firm-level business impact (validated on 201 historical cases)
>
> **Contribution 2:** An exposure-matching algorithm that pairs companies to historical precedents based on channel + exposure similarity (not just event similarity)
>
> **Contribution 3:** Empirical validation showing exposure-matched precedents predict outcomes better than event-category averages
>
> **Contribution 4:** An open-source tool where companies input specific exposure data and receive tailored, precedent-based assessments

### Remaining Technical Work

| Component | Status | What's needed |
|-----------|--------|---------------|
| Event classifier (Model 1) | Done | Paper write-up |
| Channel taxonomy (10 channels) | Done | Formal validation section |
| Channel prediction (Model 2) | Done, 0.711 F1 | Nothing |
| Exposure from SEC filings | Done for 99 companies | Already built |
| Custom exposure input | Not built | **Dashboard form** |
| Exposure-similarity matching | Not built | **New matching algorithm** |
| 128 exposure-tagged outcomes | Done | This IS the validation dataset |
| Model 3A market reaction | Done | Keep as-is |
| Model 3B revenue number | **Deprecated** | Replaced by exposure-matched ranges |
| Bayesian granular priors | Done (128 labels) | Feed into the matcher |
| 201 case studies | Done | Paper appendix |

---

## What I Learned

1. **Predicting a specific number requires understanding causation, not just correlation.** Revenue changes after geopolitical events are caused by dozens of factors — the event is one of them. Regression, Bayesian estimation, and every other method will struggle until you can isolate the event's causal contribution. That requires DiD or Synthetic Control, not fancier priors.

2. **The user knows their own situation better than any model.** Our biggest limitation was guessing at exposure from GICS sectors and SEC filings. A company that enters "40% of our components come from Taiwan" gives us more information in one input than all our XBRL extraction combined. The model should ask for input, not guess.

3. **A range with context beats a precise number.** "Companies with your exposure profile typically saw -4% to -12%, here's what they did" is more useful than "-7.3% revenue impact" because:
   - The user can calibrate against the precedents ("we're more like Nike than H&M because we didn't take a public stance")
   - The range communicates honest uncertainty
   - The precedents are actionable — they describe specific strategies with outcomes

4. **Exposure level is the critical variable.** Same event, same channel, 45x difference in outcome (JPM vs BNP Paribas). The channel tells you WHERE you're hit. The exposure level tells you HOW BAD. We had the channel right from Day 1. We only got the exposure granularity right on Day 13.

5. **Building a model and building a useful product are different problems.** We built four increasingly sophisticated models for revenue prediction (quantile regression, conformal, flat Bayesian, granular Bayesian). None produced numbers reliable enough to act on. The useful product turned out to be "show me what happened to companies like mine" — which doesn't need a prediction model at all, just a good matching algorithm and a curated case database.
