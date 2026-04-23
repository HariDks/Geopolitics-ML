# Learning Day 10 — From Model Demo to Decision Tool

**Project:** Geopolitical Muscle ML Model  
**Date:** April 18-19, 2026  
**Written for:** Understanding the transition from "cool model" to "something people would actually use" — through three rounds of product review, sign correction, severity separation, and a complete dashboard rebuild

---

## Where We Left Off

At the end of Day 9 we had:
- Channel prediction solved in text-rich mode (95.2% cross-validated)
- Blind evaluation showing 62.3% top-2 on novel event-company pairs
- Dual-mode reliability labeling (text_rich / text_poor)
- All models working, all evaluation honest
- Project on GitHub at https://github.com/HariDks/Geopolitics-ML

The models were done. The dashboard existed. But it still felt like a model demo, not a product.

Day 10 was about **closing the gap between "interesting system" and "useful tool."**

---

## Part 1: Publishing to GitHub and Streamlit Cloud

### GitHub

Published at **https://github.com/HariDks/Geopolitics-ML** as a public repo. The transparency is the USP — unlike Blackrock's closed geopolitical risk index, everything is open: code, data, methodology, and honest accuracy numbers.

### Streamlit Cloud Deployment

Three deployment issues hit:

**Issue 1: `gdeltPyR` not available on Python 3.14.**
The full `requirements.txt` included pipeline dependencies the dashboard doesn't need. Fix: created `requirements-full.txt` for the pipeline and a lightweight `requirements.txt` with dashboard-only deps.

**Issue 2: PyTorch import on Streamlit Cloud.**
The `tokenizer.json` was gitignored, so the ONNX path couldn't find it and fell back to PyTorch (which isn't installed on Streamlit Cloud). Fix: un-gitignored `tokenizer.json` (695KB).

**Issue 3: ONNX runtime failure on Python 3.14.**
Even with the ONNX model and tokenizer, `onnxruntime` crashed on Python 3.14. Fix: added a 3-tier fallback in the classifier: ONNX → PyTorch → keyword-based. The keyword classifier uses curated keyword lists per category (~70% accuracy, not as good as ONNX's 95% but ensures the app runs on any environment).

**Issue 4: SQLite cross-thread errors.**
Streamlit runs in multiple threads, SQLite connections can't be shared. Fix: wrapped all `get_db_connection()` calls in try/except so models gracefully fall back to zero features when the database isn't available.

---

## Part 2: Three Rounds of Product Review

The reviewer evaluated the dashboard three times, each time pushing it closer to a real product. The feedback followed a pattern: **the model is fine, the presentation is wrong.**

### Round 1: "It feels like a demo, not a product"

Six problems identified:

1. **No clear takeaway** — users see channels and probabilities but don't know what to conclude
2. **No immediate wow moment** — have to think before acting
3. **Explanation not punchy** — detection signals listed but not compelling
4. **System insight hidden** — the strongest differentiator (knowing when it's uncertain) wasn't surfaced
5. **Impact feels abstract** — percentages without dollar context
6. **No feedback loop** — use it and leave, no interaction

**Fixes applied:**
- Top-line summary card: "What this means for [Company]"
- 8 one-click scenario buttons as hero section
- Structured bullet explanation with match indicators
- Blue info box for system insight
- Dollar range in summary
- Inline feedback: useful? channel correct? correction dropdown

### Round 2: "The predictions feel off"

The reviewer did a case-by-case analysis of all 8 preloaded scenarios and found:

- **Sign errors**: ransomware attack predicting positive impact, boycott showing positive range
- **"Minimal impact" for $400M losses**: FedEx/NotPetya predicted "minimal" because -0.5% of revenue
- **Wrong company pairings**: BP's Rosneft write-down showing Exxon as the company
- **Magnitude too generic**: same prediction for very different companies

**Root cause analysis:**
1. Model's impact predictions cluster near zero (trained on event study data where most companies barely react)
2. Sign comes from the model's raw output which often defaults to slightly positive
3. "Minimal/Moderate/Significant" labels conflate financial percentage with business severity
4. Event descriptions described one company's experience but paired with a different company

### Round 3: "Separate what you know from what you're guessing"

The deepest feedback round. Key insight: **the model compresses three different concepts into one output:**

1. Business severity (is this a big deal?)
2. Financial magnitude (how much money?)
3. Confidence (how sure are we?)

These should be presented separately.

---

## Part 3: Sign Correction

### The Problem

The impact estimator's raw output frequently had wrong signs:
- "Ransomware destroys IT systems" → model predicts +$752M (positive)
- "Boycott wipes brand from platforms" → model predicts +$126M (positive)
- "Export controls restrict chip sales" → model predicts +$77M (positive)

### Why It Happened

The impact model was trained on event study stock reactions, where most S&P 500 companies show near-zero or slightly positive reactions to geopolitical events (because the average company isn't directly affected, and some benefit). The model learned "default to slightly positive" as a safe prediction.

### The Fix

Rule-based sign correction using 40+ negative signal keywords:

```python
NEGATIVE_SIGNALS = [
    "loss", "boycott", "restriction", "sanction", "impairment",
    "write-down", "destroy", "attack", "ransomware", "collapse",
    "tariff", "penalty", "crisis", "disruption", "shutdown", ...
]
```

If the event text contains clearly negative signals but the model predicts positive, flip the sign. This catches 60-70% of the "feels wrong" cases.

### The Tradeoff

The reviewer correctly noted: "Direction adjusted" in the output tells users "the model didn't naturally get this right, so we overrode it." That makes predictions feel patched. Solution: moved sign correction info to diagnostics, not the main summary.

---

## Part 4: Separating Financial Scale from Operational Severity

### The Problem

"Minimal impact" for a $400M ransomware loss feels wrong to any human. But mathematically, $400M on $88B revenue IS <0.5% — which our threshold called "minimal."

### The Insight

Financial percentage and operational disruption are different axes:

| | Low financial % | High financial % |
|---|---|---|
| **Low operational** | Trade agreement signed | Major market loss |
| **High operational** | Ransomware ($400M on $88B = 0.5%) | Supply chain collapse |

A cyber attack can be "low financial scale" but "high operational severity." Conflating them into one label ("minimal") is misleading.

### The Fix

Two separate labels:

- **Financial scale**: Limited / Low-to-moderate / Moderate / Significant / Severe (based on % of revenue)
- **Operational severity**: High / Medium-to-high / Medium (based on channel type — cybersecurity and logistics are inherently high-disruption channels)

FedEx/NotPetya now shows: Financial scale: Low-to-moderate | Operational severity: High.

---

## Part 5: Preloaded Scenario Fixes

### Wrong Company Pairings

The original scenarios had mismatched event-company pairs:
- "BP exited Rosneft stake" paired with Exxon Mobil
- Events describing one company's loss but showing analysis for a different company

**Fix**: Rewrote each event description to specifically describe what happened to the paired company:
- "Energy companies with Russian operations faced asset freezes..." (for Exxon)
- "FedEx subsidiary TNT Express lost $400M from the attack..." (for FedEx)

### Hardcoded Severity Tags

Original preloaded scenarios had manual tags ("Severe", "Moderate") that didn't match the model's predictions. A scenario tagged "Severe" would show "Minimal" from the model — confusing.

**Fix**: Removed all hardcoded tags. The model's own prediction is the only label shown.

---

## Part 6: The Three-Page Architecture

### Why One Page Didn't Work

The single-page layout crammed everything together:
- Landing text + scenario buttons + custom input + results + feedback
- Preloaded scenarios locked the text box (disabled=True)
- Clicking feedback radio buttons wiped analysis results (Streamlit re-render)

### The Three Pages

**Page 1: Overview** — What the tool does, why it exists, system insight, performance, feedback invite. No analysis.

**Page 2: Preloaded Examples** — 8 curated scenarios as expandable cards. Click to see full analysis. Each clearly shows "Event -> Company."

**Page 3: Custom Analysis** — Text area + company dropdown + Analyze button. Full analysis results.

### Session State

Results stored in `st.session_state.results` so they persist across re-renders. Clicking feedback no longer wipes the analysis.

---

## Part 7: The Landing Page Evolution

The landing page went through 5 iterations based on reviewer feedback:

**v1 (initial)**: Wall of text explaining the model. Reviewer: "Too heavy, can't act in 3 seconds."

**v2 (3-column)**: Three columns — What you get | Under the hood | Honest about. Reviewer: "Better but still info-first, not action-first."

**v3 (CTAs first)**: Moved "Start here" buttons above all content. Reviewer: "Good, but CTAs don't feel clickable."

**v4 (visual cards)**: Rounded gray cards for CTAs, dark blue card for system insight, 4-column feature grid. Reviewer: "Almost there — reorder sections."

**v5 (final)**: Exact structure:
```
Hero (sharp, centered)
  ↓
CTAs (action cards)
  ↓
Why this exists (credibility)
  ↓
What you get (4 feature cards)
  ↓
System insight (dark card: "better at explaining than estimating")
  ↓
Model performance (accuracy + limitations separated)
  ↓
What powers this (collapsed)
  ↓
How to interpret (collapsed)
  ↓
Feedback (hook line: "improves through real-world feedback")
  ↓
Disclaimer ("starting point, not final answer")
```

Key principle: **"do this → then understand more if you want"** — not "here is everything about the system."

---

## Part 8: The Result Page Structure

The result page was the most important redesign — where the reviewer's 8-section structure was implemented:

### Section 1: Assessment (THE takeaway)

```
This event is likely to have a mild negative financial impact on FedEx,
driven primarily by cybersecurity / IT disruption.

Confidence: High
```

This is a colored banner (red/yellow/green) — the first thing users see. One sentence that answers "so what?"

### Section 2: Core Output (2-column)

```
Key driver: Cybersecurity It         Financial scale: Low-to-moderate
Secondary: Logistics Operations      Operational severity: High
Estimated: -0.1% to -0.9%           Reliability: High
```

### Section 3: Event Interpretation (one line)

```
Event interpretation: Armed Conflict Instability (96% confidence)
```

### Section 4: Expandable Insight Blocks

Three collapsed expanders:
- **What could change this**: "If FedEx has significant regional exposure..."
- **Why this might be wrong**: "Does not include FedEx's specific revenue by geography..."
- **Historical context**: "Similar events in the armed conflict category have historically..."

### Section 5: Channel Breakdown

Primary and secondary channels with probabilities and descriptions.

### Section 6: Why This Prediction

Natural language, not detection syntax:
```
The model picks up signals like "ransomware" and "IT systems,"
which are typically associated with cybersecurity impacts.
```

Instead of:
```
[+] Detected "ransomware" -> Cybersecurity It
```

### Section 7: Full Diagnostics (collapsed)

Event classification, all probabilities, sign confidence, mode.

### Section 8: Feedback

"Was this useful?" + "Main driver correct?" + correction dropdown + comment.

---

## Part 9: What the Reviews Taught About Product Design

### Lesson 1: The model output is not the product output

The model predicts a probability distribution over 10 channels and a percentage impact range. That's useful for a data scientist. A user needs: "This event will probably hurt Apple through its supply chain, costing roughly $2-5B. We're fairly confident because the event description is detailed."

**The translation layer between model and user is as important as the model itself.**

### Lesson 2: Honesty is a feature, not a limitation

The reviewer's strongest praise was for the reliability labeling ("text_rich → high confidence"). Every time we added more transparency about what the model doesn't know, the product felt more trustworthy.

### Lesson 3: Separate what you know from what you're guessing

Financial percentage, operational severity, and confidence are three different things. Conflating them into "Minimal impact" makes the system feel stupid when it's actually making a reasonable (if narrow) prediction.

### Lesson 4: Action-first, explanation-later

The landing page went from "here's everything about the system" to "click this to see what happens" in 5 iterations. Every iteration that moved the CTA higher increased perceived quality.

### Lesson 5: Case-by-case validation reveals what metrics hide

The reviewer's manual testing of 8 scenarios found the sign error, the severity mismatch, and the wrong company pairings — none of which showed up in aggregate metrics like "82.5% macro F1" or "62.3% top-2 accuracy."

---

## Part 10: Files Changed on Day 10

```
dashboard/app.py           — Complete rewrite (5 iterations)
.gitignore                 — Un-gitignored tokenizer.json
requirements.txt           — Dashboard-only deps (no gdeltPyR)
requirements-full.txt      — Full pipeline deps (moved from requirements.txt)
packages.txt               — Streamlit Cloud system packages
models/event_classifier/predict.py — 3-tier fallback (ONNX → PyTorch → keyword)
models/exposure_scorer/predict.py  — Graceful DB fallback
models/impact_estimator/predict.py — Graceful DB fallback
```

---

## Summary: Day 10 in One Paragraph

Day 10 transformed the Geopolitical Impact Tester from a model demo into a decision tool through three rounds of product review and approximately 15 incremental commits. The biggest fixes were: rule-based sign correction (40+ negative keywords flip positive predictions for clearly negative events), separating financial scale from operational severity (a $400M ransomware loss is "low financial" but "high operational"), rebuilding as a 3-page app (Overview / Preloaded Examples / Custom Analysis), and restructuring the result page into 8 sections with the decision takeaway at the top. The landing page went through 5 iterations following the principle "do this → then understand more if you want." Three Streamlit Cloud deployment issues were fixed (requirements split, tokenizer tracking, ONNX/keyword fallback). The reviewer's deepest insight: **the model compresses business severity, financial magnitude, and confidence into one output — but users need these as three separate signals.** The system now explicitly separates them, and every prediction starts with a one-sentence assessment that answers "so what should I conclude?" The project is live on GitHub and deployable to Streamlit Cloud.
