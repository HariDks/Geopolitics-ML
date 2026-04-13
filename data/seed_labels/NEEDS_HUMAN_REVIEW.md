# Seed Labels — Requires Human Review

This directory will hold the 200-300 manually labeled event-impact pairs that seed the corporate outcome classifier (Model 2 + Model 3).

**No automated labels have been generated here.** Seed labeling requires human judgment.

---

## What needs to be labeled

Each label is a row in `seed_labels.csv` (schema below) linking a geopolitical event to a measured corporate impact.

**Target: 200-300 high-quality pairs.** Quality over quantity — a mislabeled seed degrades the semi-supervised model.

---

## Sources to draw from (in priority order)

### 1. WEF Report Case Studies (~20 pairs)
Source: "Building Geopolitical Muscle" — WEF/IMD/BCG, January 2026 (PDF in project root)

Cases to label:
- [ ] Teva Pharmaceutical — oncology drug stockpiling ahead of supply disruption
- [ ] Automotive company — $4B tariff impact (exact company may be anonymized in report)
- [ ] EDF — navigating M&A in shifting regulatory environment
- [ ] Rio Tinto — cross-functional tariff task force
- [ ] Siemens — "Value at Stake" methodology application
- [ ] Allianz — "Political Stability Grid" output
- [ ] Philips — exposure management case
- [ ] Nissan — Brexit supply chain restructuring
- [ ] Airbus — dual-use technology and ITAR compliance
- [ ] LATC (Latin American company, anonymized) — commodity exposure case

**Note from project owner:** Review the WEF PDF carefully — some case studies are anonymized or only partially described. For those, label what's described and flag confidence as "low."

### 2. Russia-Ukraine Corporate Exits (~50 pairs)
Source: Yale CELI tracker (School of Management database of corporate responses)
URL: https://som.yale.edu/story/2022/over-1000-companies-have-curtailed-operations-russia

High-value companies to prioritize (earnings calls available):
- [ ] McDonald's — sold Russia operations, ~$1.4B write-down (Q2 2022)
- [ ] Shell — Sakhalin-2 exit, ~$3.9B write-down (Q1 2022)
- [ ] BP — Rosneft stake exit, ~$24B write-down (Q1 2022)
- [ ] IKEA — manufacturing and retail closure
- [ ] Renault — AvtoVAZ stake sale
- [ ] Carlsberg — beer business exit (~$1.4B impairment)
- [ ] Heineken — Russia exit (~€300M)
- [ ] Unilever — kept operations but flagged reputational risk
- [ ] Maersk — Baltic/Black Sea route suspension

### 3. Red Sea Shipping Rerouting (~30 pairs)
Events: Houthi attacks on commercial shipping (December 2023 onwards)
Primary affected companies: Maersk, MSC, CMA CGM, Evergreen, COSCO

Label from earnings calls (Q4 2023, Q1 2024, Q2 2024):
- [ ] Maersk Q4 2023 — route diversion costs, capacity utilization impact
- [ ] Maersk Q1 2024 — full quarter of Cape of Good Hope rerouting cost
- [ ] MSC statement on Red Sea avoidance
- [ ] Retail companies (IKEA, H&M, Zara) — inventory delays disclosed in earnings
- [ ] Energy companies — LNG tanker rerouting (Qatargas, TotalEnergies)

### 4. US-China Semiconductor Controls (~50 pairs)
Events: Oct 2022 BIS rule, Oct 2023 update, Oct 2024 expansion; TSMC, ASML restrictions

Label from earnings calls:
- [ ] NVIDIA Q3 2022 — H100/A100 export control impact ($400M guidance cut)
- [ ] NVIDIA Q4 2023 — A800/H800 restriction impact
- [ ] AMD — MI300 export restrictions
- [ ] ASML — EUV shipment halt to China
- [ ] TSMC — revenue guidance adjustment for China customers
- [ ] Applied Materials, Lam Research, KLA — tool export restrictions
- [ ] Intel — Xeon China sales impact
- [ ] Qualcomm — Huawei license revocation impact

### 5. European Energy Crisis (~30 pairs)
Events: Nord Stream disruption (Sep 2022), Russian gas cutoff (2022-2023)

Label from earnings calls and press releases:
- [ ] BASF — Verbund site energy cost and production cut disclosures
- [ ] ThyssenKrupp — steel production reduction due to gas prices
- [ ] Covestro — energy surcharge and plant idling
- [ ] Wacker Chemie — silicon production cut
- [ ] Yara — fertilizer production capacity reduction
- [ ] European auto OEMs — energy cost in COGS

### 6. 2025 US Tariff Escalation (~30 pairs)
Events: April 2025 tariff announcements (Liberation Day tariffs and subsequent negotiations)

Label from Q1/Q2 2025 earnings calls:
- [ ] Apple — supply chain cost disclosure, India shift
- [ ] Tesla — China import/export exposure
- [ ] GM/Ford — steel/aluminum tariff COGS impact
- [ ] Walmart/Target — consumer goods sourcing cost
- [ ] Nike/Adidas — Vietnam/Indonesia sourcing exposure

---

## Label schema (seed_labels.csv)

```
event_id,company_ticker,company_name,sector_gics,impact_channel,quarter,
mention_text,mention_sentiment,management_action_described,
revenue_delta_pct,cogs_delta_pct,operating_income_delta_pct,capex_delta_pct,
car_1_5,car_1_30,source,confidence,labeled_by,human_reviewed,notes
```

**Key fields explained:**
- `event_id`: Link to geopolitical_events table (or use descriptive string if not yet in DB)
- `impact_channel`: One of the 10 channels from taxonomy.json
- `mention_text`: Exact quote from earnings call or press release (keep under 500 chars)
- `mention_sentiment`: Your assessment: -1 (very negative) to 1 (positive)
- `management_action_described`: What did they say they're doing about it?
- Financial deltas: quarter-over-quarter % change; NULL if not disclosed
- `confidence`: "high" (quantified impact), "medium" (described but not quantified), "low" (mentioned only)
- `labeled_by`: Your name/initials
- `human_reviewed`: Set to 1 when you're satisfied with the label

---

## Do not automate these labels

The pipeline can auto-extract keyword mentions from earnings calls (Weeks 5-7).
But the 200-300 seed labels here require:
1. Reading the actual source text, not just keyword matching
2. Verifying the financial figures are correctly linked to the geopolitical event (not other factors)
3. Assessing whether management's described action is real or boilerplate
4. Calibrating sentiment scores consistently

These are the training labels that everything else depends on. Get them right.
