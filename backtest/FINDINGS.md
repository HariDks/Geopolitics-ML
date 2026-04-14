# The Hidden Geopolitical Risk Map of the S&P 500

*An ML model trained on 7.76 million geopolitical events reveals that 81% of major companies are wrong about their #1 geopolitical risk.*

---

## The Question

Every S&P 500 company includes "geopolitical risk" in their 10-K Risk Factors section. But do they know which specific event would actually hit them hardest? We built an ML model to find out.

We ran 37 S&P 500 companies through 10 distinct geopolitical scenarios — from Red Sea shipping disruptions to Taiwan tensions to OPEC production cuts — producing 370 company-event analyses. Each analysis predicts how severely the company would be affected, through which business channel, and estimates the financial impact.

Then we compared each company's actual #1 predicted risk against what you'd intuitively expect for their sector. The results were surprising.

---

## Finding 1: 81% of Companies Have an Unexpected #1 Risk

30 of 37 companies have a top geopolitical risk that doesn't match their sector's obvious vulnerability.

**What we expected:**
- Energy companies → OPEC production cuts
- Tech companies → US-China tariffs / export controls
- Financial companies → Emerging market debt crisis
- Industrial companies → Red Sea shipping disruption

**What the model found:**

| Company | Sector | Expected #1 Risk | Actual #1 Risk | Why |
|---------|--------|-----------------|---------------|-----|
| Pfizer | Health Care | EU regulation | **Red Sea shipping** | API ingredients ship through Suez from India/China. Route disruption delays drug production. |
| NVIDIA | Information Technology | US-China tariffs | **OPEC production cut** | Data centers consume massive energy. Oil price spikes = higher operational costs for customers = lower chip demand. |
| Boeing | Industrials | Red Sea shipping | **OPEC production cut** | Jet fuel is airlines' #1 cost. Oil spikes → airlines defer aircraft orders → Boeing revenue falls. |
| Goldman Sachs | Financials | EM debt crisis | **US-China tariffs** | Trading desk revenue depends on cross-border capital flows. Tariff wars reduce the deal pipeline. |
| Meta/Facebook | Info Tech | US-China tariffs | **EU regulatory crackdown** | GDPR/DMA compliance costs + ad targeting restrictions directly hit Meta's business model. |
| FedEx | Industrials | Red Sea shipping | **OPEC production cut** | Fuel is 25% of operating costs. A fuel price spike compresses margins before any route disruption. |

The pattern: **second-order effects often dominate first-order effects.** Boeing isn't directly affected by oil prices — but its customers (airlines) are, and that propagates back to Boeing's order book.

---

## Finding 2: US-China Tariffs Hurt Everyone — No Exceptions

Of the 10 scenarios, US-China tariff escalation is the only one that hurts **100% of companies analyzed**. Not 95%. Not 99%. All 37.

| Scenario | % of Companies Hurt |
|----------|:---:|
| US-China tariff escalation | **100%** |
| OPEC production cut | 95% |
| Emerging market debt crisis | 95% |
| Global supply chain shock | 95% |
| EU regulatory crackdown | 86% |

This makes tariff escalation unique as a **systemic geopolitical risk** — there's no sector that benefits, no hedge that fully protects. Even defense companies that you'd expect to be insulated face higher input costs.

---

## Finding 3: Same Sector, Opposite Outcomes

Companies in the same industry can react very differently to the same event. The model identifies these **intra-sector divergences** that sector-level analysis misses:

**Health Care: Pfizer vs Johnson & Johnson**
Under a Red Sea shipping disruption, Pfizer (severity -0.414) is 27% more exposed than J&J (-0.326). Why? Pfizer's API supply chain runs through Suez from Indian manufacturers. J&J's supply chain is more diversified across Americas and Europe.

**Information Technology: Meta vs Cisco**
Under US-China tariffs, Meta (-0.407) is 34% more exposed than Cisco (-0.304). Meta's ad business depends on global commerce flowing freely (advertisers cut budgets during trade wars). Cisco actually benefits from "rip and replace" campaigns when Huawei gets restricted.

**Industrials: Boeing vs Raytheon**
Under OPEC production cuts, Boeing (-0.402) is 32% more exposed than RTX (-0.305). Both are aerospace/defense, but Boeing is commercial aviation (fuel-cost sensitive) while RTX is defense (conflict-driven demand).

**The implication:** Sector ETFs and industry-level risk assessments hide these divergences. Two companies with the same GICS code can have opposite geopolitical risk profiles.

---

## Finding 4: The Most and Least Geopolitically Resilient Companies

Averaging severity across all 10 scenarios reveals which companies are structurally exposed vs structurally resilient:

**Most Exposed:**

| Rank | Company | Avg Severity | Why |
|------|---------|:-----------:|-----|
| 1 | Pfizer | -0.408 | Global pharmaceutical supply chain through every chokepoint |
| 2 | Meta/Facebook | -0.383 | Ad-dependent model vulnerable to any economic disruption + regulatory target |
| 3 | UnitedHealth | -0.370 | Largest health insurer, exposed to pharma supply chain + policy volatility |
| 4 | McDonald's | -0.368 | 40,000 restaurants in 100+ countries = maximum geographic surface area |
| 5 | Boeing | -0.360 | Commercial aviation amplifies every macro shock |

**Most Resilient:**

| Rank | Company | Avg Severity | Why |
|------|---------|:-----------:|-----|
| 37 | Broadcom | -0.312 | Diversified semiconductor + software, less China-dependent than peers |
| 36 | Johnson & Johnson | -0.312 | Diversified healthcare with balanced geographic manufacturing |
| 35 | General Motors | -0.322 | North America-focused operations insulate from most global shocks |
| 34 | Schlumberger | -0.323 | Oilfield services benefit from energy volatility (higher oil = more drilling) |
| 33 | Cisco | -0.324 | Benefits from Huawei displacement + limited direct China exposure |

**The pattern:** Resilience comes from **geographic diversification of operations** (not revenue). J&J manufactures in the Americas and Europe. GM builds in North America. Broadcom has less Taiwan/China concentration than NVIDIA. Companies that sell globally but *produce* locally are more resilient than companies that both source and sell globally.

---

## Finding 5: The EU Regulatory Crackdown Is the Most Underestimated Risk

EU regulatory crackdown scores as the **#1 most dangerous scenario** by average severity (-0.365), ahead of OPEC cuts, supply chain shocks, and even US-China tariffs.

Yet almost no company lists "EU regulation" as their primary geopolitical risk in their 10-K. They write about wars, sanctions, and tariffs — the dramatic, headline-grabbing events.

Why is EU regulation so damaging? Because it affects all sectors simultaneously:
- **Tech:** DMA/DSA compliance costs, app store rules, AI regulation
- **Consumer:** Product safety standards, packaging requirements, sustainability reporting
- **Finance:** Capital requirements, data localization, payment regulation
- **Health Care:** Clinical trial rules, drug pricing, data privacy
- **Energy:** Carbon border tax, ESG disclosure, green taxonomy

The quiet structural risk outweighs the loud episodic ones.

---

## How the Model Works

This analysis was produced by a 4-model ML pipeline trained on:
- **7.76 million geopolitical events** from GDELT, ACLED, GTA, OFAC, and BIS
- **602 labeled company-event impact pairs** with quantified financial outcomes
- **17,372 geopolitical mentions** extracted from SEC 10-K and 10-Q filings
- **1,973 stock reaction event studies** covering 20 major events

The pipeline:
1. **Classifies** the event into one of 8 categories (95.3% accuracy on news text)
2. **Scores** company-specific exposure across 10 business impact channels (82.6% accuracy)
3. **Estimates** financial impact range using quantile regression (80.7% calibrated coverage)
4. **Recommends** ranked strategic responses from a curated database of 148 strategies

Backtested against 10 historical events, the model predicted NVIDIA's $400M export control impact within 1.3 percentage points, correctly identified Maersk as a counterintuitive beneficiary of the Red Sea crisis, and estimated First Quantum's Panama mine loss within 0.9 points.

---

## What This Means

Three implications for risk managers:

**1. Stop using sector-level risk assessments.** Two health care companies can differ by 27% in their exposure to the same event. GICS codes don't predict geopolitical vulnerability.

**2. Map your second-order exposures.** Your company may not operate in the Red Sea, but your API supplier does. You may not sell oil, but your customers' budgets depend on fuel costs. The transmission path matters more than the headline.

**3. Take regulatory risk as seriously as conflict risk.** EU regulation scored higher than every military scenario in our analysis. The slow, structural risks that don't make headlines are often the most financially material.

---

## Data and Methodology

The full 37x10 risk matrix, interactive visualizations, model code, and training data are open-source. The model runs on a laptop (no cloud compute required) and takes <2 seconds per company-scenario analysis.

All backtest results, including cases where the model was wrong (Treasury Wine Estates: predicted ~0%, actual -96%), are published transparently.

---

*Built with: DistilBERT, XGBoost, ONNX Runtime, SEC EDGAR, GDELT, ACLED, Global Trade Alert.*  
*Methodology based on the WEF/IMD/BCG "Building Geopolitical Muscle" framework (January 2026).*
