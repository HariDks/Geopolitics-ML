# Learning Day 1 — What We Built, Why, and How

**Project:** Geopolitical Muscle ML Model  
**Date:** April 3-4, 2026  
**Written for:** Understanding the full picture from scratch

---

## The Big Picture First (Before Any Code)

Think of what we're building like this:

> A company like Apple needs to know: "If the US bans chip exports to China tomorrow, how much money do we lose, and what should we do about it?"

Right now, companies read the news and someone in government affairs writes a report. It takes weeks, it's based on gut feel, and different companies with similar exposure give completely different answers.

**What this model does:** Automates that process. It watches thousands of news events, conflict databases, and government announcements every day. When something happens, it figures out what *type* of geopolitical event it is, which *industries and companies* are most exposed, estimates a *dollar impact*, and recommends *strategic responses* — all in minutes instead of weeks.

The model has **4 stages** that feed into each other like an assembly line:

```
Raw news/events → [Stage 1: What happened?] → [Stage 2: Who's exposed?] → [Stage 3: How much does it cost?] → [Stage 4: What should you do?]
```

Today (Day 1), we built the **foundation** that all 4 stages sit on: the database, the data pipelines, and the taxonomy mappings. Without this, there is no model.

---

## Part 1: The Taxonomy — The Most Important Concept to Understand

### What is a taxonomy?

A taxonomy is just a classification system. Think of how a doctor classifies diseases — not every sickness is called "being sick," there's influenza, pneumonia, diabetes, etc. Because things are named and categorized, doctors can look up what drug works for what disease.

We need the same thing for geopolitical events. Not every bad thing in the world is "geopolitical risk." We defined **8 specific categories**:

| # | Category | Real-world example |
|---|----------|-------------------|
| 1 | Trade Policy Actions | US tariffs on Chinese goods (2025) |
| 2 | Sanctions & Financial Restrictions | US freezing Russian bank assets (2022) |
| 3 | Armed Conflict & Instability | Russia invading Ukraine (Feb 2022) |
| 4 | Regulatory & Sovereignty Shifts | China requiring data to stay in China |
| 5 | Technology Controls | US banning chip sales to Huawei |
| 6 | Resource & Energy Disruptions | OPEC cutting oil production |
| 7 | Political Transitions & Volatility | Military coup in Sudan (2023) |
| 8 | Institutional & Alliance Realignment | UK leaving the European Union (Brexit) |

And we defined **10 ways a business gets hurt** (impact channels):

| # | Channel | Example |
|---|---------|---------|
| 1 | Procurement & Supply Chain | Can't get chips from Taiwan anymore |
| 2 | Revenue & Market Access | Can't sell products in Russia |
| 3 | Capital Allocation | Asset in Russia now worth $0 |
| 4 | Regulatory Compliance | Need to hire lawyers for new rules |
| 5 | Logistics & Operations | Ships rerouted around Africa (Red Sea crisis) |
| 6 | Innovation & IP | Can't collaborate with Chinese university |
| 7 | Workforce & Talent | Employees can't get visas |
| 8 | Reputation & Stakeholders | Boycotts for staying in Russia |
| 9 | Financial & Treasury | Currency collapses, can't repatriate profits |
| 10 | Cybersecurity & IT | State-sponsored cyberattack |

**Why does this matter for the model?** Because all 4 ML models speak this same language. Stage 1 asks "what category is this event?" Stage 2 asks "which impact channels does this hit?" Stage 3 asks "how big is the dollar impact on each channel?" Stage 4 asks "what's the response strategy for this category × channel combination?"

The taxonomy is the **shared language** of the entire system. If it's wrong or inconsistent, everything downstream is wrong.

---

## Part 2: The Data Sources — Where We Get Events From

The model needs to learn from history. We need a database of thousands of geopolitical events that happened between 2015-2025, tagged with the 8 categories above. We get this from 3 main sources:

### Source 1: GDELT (Global Database of Events, Language, and Tone)
- **What it is:** A massive project that reads every major news article in the world, every 15 minutes, and extracts events as structured data. It goes back to 1979.
- **Free to use:** Yes, completely open
- **The problem:** It uses its own coding system called CAMEO (Conflict and Mediation Event Observations). CAMEO has codes like "190" (armed fighting) or "171" (sanctions imposed). These codes don't match our 8 categories — we have to build a translation table.
- **The other problem:** GDELT is extremely noisy. A news article about a minor town hall dispute gets coded the same way as a war. We have to filter heavily.

### Source 2: ACLED (Armed Conflict Location & Event Data)
- **What it is:** A research organization that manually codes every armed conflict, protest, and political violence event in the world. Much cleaner data than GDELT.
- **Free for research:** Yes, requires registration
- **What it covers:** Mostly Armed Conflict (category 3) and Political Transitions (category 7)
- **Unique value:** It has exact GPS coordinates and fatality counts, which help us score severity

### Source 3: OFAC SDN List (US Treasury Sanctions)
- **What it is:** The US government's list of entities that are sanctioned — companies, ships, people
- **Free to use:** Yes, fully public
- **What it covers:** Sanctions & Financial Restrictions (category 2)
- **Unique value:** It's official — no interpretation needed. If something is on the SDN list, it's a sanction.

---

## Part 3: What We Actually Built Today

### Step 1: The Folder Structure

Before writing any code, we created the skeleton of the project — all the folders that will hold data, models, pipelines, and config:

```
Geopolitics ML/
├── data/
│   ├── raw/        ← downloaded files before processing
│   ├── processed/  ← cleaned data ready for the model
│   ├── mappings/   ← the translation tables (critical!)
│   └── seed_labels/ ← 200-300 human-labeled training examples
├── config/         ← the settings and taxonomy definitions
├── database/       ← the SQL schema
├── pipelines/      ← the Python scripts that download and process data
├── models/         ← where the 4 ML models will live (not built yet)
└── notebooks/      ← Jupyter notebooks for exploration (not built yet)
```

**Why this matters:** Every data science project needs a clean structure. Raw data should never be mixed with processed data. Config should be separate from code. This structure makes the project maintainable.

---

### Step 2: The .env File (Managing Secrets)

Some data sources require credentials — API keys, usernames, passwords. You never want to put these directly in code (it could get accidentally uploaded to GitHub where everyone can see it).

The solution: a `.env` file that sits on your computer locally and is never shared.

We created `.env.template` (a blank template anyone can see) and `.env` (the actual file with your credentials, never committed to version control).

```
# What .env.template looks like (safe to share):
ACLED_EMAIL=your_registered_email@example.com
ACLED_PASSWORD=your_acled_password_here

# What .env looks like (private to your machine):
ACLED_EMAIL=hari2021@uchicago.edu
ACLED_PASSWORD=<your actual password>
```

**Roadblock hit:** We initially assumed ACLED used a static API key (like most APIs). When we read their actual documentation, they switched to OAuth — a more modern system where you log in with email + password to get a temporary 24-hour access token. We updated both the `.env` template and the pipeline code to use the OAuth flow.

---

### Step 3: The Three Taxonomy Mapping Tables (The Critical First Deliverable)

These three JSON files are arguably the most important thing built today. They're the "translation dictionaries" between raw data formats and our 8-category taxonomy.

#### File 1: `cameo_to_taxonomy.json`

GDELT uses CAMEO codes. Here's what the mapping looks like:

```
CAMEO code 194 ("Engage in war") → armed_conflict_instability, confidence: high
CAMEO code 171 ("Impose sanctions") → sanctions_financial_restrictions, confidence: high
CAMEO code 166 ("Impose embargo") → trade_policy_actions, confidence: high
CAMEO code 133 ("Issue ultimatum") → armed_conflict_instability, confidence: medium
```

Each mapping also has:
- **Confidence level** (high/medium/low) — how reliable is this translation?
- **Filtering notes** — e.g., "only include if GoldsteinScale <= -5" (GoldsteinScale is GDELT's measure of how negative an event is, -10 to +10)
- **Validation notes** — e.g., "Russia invasion Feb 24 2022 should generate dense 190/193/194 events"

**Important caveat documented in the file:** GDELT is terrible at covering trade policy (category 1), technology controls (category 5), and regulatory shifts (category 4). These categories need other sources (GTA, BIS). We explicitly documented this so future work doesn't waste time trying to fix GDELT for categories it can't cover.

#### File 2: `acled_to_taxonomy.json`

ACLED has its own event types. The mapping:

```
"Battles" → armed_conflict_instability, default_severity: 4
  └── "Armed clash" → severity 3
  └── "Non-state actor overtakes territory" → severity 5

"Protests" → political_transitions_volatility, default_severity: 2
  └── "Peaceful protest" → severity 1 (filtered out unless large-scale)
  └── "Excessive force against protesters" → severity 4

"Strategic developments" → varies by sub-type
  └── "Coup" → political_transitions_volatility, severity 5 (always)
```

There's also a severity formula: fatality counts can override the default severity. 500+ deaths → severity 5, no matter what the default is.

#### File 3: `gta_to_taxonomy.json`

The Global Trade Alert database uses its own intervention type codes. Key mappings:

```
"Import tariff" → trade_policy_actions, subtype: tariff_adjustment
"Export ban" → trade_policy_actions OR technology_controls
  └── If it's semiconductors/chips → technology_controls (higher priority)
  └── If it's critical minerals → resource_energy_disruptions
  └── Everything else → trade_policy_actions
"Foreign investment screening" → regulatory_sovereignty_shifts
"Data localisation" → regulatory_sovereignty_shifts, severity 4
```

GTA also tags each intervention as "Red" (harmful/discriminatory), "Green" (liberalizing), or "Amber" (unclear). Red interventions get higher severity scores.

---

### Step 4: The Config Files

Three JSON files in `config/` define the system's parameters:

**`taxonomy.json`** — The master list of all 8 categories, 10 channels, and their metadata. Every pipeline refers to this as the source of truth.

**`lambda_rates.json`** — This is for a mathematical technique called **exponential decay weighting**. The idea: an event from 2016 should matter less than an event from 2024, because the world has changed. But *how much less* depends on the type of event.

```
Political Transitions: λ = 0.8 → A 2016 election is basically irrelevant today
Institutional Realignment: λ = 0.2 → Brexit (2016) is still shaping UK trade today
Trade Policy: λ = 0.5 → 2018 tariffs less relevant than 2025 tariffs, but still matter
```

The formula is: `Weight = e^(-λ × years_ago)`

So for Trade Policy (λ=0.5):
- An event 1 year ago: weight = 0.61 (still 61% relevant)
- An event 3 years ago: weight = 0.22 (22% relevant)
- An event 5 years ago: weight = 0.08 (8% relevant)

**`priority_matrix.json`** — The 8×10 grid of Frequency × Severity scores from Phase 1 (the expert-assessed phase that was already done before we started coding). Every cell scores 1-25. The 8 cells at score 25 (maximum) are:

- Trade Policy × Supply Chain (score 25)
- Trade Policy × Revenue (score 25)
- Sanctions × Financial/Treasury (score 25)
- Armed Conflict × Logistics (score 25)
- Technology Controls × Supply Chain (score 25)
- Technology Controls × Innovation/IP (score 25)
- Technology Controls × Cybersecurity (score 25)
- Resource Disruptions × Supply Chain (score 25)

---

### Step 5: The Database Schema

We defined the structure of our database — essentially the "column headers" for all the data we'll store. Two main tables:

**`geopolitical_events`** — One row per geopolitical event:
```
event_id: "EVT-ACLED-20220224-ABC123"
source: "acled"
event_category: "armed_conflict_instability"
event_date: 2022-02-24
affected_countries: ["UA", "RU"]
severity_estimate: 5
fatalities: 137
...
```

**`corporate_impacts`** — One row per company-event linkage (built in Weeks 5-7):
```
company_ticker: "MCD"
event_id: "EVT-ACLED-20220224-..."
impact_channel: "revenue_market_access"
revenue_delta_pct: -8.3
mention_text: "We have suspended operations in Russia..."
car_1_5: -0.032  ← stock dropped 3.2% in 5 days after event
...
```

We chose **SQLite** for prototyping (it's a single file on your computer, no server needed) with the plan to upgrade to **PostgreSQL** in production.

---

### Step 6: The Pipeline Scripts

Four Python scripts in `pipelines/`:

**`utils.py`** — Shared helper functions used by all pipelines:
- `get_db_connection()` — opens the database
- `cameo_to_taxonomy()` — looks up a CAMEO code in our mapping table
- `acled_to_taxonomy()` — looks up an ACLED event type
- `make_event_id()` — generates a unique ID like "EVT-ACLED-20220224-ABC123"
- `event_exists()` — checks if we already stored this event (prevents duplicates)

**`ingest_gdelt.py`** — Downloads GDELT data day by day, filters, maps to taxonomy, stores:
```bash
python pipelines/ingest_gdelt.py --start 2022-02-24 --end 2022-03-31
```

**`ingest_acled.py`** — Authenticates with ACLED, paginates through their API, maps, stores:
```bash
python pipelines/ingest_acled.py --start 2020-01-01 --end 2025-12-31
```

**`ingest_ofac.py`** — Downloads the 122MB OFAC sanctions XML, parses it, groups by sanction program and date, stores:
```bash
python pipelines/ingest_ofac.py --since 2020-01-01
```

**`compute_frequency_scores.py`** — Takes everything in the database, applies the exponential decay formula, and produces a "live" priority matrix to compare against the Phase 1 expert scores:
```bash
python pipelines/compute_frequency_scores.py
```

---

## Part 4: The Roadblocks We Hit (And How We Solved Them)

This is the most educational part — real-world data engineering almost never goes smoothly on the first try.

---

### Roadblock 1: ACLED Changed Their Authentication System

**What we expected:** ACLED would have a static API key — a long string like `abc123xyz` that you paste once and use forever. That's what the README assumed.

**What we found:** ACLED changed to OAuth (Open Authorization) — a modern auth system where instead of a static key, you:
1. Send your email + password to an endpoint
2. Get back a temporary "access token" valid for 24 hours
3. Include that token in every subsequent API request

**Why OAuth is better:** If your static API key leaks (e.g., accidentally committed to GitHub), anyone can use it forever. With OAuth tokens, leaked tokens expire in 24 hours and are useless after that.

**How we fixed it:**
1. Updated `.env` — removed `ACLED_API_KEY`, added `ACLED_PASSWORD`
2. Added `get_acled_token()` function in the pipeline that POSTs credentials and extracts the token
3. Changed all API calls to include `Authorization: Bearer {token}` header

**Test result:** Worked. 563 events fetched for Ukraine/Russia Feb 24-28, 2022.

---

### Roadblock 2: ACLED's Field Names Changed

**What we expected:** Events would have a `data_id` field as their unique identifier (based on old documentation).

**What we found:** The field is now called `event_id_cnty` (a composite of country + sequential number, like "TUR10582").

Also: we were using `count` to track pagination (how many total records exist), but `count` just echoes back the page size limit. The actual total is in `total_count`.

**Impact of the bug:** The pipeline was stopping after page 1 (5,000 records) thinking it had fetched everything, when there are 1.74 *million* events globally for 2020-2025.

**How we found it:** The first run showed 5,000 fetched → 1,964 duplicates. A fresh database shouldn't have any duplicates. That was the red flag that made us inspect the API response directly.

**How we fixed it:**
1. Changed `data_id` → `event_id_cnty` in the fields request and the storage code
2. Changed pagination check from `len(all_events) >= count` to `len(data) < ACLED_PAGE_SIZE` (stop when you get a partial page, not when you hit the wrong total)

---

### Roadblock 3: OFAC Moved Their XML File

**What we expected:** The OFAC SDN XML file would be at the URL in the original code: `https://www.treasury.gov/ofac/downloads/sdn_advanced.xml`

**What we found:** 404 Not Found. OFAC completely rebuilt their sanctions list portal in 2024. The file now lives at a new API endpoint: `https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN_ADVANCED.XML`

**How we found the new URL:** Tried several candidate URLs, checked HTTP status codes, and used a research agent to look at OFAC's own documentation and recent technical announcements. They published a notice about this change.

**How we fixed it:** One line change — updated `OFAC_XML_URL` in the pipeline.

---

### Roadblock 4: OFAC Changed Their XML Schema

**What we expected:** The XML would have elements like `<sdnEntry>`, `<uid>`, `<sdnType>`, `<programList>` — the classic SDN format that existed for years.

**What we found:** After downloading the new 122MB file and getting 0 parsed entities, we inspected the actual XML structure. It's completely restructured:

Old format:
```xml
<sdnEntry>
  <uid>36</uid>
  <lastName>AERO-CARIBBEAN</lastName>
  <sdnType>Entity</sdnType>
  <programList><program>CUBA</program></programList>
</sdnEntry>
```

New format:
```xml
<DistinctParty>
  <Profile ID="36" PartySubTypeID="3">
    <Identity>
      <Alias Primary="true">
        <NamePartValue>AERO-CARIBBEAN</NamePartValue>
      </Alias>
    </Identity>
  </Profile>
</DistinctParty>

<SanctionsEntry ProfileID="36">
  <EntryEvent>
    <Date><Year>1991</Year><Month>1</Month><Day>22</Day></Date>
  </EntryEvent>
  <SanctionsMeasure>
    <Comment>CUBA</Comment>
  </SanctionsMeasure>
</SanctionsEntry>
```

The names, types, dates, and programs are now in completely separate sections of the file, linked by a `ProfileID` number.

**How we fixed it:** Rewrote `parse_sdn_xml()` entirely with a three-pass approach:
1. First pass: read all `DistinctParty` elements → build `profile_id → name` and `profile_id → type` dictionaries
2. Second pass: read all `SanctionsEntry` elements → build `profile_id → date` and `profile_id → programs` dictionaries  
3. Final pass: join the two dictionaries on `profile_id` to create complete entity records

**Test result:** 9,518 entities parsed → 586 sanction events stored. ✓

---

## Part 5: What's Running Right Now (As of End of Day 1)

```
OFAC: ✅ DONE — 586 sanction events stored (2020-present)
ACLED: 🔄 RUNNING — year-by-year fetch of 1.74M global events (2020-2025)
GDELT: ⏳ PENDING — needs to be run (will take hours; ~1,825 daily files)
```

The ACLED run is in the background. It's doing:
- 2020: ~200k events
- 2021: ~250k events
- 2022: ~350k events (Ukraine war explodes the numbers)
- 2023: ~350k events (Sudan civil war, Gaza)
- 2024: ~350k events
- 2025: ~200k events so far

---

## Part 6: How All of This Connects to the Final Model

Here's the full picture of how Day 1's work feeds into everything:

```
Day 1 Work                    → What It Enables
─────────────────────────────────────────────────────────────────

Taxonomy mapping tables        → Stage 1 (Event Classifier) knows what labels to predict
(cameo/acled/gta → taxonomy)     Stage 2-4 all speak the same 8-category language

Data pipelines                 → Builds the labeled dataset that trains Stage 1
(GDELT + ACLED + OFAC)           ~100,000+ historical events with categories assigned

Database schema                → Stores events + corporate impacts in queryable format
                                 Links events to company outcomes for training Stage 2 + 3

Lambda rates config            → compute_frequency_scores.py calculates the "live" matrix
                                 This validates our taxonomy (Week 3-4 task)

Priority matrix config         → Stage 4 (Strategy Recommender) uses this to rank
                                 which event × channel combinations need which responses

.env + credentials             → ACLED running; OFAC running; EDGAR will need EDGAR_USER_AGENT

Seed label stub                → Reminds us that 200-300 human labels are still needed
(NEEDS_HUMAN_REVIEW.md)          This is the single most important human input left
```

**The critical path to the first working model:**

```
[Week 1-2: Today] Build DB + ingest GDELT/ACLED/OFAC
       ↓
[Week 3-4] Run compute_frequency_scores.py, compare to expert scores,
           fix any taxonomy mappings that are wrong
       ↓
[Week 5-7] Build earnings call scraper, extract geopolitical mentions,
           you manually label ~300 event-impact pairs
       ↓
[Week 8-10] Train the 4 models, build end-to-end pipeline
```

---

## Part 7: The One Thing That Can't Be Automated (Your Job)

The 200-300 seed labels in `data/seed_labels/NEEDS_HUMAN_REVIEW.md` need human judgment. Here's why this matters:

The model learns from examples. If you show it 300 examples of "geopolitical event X caused $Y million loss for company Z via impact channel W," it starts to learn the pattern.

But if those 300 examples are wrong — if, say, the $4B automotive tariff impact is actually mis-attributed to the wrong event, or the Teva pharmaceutical case is labeled as "supply chain" when it was actually "regulatory compliance" — the model learns wrong patterns and makes bad predictions.

No amount of code can replace reading the WEF report's case studies and making the judgment call: "This is clearly a Supply Chain impact, not a Revenue impact." That's why the seed labels are flagged as a human gate.

The priority cases for your review are:
1. **WEF report case studies** — you've read the paper, you have the context
2. **NVIDIA's chip export control earnings calls** — quantified impact is disclosed explicitly
3. **Maersk's Red Sea rerouting** — they gave very specific cost figures
4. **BP's Russia exit** — the $24B write-down is well-documented

These are the highest-confidence labels to start with. Everything else gets labeled with medium/low confidence and gets validated later.

---

## Summary: What Day 1 Is in One Paragraph

Day 1 built the data foundation: a classification system (taxonomy) for 8 types of geopolitical events and 10 types of business impact, translation tables that convert three different data formats (CAMEO, ACLED, GTA) into that taxonomy, Python scripts that download historical event data from three sources (GDELT, ACLED, OFAC) and store it in a local database, and a scoring system that weights recent events more heavily than old ones. We hit four real-world roadblocks — ACLED switching from API keys to OAuth, a field name change in ACLED's data, OFAC moving their XML file, and OFAC completely restructuring their XML schema — and fixed all four by inspecting the actual live data rather than relying on outdated documentation. At the end of Day 1, 586 sanction events are in the database and 1.74 million conflict events are being ingested in the background.
