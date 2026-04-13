-- Geopolitical Muscle ML — Database Schema
-- Supports both PostgreSQL (production) and SQLite (prototyping).
-- PostgreSQL-specific syntax noted inline where it differs from SQLite.

-- ─── GEOPOLITICAL EVENTS ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS geopolitical_events (
    event_id            TEXT PRIMARY KEY,           -- "EVT-{source}-{date}-{seq}"
    source              TEXT NOT NULL,              -- gdelt / acled / gta / ofac / bis / manual
    source_event_id     TEXT,                       -- original ID from the data source
    event_category      TEXT NOT NULL,              -- one of 8 taxonomy categories
    event_subtype       TEXT,                       -- specific action within category
    event_date          DATE NOT NULL,
    event_end_date      DATE,                       -- NULL if ongoing or point event
    -- PostgreSQL: affected_countries TEXT[], affected_sectors TEXT[]
    -- SQLite: store as JSON strings
    affected_countries  TEXT,                       -- JSON array of ISO 3166-1 alpha-2 codes
    affected_sectors    TEXT,                       -- JSON array of NAICS/GICS codes
    severity_estimate   INTEGER,                    -- 1-5
    onset_speed         TEXT,                       -- sudden / gradual / phased
    expected_duration   TEXT,                       -- transient / short_term / medium_term / structural
    description_text    TEXT,
    source_url          TEXT,
    goldstein_scale     REAL,                       -- GDELT-specific; NULL for other sources
    fatalities          INTEGER,                    -- ACLED-specific; NULL for other sources
    num_mentions        INTEGER,                    -- GDELT-specific; NULL for other sources
    avg_tone            REAL,                       -- GDELT AvgTone field
    mapping_confidence  TEXT,                       -- high / medium / low (from taxonomy mapping)
    created_at          TEXT DEFAULT (datetime('now'))  -- PostgreSQL: TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_date        ON geopolitical_events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_category    ON geopolitical_events(event_category);
CREATE INDEX IF NOT EXISTS idx_events_source      ON geopolitical_events(source);
-- PostgreSQL only: CREATE INDEX idx_events_countries ON geopolitical_events USING GIN(affected_countries);

-- ─── CORPORATE IMPACTS ───────────────────────────────────────────────────────
-- Populated in Weeks 5-7. Schema defined here for reference.

CREATE TABLE IF NOT EXISTS corporate_impacts (
    impact_id                   TEXT PRIMARY KEY,
    event_id                    TEXT REFERENCES geopolitical_events(event_id),
    company_ticker              TEXT NOT NULL,
    company_name                TEXT,
    sector_gics                 TEXT,
    impact_channel              TEXT NOT NULL,          -- one of 10 channels
    quarter                     TEXT,                   -- "2025Q2"

    -- Qualitative (from earnings call NLP)
    mention_text                TEXT,
    mention_sentiment           REAL,                   -- -1 to 1
    management_action_described TEXT,

    -- Quantitative financial deltas (from EDGAR XBRL)
    revenue_delta_pct           REAL,
    cogs_delta_pct              REAL,
    operating_income_delta_pct  REAL,
    capex_delta_pct             REAL,

    -- Market reaction (from event study)
    car_1_5                     REAL,                   -- cumulative abnormal return [-1, +5]
    car_1_30                    REAL,                   -- cumulative abnormal return [-1, +30]

    -- Labeling metadata
    source                      TEXT,                   -- earnings_call / 10k / event_study / manual
    confidence                  TEXT,                   -- high / medium / low
    labeled_by                  TEXT,                   -- human_review / auto_nlp / event_study
    human_reviewed              INTEGER DEFAULT 0,      -- 0/1 boolean; 1 = validated by human
    created_at                  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_impacts_event      ON corporate_impacts(event_id);
CREATE INDEX IF NOT EXISTS idx_impacts_ticker     ON corporate_impacts(company_ticker);
CREATE INDEX IF NOT EXISTS idx_impacts_channel    ON corporate_impacts(impact_channel);
CREATE INDEX IF NOT EXISTS idx_impacts_reviewed   ON corporate_impacts(human_reviewed);

-- ─── STRATEGIES ──────────────────────────────────────────────────────────────
-- Populated from Phase 1 Top 34 cell documentation. Used by Model 4.

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id             TEXT PRIMARY KEY,
    event_category          TEXT NOT NULL,
    impact_channel          TEXT NOT NULL,
    strategy_name           TEXT NOT NULL,
    strategy_category       TEXT,               -- mitigate / hedge / exit / capture / engage / monitor
    description             TEXT,
    typical_cost_range      TEXT,               -- "low" / "medium" / "high" or dollar range string
    implementation_time     TEXT,               -- e.g., "3-6 months"
    prerequisites           TEXT,               -- JSON array of required capabilities
    historical_precedents   TEXT,               -- JSON array of company examples
    success_conditions      TEXT,
    precedent_count         INTEGER DEFAULT 0,
    precedent_success_rate  REAL,               -- 0-1
    created_at              TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_strategies_event   ON strategies(event_category);
CREATE INDEX IF NOT EXISTS idx_strategies_channel ON strategies(impact_channel);

-- ─── INGESTION LOG ───────────────────────────────────────────────────────────
-- Track ingestion runs for idempotency and debugging.

CREATE TABLE IF NOT EXISTS ingestion_log (
    run_id          TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    start_date      DATE,
    end_date        DATE,
    records_fetched INTEGER DEFAULT 0,
    records_stored  INTEGER DEFAULT 0,
    records_skipped INTEGER DEFAULT 0,  -- duplicates
    status          TEXT,               -- success / failed / partial
    error_message   TEXT,
    run_at          TEXT DEFAULT (datetime('now'))
);
