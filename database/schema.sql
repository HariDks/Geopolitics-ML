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

-- ─── PREDICTIONS LOG ────────────────────────────────────────────────────────
-- Every prediction the system makes, with full inputs and outputs.

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       TEXT PRIMARY KEY,
    timestamp           TEXT DEFAULT (datetime('now')),
    user_id             TEXT,
    input_text          TEXT NOT NULL,
    input_ticker        TEXT,
    input_company       TEXT,
    input_revenue       REAL,
    predicted_category  TEXT,
    predicted_category_confidence REAL,
    predicted_channel_1 TEXT,
    predicted_channel_2 TEXT,
    predicted_channel_confidence REAL,
    channel_mode        TEXT,               -- text_rich / text_partial / text_poor
    channel_reliability TEXT,               -- high / moderate / low
    predicted_impact_low  REAL,
    predicted_impact_mid  REAL,
    predicted_impact_high REAL,
    predicted_severity    TEXT,              -- Limited / Low-to-moderate / Moderate / etc.
    predicted_ops_severity TEXT,            -- High / Medium-to-high / Medium
    model_version       TEXT DEFAULT 'v1'
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(input_ticker);
CREATE INDEX IF NOT EXISTS idx_pred_date   ON predictions(timestamp);

-- ─── CORRECTIONS ────────────────────────────────────────────────────────────
-- Human corrections linked to specific predictions.

CREATE TABLE IF NOT EXISTS corrections (
    correction_id   TEXT PRIMARY KEY,
    prediction_id   TEXT REFERENCES predictions(prediction_id),
    timestamp       TEXT DEFAULT (datetime('now')),
    reviewer_id     TEXT,
    useful          TEXT,                   -- Yes / No
    correct_category TEXT,                  -- NULL if category was correct
    correct_channel  TEXT,                  -- NULL if channel was correct
    actual_impact_pct REAL,                -- actual revenue delta if known
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_corr_pred ON corrections(prediction_id);

-- ─── MODEL VERSIONS ─────────────────────────────────────────────────────────
-- Track retraining runs and accuracy over time.

CREATE TABLE IF NOT EXISTS model_versions (
    version_id      TEXT PRIMARY KEY,
    trained_at      TEXT DEFAULT (datetime('now')),
    training_data_size INTEGER,
    gold_labels     INTEGER,
    weak_labels     INTEGER,
    holdout_accuracy_category  REAL,
    holdout_accuracy_channel   REAL,
    holdout_coverage_impact    REAL,
    split_type      TEXT DEFAULT 'temporal',  -- temporal / random
    notes           TEXT
);

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
