-- MedXAI Review-II: fall prediction tables
-- Run once in the Supabase SQL editor. Safe to re-run.

-- 17. Fall events (written by the app's fall detector, or the demo seed)
--     event_type: 'fall'      = detected fall, not cancelled by the user
--                 'near_fall' = impact detected but the user recovered / cancelled
CREATE TABLE IF NOT EXISTS user_fall_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type TEXT NOT NULL DEFAULT 'fall',
    peak_g REAL,
    user_cancelled BOOLEAN DEFAULT FALSE,
    dispatched BOOLEAN DEFAULT FALSE,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fall_events_user_time ON user_fall_events (user_id, detected_at DESC);

-- 18. Fall-risk audit log (one row per engine computation; used for traceability)
CREATE TABLE IF NOT EXISTS user_fall_risk (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    frs_24 INT,
    tier TEXT,
    coverage_q INT,
    layer_p INT,
    layer_rc INT,
    layer_mo INT,
    layer_fh INT,
    weights JSONB,
    top_contributors JSONB,
    cycle_phase TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fall_risk_user_time ON user_fall_risk (user_id, computed_at DESC);
