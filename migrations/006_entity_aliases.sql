-- Entity alias table for merging similar entities
-- canonical_entity_id: the main entity that aliases point to
-- alias_name: alternative name that should be merged into canonical

CREATE TABLE IF NOT EXISTS entity_aliases (
    id SERIAL PRIMARY KEY,
    canonical_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias_name VARCHAR(512) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(64) DEFAULT 'system',  -- 'system', 'agent', 'user'
    UNIQUE(alias_name)
);

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_name ON entity_aliases(alias_name);

-- Comment
COMMENT ON TABLE entity_aliases IS 'Maps alternative entity names to canonical entities for deduplication';
COMMENT ON COLUMN entity_aliases.canonical_entity_id IS 'The main entity ID that this alias points to';
COMMENT ON COLUMN entity_aliases.alias_name IS 'Alternative name (case-insensitive matching)';
COMMENT ON COLUMN entity_aliases.created_by IS 'Who created this alias: system, agent, or user';
