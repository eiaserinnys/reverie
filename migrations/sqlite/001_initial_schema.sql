-- Reverie SQLite Schema (using sqlite-vec for vector operations)
-- This schema mirrors PostgreSQL but uses SQLite-compatible syntax and types

-- Enable foreign keys (SQLite has this disabled by default)
PRAGMA foreign_keys = ON;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Processed sessions tracking (legacy, kept for compatibility)
CREATE TABLE IF NOT EXISTS processed_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    processed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'completed',
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT
);

-- Passages (chunks from session JSONL)
-- embedding stored as BLOB using struct.pack('f' * 1536, ...)
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',  -- JSON stored as TEXT
    embedding BLOB,              -- vector(1536) as packed floats
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decay_score REAL DEFAULT 1.0,

    -- User/Assistant split columns (v2)
    user_text TEXT,
    user_embedding BLOB,
    assistant_text TEXT,
    assistant_embedding BLOB,

    -- Profile support
    profile_id TEXT NOT NULL DEFAULT 'default',
    session_seq INTEGER,

    -- PPR support
    out_degree INTEGER DEFAULT 0,
    importance REAL DEFAULT 1.0
);

-- Entities (named entities extracted via OpenIE)
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    profile_id TEXT NOT NULL DEFAULT 'default',
    out_degree INTEGER DEFAULT 0,
    importance REAL DEFAULT 1.0
);

-- Entity-Passage links (bipartite graph)
CREATE TABLE IF NOT EXISTS entity_passages (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, passage_id)
);

-- Passage adjacency (temporal graph)
CREATE TABLE IF NOT EXISTS passage_adjacency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    passage_id_a INTEGER REFERENCES passages(id) ON DELETE CASCADE,
    passage_id_b INTEGER REFERENCES passages(id) ON DELETE CASCADE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(passage_id_a, passage_id_b),
    CHECK(passage_id_a < passage_id_b)  -- Normalize: a < b
);

-- ============================================================================
-- SESSION INDEXING STATE
-- ============================================================================

CREATE TABLE IF NOT EXISTS session_index_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    session_seq INTEGER NOT NULL,
    file_path TEXT NOT NULL,

    -- File state tracking
    file_mtime DATETIME NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_lines_count INTEGER NOT NULL DEFAULT 0,

    -- Indexing progress
    last_indexed_turn INTEGER DEFAULT -1,
    last_indexed_line INTEGER DEFAULT 0,
    last_passage_id INTEGER,
    total_passages_indexed INTEGER DEFAULT 0,
    total_passages_dropped INTEGER DEFAULT 0,

    -- Status: 'pending', 'in_progress', 'completed', 'failed'
    status TEXT DEFAULT 'pending',
    error_message TEXT,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_indexed_at DATETIME,

    UNIQUE(profile_id, session_id)
);

-- Passage processing history
CREATE TABLE IF NOT EXISTS passage_processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,

    -- Processing result: 'indexed', 'dropped', 'skipped', 'failed'
    action TEXT NOT NULL,
    drop_reason TEXT,

    -- Metadata
    raw_text TEXT,
    preprocessed_text TEXT,
    entities_extracted TEXT,  -- JSON array as TEXT

    -- Connection
    passage_id INTEGER REFERENCES passages(id) ON DELETE SET NULL,

    -- Timestamp
    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(profile_id, session_id, turn_index)
);

-- ============================================================================
-- EXTENDED GRAPH TABLES (from 011_unified_graph_model)
-- ============================================================================

-- Passage-Passage connections with decay weights
CREATE TABLE IF NOT EXISTS passage_connections (
    passage_a_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    passage_b_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1,
    weight REAL NOT NULL DEFAULT 1.0,
    turn_distance INTEGER NOT NULL DEFAULT 1,
    decay_weight REAL NOT NULL DEFAULT 0.8,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (passage_a_id, passage_b_id),
    CHECK (passage_a_id < passage_b_id)
);

-- Entity-Passage connections with decay weights
CREATE TABLE IF NOT EXISTS entity_passage_connections (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1,
    turn_distance INTEGER NOT NULL DEFAULT 0,
    decay_weight REAL NOT NULL DEFAULT 1.0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id, passage_id)
);

-- Entity co-occurrences
CREATE TABLE IF NOT EXISTS entity_cooccurrences (
    entity_a_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_a_id, entity_b_id),
    CHECK (entity_a_id < entity_b_id)
);

-- Entity aliases
CREATE TABLE IF NOT EXISTS entity_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alias_name TEXT NOT NULL UNIQUE,
    canonical_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    similarity_score REAL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SCHEMA MIGRATIONS TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Passages
CREATE INDEX IF NOT EXISTS idx_passages_session_id ON passages(session_id);
CREATE INDEX IF NOT EXISTS idx_passages_profile ON passages(profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_passages_profile_seq ON passages(profile_id, session_seq);
CREATE INDEX IF NOT EXISTS idx_passages_decay_score ON passages(decay_score);
CREATE INDEX IF NOT EXISTS idx_passages_out_degree ON passages(out_degree);
CREATE INDEX IF NOT EXISTS idx_passages_importance ON passages(importance DESC);

-- Entities
CREATE INDEX IF NOT EXISTS idx_entities_profile ON entities(profile_id, name);
CREATE INDEX IF NOT EXISTS idx_entities_out_degree ON entities(out_degree);
CREATE INDEX IF NOT EXISTS idx_entities_importance ON entities(importance DESC);

-- Entity-Passage links
CREATE INDEX IF NOT EXISTS idx_entity_passages_entity ON entity_passages(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_passages_passage ON entity_passages(passage_id);

-- Passage adjacency
CREATE INDEX IF NOT EXISTS idx_passage_adjacency_a ON passage_adjacency(passage_id_a);
CREATE INDEX IF NOT EXISTS idx_passage_adjacency_b ON passage_adjacency(passage_id_b);

-- Session state
CREATE INDEX IF NOT EXISTS idx_session_index_state_profile ON session_index_state(profile_id);
CREATE INDEX IF NOT EXISTS idx_session_index_state_status ON session_index_state(profile_id, status);

-- Processing history
CREATE INDEX IF NOT EXISTS idx_passage_history_profile ON passage_processing_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_passage_history_session ON passage_processing_history(profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_passage_history_action ON passage_processing_history(profile_id, action);

-- Passage connections
CREATE INDEX IF NOT EXISTS idx_passage_connections_a ON passage_connections(passage_a_id);
CREATE INDEX IF NOT EXISTS idx_passage_connections_b ON passage_connections(passage_b_id);
CREATE INDEX IF NOT EXISTS idx_passage_connections_decay ON passage_connections(decay_weight);

-- Entity-Passage connections
CREATE INDEX IF NOT EXISTS idx_entity_passage_conn_entity ON entity_passage_connections(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_passage_conn_passage ON entity_passage_connections(passage_id);
CREATE INDEX IF NOT EXISTS idx_entity_passage_conn_decay ON entity_passage_connections(decay_weight);

-- Entity co-occurrences
CREATE INDEX IF NOT EXISTS idx_cooccurrences_a ON entity_cooccurrences(entity_a_id);
CREATE INDEX IF NOT EXISTS idx_cooccurrences_b ON entity_cooccurrences(entity_b_id);

-- Entity aliases
CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical_entity_id);

-- Processed sessions
CREATE INDEX IF NOT EXISTS idx_processed_sessions_status ON processed_sessions(status);

-- ============================================================================
-- Note: Vector indexes (HNSW) are NOT supported in SQLite with sqlite-vec.
-- Vector search uses brute-force linear scan, which is acceptable for
-- small to medium datasets (< 100k passages).
-- For production scale, use PostgreSQL with pgvector HNSW indexes.
-- ============================================================================
