-- reverie initial schema
-- Requires: PostgreSQL 16+ with pgvector extension

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Processed sessions tracking
CREATE TABLE IF NOT EXISTS processed_sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'completed',
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT
);

-- Passages (chunks from session JSONL)
CREATE TABLE IF NOT EXISTS passages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decay_score REAL DEFAULT 1.0
);

-- Entities (named entities extracted via OpenIE)
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Facts (triples: subject-predicate-object)
CREATE TABLE IF NOT EXISTS facts (
    id SERIAL PRIMARY KEY,
    subject_entity_id INTEGER REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_entity_id INTEGER REFERENCES entities(id),
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Passage-Fact mapping
CREATE TABLE IF NOT EXISTS passage_facts (
    passage_id INTEGER REFERENCES passages(id) ON DELETE CASCADE,
    fact_id INTEGER REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (passage_id, fact_id)
);

-- Indexes for pgvector KNN search (HNSW)
CREATE INDEX IF NOT EXISTS idx_passages_embedding
    ON passages USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_entities_embedding
    ON entities USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_facts_embedding
    ON facts USING hnsw (embedding vector_cosine_ops);

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_passages_session_id ON passages(session_id);
CREATE INDEX IF NOT EXISTS idx_passages_decay_score ON passages(decay_score);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_processed_sessions_status ON processed_sessions(status);
