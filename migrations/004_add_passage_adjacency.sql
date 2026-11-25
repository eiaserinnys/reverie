-- MemoryLink v2: Passage-Passage adjacency table
-- Creates passage_adjacency table for storing temporal edges between passages

-- passage_adjacency: Passage↔Passage 시계열 연결
CREATE TABLE IF NOT EXISTS passage_adjacency (
    id SERIAL PRIMARY KEY,
    passage_id_a INTEGER REFERENCES passages(id) ON DELETE CASCADE,
    passage_id_b INTEGER REFERENCES passages(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(passage_id_a, passage_id_b),
    CHECK(passage_id_a < passage_id_b)  -- 중복 방지: a는 항상 b보다 작음
);

-- Indexes for efficient lookup
CREATE INDEX IF NOT EXISTS idx_passage_adjacency_a ON passage_adjacency(passage_id_a);
CREATE INDEX IF NOT EXISTS idx_passage_adjacency_b ON passage_adjacency(passage_id_b);
