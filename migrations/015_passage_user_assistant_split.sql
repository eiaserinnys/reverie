-- Passage User/Assistant Split Migration
-- Separates user and assistant text/embeddings for better retrieval quality
--
-- Key changes:
-- 1. Add user_text, user_embedding, assistant_text, assistant_embedding columns
-- 2. Keep legacy text, embedding columns for backward compatibility (deprecated)
-- 3. Update HNSW index for dual embedding search
-- 4. Add harmonic mean scoring function for KNN search

-- ============================================================================
-- SCHEMA CHANGES
-- ============================================================================

-- Add user text and embedding columns
ALTER TABLE passages ADD COLUMN IF NOT EXISTS user_text TEXT;
ALTER TABLE passages ADD COLUMN IF NOT EXISTS user_embedding vector(1536);

-- Add assistant text and embedding columns
ALTER TABLE passages ADD COLUMN IF NOT EXISTS assistant_text TEXT;
ALTER TABLE passages ADD COLUMN IF NOT EXISTS assistant_embedding vector(1536);

-- Mark legacy columns as deprecated (via comment)
COMMENT ON COLUMN passages.text IS 'DEPRECATED: Use user_text + assistant_text instead. Combined turn text.';
COMMENT ON COLUMN passages.embedding IS 'DEPRECATED: Use user_embedding + assistant_embedding instead. Combined turn embedding.';

-- New column comments
COMMENT ON COLUMN passages.user_text IS 'User message text for this turn';
COMMENT ON COLUMN passages.user_embedding IS 'Embedding vector for user message';
COMMENT ON COLUMN passages.assistant_text IS 'Assistant response text for this turn';
COMMENT ON COLUMN passages.assistant_embedding IS 'Embedding vector for assistant response';


-- ============================================================================
-- INDEXES FOR DUAL EMBEDDING SEARCH
-- ============================================================================

-- HNSW indexes for user and assistant embeddings
CREATE INDEX IF NOT EXISTS idx_passages_user_embedding
    ON passages USING hnsw (user_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_passages_assistant_embedding
    ON passages USING hnsw (assistant_embedding vector_cosine_ops);


-- ============================================================================
-- HARMONIC MEAN SCORING FUNCTION
-- ============================================================================

-- Calculate harmonic mean of two similarity scores
-- Used for combining user and assistant similarity in retrieval
-- Returns 0 if either score is 0 (requires both to be relevant)
CREATE OR REPLACE FUNCTION harmonic_mean(a REAL, b REAL)
RETURNS REAL AS $$
DECLARE
    epsilon REAL := 0.0001;
BEGIN
    IF a <= 0 OR b <= 0 THEN
        RETURN 0;
    END IF;
    RETURN 2.0 * a * b / (a + b + epsilon);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION harmonic_mean IS 'Calculate harmonic mean of two scores. Returns 0 if either is 0.';


-- ============================================================================
-- DUAL EMBEDDING KNN SEARCH FUNCTION
-- ============================================================================

-- Search passages using both user and assistant embeddings
-- Combines scores using harmonic mean for better relevance
-- Uses HNSW index for efficient KNN search on each embedding type
CREATE OR REPLACE FUNCTION search_passages_dual_embedding(
    p_query_embedding vector,
    p_top_k INTEGER DEFAULT 10
) RETURNS TABLE(
    passage_id INTEGER,
    user_similarity REAL,
    assistant_similarity REAL,
    combined_score REAL,
    user_text TEXT,
    assistant_text TEXT
) AS $$
DECLARE
    -- Fetch more candidates than needed to ensure good coverage after merge
    v_candidate_multiplier INTEGER := 3;
BEGIN
    RETURN QUERY
    WITH user_top AS (
        -- Get top-K candidates by user embedding using HNSW index
        SELECT
            p.id,
            1.0 - (p.user_embedding <=> p_query_embedding) / 2.0 AS user_sim
        FROM passages p
        WHERE p.user_embedding IS NOT NULL
        ORDER BY p.user_embedding <=> p_query_embedding
        LIMIT p_top_k * v_candidate_multiplier
    ),
    assistant_top AS (
        -- Get top-K candidates by assistant embedding using HNSW index
        SELECT
            p.id,
            1.0 - (p.assistant_embedding <=> p_query_embedding) / 2.0 AS assistant_sim
        FROM passages p
        WHERE p.assistant_embedding IS NOT NULL
        ORDER BY p.assistant_embedding <=> p_query_embedding
        LIMIT p_top_k * v_candidate_multiplier
    ),
    combined AS (
        -- Combine scores with harmonic mean
        -- Only consider passages that appear in both candidate sets
        -- or have scores high enough in at least one set
        SELECT
            COALESCE(u.id, a.id) AS pid,
            COALESCE(u.user_sim, 0)::REAL AS u_sim,
            COALESCE(a.assistant_sim, 0)::REAL AS a_sim,
            harmonic_mean(
                COALESCE(u.user_sim, 0)::REAL,
                COALESCE(a.assistant_sim, 0)::REAL
            ) AS score
        FROM user_top u
        FULL OUTER JOIN assistant_top a ON u.id = a.id
    )
    SELECT
        c.pid,
        c.u_sim,
        c.a_sim,
        c.score,
        p.user_text,
        p.assistant_text
    FROM combined c
    JOIN passages p ON p.id = c.pid
    WHERE c.score > 0
    ORDER BY c.score DESC
    LIMIT p_top_k;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_passages_dual_embedding IS 'Search passages using harmonic mean of user/assistant embedding similarities (HNSW-accelerated)';


-- ============================================================================
-- MIGRATION STATUS HELPER
-- ============================================================================

-- Check migration status: how many passages have been migrated
CREATE OR REPLACE FUNCTION get_passage_migration_status()
RETURNS TABLE(
    total_passages BIGINT,
    migrated_passages BIGINT,
    pending_passages BIGINT,
    migration_percentage REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT AS total_passages,
        COUNT(CASE WHEN user_embedding IS NOT NULL AND assistant_embedding IS NOT NULL THEN 1 END)::BIGINT AS migrated_passages,
        COUNT(CASE WHEN user_embedding IS NULL OR assistant_embedding IS NULL THEN 1 END)::BIGINT AS pending_passages,
        CASE WHEN COUNT(*) > 0 THEN
            (COUNT(CASE WHEN user_embedding IS NOT NULL AND assistant_embedding IS NOT NULL THEN 1 END)::REAL / COUNT(*)::REAL) * 100
        ELSE 0 END AS migration_percentage
    FROM passages;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_passage_migration_status IS 'Check how many passages have been migrated to dual embeddings';
