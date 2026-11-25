-- Entity importance (PageRank-style) for MemoryLink enrich pipeline
-- Progressive calculation based on co-occurrence graph

-- ============================================================================
-- SCHEMA CHANGES
-- ============================================================================

-- Add importance column to entities
ALTER TABLE entities ADD COLUMN IF NOT EXISTS importance REAL DEFAULT 1.0;

-- Index for importance-based queries
CREATE INDEX IF NOT EXISTS idx_entities_importance ON entities(importance DESC);


-- ============================================================================
-- PAGERANK PROCEDURES
-- ============================================================================

-- Initialize importance for all entities (one-time bootstrap)
-- Sets importance = 1.0 for all entities
CREATE OR REPLACE FUNCTION initialize_entity_importance()
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    UPDATE entities SET importance = 1.0;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;


-- Full PageRank calculation (expensive, use sparingly)
-- damping: probability of following a link (typically 0.85)
-- iterations: number of power iteration steps
CREATE OR REPLACE FUNCTION calculate_pagerank_full(
    p_damping REAL DEFAULT 0.85,
    p_iterations INTEGER DEFAULT 20
) RETURNS TABLE(
    entities_updated INTEGER,
    max_change REAL
) AS $$
DECLARE
    v_num_entities INTEGER;
    v_iteration INTEGER;
    v_max_change REAL;
    v_base_score REAL;
BEGIN
    -- Get total entity count
    SELECT COUNT(*) INTO v_num_entities FROM entities;

    IF v_num_entities = 0 THEN
        RETURN QUERY SELECT 0, 0.0::REAL;
        RETURN;
    END IF;

    -- Base score for random jump
    v_base_score := (1.0 - p_damping) / v_num_entities;

    -- Initialize all to 1/N
    UPDATE entities SET importance = 1.0 / v_num_entities;

    -- Power iteration
    FOR v_iteration IN 1..p_iterations LOOP
        -- Create temp table with new scores
        CREATE TEMP TABLE IF NOT EXISTS _pagerank_temp (
            entity_id INTEGER PRIMARY KEY,
            new_importance REAL
        ) ON COMMIT DROP;

        TRUNCATE _pagerank_temp;

        -- Calculate new importance for each entity
        INSERT INTO _pagerank_temp (entity_id, new_importance)
        SELECT
            e.id,
            v_base_score + p_damping * COALESCE(
                (
                    -- Sum of importance from incoming edges
                    SELECT SUM(
                        src.importance * co.count / NULLIF(
                            (SELECT SUM(count) FROM entity_cooccurrences
                             WHERE entity_a_id = src.id OR entity_b_id = src.id),
                            0
                        )
                    )
                    FROM entity_cooccurrences co
                    JOIN entities src ON (
                        (co.entity_a_id = src.id AND co.entity_b_id = e.id) OR
                        (co.entity_b_id = src.id AND co.entity_a_id = e.id)
                    )
                ),
                0
            )
        FROM entities e;

        -- Calculate max change for convergence check
        SELECT MAX(ABS(t.new_importance - e.importance))
        INTO v_max_change
        FROM _pagerank_temp t
        JOIN entities e ON e.id = t.entity_id;

        -- Update entities with new scores
        UPDATE entities e
        SET importance = t.new_importance
        FROM _pagerank_temp t
        WHERE e.id = t.entity_id;

        -- Early exit if converged
        IF v_max_change < 0.0001 THEN
            EXIT;
        END IF;
    END LOOP;

    RETURN QUERY SELECT v_num_entities, v_max_change;
END;
$$ LANGUAGE plpgsql;


-- Local PageRank update (progressive, for incremental indexing)
-- Updates only affected entities and their immediate neighbors
CREATE OR REPLACE FUNCTION update_entity_importance_local(
    p_entity_ids INTEGER[],
    p_damping REAL DEFAULT 0.85,
    p_iterations INTEGER DEFAULT 3
) RETURNS TABLE(
    entities_updated INTEGER,
    max_change REAL
) AS $$
DECLARE
    v_affected_ids INTEGER[];
    v_num_entities INTEGER;
    v_iteration INTEGER;
    v_max_change REAL := 0;
    v_base_score REAL;
BEGIN
    -- Get total entity count for base score
    SELECT COUNT(*) INTO v_num_entities FROM entities;

    IF v_num_entities = 0 OR array_length(p_entity_ids, 1) IS NULL THEN
        RETURN QUERY SELECT 0, 0.0::REAL;
        RETURN;
    END IF;

    v_base_score := (1.0 - p_damping) / v_num_entities;

    -- Collect affected entities (input + 1-hop neighbors)
    SELECT ARRAY(
        SELECT DISTINCT id FROM (
            -- Input entities
            SELECT UNNEST(p_entity_ids) AS id
            UNION
            -- Neighbors via entity_a
            SELECT entity_b_id AS id
            FROM entity_cooccurrences
            WHERE entity_a_id = ANY(p_entity_ids)
            UNION
            -- Neighbors via entity_b
            SELECT entity_a_id AS id
            FROM entity_cooccurrences
            WHERE entity_b_id = ANY(p_entity_ids)
        ) sub
    ) INTO v_affected_ids;

    -- Power iteration on affected subset
    FOR v_iteration IN 1..p_iterations LOOP
        -- Create temp table with new scores
        CREATE TEMP TABLE IF NOT EXISTS _pagerank_local_temp (
            entity_id INTEGER PRIMARY KEY,
            new_importance REAL
        ) ON COMMIT DROP;

        TRUNCATE _pagerank_local_temp;

        -- Calculate new importance for affected entities only
        INSERT INTO _pagerank_local_temp (entity_id, new_importance)
        SELECT
            e.id,
            v_base_score + p_damping * COALESCE(
                (
                    SELECT SUM(
                        src.importance * co.count / NULLIF(
                            (SELECT SUM(count) FROM entity_cooccurrences
                             WHERE entity_a_id = src.id OR entity_b_id = src.id),
                            0
                        )
                    )
                    FROM entity_cooccurrences co
                    JOIN entities src ON (
                        (co.entity_a_id = src.id AND co.entity_b_id = e.id) OR
                        (co.entity_b_id = src.id AND co.entity_a_id = e.id)
                    )
                ),
                0
            )
        FROM entities e
        WHERE e.id = ANY(v_affected_ids);

        -- Calculate max change
        SELECT COALESCE(MAX(ABS(t.new_importance - e.importance)), 0)
        INTO v_max_change
        FROM _pagerank_local_temp t
        JOIN entities e ON e.id = t.entity_id;

        -- Update affected entities
        UPDATE entities e
        SET importance = t.new_importance
        FROM _pagerank_local_temp t
        WHERE e.id = t.entity_id;

        -- Early exit if converged
        IF v_max_change < 0.0001 THEN
            EXIT;
        END IF;
    END LOOP;

    RETURN QUERY SELECT array_length(v_affected_ids, 1), v_max_change;
END;
$$ LANGUAGE plpgsql;


-- Wrapper to update importance after fact insertion
-- Called automatically by insert_fact_with_cooccurrence (next migration)
CREATE OR REPLACE FUNCTION update_importance_after_fact(
    p_subject_entity_id INTEGER,
    p_object_entity_id INTEGER
) RETURNS void AS $$
BEGIN
    -- Update local PageRank for affected entities
    PERFORM update_entity_importance_local(
        ARRAY[p_subject_entity_id, p_object_entity_id]
    );
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Get top N most important entities
CREATE OR REPLACE FUNCTION get_top_entities(
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    entity_id INTEGER,
    entity_name TEXT,
    importance REAL,
    cooccurrence_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.name,
        e.importance,
        COALESCE(
            (SELECT SUM(count) FROM entity_cooccurrences
             WHERE entity_a_id = e.id OR entity_b_id = e.id),
            0
        )::BIGINT
    FROM entities e
    ORDER BY e.importance DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN entities.importance IS 'PageRank-style importance score (0-1), updated progressively';
COMMENT ON FUNCTION initialize_entity_importance IS 'Reset all entity importance to 1.0';
COMMENT ON FUNCTION calculate_pagerank_full IS 'Full PageRank calculation (expensive, use for initial bootstrap or periodic recalculation)';
COMMENT ON FUNCTION update_entity_importance_local IS 'Local PageRank update for incremental indexing';
COMMENT ON FUNCTION update_importance_after_fact IS 'Wrapper called after fact insertion to update importance';
COMMENT ON FUNCTION get_top_entities IS 'Get top N entities by importance';
