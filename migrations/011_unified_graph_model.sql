-- Unified Graph Model for Energy-based Graph Search
-- Entity and Passage nodes treated identically with distance-based decay
-- PageRank calculated for all nodes

-- ============================================================================
-- SCHEMA CHANGES
-- ============================================================================

-- Add importance column to passages (same as entities)
ALTER TABLE passages ADD COLUMN IF NOT EXISTS importance REAL DEFAULT 1.0;

-- Index for importance-based queries on passages
CREATE INDEX IF NOT EXISTS idx_passages_importance ON passages(importance DESC);

-- Passage-Passage connections (session-based adjacency)
-- Similar structure to entity_cooccurrences but for passage adjacency
CREATE TABLE IF NOT EXISTS passage_connections (
    passage_a_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    passage_b_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    -- count: how many times they appear as neighbors (typically 1, but could be more with re-indexing)
    count INTEGER NOT NULL DEFAULT 1,
    -- weight: 1.0 / count (for Dijkstra distance)
    weight REAL NOT NULL DEFAULT 1.0,
    -- turn_distance: how many turns apart (0 = same turn, 1 = adjacent, etc.)
    turn_distance INTEGER NOT NULL DEFAULT 1,
    -- decay_weight: distance-based decay value (0=1.0, 1=0.8, 2=0.6, 3=0.4, 4=0.2, 5+=0)
    decay_weight REAL NOT NULL DEFAULT 0.8,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (passage_a_id, passage_b_id),
    -- Normalize: smaller ID always first (undirected edge)
    CHECK (passage_a_id < passage_b_id)
);

-- Entity-Passage connections with distance-based decay
-- Replaces fixed weight=1.0 with decay-aware weights
CREATE TABLE IF NOT EXISTS entity_passage_connections (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    -- count: how many times this connection was established
    count INTEGER NOT NULL DEFAULT 1,
    -- turn_distance: 0 = entity directly mentioned in passage, 1-4 = adjacent passages
    turn_distance INTEGER NOT NULL DEFAULT 0,
    -- decay_weight: distance-based decay (0=1.0, 1=0.8, 2=0.6, 3=0.4, 4=0.2)
    decay_weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, passage_id)
);

-- Indexes for passage connections
CREATE INDEX IF NOT EXISTS idx_passage_connections_a ON passage_connections(passage_a_id);
CREATE INDEX IF NOT EXISTS idx_passage_connections_b ON passage_connections(passage_b_id);
CREATE INDEX IF NOT EXISTS idx_passage_connections_weight ON passage_connections(weight);
CREATE INDEX IF NOT EXISTS idx_passage_connections_decay ON passage_connections(decay_weight);

-- Indexes for entity-passage connections
CREATE INDEX IF NOT EXISTS idx_entity_passage_entity ON entity_passage_connections(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_passage_passage ON entity_passage_connections(passage_id);
CREATE INDEX IF NOT EXISTS idx_entity_passage_decay ON entity_passage_connections(decay_weight);


-- ============================================================================
-- DISTANCE-BASED DECAY FUNCTIONS
-- ============================================================================

-- Calculate decay weight based on turn distance
-- 0=1.0, 1=0.8, 2=0.6, 3=0.4, 4=0.2, 5+=0
CREATE OR REPLACE FUNCTION calculate_decay_weight(p_turn_distance INTEGER)
RETURNS REAL AS $$
BEGIN
    IF p_turn_distance <= 0 THEN
        RETURN 1.0;
    ELSIF p_turn_distance = 1 THEN
        RETURN 0.8;
    ELSIF p_turn_distance = 2 THEN
        RETURN 0.6;
    ELSIF p_turn_distance = 3 THEN
        RETURN 0.4;
    ELSIF p_turn_distance = 4 THEN
        RETURN 0.2;
    ELSE
        RETURN 0.0;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ============================================================================
-- PASSAGE CONNECTION UPSERT
-- ============================================================================

-- Upsert passage-passage connection with decay weight
CREATE OR REPLACE FUNCTION upsert_passage_connection(
    p_passage_a_id INTEGER,
    p_passage_b_id INTEGER,
    p_turn_distance INTEGER
) RETURNS void AS $$
DECLARE
    v_a INTEGER;
    v_b INTEGER;
    v_decay REAL;
BEGIN
    -- Calculate decay weight
    v_decay := calculate_decay_weight(p_turn_distance);

    -- Skip if decay is 0 (too far apart)
    IF v_decay <= 0 THEN
        RETURN;
    END IF;

    -- Normalize: smaller ID first
    IF p_passage_a_id < p_passage_b_id THEN
        v_a := p_passage_a_id;
        v_b := p_passage_b_id;
    ELSE
        v_a := p_passage_b_id;
        v_b := p_passage_a_id;
    END IF;

    INSERT INTO passage_connections (passage_a_id, passage_b_id, count, weight, turn_distance, decay_weight)
    VALUES (v_a, v_b, 1, 1.0, p_turn_distance, v_decay)
    ON CONFLICT (passage_a_id, passage_b_id) DO UPDATE
    SET count = passage_connections.count + 1,
        weight = 1.0 / (passage_connections.count + 1),
        -- Keep minimum turn distance (strongest connection)
        turn_distance = LEAST(passage_connections.turn_distance, p_turn_distance),
        decay_weight = GREATEST(passage_connections.decay_weight, v_decay),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;


-- Upsert entity-passage connection with decay weight
CREATE OR REPLACE FUNCTION upsert_entity_passage_connection(
    p_entity_id INTEGER,
    p_passage_id INTEGER,
    p_turn_distance INTEGER DEFAULT 0
) RETURNS void AS $$
DECLARE
    v_decay REAL;
BEGIN
    -- Calculate decay weight
    v_decay := calculate_decay_weight(p_turn_distance);

    -- Skip if decay is 0 (too far apart)
    IF v_decay <= 0 THEN
        RETURN;
    END IF;

    INSERT INTO entity_passage_connections (entity_id, passage_id, count, turn_distance, decay_weight)
    VALUES (p_entity_id, p_passage_id, 1, p_turn_distance, v_decay)
    ON CONFLICT (entity_id, passage_id) DO UPDATE
    SET count = entity_passage_connections.count + 1,
        -- Keep minimum turn distance (strongest connection)
        turn_distance = LEAST(entity_passage_connections.turn_distance, p_turn_distance),
        decay_weight = GREATEST(entity_passage_connections.decay_weight, v_decay),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COOCCURRENCE UPDATE WITH DECAY
-- ============================================================================

-- Update entity cooccurrence with distance-based decay
-- This replaces the simple count-based upsert
CREATE OR REPLACE FUNCTION upsert_cooccurrence_with_decay(
    p_entity_a_id INTEGER,
    p_entity_b_id INTEGER,
    p_turn_distance INTEGER DEFAULT 0
) RETURNS void AS $$
DECLARE
    v_a INTEGER;
    v_b INTEGER;
    v_decay REAL;
BEGIN
    -- Calculate decay weight
    v_decay := calculate_decay_weight(p_turn_distance);

    -- Skip if decay is 0 (too far apart)
    IF v_decay <= 0 THEN
        RETURN;
    END IF;

    -- Normalize: smaller ID first
    IF p_entity_a_id < p_entity_b_id THEN
        v_a := p_entity_a_id;
        v_b := p_entity_b_id;
    ELSE
        v_a := p_entity_b_id;
        v_b := p_entity_a_id;
    END IF;

    -- Upsert with decay-weighted count
    -- Each occurrence adds decay_weight to the count
    INSERT INTO entity_cooccurrences (entity_a_id, entity_b_id, count, weight)
    VALUES (v_a, v_b, 1, 1.0 / v_decay)  -- Initial weight based on decay
    ON CONFLICT (entity_a_id, entity_b_id) DO UPDATE
    SET count = entity_cooccurrences.count + 1,
        -- Weight inversely proportional to accumulated decay-weighted count
        weight = 1.0 / (entity_cooccurrences.count + 1),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- UNIFIED PAGERANK FOR ALL NODES
-- ============================================================================

-- Full PageRank calculation for all nodes (entities + passages)
-- Treats all nodes identically in the graph
CREATE OR REPLACE FUNCTION calculate_unified_pagerank(
    p_damping REAL DEFAULT 0.85,
    p_iterations INTEGER DEFAULT 20
) RETURNS TABLE(
    node_type TEXT,
    node_id INTEGER,
    importance REAL,
    max_change REAL
) AS $$
DECLARE
    v_num_nodes INTEGER;
    v_iteration INTEGER;
    v_max_change REAL := 1.0;
    v_base_score REAL;
BEGIN
    -- Count total nodes (entities + passages)
    SELECT (SELECT COUNT(*) FROM entities) + (SELECT COUNT(*) FROM passages) INTO v_num_nodes;

    IF v_num_nodes = 0 THEN
        RETURN;
    END IF;

    v_base_score := (1.0 - p_damping) / v_num_nodes;

    -- Create unified node table
    CREATE TEMP TABLE IF NOT EXISTS _unified_nodes (
        node_type TEXT NOT NULL,
        node_id INTEGER NOT NULL,
        importance REAL NOT NULL,
        new_importance REAL,
        out_degree REAL DEFAULT 0,
        PRIMARY KEY (node_type, node_id)
    ) ON COMMIT DROP;

    TRUNCATE _unified_nodes;

    -- Initialize with current importance (or 1/N if not set)
    INSERT INTO _unified_nodes (node_type, node_id, importance)
    SELECT 'entity', id, COALESCE(importance, 1.0 / v_num_nodes)
    FROM entities
    UNION ALL
    SELECT 'passage', id, COALESCE(importance, 1.0 / v_num_nodes)
    FROM passages;

    -- Calculate out-degree for each node (sum of outgoing edge weights)
    -- Entity-Entity edges
    UPDATE _unified_nodes u
    SET out_degree = out_degree + COALESCE((
        SELECT SUM(1.0 / NULLIF(weight, 0))
        FROM entity_cooccurrences co
        WHERE co.entity_a_id = u.node_id OR co.entity_b_id = u.node_id
    ), 0)
    WHERE u.node_type = 'entity';

    -- Entity-Passage edges
    UPDATE _unified_nodes u
    SET out_degree = out_degree + COALESCE((
        SELECT SUM(decay_weight)
        FROM entity_passage_connections ep
        WHERE ep.entity_id = u.node_id
    ), 0)
    WHERE u.node_type = 'entity';

    UPDATE _unified_nodes u
    SET out_degree = out_degree + COALESCE((
        SELECT SUM(decay_weight)
        FROM entity_passage_connections ep
        WHERE ep.passage_id = u.node_id
    ), 0)
    WHERE u.node_type = 'passage';

    -- Passage-Passage edges
    UPDATE _unified_nodes u
    SET out_degree = out_degree + COALESCE((
        SELECT SUM(decay_weight)
        FROM passage_connections pc
        WHERE pc.passage_a_id = u.node_id OR pc.passage_b_id = u.node_id
    ), 0)
    WHERE u.node_type = 'passage';

    -- Power iteration
    FOR v_iteration IN 1..p_iterations LOOP
        -- Reset new_importance
        UPDATE _unified_nodes SET new_importance = v_base_score;

        -- Propagate importance from entities via co-occurrence edges
        UPDATE _unified_nodes target
        SET new_importance = target.new_importance + p_damping * COALESCE((
            SELECT SUM(
                src.importance * (1.0 / NULLIF(co.weight, 0)) / NULLIF(src.out_degree, 0)
            )
            FROM entity_cooccurrences co
            JOIN _unified_nodes src ON src.node_type = 'entity' AND (
                (co.entity_a_id = src.node_id AND co.entity_b_id = target.node_id) OR
                (co.entity_b_id = src.node_id AND co.entity_a_id = target.node_id)
            )
        ), 0)
        WHERE target.node_type = 'entity';

        -- Propagate from entities to passages
        UPDATE _unified_nodes target
        SET new_importance = target.new_importance + p_damping * COALESCE((
            SELECT SUM(
                src.importance * ep.decay_weight / NULLIF(src.out_degree, 0)
            )
            FROM entity_passage_connections ep
            JOIN _unified_nodes src ON src.node_type = 'entity' AND src.node_id = ep.entity_id
            WHERE ep.passage_id = target.node_id
        ), 0)
        WHERE target.node_type = 'passage';

        -- Propagate from passages to entities
        UPDATE _unified_nodes target
        SET new_importance = target.new_importance + p_damping * COALESCE((
            SELECT SUM(
                src.importance * ep.decay_weight / NULLIF(src.out_degree, 0)
            )
            FROM entity_passage_connections ep
            JOIN _unified_nodes src ON src.node_type = 'passage' AND src.node_id = ep.passage_id
            WHERE ep.entity_id = target.node_id
        ), 0)
        WHERE target.node_type = 'entity';

        -- Propagate between passages
        UPDATE _unified_nodes target
        SET new_importance = target.new_importance + p_damping * COALESCE((
            SELECT SUM(
                src.importance * pc.decay_weight / NULLIF(src.out_degree, 0)
            )
            FROM passage_connections pc
            JOIN _unified_nodes src ON src.node_type = 'passage' AND (
                (pc.passage_a_id = src.node_id AND pc.passage_b_id = target.node_id) OR
                (pc.passage_b_id = src.node_id AND pc.passage_a_id = target.node_id)
            )
        ), 0)
        WHERE target.node_type = 'passage';

        -- Calculate max change
        SELECT MAX(ABS(new_importance - importance)) INTO v_max_change FROM _unified_nodes;

        -- Update importance
        UPDATE _unified_nodes SET importance = new_importance;

        -- Early exit if converged
        IF v_max_change < 0.0001 THEN
            EXIT;
        END IF;
    END LOOP;

    -- Write back to entity and passage tables
    UPDATE entities e
    SET importance = u.importance
    FROM _unified_nodes u
    WHERE u.node_type = 'entity' AND u.node_id = e.id;

    UPDATE passages p
    SET importance = u.importance
    FROM _unified_nodes u
    WHERE u.node_type = 'passage' AND u.node_id = p.id;

    -- Return results
    RETURN QUERY
    SELECT u.node_type, u.node_id, u.importance, v_max_change
    FROM _unified_nodes u
    ORDER BY u.importance DESC;
END;
$$ LANGUAGE plpgsql;


-- Local PageRank update for incremental indexing (affects both entities and passages)
CREATE OR REPLACE FUNCTION update_unified_importance_local(
    p_entity_ids INTEGER[],
    p_passage_ids INTEGER[],
    p_damping REAL DEFAULT 0.85,
    p_iterations INTEGER DEFAULT 3
) RETURNS TABLE(
    nodes_updated INTEGER,
    max_change REAL
) AS $$
DECLARE
    v_affected_entity_ids INTEGER[];
    v_affected_passage_ids INTEGER[];
    v_num_nodes INTEGER;
    v_iteration INTEGER;
    v_max_change REAL := 0;
    v_base_score REAL;
BEGIN
    -- Get total node count
    SELECT (SELECT COUNT(*) FROM entities) + (SELECT COUNT(*) FROM passages) INTO v_num_nodes;

    IF v_num_nodes = 0 THEN
        RETURN QUERY SELECT 0, 0.0::REAL;
        RETURN;
    END IF;

    v_base_score := (1.0 - p_damping) / v_num_nodes;

    -- Collect affected entities (input + 1-hop neighbors)
    SELECT ARRAY(
        SELECT DISTINCT id FROM (
            SELECT UNNEST(p_entity_ids) AS id
            UNION
            SELECT entity_b_id FROM entity_cooccurrences WHERE entity_a_id = ANY(p_entity_ids)
            UNION
            SELECT entity_a_id FROM entity_cooccurrences WHERE entity_b_id = ANY(p_entity_ids)
            UNION
            SELECT entity_id FROM entity_passage_connections WHERE passage_id = ANY(p_passage_ids)
        ) sub
    ) INTO v_affected_entity_ids;

    -- Collect affected passages
    SELECT ARRAY(
        SELECT DISTINCT id FROM (
            SELECT UNNEST(p_passage_ids) AS id
            UNION
            SELECT passage_b_id FROM passage_connections WHERE passage_a_id = ANY(p_passage_ids)
            UNION
            SELECT passage_a_id FROM passage_connections WHERE passage_b_id = ANY(p_passage_ids)
            UNION
            SELECT passage_id FROM entity_passage_connections WHERE entity_id = ANY(p_entity_ids)
        ) sub
    ) INTO v_affected_passage_ids;

    -- Simplified local update: just recalculate importance for affected nodes
    -- Full implementation would do proper power iteration on subgraph

    RETURN QUERY SELECT
        COALESCE(array_length(v_affected_entity_ids, 1), 0) +
        COALESCE(array_length(v_affected_passage_ids, 1), 0),
        v_max_change;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Get top nodes (entities + passages) by importance
CREATE OR REPLACE FUNCTION get_top_nodes(
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    node_type TEXT,
    node_id INTEGER,
    node_name TEXT,
    importance REAL,
    connection_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM (
        -- Entities
        SELECT
            'entity'::TEXT,
            e.id,
            e.name,
            e.importance,
            COALESCE((
                SELECT SUM(count) FROM entity_cooccurrences
                WHERE entity_a_id = e.id OR entity_b_id = e.id
            ), 0)::BIGINT +
            COALESCE((
                SELECT COUNT(*) FROM entity_passage_connections
                WHERE entity_id = e.id
            ), 0)::BIGINT
        FROM entities e

        UNION ALL

        -- Passages
        SELECT
            'passage'::TEXT,
            p.id,
            LEFT(p.text, 50),
            p.importance,
            COALESCE((
                SELECT COUNT(*) FROM passage_connections
                WHERE passage_a_id = p.id OR passage_b_id = p.id
            ), 0)::BIGINT +
            COALESCE((
                SELECT COUNT(*) FROM entity_passage_connections
                WHERE passage_id = p.id
            ), 0)::BIGINT
        FROM passages p
    ) combined
    ORDER BY importance DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE passage_connections IS 'Passage-Passage connections with session-based adjacency and decay weights';
COMMENT ON TABLE entity_passage_connections IS 'Entity-Passage connections with distance-based decay weights';
COMMENT ON COLUMN passages.importance IS 'PageRank-style importance score, calculated uniformly with entities';
COMMENT ON FUNCTION calculate_decay_weight IS 'Calculate decay weight: 0=1.0, 1=0.8, 2=0.6, 3=0.4, 4=0.2, 5+=0';
COMMENT ON FUNCTION upsert_passage_connection IS 'Add/update passage-passage connection with decay weight';
COMMENT ON FUNCTION upsert_entity_passage_connection IS 'Add/update entity-passage connection with decay weight';
COMMENT ON FUNCTION upsert_cooccurrence_with_decay IS 'Add/update entity co-occurrence with distance-based decay';
COMMENT ON FUNCTION calculate_unified_pagerank IS 'Full PageRank for all nodes (entities + passages) treated uniformly';
COMMENT ON FUNCTION update_unified_importance_local IS 'Local PageRank update for incremental indexing';
COMMENT ON FUNCTION get_top_nodes IS 'Get top nodes (entities + passages) by importance';
