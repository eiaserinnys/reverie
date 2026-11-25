-- Enrich Pipeline Procedures
-- DB procedures for query-based entity exploration and scoring
-- All weight/score calculations are encapsulated in these procedures

-- ============================================================================
-- ENTITY SCORING
-- ============================================================================

-- Calculate combined score for an entity relative to seed entities and query
-- Score = cooccurrence_strength * importance * embedding_similarity * time_decay
CREATE OR REPLACE FUNCTION calculate_entity_score(
    p_seed_entity_ids INTEGER[],
    p_candidate_entity_id INTEGER,
    p_query_embedding VECTOR(1536) DEFAULT NULL
) RETURNS TABLE(
    entity_id INTEGER,
    entity_name TEXT,
    importance REAL,
    cooccurrence_score REAL,
    embedding_score REAL,
    combined_score REAL
) AS $$
DECLARE
    v_importance REAL;
    v_cooc_score REAL := 0;
    v_embedding_score REAL := 1.0;
    v_combined REAL;
BEGIN
    -- Get entity importance
    SELECT e.importance, e.name INTO v_importance, entity_name
    FROM entities e
    WHERE e.id = p_candidate_entity_id;

    IF v_importance IS NULL THEN
        RETURN;
    END IF;

    entity_id := p_candidate_entity_id;
    importance := v_importance;

    -- Calculate co-occurrence score (sum of connection strengths to seeds)
    SELECT COALESCE(SUM(
        CASE
            WHEN co.entity_a_id = p_candidate_entity_id THEN co.count
            WHEN co.entity_b_id = p_candidate_entity_id THEN co.count
            ELSE 0
        END
    ), 0) INTO v_cooc_score
    FROM entity_cooccurrences co
    WHERE (co.entity_a_id = p_candidate_entity_id AND co.entity_b_id = ANY(p_seed_entity_ids))
       OR (co.entity_b_id = p_candidate_entity_id AND co.entity_a_id = ANY(p_seed_entity_ids));

    cooccurrence_score := v_cooc_score;

    -- Calculate embedding similarity if query embedding provided
    IF p_query_embedding IS NOT NULL THEN
        SELECT 1 - (e.embedding <=> p_query_embedding) INTO v_embedding_score
        FROM entities e
        WHERE e.id = p_candidate_entity_id AND e.embedding IS NOT NULL;

        v_embedding_score := COALESCE(v_embedding_score, 0.5);
    END IF;

    embedding_score := v_embedding_score;

    -- Combined score (multiplicative)
    -- Normalize cooccurrence to 0-1 range (assuming max count ~100)
    v_combined := v_importance * (1 + v_cooc_score / 10.0) * v_embedding_score;
    combined_score := v_combined;

    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- ENTITY EXPLORATION
-- ============================================================================

-- Explore entities from seed entities using co-occurrence graph
-- Returns scored entities within max_hops of any seed
CREATE OR REPLACE FUNCTION explore_entities_from_seeds(
    p_seed_entity_ids INTEGER[],
    p_query_embedding VECTOR(1536) DEFAULT NULL,
    p_max_hops INTEGER DEFAULT 2,
    p_limit INTEGER DEFAULT 50
) RETURNS TABLE(
    out_entity_id INTEGER,
    out_entity_name TEXT,
    out_importance REAL,
    out_cooccurrence_score REAL,
    out_embedding_score REAL,
    out_combined_score REAL,
    out_hop_distance INTEGER,
    out_connected_from INTEGER
) AS $$
DECLARE
    v_current_hop INTEGER := 0;
    v_current_layer INTEGER[];
    v_next_layer INTEGER[];
    v_visited INTEGER[] := p_seed_entity_ids;
    v_eid INTEGER;
    v_neighbor_id INTEGER;
BEGIN
    -- Create temp table for results
    CREATE TEMP TABLE IF NOT EXISTS _explore_results (
        ent_id INTEGER PRIMARY KEY,
        ent_name TEXT,
        ent_importance REAL,
        ent_cooccurrence_score REAL,
        ent_embedding_score REAL,
        ent_combined_score REAL,
        ent_hop_distance INTEGER,
        ent_connected_from INTEGER
    ) ON COMMIT DROP;

    TRUNCATE _explore_results;

    -- Add seeds with hop_distance = 0
    FOR v_eid IN SELECT UNNEST(p_seed_entity_ids)
    LOOP
        INSERT INTO _explore_results (ent_id, ent_name, ent_importance, ent_cooccurrence_score,
                                       ent_embedding_score, ent_combined_score, ent_hop_distance, ent_connected_from)
        SELECT
            cs.entity_id,
            cs.entity_name,
            cs.importance,
            cs.cooccurrence_score,
            cs.embedding_score,
            cs.combined_score,
            0,
            NULL::INTEGER
        FROM calculate_entity_score(p_seed_entity_ids, v_eid, p_query_embedding) cs
        ON CONFLICT (ent_id) DO NOTHING;
    END LOOP;

    -- BFS exploration
    v_current_layer := p_seed_entity_ids;

    FOR v_current_hop IN 1..p_max_hops
    LOOP
        v_next_layer := ARRAY[]::INTEGER[];

        FOR v_eid IN SELECT UNNEST(v_current_layer)
        LOOP
            -- Get neighbors from co-occurrence graph
            FOR v_neighbor_id IN
                SELECT CASE
                    WHEN co.entity_a_id = v_eid THEN co.entity_b_id
                    ELSE co.entity_a_id
                END
                FROM entity_cooccurrences co
                WHERE co.entity_a_id = v_eid OR co.entity_b_id = v_eid
            LOOP
                IF NOT v_neighbor_id = ANY(v_visited) THEN
                    v_visited := array_append(v_visited, v_neighbor_id);
                    v_next_layer := array_append(v_next_layer, v_neighbor_id);

                    INSERT INTO _explore_results (ent_id, ent_name, ent_importance, ent_cooccurrence_score,
                                                   ent_embedding_score, ent_combined_score, ent_hop_distance, ent_connected_from)
                    SELECT
                        cs.entity_id,
                        cs.entity_name,
                        cs.importance,
                        cs.cooccurrence_score,
                        cs.embedding_score,
                        cs.combined_score,
                        v_current_hop,
                        v_eid
                    FROM calculate_entity_score(p_seed_entity_ids, v_neighbor_id, p_query_embedding) cs
                    ON CONFLICT (ent_id) DO NOTHING;
                END IF;
            END LOOP;
        END LOOP;

        v_current_layer := v_next_layer;
        EXIT WHEN array_length(v_current_layer, 1) IS NULL;
    END LOOP;

    -- Return results ordered by combined score
    RETURN QUERY
    SELECT r.ent_id, r.ent_name, r.ent_importance, r.ent_cooccurrence_score,
           r.ent_embedding_score, r.ent_combined_score, r.ent_hop_distance, r.ent_connected_from
    FROM _explore_results r
    ORDER BY r.ent_combined_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- VISUALIZATION HELPERS
-- ============================================================================

-- Get entities with importance for visualization (around a center entity)
CREATE OR REPLACE FUNCTION get_entities_for_visualization(
    p_center_entity_id INTEGER,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    entity_id INTEGER,
    entity_name TEXT,
    importance REAL,
    cooccurrence_count BIGINT,
    hop_distance INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH neighbors AS (
        SELECT
            CASE
                WHEN co.entity_a_id = p_center_entity_id THEN co.entity_b_id
                ELSE co.entity_a_id
            END AS neighbor_id,
            co.count
        FROM entity_cooccurrences co
        WHERE co.entity_a_id = p_center_entity_id OR co.entity_b_id = p_center_entity_id
    )
    SELECT
        e.id,
        e.name,
        e.importance,
        COALESCE(n.count, 0)::BIGINT,
        CASE WHEN e.id = p_center_entity_id THEN 0 ELSE 1 END
    FROM entities e
    LEFT JOIN neighbors n ON n.neighbor_id = e.id
    WHERE e.id = p_center_entity_id
       OR e.id IN (SELECT neighbor_id FROM neighbors)
    ORDER BY e.importance DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- Get co-occurrence edges with combined scores for visualization
CREATE OR REPLACE FUNCTION get_cooccurrence_edges(
    p_entity_ids INTEGER[]
) RETURNS TABLE(
    out_entity_a_id INTEGER,
    out_entity_b_id INTEGER,
    out_count INTEGER,
    out_weight REAL,
    out_importance_a REAL,
    out_importance_b REAL,
    out_combined_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        co.entity_a_id::INTEGER,
        co.entity_b_id::INTEGER,
        co.count::INTEGER,
        co.weight::REAL,
        ea.importance::REAL,
        eb.importance::REAL,
        -- Combined score: count * sqrt(importance_a * importance_b)
        (co.count::REAL * SQRT(COALESCE(ea.importance, 0.01) * COALESCE(eb.importance, 0.01)))::REAL
    FROM entity_cooccurrences co
    JOIN entities ea ON ea.id = co.entity_a_id
    JOIN entities eb ON eb.id = co.entity_b_id
    WHERE co.entity_a_id = ANY(p_entity_ids)
       OR co.entity_b_id = ANY(p_entity_ids);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- FACT RETRIEVAL
-- ============================================================================

-- Get facts connected to a set of entities, scored by relevance
CREATE OR REPLACE FUNCTION get_facts_for_entities(
    p_entity_ids INTEGER[],
    p_query_embedding VECTOR(1536) DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE(
    fact_id INTEGER,
    subject_id INTEGER,
    subject_name TEXT,
    predicate TEXT,
    object_id INTEGER,
    object_name TEXT,
    relevance_score REAL,
    recency_score REAL,
    combined_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.id,
        f.subject_entity_id,
        es.name,
        f.predicate,
        f.object_entity_id,
        eo.name,
        -- Relevance: embedding similarity if available
        CASE
            WHEN p_query_embedding IS NOT NULL AND f.embedding IS NOT NULL
            THEN 1 - (f.embedding <=> p_query_embedding)
            ELSE 0.5
        END::REAL,
        -- Recency: decay based on age (half-life ~30 days)
        EXP(-EXTRACT(EPOCH FROM (NOW() - f.created_at)) / (30 * 24 * 3600))::REAL,
        -- Combined
        CASE
            WHEN p_query_embedding IS NOT NULL AND f.embedding IS NOT NULL
            THEN (1 - (f.embedding <=> p_query_embedding)) *
                 EXP(-EXTRACT(EPOCH FROM (NOW() - f.created_at)) / (30 * 24 * 3600))
            ELSE 0.5 * EXP(-EXTRACT(EPOCH FROM (NOW() - f.created_at)) / (30 * 24 * 3600))
        END::REAL
    FROM facts f
    JOIN entities es ON es.id = f.subject_entity_id
    JOIN entities eo ON eo.id = f.object_entity_id
    WHERE f.subject_entity_id = ANY(p_entity_ids)
       OR f.object_entity_id = ANY(p_entity_ids)
    ORDER BY combined_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- FULL ENRICH PIPELINE
-- ============================================================================

-- Complete enrich query: from seeds to scored entities and facts
CREATE OR REPLACE FUNCTION enrich_query(
    p_seed_entity_ids INTEGER[],
    p_query_embedding VECTOR(1536) DEFAULT NULL,
    p_max_hops INTEGER DEFAULT 2,
    p_max_entities INTEGER DEFAULT 50,
    p_max_facts INTEGER DEFAULT 20
) RETURNS TABLE(
    out_result_type TEXT,  -- 'entity' or 'fact'
    out_entity_id INTEGER,
    out_entity_name TEXT,
    out_importance REAL,
    out_hop_distance INTEGER,
    out_fact_id INTEGER,
    out_subject_name TEXT,
    out_predicate TEXT,
    out_object_name TEXT,
    out_score REAL
) AS $$
DECLARE
    v_entity_ids INTEGER[];
BEGIN
    -- Get explored entities
    CREATE TEMP TABLE IF NOT EXISTS _enrich_entities (
        ent_id INTEGER,
        ent_name TEXT,
        ent_importance REAL,
        ent_hop_distance INTEGER,
        ent_score REAL
    ) ON COMMIT DROP;

    TRUNCATE _enrich_entities;

    INSERT INTO _enrich_entities (ent_id, ent_name, ent_importance, ent_hop_distance, ent_score)
    SELECT e.out_entity_id, e.out_entity_name, e.out_importance, e.out_hop_distance, e.out_combined_score
    FROM explore_entities_from_seeds(p_seed_entity_ids, p_query_embedding, p_max_hops, p_max_entities) e;

    -- Return entities
    RETURN QUERY
    SELECT
        'entity'::TEXT,
        ee.ent_id,
        ee.ent_name,
        ee.ent_importance,
        ee.ent_hop_distance,
        NULL::INTEGER,
        NULL::TEXT,
        NULL::TEXT,
        NULL::TEXT,
        ee.ent_score
    FROM _enrich_entities ee;

    -- Collect entity IDs for fact lookup
    SELECT ARRAY_AGG(ee.ent_id) INTO v_entity_ids FROM _enrich_entities ee;

    -- Return facts
    IF v_entity_ids IS NOT NULL THEN
        RETURN QUERY
        SELECT
            'fact'::TEXT,
            NULL::INTEGER,
            NULL::TEXT,
            NULL::REAL,
            NULL::INTEGER,
            gf.fact_id,
            gf.subject_name,
            gf.predicate,
            gf.object_name,
            gf.combined_score
        FROM get_facts_for_entities(v_entity_ids, p_query_embedding, p_max_facts) gf;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION calculate_entity_score IS 'Calculate combined score for entity based on co-occurrence, importance, and embedding similarity';
COMMENT ON FUNCTION explore_entities_from_seeds IS 'BFS exploration from seed entities with scoring';
COMMENT ON FUNCTION get_entities_for_visualization IS 'Get entities around a center for graph visualization';
COMMENT ON FUNCTION get_cooccurrence_edges IS 'Get co-occurrence edges with combined scores';
COMMENT ON FUNCTION get_facts_for_entities IS 'Get facts connected to entities, scored by relevance';
COMMENT ON FUNCTION enrich_query IS 'Complete enrich pipeline: seeds -> entities + facts';
