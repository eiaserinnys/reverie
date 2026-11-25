-- Stored procedures for atomic, progressive indexing
-- These procedures handle entity resolution (aliases) and co-occurrence updates

-- ============================================================================
-- ENTITY PROCEDURES
-- ============================================================================

-- Upsert entity with automatic alias resolution
-- If the name matches an alias, returns the canonical entity ID instead
-- Note: p_embedding uses VECTOR(1536) for OpenAI ada-002 embeddings
CREATE OR REPLACE FUNCTION upsert_entity(
    p_name VARCHAR(512),
    p_embedding VECTOR(1536)
) RETURNS TABLE(
    entity_id INTEGER,
    is_new BOOLEAN,
    resolved_from_alias BOOLEAN,
    canonical_name VARCHAR(512)
) AS $$
DECLARE
    v_entity_id INTEGER;
    v_canonical_id INTEGER;
    v_canonical_name VARCHAR(512);
    v_is_new BOOLEAN := FALSE;
BEGIN
    -- Check if name matches an existing alias
    SELECT ea.canonical_entity_id, e.name
    INTO v_canonical_id, v_canonical_name
    FROM entity_aliases ea
    JOIN entities e ON ea.canonical_entity_id = e.id
    WHERE LOWER(ea.alias_name) = LOWER(p_name);

    IF v_canonical_id IS NOT NULL THEN
        -- Return canonical entity (alias resolved)
        RETURN QUERY SELECT v_canonical_id, FALSE, TRUE, v_canonical_name;
        RETURN;
    END IF;

    -- Check if entity already exists
    SELECT id INTO v_entity_id FROM entities
    WHERE LOWER(name) = LOWER(p_name);

    IF v_entity_id IS NOT NULL THEN
        -- Update embedding if provided
        IF p_embedding IS NOT NULL THEN
            UPDATE entities SET embedding = p_embedding WHERE id = v_entity_id;
        END IF;
        RETURN QUERY SELECT v_entity_id, FALSE, FALSE, p_name;
    ELSE
        -- Insert new entity with ON CONFLICT to handle race conditions
        INSERT INTO entities (name, embedding)
        VALUES (p_name, p_embedding)
        ON CONFLICT (name) DO UPDATE SET
            embedding = COALESCE(EXCLUDED.embedding, entities.embedding)
        RETURNING id, (xmax = 0) INTO v_entity_id, v_is_new;

        RETURN QUERY SELECT v_entity_id, v_is_new, FALSE, p_name;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- PASSAGE PROCEDURES
-- ============================================================================

-- Insert passage and create entity-passage links
CREATE OR REPLACE FUNCTION insert_passage_with_entities(
    p_session_id VARCHAR(255),
    p_turn_index INTEGER,
    p_text TEXT,
    p_source VARCHAR(50),
    p_embedding VECTOR(1536),
    p_entity_ids INTEGER[]
) RETURNS TABLE(
    passage_id INTEGER,
    entity_links_created INTEGER
) AS $$
DECLARE
    v_passage_id INTEGER;
    v_entity_id INTEGER;
    v_links_created INTEGER := 0;
BEGIN
    -- Insert passage
    INSERT INTO passages (session_id, turn_index, text, source, embedding)
    VALUES (p_session_id, p_turn_index, p_text, p_source, p_embedding)
    RETURNING id INTO v_passage_id;

    -- Create entity-passage links
    IF p_entity_ids IS NOT NULL THEN
        FOREACH v_entity_id IN ARRAY p_entity_ids
        LOOP
            INSERT INTO entity_passages (entity_id, passage_id)
            VALUES (v_entity_id, v_passage_id)
            ON CONFLICT (entity_id, passage_id) DO NOTHING;

            IF FOUND THEN
                v_links_created := v_links_created + 1;
            END IF;
        END LOOP;
    END IF;

    RETURN QUERY SELECT v_passage_id, v_links_created;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- FACT PROCEDURES
-- ============================================================================

-- Insert fact and update co-occurrence
-- Automatically resolves entity aliases and updates co-occurrence weights
CREATE OR REPLACE FUNCTION insert_fact_with_cooccurrence(
    p_subject_entity_id INTEGER,
    p_predicate VARCHAR(255),
    p_object_entity_id INTEGER,
    p_passage_id INTEGER,
    p_turn_index INTEGER DEFAULT 0
) RETURNS TABLE(
    fact_id INTEGER,
    cooccurrence_updated BOOLEAN,
    new_count INTEGER,
    new_weight REAL
) AS $$
DECLARE
    v_fact_id INTEGER;
    v_entity_a INTEGER;
    v_entity_b INTEGER;
    v_current_count INTEGER;
    v_new_count INTEGER;
    v_new_weight REAL;
    v_cooc_updated BOOLEAN := FALSE;
BEGIN
    -- Resolve aliases for subject and object
    SELECT entity_id INTO p_subject_entity_id
    FROM upsert_entity(
        (SELECT name FROM entities WHERE id = p_subject_entity_id),
        NULL
    );

    SELECT entity_id INTO p_object_entity_id
    FROM upsert_entity(
        (SELECT name FROM entities WHERE id = p_object_entity_id),
        NULL
    );

    -- Insert fact
    INSERT INTO facts (subject_entity_id, predicate, object_entity_id)
    VALUES (p_subject_entity_id, p_predicate, p_object_entity_id)
    RETURNING id INTO v_fact_id;

    -- Link fact to passage
    IF p_passage_id IS NOT NULL THEN
        INSERT INTO passage_facts (passage_id, fact_id)
        VALUES (p_passage_id, v_fact_id)
        ON CONFLICT DO NOTHING;
    END IF;

    -- Update co-occurrence (if different entities)
    IF p_subject_entity_id != p_object_entity_id THEN
        -- Ensure entity_a_id < entity_b_id for constraint
        v_entity_a := LEAST(p_subject_entity_id, p_object_entity_id);
        v_entity_b := GREATEST(p_subject_entity_id, p_object_entity_id);

        -- Get current count
        SELECT count INTO v_current_count
        FROM entity_cooccurrences
        WHERE entity_a_id = v_entity_a AND entity_b_id = v_entity_b;

        IF v_current_count IS NOT NULL THEN
            -- Update existing
            v_new_count := v_current_count + 1;
            v_new_weight := 1.0 / v_new_count;

            UPDATE entity_cooccurrences
            SET count = v_new_count, weight = v_new_weight, updated_at = NOW()
            WHERE entity_a_id = v_entity_a AND entity_b_id = v_entity_b;
        ELSE
            -- Insert new
            v_new_count := 1;
            v_new_weight := 1.0;

            INSERT INTO entity_cooccurrences (entity_a_id, entity_b_id, count, weight)
            VALUES (v_entity_a, v_entity_b, v_new_count, v_new_weight);
        END IF;

        v_cooc_updated := TRUE;
    ELSE
        v_new_count := 0;
        v_new_weight := 0;
    END IF;

    RETURN QUERY SELECT v_fact_id, v_cooc_updated, v_new_count, v_new_weight;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- CO-OCCURRENCE PROCEDURES
-- ============================================================================

-- Update co-occurrence between two entities with turn distance weighting
-- weight_contribution: 1.0 for same turn, decreases with distance
CREATE OR REPLACE FUNCTION update_cooccurrence_weighted(
    p_entity_a_id INTEGER,
    p_entity_b_id INTEGER,
    p_weight_contribution REAL DEFAULT 1.0
) RETURNS TABLE(
    entity_a INTEGER,
    entity_b INTEGER,
    new_count INTEGER,
    new_weight REAL
) AS $$
DECLARE
    v_entity_a INTEGER;
    v_entity_b INTEGER;
    v_current_count INTEGER;
    v_current_total_weight REAL;
    v_new_count INTEGER;
    v_new_total_weight REAL;
    v_new_weight REAL;
BEGIN
    -- Skip if same entity
    IF p_entity_a_id = p_entity_b_id THEN
        RETURN;
    END IF;

    -- Ensure entity_a < entity_b
    v_entity_a := LEAST(p_entity_a_id, p_entity_b_id);
    v_entity_b := GREATEST(p_entity_a_id, p_entity_b_id);

    -- Get current values
    SELECT count, 1.0 / NULLIF(weight, 0) INTO v_current_count, v_current_total_weight
    FROM entity_cooccurrences
    WHERE entity_a_id = v_entity_a AND entity_b_id = v_entity_b;

    IF v_current_count IS NOT NULL THEN
        -- Update: add to count and total weight
        v_new_count := v_current_count + 1;
        v_new_total_weight := COALESCE(v_current_total_weight, 0) + p_weight_contribution;
        v_new_weight := 1.0 / v_new_total_weight;

        UPDATE entity_cooccurrences
        SET count = v_new_count, weight = v_new_weight, updated_at = NOW()
        WHERE entity_a_id = v_entity_a AND entity_b_id = v_entity_b;
    ELSE
        -- Insert new
        v_new_count := 1;
        v_new_total_weight := p_weight_contribution;
        v_new_weight := 1.0 / v_new_total_weight;

        INSERT INTO entity_cooccurrences (entity_a_id, entity_b_id, count, weight)
        VALUES (v_entity_a, v_entity_b, v_new_count, v_new_weight);
    END IF;

    RETURN QUERY SELECT v_entity_a, v_entity_b, v_new_count, v_new_weight;
END;
$$ LANGUAGE plpgsql;


-- Batch update co-occurrences for entities in a passage/turn
-- p_entity_ids: array of entity IDs that co-occur
-- p_turn_distances: optional array of turn distances (0 = same turn)
CREATE OR REPLACE FUNCTION update_cooccurrences_batch(
    p_entity_ids INTEGER[],
    p_base_weight REAL DEFAULT 1.0
) RETURNS TABLE(
    pairs_updated INTEGER
) AS $$
DECLARE
    v_pairs_updated INTEGER := 0;
    i INTEGER;
    j INTEGER;
BEGIN
    -- Update all pairs
    FOR i IN 1..array_length(p_entity_ids, 1) LOOP
        FOR j IN (i+1)..array_length(p_entity_ids, 1) LOOP
            PERFORM update_cooccurrence_weighted(
                p_entity_ids[i],
                p_entity_ids[j],
                p_base_weight
            );
            v_pairs_updated := v_pairs_updated + 1;
        END LOOP;
    END LOOP;

    RETURN QUERY SELECT v_pairs_updated;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION upsert_entity IS 'Insert/update entity with automatic alias resolution';
COMMENT ON FUNCTION insert_passage_with_entities IS 'Insert passage and create entity-passage links atomically';
COMMENT ON FUNCTION insert_fact_with_cooccurrence IS 'Insert fact and auto-update co-occurrence counts';
COMMENT ON FUNCTION update_cooccurrence_weighted IS 'Update co-occurrence with weight contribution';
COMMENT ON FUNCTION update_cooccurrences_batch IS 'Batch update co-occurrences for multiple entities';
