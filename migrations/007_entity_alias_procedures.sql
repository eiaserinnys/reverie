-- Stored procedures for entity alias operations
-- merge_entity_alias: Add alias and merge matching entity into canonical
-- remove_entity_alias: Remove alias (entities stay merged - no undo)

-- Function to merge a source entity into a target entity
-- Updates all references and deletes source
-- IMPORTANT: Cooccurrence weights are aggregated (counts summed, weight recalculated)
CREATE OR REPLACE FUNCTION merge_entity_into(
    p_source_id INTEGER,
    p_target_id INTEGER
) RETURNS TABLE(
    facts_updated INTEGER,
    cooccurrences_updated INTEGER,
    passages_updated INTEGER
) AS $$
DECLARE
    v_facts_updated INTEGER := 0;
    v_cooc_updated INTEGER := 0;
    v_passages_updated INTEGER := 0;
    v_count INTEGER;
    r RECORD;
BEGIN
    -- Validate inputs
    IF p_source_id = p_target_id THEN
        RAISE EXCEPTION 'Source and target cannot be the same entity';
    END IF;

    -- Update facts - subject_entity_id
    UPDATE facts SET subject_entity_id = p_target_id
    WHERE subject_entity_id = p_source_id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    v_facts_updated := v_facts_updated + v_count;

    -- Update facts - object_entity_id
    UPDATE facts SET object_entity_id = p_target_id
    WHERE object_entity_id = p_source_id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    v_facts_updated := v_facts_updated + v_count;

    -- entity_cooccurrences: Delete self-referencing edges (source-target)
    DELETE FROM entity_cooccurrences
    WHERE (entity_a_id = p_source_id AND entity_b_id = p_target_id)
       OR (entity_a_id = p_target_id AND entity_b_id = p_source_id);

    -- Merge cooccurrence weights for edges that would become duplicates
    -- For each source edge, if a matching target edge exists, aggregate them
    FOR r IN
        SELECT
            ec_src.entity_a_id AS src_a,
            ec_src.entity_b_id AS src_b,
            ec_src.count AS src_count,
            ec_src.weight AS src_weight,
            -- The neighbor is the one that's not source_id
            CASE WHEN ec_src.entity_a_id = p_source_id THEN ec_src.entity_b_id ELSE ec_src.entity_a_id END AS neighbor_id
        FROM entity_cooccurrences ec_src
        WHERE ec_src.entity_a_id = p_source_id OR ec_src.entity_b_id = p_source_id
    LOOP
        DECLARE
            v_target_a INTEGER := LEAST(p_target_id, r.neighbor_id);
            v_target_b INTEGER := GREATEST(p_target_id, r.neighbor_id);
            v_existing_count INTEGER;
            v_existing_weight FLOAT;
            v_new_count INTEGER;
            v_new_weight FLOAT;
        BEGIN
            -- Check if target already has an edge to this neighbor
            SELECT count, weight INTO v_existing_count, v_existing_weight
            FROM entity_cooccurrences
            WHERE entity_a_id = v_target_a AND entity_b_id = v_target_b;

            IF FOUND THEN
                -- Aggregate: sum counts, recalculate weight as 1/total_count
                v_new_count := v_existing_count + r.src_count;
                v_new_weight := 1.0 / v_new_count;

                UPDATE entity_cooccurrences
                SET count = v_new_count, weight = v_new_weight, updated_at = NOW()
                WHERE entity_a_id = v_target_a AND entity_b_id = v_target_b;

                -- Delete the source edge
                DELETE FROM entity_cooccurrences
                WHERE entity_a_id = r.src_a AND entity_b_id = r.src_b;
            ELSE
                -- No existing target edge, just update source edge to point to target
                UPDATE entity_cooccurrences
                SET entity_a_id = v_target_a, entity_b_id = v_target_b, updated_at = NOW()
                WHERE entity_a_id = r.src_a AND entity_b_id = r.src_b;
            END IF;

            v_cooc_updated := v_cooc_updated + 1;
        END;
    END LOOP;

    -- Delete any self-references that might have been created
    DELETE FROM entity_cooccurrences WHERE entity_a_id = entity_b_id;

    -- entity_passages: Delete duplicates first
    DELETE FROM entity_passages ep
    WHERE ep.entity_id = p_source_id
    AND EXISTS (
        SELECT 1 FROM entity_passages ep2
        WHERE ep2.entity_id = p_target_id AND ep2.passage_id = ep.passage_id
    );

    UPDATE entity_passages SET entity_id = p_target_id
    WHERE entity_id = p_source_id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    v_passages_updated := v_passages_updated + v_count;

    -- Delete the source entity
    DELETE FROM entities WHERE id = p_source_id;

    RETURN QUERY SELECT v_facts_updated, v_cooc_updated, v_passages_updated;
END;
$$ LANGUAGE plpgsql;


-- Main procedure: Add alias and merge matching entity
CREATE OR REPLACE FUNCTION merge_entity_alias(
    p_alias_name VARCHAR(512),
    p_canonical_name VARCHAR(512),
    p_created_by VARCHAR(64) DEFAULT 'user'
) RETURNS TABLE(
    status TEXT,
    canonical_id INTEGER,
    source_id INTEGER,
    facts_updated INTEGER,
    cooccurrences_updated INTEGER,
    passages_updated INTEGER
) AS $$
DECLARE
    v_canonical_id INTEGER;
    v_source_id INTEGER;
    v_facts INTEGER := 0;
    v_cooc INTEGER := 0;
    v_passages INTEGER := 0;
    v_result RECORD;
BEGIN
    -- Find canonical entity
    SELECT id INTO v_canonical_id FROM entities
    WHERE LOWER(name) = LOWER(p_canonical_name);

    IF v_canonical_id IS NULL THEN
        RETURN QUERY SELECT
            'error: canonical entity not found'::TEXT,
            NULL::INTEGER, NULL::INTEGER, 0, 0, 0;
        RETURN;
    END IF;

    -- Check if alias already exists
    IF EXISTS (SELECT 1 FROM entity_aliases WHERE LOWER(alias_name) = LOWER(p_alias_name)) THEN
        RETURN QUERY SELECT
            'error: alias already exists'::TEXT,
            v_canonical_id, NULL::INTEGER, 0, 0, 0;
        RETURN;
    END IF;

    -- Find source entity matching alias name
    SELECT id INTO v_source_id FROM entities
    WHERE LOWER(name) = LOWER(p_alias_name) AND id != v_canonical_id;

    -- Insert alias
    INSERT INTO entity_aliases (canonical_entity_id, alias_name, created_by)
    VALUES (v_canonical_id, p_alias_name, p_created_by);

    -- If source entity exists, merge it
    IF v_source_id IS NOT NULL THEN
        SELECT * INTO v_result FROM merge_entity_into(v_source_id, v_canonical_id);
        v_facts := v_result.facts_updated;
        v_cooc := v_result.cooccurrences_updated;
        v_passages := v_result.passages_updated;

        RETURN QUERY SELECT
            'merged'::TEXT,
            v_canonical_id, v_source_id, v_facts, v_cooc, v_passages;
    ELSE
        RETURN QUERY SELECT
            'alias_added'::TEXT,
            v_canonical_id, NULL::INTEGER, 0, 0, 0;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- Remove an alias (does not undo merge - entities stay merged)
CREATE OR REPLACE FUNCTION remove_entity_alias(
    p_alias_name VARCHAR(512)
) RETURNS TABLE(
    status TEXT,
    removed_alias VARCHAR(512),
    canonical_name VARCHAR(512)
) AS $$
DECLARE
    v_canonical_id INTEGER;
    v_canonical_name VARCHAR(512);
BEGIN
    -- Find and delete the alias
    SELECT ea.canonical_entity_id, e.name
    INTO v_canonical_id, v_canonical_name
    FROM entity_aliases ea
    JOIN entities e ON ea.canonical_entity_id = e.id
    WHERE LOWER(ea.alias_name) = LOWER(p_alias_name);

    IF v_canonical_id IS NULL THEN
        RETURN QUERY SELECT
            'error: alias not found'::TEXT,
            p_alias_name,
            NULL::VARCHAR(512);
        RETURN;
    END IF;

    DELETE FROM entity_aliases WHERE LOWER(alias_name) = LOWER(p_alias_name);

    RETURN QUERY SELECT
        'removed'::TEXT,
        p_alias_name,
        v_canonical_name;
END;
$$ LANGUAGE plpgsql;


-- List all aliases (convenience function)
CREATE OR REPLACE FUNCTION list_entity_aliases()
RETURNS TABLE(
    canonical_name TEXT,
    alias_name TEXT,
    created_by TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT e.name::TEXT, ea.alias_name::TEXT, ea.created_by::TEXT, ea.created_at
    FROM entity_aliases ea
    JOIN entities e ON ea.canonical_entity_id = e.id
    ORDER BY e.name, ea.alias_name;
END;
$$ LANGUAGE plpgsql;


-- Comments
COMMENT ON FUNCTION merge_entity_alias IS 'Add alias and merge matching entity into canonical. Cooccurrence weights are aggregated.';
COMMENT ON FUNCTION remove_entity_alias IS 'Remove alias. Note: merged entities stay merged.';
COMMENT ON FUNCTION merge_entity_into IS 'Internal: merge source entity into target, aggregating cooccurrence weights.';
COMMENT ON FUNCTION list_entity_aliases IS 'List all entity aliases with their canonical names.';
