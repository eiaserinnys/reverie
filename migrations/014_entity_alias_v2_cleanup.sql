-- Entity alias procedures v2.1: Remove legacy table references
--
-- Migration 013 dropped facts, entity_cooccurrences, entity_passage_connections
-- This migration updates the alias procedures to only use v2 tables:
-- - entities, passages, entity_passages (bipartite graph)

-- ============================================================================
-- DROP OLD FUNCTIONS (required due to signature changes)
-- ============================================================================

DROP FUNCTION IF EXISTS merge_entity_into(integer, integer);
DROP FUNCTION IF EXISTS merge_entity_alias(varchar, varchar, varchar);

-- ============================================================================
-- UPDATED MERGE FUNCTION (v2 - no facts, cooccurrences, or passage_connections)
-- ============================================================================

-- Function to merge a source entity into a target entity (v2.1)
-- Only handles entity_passages (bipartite graph)
CREATE OR REPLACE FUNCTION merge_entity_into(
    p_source_id INTEGER,
    p_target_id INTEGER
) RETURNS TABLE(
    passages_updated INTEGER
) AS $$
DECLARE
    v_passages_updated INTEGER := 0;
    v_count INTEGER;
BEGIN
    -- Validate inputs
    IF p_source_id = p_target_id THEN
        RAISE EXCEPTION 'Source and target cannot be the same entity';
    END IF;

    -- entity_passages: Delete duplicates first, then update
    -- (If target already has an edge to the same passage, delete source's edge)
    DELETE FROM entity_passages ep
    WHERE ep.entity_id = p_source_id
    AND EXISTS (
        SELECT 1 FROM entity_passages ep2
        WHERE ep2.entity_id = p_target_id AND ep2.passage_id = ep.passage_id
    );

    -- Update remaining edges to point to target entity
    UPDATE entity_passages SET entity_id = p_target_id
    WHERE entity_id = p_source_id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    v_passages_updated := v_passages_updated + v_count;

    -- Delete the source entity
    DELETE FROM entities WHERE id = p_source_id;

    RETURN QUERY SELECT v_passages_updated;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- UPDATED MERGE ALIAS FUNCTION (v2.1)
-- ============================================================================

-- Main procedure: Add alias and merge matching entity (v2.1)
-- Simplified return signature for v2 architecture
CREATE OR REPLACE FUNCTION merge_entity_alias(
    p_alias_name VARCHAR(512),
    p_canonical_name VARCHAR(512),
    p_created_by VARCHAR(64) DEFAULT 'user'
) RETURNS TABLE(
    status TEXT,
    canonical_id INTEGER,
    source_id INTEGER,
    passages_updated INTEGER
) AS $$
DECLARE
    v_canonical_id INTEGER;
    v_source_id INTEGER;
    v_passages INTEGER := 0;
    v_result RECORD;
BEGIN
    -- Find canonical entity
    SELECT id INTO v_canonical_id FROM entities
    WHERE LOWER(name) = LOWER(p_canonical_name);

    IF v_canonical_id IS NULL THEN
        RETURN QUERY SELECT
            'error: canonical entity not found'::TEXT,
            NULL::INTEGER, NULL::INTEGER, 0;
        RETURN;
    END IF;

    -- Check if alias already exists
    IF EXISTS (SELECT 1 FROM entity_aliases WHERE LOWER(alias_name) = LOWER(p_alias_name)) THEN
        RETURN QUERY SELECT
            'error: alias already exists'::TEXT,
            v_canonical_id, NULL::INTEGER, 0;
        RETURN;
    END IF;

    -- Find source entity matching alias name
    SELECT id INTO v_source_id FROM entities
    WHERE LOWER(name) = LOWER(p_alias_name) AND id != v_canonical_id;

    -- Insert alias with original_entity_id for rollback support
    INSERT INTO entity_aliases (canonical_entity_id, alias_name, created_by, original_entity_id)
    VALUES (v_canonical_id, p_alias_name, p_created_by, v_source_id);

    -- If source entity exists, merge it
    IF v_source_id IS NOT NULL THEN
        SELECT * INTO v_result FROM merge_entity_into(v_source_id, v_canonical_id);
        v_passages := v_result.passages_updated;

        RETURN QUERY SELECT
            'merged'::TEXT,
            v_canonical_id, v_source_id, v_passages;
    ELSE
        RETURN QUERY SELECT
            'alias_added'::TEXT,
            v_canonical_id, NULL::INTEGER, 0;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION merge_entity_into IS 'Merge source entity into target. v2.1: Only handles entity_passages (bipartite graph).';
COMMENT ON FUNCTION merge_entity_alias IS 'Add alias and merge matching entity. v2.1: Simplified for bipartite graph.';
