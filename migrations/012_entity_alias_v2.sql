-- Entity alias procedures v2
-- Adds support for entity_passage_connections and split (undo merge) functionality

-- ============================================================================
-- DROP OLD FUNCTIONS (required due to signature changes)
-- ============================================================================

DROP FUNCTION IF EXISTS merge_entity_into(integer, integer);
DROP FUNCTION IF EXISTS merge_entity_alias(varchar, varchar, varchar);
DROP FUNCTION IF EXISTS list_entity_aliases();

-- ============================================================================
-- SCHEMA CHANGES
-- ============================================================================

-- Add original_entity_id to track the source entity for rollback
-- Note: No FK constraint because the original entity is deleted during merge
-- We just store the ID for reference/logging purposes
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'entity_aliases' AND column_name = 'original_entity_id'
    ) THEN
        ALTER TABLE entity_aliases ADD COLUMN original_entity_id INTEGER;
    END IF;
END $$;

-- Drop FK constraint if it exists (from previous version)
DO $$
BEGIN
    ALTER TABLE entity_aliases DROP CONSTRAINT IF EXISTS entity_aliases_original_entity_id_fkey;
EXCEPTION
    WHEN undefined_object THEN NULL;
END $$;

-- Comment
COMMENT ON COLUMN entity_aliases.original_entity_id IS 'Original entity ID that was merged (stored for reference, entity is deleted)';


-- ============================================================================
-- UPDATED MERGE FUNCTION
-- ============================================================================

-- Function to merge a source entity into a target entity (v2)
-- Now handles entity_passage_connections in addition to entity_passages
CREATE OR REPLACE FUNCTION merge_entity_into(
    p_source_id INTEGER,
    p_target_id INTEGER
) RETURNS TABLE(
    facts_updated INTEGER,
    cooccurrences_updated INTEGER,
    passages_updated INTEGER,
    passage_connections_updated INTEGER
) AS $$
DECLARE
    v_facts_updated INTEGER := 0;
    v_cooc_updated INTEGER := 0;
    v_passages_updated INTEGER := 0;
    v_passage_conns_updated INTEGER := 0;
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
    FOR r IN
        SELECT
            ec_src.entity_a_id AS src_a,
            ec_src.entity_b_id AS src_b,
            ec_src.count AS src_count,
            ec_src.weight AS src_weight,
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
                -- Aggregate: sum counts, recalculate weight
                v_new_count := v_existing_count + r.src_count;
                v_new_weight := 1.0 / v_new_count;

                UPDATE entity_cooccurrences
                SET count = v_new_count, weight = v_new_weight, updated_at = NOW()
                WHERE entity_a_id = v_target_a AND entity_b_id = v_target_b;

                DELETE FROM entity_cooccurrences
                WHERE entity_a_id = r.src_a AND entity_b_id = r.src_b;
            ELSE
                UPDATE entity_cooccurrences
                SET entity_a_id = v_target_a, entity_b_id = v_target_b, updated_at = NOW()
                WHERE entity_a_id = r.src_a AND entity_b_id = r.src_b;
            END IF;

            v_cooc_updated := v_cooc_updated + 1;
        END;
    END LOOP;

    -- Delete any self-references
    DELETE FROM entity_cooccurrences WHERE entity_a_id = entity_b_id;

    -- entity_passages: Delete duplicates first, then update
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

    -- NEW: entity_passage_connections: Same logic - delete duplicates, then merge
    DELETE FROM entity_passage_connections epc
    WHERE epc.entity_id = p_source_id
    AND EXISTS (
        SELECT 1 FROM entity_passage_connections epc2
        WHERE epc2.entity_id = p_target_id AND epc2.passage_id = epc.passage_id
    );

    UPDATE entity_passage_connections SET entity_id = p_target_id
    WHERE entity_id = p_source_id;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    v_passage_conns_updated := v_passage_conns_updated + v_count;

    -- Delete the source entity
    DELETE FROM entities WHERE id = p_source_id;

    RETURN QUERY SELECT v_facts_updated, v_cooc_updated, v_passages_updated, v_passage_conns_updated;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- UPDATED MERGE ALIAS FUNCTION
-- ============================================================================

-- Main procedure: Add alias and merge matching entity (v2)
-- Now records original_entity_id for split support
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
    passages_updated INTEGER,
    passage_connections_updated INTEGER
) AS $$
DECLARE
    v_canonical_id INTEGER;
    v_source_id INTEGER;
    v_facts INTEGER := 0;
    v_cooc INTEGER := 0;
    v_passages INTEGER := 0;
    v_passage_conns INTEGER := 0;
    v_result RECORD;
BEGIN
    -- Find canonical entity
    SELECT id INTO v_canonical_id FROM entities
    WHERE LOWER(name) = LOWER(p_canonical_name);

    IF v_canonical_id IS NULL THEN
        RETURN QUERY SELECT
            'error: canonical entity not found'::TEXT,
            NULL::INTEGER, NULL::INTEGER, 0, 0, 0, 0;
        RETURN;
    END IF;

    -- Check if alias already exists
    IF EXISTS (SELECT 1 FROM entity_aliases WHERE LOWER(alias_name) = LOWER(p_alias_name)) THEN
        RETURN QUERY SELECT
            'error: alias already exists'::TEXT,
            v_canonical_id, NULL::INTEGER, 0, 0, 0, 0;
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
        v_facts := v_result.facts_updated;
        v_cooc := v_result.cooccurrences_updated;
        v_passages := v_result.passages_updated;
        v_passage_conns := v_result.passage_connections_updated;

        RETURN QUERY SELECT
            'merged'::TEXT,
            v_canonical_id, v_source_id, v_facts, v_cooc, v_passages, v_passage_conns;
    ELSE
        RETURN QUERY SELECT
            'alias_added'::TEXT,
            v_canonical_id, NULL::INTEGER, 0, 0, 0, 0;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SPLIT (UNDO MERGE) FUNCTION
-- ============================================================================

-- Split an entity back from its canonical form
-- Recreates the original entity and restores edges based on saved data
-- Note: This is a best-effort restoration; some data may not be fully recoverable
CREATE OR REPLACE FUNCTION split_entity_alias(
    p_alias_name VARCHAR(512)
) RETURNS TABLE(
    status TEXT,
    new_entity_id INTEGER,
    canonical_id INTEGER,
    message TEXT
) AS $$
DECLARE
    v_alias_record RECORD;
    v_canonical_id INTEGER;
    v_new_entity_id INTEGER;
    v_original_entity_id INTEGER;
BEGIN
    -- Find the alias record
    SELECT ea.*, e.name as canonical_name
    INTO v_alias_record
    FROM entity_aliases ea
    JOIN entities e ON ea.canonical_entity_id = e.id
    WHERE LOWER(ea.alias_name) = LOWER(p_alias_name);

    IF v_alias_record IS NULL THEN
        RETURN QUERY SELECT
            'error: alias not found'::TEXT,
            NULL::INTEGER,
            NULL::INTEGER,
            'Alias does not exist'::TEXT;
        RETURN;
    END IF;

    v_canonical_id := v_alias_record.canonical_entity_id;
    v_original_entity_id := v_alias_record.original_entity_id;

    -- Check if original_entity_id was recorded
    IF v_original_entity_id IS NULL THEN
        RETURN QUERY SELECT
            'error: cannot split'::TEXT,
            NULL::INTEGER,
            v_canonical_id,
            'No original entity ID recorded. This alias was added without an existing entity to merge.'::TEXT;
        RETURN;
    END IF;

    -- Check if entity with same name already exists
    IF EXISTS (SELECT 1 FROM entities WHERE LOWER(name) = LOWER(p_alias_name)) THEN
        RETURN QUERY SELECT
            'error: entity exists'::TEXT,
            NULL::INTEGER,
            v_canonical_id,
            format('Entity with name "%s" already exists', p_alias_name)::TEXT;
        RETURN;
    END IF;

    -- Create a new entity with the alias name
    INSERT INTO entities (name)
    VALUES (p_alias_name)
    RETURNING id INTO v_new_entity_id;

    -- Remove the alias record
    DELETE FROM entity_aliases WHERE id = v_alias_record.id;

    RETURN QUERY SELECT
        'split_complete'::TEXT,
        v_new_entity_id,
        v_canonical_id,
        format('Entity "%s" created with ID %s. Note: Edge data was not restored; re-indexing recommended.', p_alias_name, v_new_entity_id)::TEXT;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- BATCH MERGE FUNCTION
-- ============================================================================

-- Batch merge multiple aliases to a canonical entity
-- Useful for processing alias lists from JSON
CREATE OR REPLACE FUNCTION batch_merge_aliases(
    p_canonical_name VARCHAR(512),
    p_alias_names TEXT[],
    p_created_by VARCHAR(64) DEFAULT 'user'
) RETURNS TABLE(
    alias_name TEXT,
    status TEXT,
    source_id INTEGER
) AS $$
DECLARE
    v_alias TEXT;
    v_result RECORD;
BEGIN
    FOREACH v_alias IN ARRAY p_alias_names
    LOOP
        SELECT * INTO v_result FROM merge_entity_alias(v_alias, p_canonical_name, p_created_by);
        RETURN QUERY SELECT v_alias, v_result.status, v_result.source_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- UPDATED LIST FUNCTION
-- ============================================================================

-- List all aliases with original_entity_id (for split support visibility)
CREATE OR REPLACE FUNCTION list_entity_aliases()
RETURNS TABLE(
    canonical_name TEXT,
    alias_name TEXT,
    original_entity_id INTEGER,
    created_by TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT e.name::TEXT, ea.alias_name::TEXT, ea.original_entity_id, ea.created_by::TEXT, ea.created_at
    FROM entity_aliases ea
    JOIN entities e ON ea.canonical_entity_id = e.id
    ORDER BY e.name, ea.alias_name;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON FUNCTION merge_entity_into IS 'Merge source entity into target. Now includes entity_passage_connections.';
COMMENT ON FUNCTION merge_entity_alias IS 'Add alias and merge matching entity. Records original_entity_id for split support.';
COMMENT ON FUNCTION split_entity_alias IS 'Undo a merge by recreating the original entity. Best-effort restoration.';
COMMENT ON FUNCTION batch_merge_aliases IS 'Batch merge multiple aliases to a canonical entity.';
COMMENT ON FUNCTION list_entity_aliases IS 'List all aliases with original_entity_id for split visibility.';
