-- Fix deadlock in upsert_entity by adding advisory lock
-- When multiple workers upsert the same entity concurrently,
-- INSERT ... ON CONFLICT causes deadlocks due to conflicting row locks.
-- pg_advisory_xact_lock serializes access per entity name within a transaction.

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
    -- Advisory lock on entity name to prevent deadlocks
    -- Uses two-int form: namespace (constant) + hash of lowercased name
    PERFORM pg_advisory_xact_lock(
        hashtext('upsert_entity'),
        hashtext(LOWER(p_name))
    );

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
