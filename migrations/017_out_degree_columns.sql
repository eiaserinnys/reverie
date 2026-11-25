-- Migration: out_degree columns for Lazy PPR
-- Adds out_degree column to entities and passages tables for PPR normalization

-- 1. Add out_degree column to entities table
-- out_degree = number of passages this entity appears in
ALTER TABLE entities ADD COLUMN IF NOT EXISTS out_degree INT DEFAULT 0;

-- 2. Add out_degree column to passages table
-- out_degree = number of entity connections + number of adjacent passages (bidirectional)
ALTER TABLE passages ADD COLUMN IF NOT EXISTS out_degree INT DEFAULT 0;

-- 3. Create index for efficient out_degree lookups
CREATE INDEX IF NOT EXISTS idx_entities_out_degree ON entities(out_degree) WHERE out_degree > 0;
CREATE INDEX IF NOT EXISTS idx_passages_out_degree ON passages(out_degree) WHERE out_degree > 0;

-- 4. Calculate out_degree for existing entities
-- entity.out_degree = COUNT(*) from entity_passages where entity_id = entity.id
UPDATE entities e SET out_degree = (
    SELECT COUNT(*) FROM entity_passages ep WHERE ep.entity_id = e.id
);

-- 5. Calculate out_degree for existing passages
-- passage.out_degree = entity connections + bidirectional adjacency connections
UPDATE passages p SET out_degree = (
    -- Number of entities connected to this passage
    SELECT COUNT(*) FROM entity_passages ep WHERE ep.passage_id = p.id
) + (
    -- Number of adjacent passages (bidirectional: both a→b and b→a count)
    SELECT COUNT(*) FROM passage_adjacency a
    WHERE a.passage_id_a = p.id OR a.passage_id_b = p.id
);
