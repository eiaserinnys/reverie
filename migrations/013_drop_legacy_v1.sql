-- MemoryLink v2: Drop legacy v1 tables, functions, and columns
--
-- This migration removes all v1-era components that are no longer used:
-- - Fact-based triple extraction tables (facts, passage_facts)
-- - Entity co-occurrence tables (entity_cooccurrences)
-- - v1 unified graph tables (passage_connections, entity_passage_connections)
-- - Legacy functions for co-occurrence, decay weight, and pagerank
--
-- v2 uses:
-- - entities, passages (core nodes)
-- - entity_passages (contain edges)
-- - passage_adjacency (temporal edges)

-- ============================================================================
-- PHASE 1: DROP LEGACY TABLES
-- ============================================================================

-- Drop passage_facts first (depends on facts)
DROP TABLE IF EXISTS passage_facts CASCADE;

-- Drop facts table (LLM triple extraction - removed in v2)
DROP TABLE IF EXISTS facts CASCADE;

-- Drop entity_cooccurrences (v2 uses passage as implicit connection)
DROP TABLE IF EXISTS entity_cooccurrences CASCADE;

-- Drop entity_passage_connections (v2 uses entity_passages instead)
DROP TABLE IF EXISTS entity_passage_connections CASCADE;

-- Drop passage_connections (v2 uses passage_adjacency instead)
DROP TABLE IF EXISTS passage_connections CASCADE;

-- ============================================================================
-- PHASE 2: DROP LEGACY FUNCTIONS
-- ============================================================================

-- Co-occurrence functions (no longer needed - v2 has no entity↔entity edges)
DROP FUNCTION IF EXISTS upsert_cooccurrence(INTEGER, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS decrement_cooccurrence(INTEGER, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS upsert_cooccurrence_with_decay(INTEGER, INTEGER, INTEGER) CASCADE;

-- Decay weight function (v2 uses weight=1 for all edges)
DROP FUNCTION IF EXISTS calculate_decay_weight(INTEGER) CASCADE;

-- v1 unified graph functions
DROP FUNCTION IF EXISTS upsert_passage_connection(INTEGER, INTEGER, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS upsert_entity_passage_connection(INTEGER, INTEGER, INTEGER) CASCADE;

-- PageRank functions (v2 uses BipartitePPR instead)
DROP FUNCTION IF EXISTS calculate_unified_pagerank(REAL, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS update_unified_importance_local(INTEGER[], INTEGER[], REAL, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS get_top_nodes(INTEGER) CASCADE;

-- ============================================================================
-- PHASE 3: DROP LEGACY INDEXES (if tables were already dropped, these are gone)
-- ============================================================================

-- Co-occurrence indexes
DROP INDEX IF EXISTS idx_cooccurrences_entity_a;
DROP INDEX IF EXISTS idx_cooccurrences_entity_b;
DROP INDEX IF EXISTS idx_cooccurrences_weight;

-- Entity-passage-connections indexes
DROP INDEX IF EXISTS idx_entity_passage_entity;
DROP INDEX IF EXISTS idx_entity_passage_passage;
DROP INDEX IF EXISTS idx_entity_passage_decay;

-- Passage-connections indexes
DROP INDEX IF EXISTS idx_passage_connections_a;
DROP INDEX IF EXISTS idx_passage_connections_b;
DROP INDEX IF EXISTS idx_passage_connections_weight;
DROP INDEX IF EXISTS idx_passage_connections_decay;

-- Facts indexes
DROP INDEX IF EXISTS idx_facts_subject;
DROP INDEX IF EXISTS idx_facts_object;
DROP INDEX IF EXISTS idx_facts_embedding;

-- ============================================================================
-- PHASE 4: DROP UNUSED COLUMNS
-- ============================================================================

-- Remove decay_score from passages (v2 doesn't use decay)
ALTER TABLE passages DROP COLUMN IF EXISTS decay_score;

-- Note: importance column is kept for potential future use in v2
-- (e.g., entity importance for seed selection)

-- ============================================================================
-- VERIFICATION QUERIES (for manual inspection)
-- ============================================================================

-- Run these manually to verify cleanup:
--
-- SELECT tablename FROM pg_tables
-- WHERE schemaname = 'public'
-- AND tablename IN ('facts', 'passage_facts', 'entity_cooccurrences',
--                   'entity_passage_connections', 'passage_connections');
--
-- SELECT routine_name FROM information_schema.routines
-- WHERE routine_schema = 'public'
-- AND routine_name IN ('upsert_cooccurrence', 'decrement_cooccurrence',
--                      'calculate_decay_weight', 'upsert_passage_connection',
--                      'upsert_entity_passage_connection', 'calculate_unified_pagerank',
--                      'update_unified_importance_local', 'get_top_nodes');

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE entities IS 'Entity nodes for MemoryLink v2 bipartite graph';
COMMENT ON TABLE passages IS 'Passage nodes for MemoryLink v2 bipartite graph';
COMMENT ON TABLE entity_passages IS 'Entity↔Passage contain edges (v2)';
COMMENT ON TABLE passage_adjacency IS 'Passage↔Passage temporal adjacency edges (v2)';
