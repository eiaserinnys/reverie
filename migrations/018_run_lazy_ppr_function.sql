-- Migration: Server-side Lazy PPR function
-- Moves PPR computation entirely into PostgreSQL to eliminate network round-trips
-- Related: artifacts/ppr-optimization-plan.md

-- Drop existing function if exists (for re-run safety)
DROP FUNCTION IF EXISTS run_lazy_ppr(int[], float[], float, int, float, float);

-- Create the server-side PPR function
CREATE OR REPLACE FUNCTION run_lazy_ppr(
    seed_entity_ids int[],
    seed_scores float[],
    damping float DEFAULT 0.85,
    max_iter int DEFAULT 10,
    min_score float DEFAULT 0.0001,
    tol float DEFAULT 0.000001
)
RETURNS TABLE(node_type char(1), node_id int, node_score float)
LANGUAGE plpgsql AS $$
DECLARE
    iter int := 0;
    max_diff float := 0.0;
    arr_len int;
BEGIN
    -- Validate input arrays have same length
    arr_len := array_length(seed_entity_ids, 1);
    IF arr_len IS NULL OR arr_len = 0 THEN
        RETURN;
    END IF;

    IF array_length(seed_scores, 1) != arr_len THEN
        RAISE EXCEPTION 'seed_entity_ids and seed_scores must have same length';
    END IF;

    -- Create temporary tables for PPR scores
    -- Using TEMPORARY tables that are automatically dropped at end of transaction
    CREATE TEMP TABLE IF NOT EXISTS _ppr_scores (
        ntype char(1) NOT NULL,
        nid int NOT NULL,
        score float NOT NULL,
        PRIMARY KEY (ntype, nid)
    ) ON COMMIT DROP;

    CREATE TEMP TABLE IF NOT EXISTS _ppr_scores_new (
        ntype char(1) NOT NULL,
        nid int NOT NULL,
        score float NOT NULL
    ) ON COMMIT DROP;

    -- Clear any existing data (in case of re-use within same transaction)
    TRUNCATE _ppr_scores;
    TRUNCATE _ppr_scores_new;

    -- Initialize with seed entities (scores already normalized by caller)
    INSERT INTO _ppr_scores (ntype, nid, score)
    SELECT 'e', unnest(seed_entity_ids), unnest(seed_scores);

    -- PPR iteration loop
    WHILE iter < max_iter LOOP
        -- Clear new scores table
        TRUNCATE _ppr_scores_new;

        -- 1. e2p: Entity → Passage propagation
        -- Score flows from entities to their connected passages
        INSERT INTO _ppr_scores_new (ntype, nid, score)
        SELECT 'p', ep.passage_id, SUM(ps.score / e.out_degree)
        FROM _ppr_scores ps
        JOIN entity_passages ep ON ep.entity_id = ps.nid AND ps.ntype = 'e'
        JOIN entities e ON e.id = ep.entity_id
        WHERE ps.score >= min_score AND e.out_degree > 0
        GROUP BY ep.passage_id;

        -- 2. p2e: Passage → Entity propagation
        -- Score flows from passages to their connected entities
        INSERT INTO _ppr_scores_new (ntype, nid, score)
        SELECT 'e', ep.entity_id, SUM(ps.score / p.out_degree)
        FROM _ppr_scores ps
        JOIN entity_passages ep ON ep.passage_id = ps.nid AND ps.ntype = 'p'
        JOIN passages p ON p.id = ep.passage_id
        WHERE ps.score >= min_score AND p.out_degree > 0
        GROUP BY ep.entity_id;

        -- 3. p2p: Passage → Passage propagation (bidirectional)
        -- Score flows between adjacent passages
        INSERT INTO _ppr_scores_new (ntype, nid, score)
        SELECT 'p', adj.target_id, SUM(ps.score / p.out_degree)
        FROM _ppr_scores ps
        JOIN (
            -- Both directions of the bidirectional adjacency
            SELECT passage_id_a AS source_id, passage_id_b AS target_id FROM passage_adjacency
            UNION ALL
            SELECT passage_id_b AS source_id, passage_id_a AS target_id FROM passage_adjacency
        ) adj ON adj.source_id = ps.nid AND ps.ntype = 'p'
        JOIN passages p ON p.id = adj.source_id
        WHERE ps.score >= min_score AND p.out_degree > 0
        GROUP BY adj.target_id;

        -- 4. Aggregate propagated scores and apply teleport
        -- new_score = (1-d) * seed_score + d * propagated_score
        WITH aggregated AS (
            SELECT ntype, nid, SUM(score) as propagated
            FROM _ppr_scores_new
            GROUP BY ntype, nid
        ),
        seeds AS (
            SELECT 'e'::char(1) as ntype,
                   unnest(seed_entity_ids) as nid,
                   unnest(seed_scores) as seed_score
        ),
        combined AS (
            SELECT
                COALESCE(a.ntype, s.ntype) as ntype,
                COALESCE(a.nid, s.nid) as nid,
                (1 - damping) * COALESCE(s.seed_score, 0) + damping * COALESCE(a.propagated, 0) as new_score
            FROM aggregated a
            FULL OUTER JOIN seeds s ON a.ntype = s.ntype AND a.nid = s.nid
        )
        -- Update scores using UPSERT
        INSERT INTO _ppr_scores (ntype, nid, score)
        SELECT ntype, nid, new_score FROM combined
        WHERE new_score >= min_score
        ON CONFLICT (ntype, nid) DO UPDATE SET score = EXCLUDED.score;

        -- Remove nodes that fell below min_score threshold
        DELETE FROM _ppr_scores WHERE score < min_score;

        iter := iter + 1;
    END LOOP;

    -- Return final scores
    RETURN QUERY
    SELECT ps.ntype, ps.nid, ps.score AS node_score
    FROM _ppr_scores ps
    WHERE ps.score >= min_score
    ORDER BY ps.score DESC;
END;
$$;

-- Add comment for documentation
COMMENT ON FUNCTION run_lazy_ppr(int[], float[], float, int, float, float) IS
'Server-side Lazy Personalized PageRank computation.
Eliminates network round-trips by running all PPR iterations in PostgreSQL.

Parameters:
- seed_entity_ids: Array of entity IDs to start PPR from
- seed_scores: Array of normalized scores for seed entities (must sum to 1)
- damping: PPR damping factor (default 0.85)
- max_iter: Maximum iterations (default 10)
- min_score: Minimum score threshold for pruning (default 0.0001)
- tol: Convergence tolerance (currently unused, for future extension)

Returns table of (node_type, node_id, node_score) where:
- node_type: ''e'' for entity, ''p'' for passage
- node_id: ID of the entity or passage
- node_score: Final PPR score

Example:
  SELECT * FROM run_lazy_ppr(
    ARRAY[1, 2, 3]::int[],
    ARRAY[0.5, 0.3, 0.2]::float[],
    0.85, 10, 0.0001, 0.000001
  );
';
