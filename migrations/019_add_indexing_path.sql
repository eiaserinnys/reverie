-- Migration: Add indexing_path column for multi-folder indexing support
-- Allows tracking indexing state per folder within a profile

-- 1. Add indexing_path column to session_index_state
ALTER TABLE session_index_state
ADD COLUMN IF NOT EXISTS indexing_path TEXT NOT NULL DEFAULT '';

-- 2. Add indexing_path column to passage_processing_history
ALTER TABLE passage_processing_history
ADD COLUMN IF NOT EXISTS indexing_path TEXT NOT NULL DEFAULT '';

-- 3. Drop existing unique constraint
ALTER TABLE session_index_state
DROP CONSTRAINT IF EXISTS session_index_state_profile_id_session_id_key;

-- 4. Create new unique constraint including indexing_path
ALTER TABLE session_index_state
ADD CONSTRAINT session_index_state_profile_session_path_key
UNIQUE (profile_id, session_id, indexing_path);

-- 5. Drop existing unique constraint on passage_processing_history
ALTER TABLE passage_processing_history
DROP CONSTRAINT IF EXISTS passage_processing_history_profile_id_session_id_turn_index_key;

-- 6. Create new unique constraint including indexing_path
ALTER TABLE passage_processing_history
ADD CONSTRAINT passage_processing_history_profile_session_turn_path_key
UNIQUE (profile_id, session_id, turn_index, indexing_path);

-- 7. Create index for efficient indexing_path queries
CREATE INDEX IF NOT EXISTS idx_session_index_state_path
ON session_index_state(profile_id, indexing_path);

-- 8. Create index for passage_processing_history
CREATE INDEX IF NOT EXISTS idx_passage_history_path
ON passage_processing_history(profile_id, indexing_path);
