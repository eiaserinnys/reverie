-- Migration: Incremental Indexing Infrastructure
-- Adds profile support and session indexing state tracking

-- 2.1 passages 테이블에 profile_id, session_seq 컬럼 추가
ALTER TABLE passages ADD COLUMN IF NOT EXISTS profile_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE passages ADD COLUMN IF NOT EXISTS session_seq INTEGER;

-- 2.2 entities 테이블에 profile_id 컬럼 추가
ALTER TABLE entities ADD COLUMN IF NOT EXISTS profile_id TEXT NOT NULL DEFAULT 'default';

-- 2.3 Composite indexes for profile-based queries
CREATE INDEX IF NOT EXISTS idx_passages_profile ON passages(profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_passages_profile_seq ON passages(profile_id, session_seq);
CREATE INDEX IF NOT EXISTS idx_entities_profile ON entities(profile_id, name);

-- 3.1 session_index_state 테이블: 세션별 인덱싱 상태 추적
CREATE TABLE IF NOT EXISTS session_index_state (
    id SERIAL PRIMARY KEY,
    profile_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    session_seq INTEGER NOT NULL,
    file_path TEXT NOT NULL,

    -- 파일 상태 추적
    file_mtime TIMESTAMPTZ NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_lines_count INTEGER NOT NULL DEFAULT 0,

    -- 인덱싱 진행 상태
    last_indexed_turn INTEGER DEFAULT -1,
    last_indexed_line INTEGER DEFAULT 0,
    last_passage_id INTEGER,
    total_passages_indexed INTEGER DEFAULT 0,
    total_passages_dropped INTEGER DEFAULT 0,

    -- 상태: 'pending', 'in_progress', 'completed', 'failed'
    status TEXT DEFAULT 'pending',
    error_message TEXT,

    -- 타임스탬프
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ,

    UNIQUE(profile_id, session_id)
);

-- session_index_state 인덱스
CREATE INDEX IF NOT EXISTS idx_session_index_state_profile ON session_index_state(profile_id);
CREATE INDEX IF NOT EXISTS idx_session_index_state_status ON session_index_state(profile_id, status);

-- 3.2 passage_processing_history 테이블: 드롭 사유 등 이력 기록
CREATE TABLE IF NOT EXISTS passage_processing_history (
    id SERIAL PRIMARY KEY,
    profile_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,

    -- 처리 결과: 'indexed', 'dropped', 'skipped', 'failed'
    action TEXT NOT NULL,
    drop_reason TEXT,

    -- 메타데이터
    raw_text TEXT,
    preprocessed_text TEXT,
    entities_extracted TEXT[],

    -- 연결
    passage_id INTEGER REFERENCES passages(id) ON DELETE SET NULL,

    -- 타임스탬프
    processed_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(profile_id, session_id, turn_index)
);

-- passage_processing_history 인덱스
CREATE INDEX IF NOT EXISTS idx_passage_history_profile ON passage_processing_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_passage_history_session ON passage_processing_history(profile_id, session_id);
CREATE INDEX IF NOT EXISTS idx_passage_history_action ON passage_processing_history(profile_id, action);

-- 3.3 get_next_session_seq() 함수: 프로필 내 다음 세션 시퀀스 반환
CREATE OR REPLACE FUNCTION get_next_session_seq(p_profile_id TEXT)
RETURNS INTEGER AS $$
DECLARE
    next_seq INTEGER;
BEGIN
    SELECT COALESCE(MAX(session_seq), 0) + 1 INTO next_seq
    FROM session_index_state
    WHERE profile_id = p_profile_id;
    RETURN next_seq;
END;
$$ LANGUAGE plpgsql;

-- updated_at 트리거 함수 (재사용)
CREATE OR REPLACE FUNCTION update_session_index_state_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- updated_at 자동 갱신 트리거
DROP TRIGGER IF EXISTS trigger_session_index_state_updated_at ON session_index_state;
CREATE TRIGGER trigger_session_index_state_updated_at
    BEFORE UPDATE ON session_index_state
    FOR EACH ROW
    EXECUTE FUNCTION update_session_index_state_updated_at();
