-- MemoryLink 공출현 그래프 스키마
-- Reverie v2: Co-occurrence based retrieval

-- 엔티티 공출현 테이블
-- 같은 passage에 함께 등장한 엔티티 쌍의 관계
CREATE TABLE IF NOT EXISTS entity_cooccurrences (
    entity_a_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1,
    weight REAL NOT NULL DEFAULT 1.0,  -- 1.0 / count (거리)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_a_id, entity_b_id),
    -- 정규화: 항상 작은 ID가 앞 (무방향 엣지)
    CHECK (entity_a_id < entity_b_id)
);

-- 엔티티-패시지 연결 테이블
-- 엔티티가 어떤 passage에 등장했는지 추적
CREATE TABLE IF NOT EXISTS entity_passages (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, passage_id)
);

-- 인덱스: 공출현 조회 최적화
CREATE INDEX IF NOT EXISTS idx_cooccurrences_entity_a
    ON entity_cooccurrences(entity_a_id);
CREATE INDEX IF NOT EXISTS idx_cooccurrences_entity_b
    ON entity_cooccurrences(entity_b_id);
CREATE INDEX IF NOT EXISTS idx_cooccurrences_weight
    ON entity_cooccurrences(weight);

-- 인덱스: 엔티티-패시지 조회 최적화
CREATE INDEX IF NOT EXISTS idx_entity_passages_entity
    ON entity_passages(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_passages_passage
    ON entity_passages(passage_id);

-- 공출현 UPSERT 함수
-- 새로운 공출현 추가 또는 기존 카운트 증가
CREATE OR REPLACE FUNCTION upsert_cooccurrence(
    p_entity_a_id INTEGER,
    p_entity_b_id INTEGER
) RETURNS void AS $$
DECLARE
    v_a INTEGER;
    v_b INTEGER;
BEGIN
    -- 정규화: 작은 ID가 항상 앞
    IF p_entity_a_id < p_entity_b_id THEN
        v_a := p_entity_a_id;
        v_b := p_entity_b_id;
    ELSE
        v_a := p_entity_b_id;
        v_b := p_entity_a_id;
    END IF;

    INSERT INTO entity_cooccurrences (entity_a_id, entity_b_id, count, weight)
    VALUES (v_a, v_b, 1, 1.0)
    ON CONFLICT (entity_a_id, entity_b_id) DO UPDATE
    SET count = entity_cooccurrences.count + 1,
        weight = 1.0 / (entity_cooccurrences.count + 1),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- 공출현 감소 함수 (passage 삭제 시 사용)
CREATE OR REPLACE FUNCTION decrement_cooccurrence(
    p_entity_a_id INTEGER,
    p_entity_b_id INTEGER
) RETURNS void AS $$
DECLARE
    v_a INTEGER;
    v_b INTEGER;
    v_new_count INTEGER;
BEGIN
    -- 정규화
    IF p_entity_a_id < p_entity_b_id THEN
        v_a := p_entity_a_id;
        v_b := p_entity_b_id;
    ELSE
        v_a := p_entity_b_id;
        v_b := p_entity_a_id;
    END IF;

    -- 카운트 감소
    UPDATE entity_cooccurrences
    SET count = count - 1,
        weight = CASE WHEN count - 1 > 0 THEN 1.0 / (count - 1) ELSE NULL END,
        updated_at = NOW()
    WHERE entity_a_id = v_a AND entity_b_id = v_b
    RETURNING count INTO v_new_count;

    -- 카운트가 0이 되면 삭제
    IF v_new_count IS NOT NULL AND v_new_count <= 0 THEN
        DELETE FROM entity_cooccurrences
        WHERE entity_a_id = v_a AND entity_b_id = v_b;
    END IF;
END;
$$ LANGUAGE plpgsql;
