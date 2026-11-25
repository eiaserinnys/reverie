"""Tests for MemoryLinkV2Pipeline (Fully Lazy).

Tests for the integrated retrieval pipeline using:
- LazySeedSelector for Progressive Widening seed selection
- LazyBipartitePPR for DB-backed PPR iteration
"""

import math
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from reverie.retriever.pipeline_v2 import (
    MemoryLinkV2Pipeline,
    calculate_time_decay,
)


class MockPassageRow:
    """Mock for PassageRow."""
    def __init__(
        self,
        id: int,
        session_id: str,
        turn_index: int,
        text: str,
        embedding=None,
        session_seq: int = None,
        created_at: datetime = None,
    ):
        self.id = id
        self.session_id = session_id
        self.turn_index = turn_index
        self.text = text
        self.embedding = embedding
        self.metadata = {}
        self.created_at = created_at
        self.session_seq = session_seq


class MockSeedSelectionResult:
    """Mock for SeedSelectionResult."""
    def __init__(self, seeds: dict[int, float]):
        self.seeds = seeds
        self.query_count = 1
        self.total_candidates = len(seeds)
        self.timing_details = {}


class MockPPRResult:
    """Mock for LazyPPRResult."""
    def __init__(self, passage_scores: dict[int, float]):
        self.entity_scores = {}
        self.passage_scores = passage_scores
        self.iterations = 5
        self.converged = True
        self.timing_details = MagicMock(
            total_ms=100.0,
            iterations=5,
            db_query_ms=80.0,
            propagation_ms=20.0,
            per_iter_avg_ms=20.0,
            per_iter_timings=[(16.0, 4.0)] * 5,  # 5 iterations, each (db_ms, prop_ms)
        )


class TestMemoryLinkV2Pipeline:
    """Test MemoryLinkV2Pipeline (Fully Lazy)."""

    @pytest.fixture
    def mock_db(self):
        """Create mock ReverieDB instance."""
        db = MagicMock()
        # Default: return passages by ID
        db.get_passages_by_ids.side_effect = lambda pids: [
            MockPassageRow(pid, "session-1", pid - 10, f"Text for passage {pid}")
            for pid in pids
        ]
        db.search_passages_dual_embedding.return_value = [
            MagicMock(passage_id=10, combined_score=0.9),
            MagicMock(passage_id=11, combined_score=0.8),
        ]
        return db

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = MagicMock()
        # Return consistent embedding for testing
        client.embed_text.return_value = [0.9] + [0.1] * 1535
        return client

    def test_retrieve_knn_fallback(self, mock_db, mock_embedding_client):
        """Should fallback to KNN when no seeds are found."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            # Configure seed selector - no seeds found
            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,  # Skip NER for simplicity
            )

            results = pipeline.retrieve("test query", top_k=2)

            # Should return KNN results
            assert len(results) == 2
            for result in results:
                assert hasattr(result, 'passage_id')
                assert hasattr(result, 'text')
                assert hasattr(result, 'score')
                assert hasattr(result, 'source')
                assert result.source == "knn"  # No seeds = KNN fallback

    def test_retrieve_with_ppr(self, mock_db, mock_embedding_client):
        """Should use PPR when seeds are found."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            # Configure seed selector - seeds found
            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({
                1: 0.9,  # entity_id 1 with score 0.9
                2: 0.8,  # entity_id 2 with score 0.8
            })
            MockSeedSelector.return_value = mock_seed_selector

            # Configure PPR
            mock_ppr = MagicMock()
            mock_ppr.run_full.return_value = MockPPRResult({10: 0.9, 11: 0.7})
            MockPPR.return_value = mock_ppr

            # Configure mock_db
            mock_db.search_passages_dual_embedding.return_value = [
                MagicMock(passage_id=10, combined_score=0.8),
            ]

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            results = pipeline.retrieve("test query", top_k=2)

            # Should use memorylink-v2 source
            assert len(results) > 0
            assert any(r.source == "memorylink-v2" for r in results)

    def test_hybrid_merge(self, mock_db, mock_embedding_client):
        """Should merge PPR and KNN results."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
            beta=0.5,  # 50% PPR, 50% KNN
        )

        ppr_results = [(10, 1.0), (11, 0.5)]
        knn_results = [(10, 0.8), (12, 0.9)]

        merged = pipeline._hybrid_merge(ppr_results, knn_results, 0.5)

        # Should combine results
        passage_ids = [r[0] for r in merged]
        assert 10 in passage_ids  # In both
        assert 11 in passage_ids  # PPR only
        assert 12 in passage_ids  # KNN only

        # Passage 10 should have combined score
        scores = {r[0]: r[1] for r in merged}
        assert scores[10] > 0

    def test_hybrid_merge_ppr_heavy(self, mock_db, mock_embedding_client):
        """Should weight PPR heavily when beta is high."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
            beta=0.9,  # 90% PPR
        )

        ppr_results = [(10, 1.0)]
        knn_results = [(11, 1.0)]

        merged = pipeline._hybrid_merge(ppr_results, knn_results, 0.9)

        scores = {r[0]: r[1] for r in merged}
        # PPR result should have higher score
        assert scores[10] > scores[11]

    def test_hybrid_merge_knn_fallback(self, mock_db, mock_embedding_client):
        """Should fallback to KNN when no PPR results."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )

        ppr_results = []
        knn_results = [(10, 0.9), (11, 0.8)]

        merged = pipeline._hybrid_merge(ppr_results, knn_results, 0.8)

        # Should return KNN results only
        assert len(merged) == 2
        assert merged[0][0] == 10
        assert merged[1][0] == 11

    def test_invalidate_graph(self, mock_db, mock_embedding_client):
        """Should clear cached state."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )

        # Set some cached values
        pipeline._last_entity_seeds = {1: 0.9}
        pipeline._last_query_entities = ["test"]

        pipeline.invalidate_graph()

        assert pipeline._last_entity_seeds is None
        assert pipeline._last_query_entities is None

    def test_hyperparameters(self, mock_db, mock_embedding_client):
        """Should use provided hyperparameters."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
            tau=0.7,
            damping=0.9,
            beta=0.6,
            ppr_max_iter=20,
            ppr_min_score=0.001,
        )

        assert pipeline._tau == 0.7
        assert pipeline._damping == 0.9
        assert pipeline._beta == 0.6
        assert pipeline._ppr_max_iter == 20
        assert pipeline._ppr_min_score == 0.001

    def test_ppr_min_score_per_query(self, mock_db, mock_embedding_client):
        """Should allow overriding ppr_min_score per query."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            # Configure seed selector
            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({1: 0.9})
            MockSeedSelector.return_value = mock_seed_selector

            # Track PPR instantiation
            ppr_instances = []
            def track_ppr(*args, **kwargs):
                instance = MagicMock()
                instance.run.return_value = [(10, 0.9)]
                ppr_instances.append(kwargs)
                return instance
            MockPPR.side_effect = track_ppr

            # Configure mock_db
            mock_db.get_passages_by_ids.return_value = [MockPassageRow(10, "s1", 0, "text")]
            mock_db.search_passages_dual_embedding.return_value = []

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                ppr_min_score=0.0001,  # Default
                use_query_ner=False,
            )

            # Query with custom ppr_min_score
            pipeline.retrieve_with_timing("test", top_k=5, ppr_min_score=0.00001)

            # Should have created PPR with custom min_score
            assert len(ppr_instances) >= 1
            # Last PPR instance should have custom min_score
            assert any(p.get('min_score') == 0.00001 for p in ppr_instances)


class TestMemoryLinkV2PipelineQueryNER:
    """Test MemoryLinkV2Pipeline with Query NER."""

    @pytest.fixture
    def mock_db(self):
        """Create mock ReverieDB instance."""
        db = MagicMock()
        db.get_passages_by_ids.side_effect = lambda pids: [
            MockPassageRow(pid, "session-1", pid - 10, f"Text for passage {pid}")
            for pid in pids
        ]
        db.search_passages_dual_embedding.return_value = []
        return db

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = MagicMock()
        def embed_text(text):
            if "agent" in text.lower():
                return [0.9] + [0.1] * 1535
            elif "memory" in text.lower():
                return [0.1] + [0.9] + [0.0] * 1534
            else:
                return [0.5] + [0.5] + [0.0] * 1534
        client.embed_text.side_effect = embed_text
        return client

    @pytest.fixture
    def mock_query_ner(self):
        """Create mock QueryNERClient."""
        client = MagicMock()
        client.extract_entities.return_value = ["agent-soul", "memory"]
        return client

    def test_retrieve_with_query_ner(
        self, mock_db, mock_embedding_client, mock_query_ner
    ):
        """Should use Query NER for seed entity selection."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({1: 0.9})
            MockSeedSelector.return_value = mock_seed_selector

            mock_ppr = MagicMock()
            mock_ppr.run.return_value = [(10, 0.9)]
            MockPPR.return_value = mock_ppr

            mock_db.get_passages_by_ids.return_value = [MockPassageRow(10, "s1", 0, "text")]

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                query_ner_client=mock_query_ner,
                use_query_ner=True,
            )

            pipeline.retrieve("에이전트 봇의 메모리 시스템은?", top_k=2)

            # Query NER should be called
            mock_query_ner.extract_entities.assert_called_once()

            # Seed selector should be called with query_entity_embeddings
            call_args = mock_seed_selector.select_seeds.call_args
            assert call_args[1].get('query_entity_embeddings') is not None

    def test_retrieve_without_query_ner(self, mock_db, mock_embedding_client):
        """Should skip Query NER when use_query_ner=False."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            mock_db.search_passages_dual_embedding.return_value = []

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            pipeline.retrieve("test query", top_k=2)

            # Seed selector should be called without query_entity_embeddings
            call_args = mock_seed_selector.select_seeds.call_args
            assert call_args[1].get('query_entity_embeddings') is None

    def test_get_query_entities(self, mock_db, mock_embedding_client, mock_query_ner):
        """Should return extracted query entities after retrieval."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            mock_db.search_passages_dual_embedding.return_value = []

            mock_query_ner.extract_entities.return_value = ["entity1", "entity2"]

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                query_ner_client=mock_query_ner,
                use_query_ner=True,
            )

            # Before retrieval
            assert pipeline.get_query_entities() is None

            # Retrieve triggers NER
            pipeline.retrieve("test query", top_k=5)

            # After retrieval
            entities = pipeline.get_query_entities()
            assert entities == ["entity1", "entity2"]


class TestTimeDecay:
    """Test time decay functionality."""

    def test_calculate_time_decay_no_decay(self):
        """Should return 1.0 for current time (no decay)."""
        now = datetime.now(timezone.utc)
        weight = calculate_time_decay(now, half_life_days=365, reference_time=now)
        assert abs(weight - 1.0) < 0.001

    def test_calculate_time_decay_half_life(self):
        """Should return 0.5 after exactly half_life_days."""
        now = datetime.now(timezone.utc)
        one_year_ago = now - timedelta(days=365)
        weight = calculate_time_decay(one_year_ago, half_life_days=365, reference_time=now)
        assert abs(weight - 0.5) < 0.01

    def test_calculate_time_decay_two_half_lives(self):
        """Should return 0.25 after two half-lives."""
        now = datetime.now(timezone.utc)
        two_years_ago = now - timedelta(days=730)
        weight = calculate_time_decay(two_years_ago, half_life_days=365, reference_time=now)
        assert abs(weight - 0.25) < 0.01

    def test_calculate_time_decay_short_half_life(self):
        """Should decay faster with shorter half-life."""
        now = datetime.now(timezone.utc)
        one_week_ago = now - timedelta(days=7)

        weight_365 = calculate_time_decay(one_week_ago, half_life_days=365, reference_time=now)
        weight_7 = calculate_time_decay(one_week_ago, half_life_days=7, reference_time=now)

        # 7-day half-life should decay more (smaller weight)
        assert weight_7 < weight_365
        assert abs(weight_7 - 0.5) < 0.01  # Should be 0.5 after 7 days with 7-day half-life

    def test_calculate_time_decay_future_timestamp(self):
        """Should return 1.0 for future timestamps."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=30)
        weight = calculate_time_decay(future, half_life_days=365, reference_time=now)
        assert weight == 1.0

    def test_calculate_time_decay_naive_datetime(self):
        """Should handle naive datetime (assume UTC)."""
        now = datetime.now(timezone.utc)
        # Create naive datetime 365 days ago
        naive_past = datetime.now() - timedelta(days=365)
        weight = calculate_time_decay(naive_past, half_life_days=365, reference_time=now)
        # Should be approximately 0.5
        assert abs(weight - 0.5) < 0.1


class TestMemoryLinkV2PipelineTimeDecay:
    """Test MemoryLinkV2Pipeline with time decay."""

    @pytest.fixture
    def mock_db(self):
        """Create mock ReverieDB instance."""
        db = MagicMock()
        return db

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = MagicMock()
        client.embed_text.return_value = [0.9] + [0.1] * 1535
        return client

    def test_time_decay_parameter(self, mock_db, mock_embedding_client):
        """Should store half_life_days parameter."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
            half_life_days=365,
        )
        assert pipeline._half_life_days == 365

        pipeline_no_decay = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )
        assert pipeline_no_decay._half_life_days is None

    def test_retrieve_with_time_decay(self, mock_db, mock_embedding_client):
        """Should apply time decay to retrieval results."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            # Create passages with different timestamps
            now = datetime.now(timezone.utc)
            old_passage = MockPassageRow(10, "s1", 0, "Old", created_at=now - timedelta(days=365))
            new_passage = MockPassageRow(11, "s1", 1, "New", created_at=now)

            passages_map = {10: old_passage, 11: new_passage}
            mock_db.get_passages_by_ids.side_effect = lambda pids: [
                passages_map[pid] for pid in pids if pid in passages_map
            ]
            mock_db.search_passages_dual_embedding.return_value = [
                MagicMock(passage_id=10, combined_score=0.9),
                MagicMock(passage_id=11, combined_score=0.8),
            ]

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                half_life_days=365,
                use_query_ner=False,
            )

            results = pipeline.retrieve("test query", top_k=2)

            assert len(results) == 2

            old_result = next(r for r in results if r.passage_id == 10)
            new_result = next(r for r in results if r.passage_id == 11)

            # Old passage should have decay_weight ~0.5
            assert old_result.decay_weight is not None
            assert abs(old_result.decay_weight - 0.5) < 0.1

            # New passage should have decay_weight ~1.0
            assert new_result.decay_weight is not None
            assert abs(new_result.decay_weight - 1.0) < 0.1

    def test_retrieve_without_time_decay(self, mock_db, mock_embedding_client):
        """Should not apply time decay when half_life_days is None."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            now = datetime.now(timezone.utc)
            passage = MockPassageRow(10, "s1", 0, "Test", created_at=now - timedelta(days=365))

            mock_db.get_passages_by_ids.return_value = [passage]
            mock_db.search_passages_dual_embedding.return_value = [
                MagicMock(passage_id=10, combined_score=0.9),
            ]

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
                # No half_life_days = no decay
            )

            results = pipeline.retrieve("test query", top_k=1)

            assert len(results) == 1
            assert results[0].decay_weight is None
            assert results[0].original_score is None


class TestRetrieveWithTiming:
    """Test retrieve_with_timing() method for profiling."""

    @pytest.fixture
    def mock_db(self):
        """Create mock ReverieDB instance."""
        db = MagicMock()
        db.get_passages_by_ids.side_effect = lambda pids: [
            MockPassageRow(pid, "session-1", pid - 10, f"Text for passage {pid}")
            for pid in pids
        ]
        db.search_passages_dual_embedding.return_value = [
            MagicMock(passage_id=10, combined_score=0.9),
        ]
        return db

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = MagicMock()
        client.embed_text.return_value = [0.9] + [0.1] * 1535
        return client

    def test_timing_returned(self, mock_db, mock_embedding_client):
        """retrieve_with_timing should return timing data."""
        from reverie.retriever.pipeline_v2 import RetrievalResponse

        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            response = pipeline.retrieve_with_timing("test query", top_k=2)

            assert isinstance(response, RetrievalResponse)
            assert response.timing is not None
            assert response.timing.total_ms >= 0
            assert "embed_query" in response.timing.stages
            assert "knn_search" in response.timing.stages
            assert "load_passages" in response.timing.stages

    def test_timing_always_cache_hit(self, mock_db, mock_embedding_client):
        """Fully lazy pipeline should always report cache hit."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            # First call
            response1 = pipeline.retrieve_with_timing("test query 1", top_k=2)
            assert response1.timing.cache_hit is True  # Always true for lazy

            # Second call
            response2 = pipeline.retrieve_with_timing("test query 2", top_k=2)
            assert response2.timing.cache_hit is True  # Always true for lazy

    def test_timing_with_full_pipeline(self, mock_db, mock_embedding_client):
        """Full pipeline with entities should include all stages."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR, \
             patch('reverie.retriever.pipeline_v2.QueryNERClient') as MockNER:

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({1: 0.9})
            MockSeedSelector.return_value = mock_seed_selector

            mock_ppr = MagicMock()
            mock_ppr.run_full.return_value = MockPPRResult({10: 0.9, 11: 0.7})
            MockPPR.return_value = mock_ppr

            mock_ner = MagicMock()
            mock_ner.extract_entities.return_value = ["TestEntity"]
            MockNER.return_value = mock_ner

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=True,
            )

            response = pipeline.retrieve_with_timing("test query", top_k=2)

            # Should have all stages
            stages = response.timing.stages
            assert "embed_query" in stages  # Includes NER (parallel execution)
            assert "knn_search" in stages
            # Note: ner_extract is now parallel with embed_query, so no separate stage
            assert "ner_embed" in stages  # Entity embedding
            assert "semantic_prop" in stages  # seed selection timing
            # PPR per-iteration stages (at least iteration 0)
            assert "ppr_i0_db" in stages  # PPR iteration 0 DB query
            assert "ppr_i0_prop" in stages  # PPR iteration 0 propagation
            assert "hybrid_merge" in stages
            assert "load_passages" in stages

    def test_timing_to_dict(self, mock_db, mock_embedding_client):
        """Timing should serialize to dict."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR'):

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({})
            MockSeedSelector.return_value = mock_seed_selector

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            response = pipeline.retrieve_with_timing("test", top_k=1)
            timing_dict = response.timing.to_dict()

            assert "total_ms" in timing_dict
            assert "stages" in timing_dict
            assert "cache_hit" in timing_dict
            assert isinstance(timing_dict["stages"], dict)


class TestGetTraversalGraph:
    """Tests for get_traversal_graph method."""

    @pytest.fixture
    def mock_db(self):
        """Create mock ReverieDB instance."""
        db = MagicMock()
        db.get_passages_by_ids.side_effect = lambda pids: [
            MockPassageRow(pid, "session-1", pid - 10, f"Text for passage {pid}")
            for pid in pids
        ]
        db.search_passages_dual_embedding.return_value = []
        # Mock get_ppr_neighbors to return edges
        db.get_ppr_neighbors.return_value = []
        return db

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        client = MagicMock()
        client.embed_text.return_value = [0.9] + [0.1] * 1535
        return client

    def test_get_traversal_graph_returns_none_without_retrieve(
        self, mock_db, mock_embedding_client
    ):
        """Should return None if retrieve() hasn't been called."""
        pipeline = MemoryLinkV2Pipeline(
            db=mock_db,
            embedding_client=mock_embedding_client,
            use_query_ner=False,
        )

        result = pipeline.get_traversal_graph()
        assert result is None

    def test_get_traversal_graph_returns_graph_after_retrieve(
        self, mock_db, mock_embedding_client
    ):
        """Should return graph dict after retrieve() with PPR results."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({1: 0.9})
            MockSeedSelector.return_value = mock_seed_selector

            # PPR result with entity and passage scores
            mock_ppr_result = MagicMock()
            mock_ppr_result.entity_scores = {1: 0.9, 2: 0.5}
            mock_ppr_result.passage_scores = {10: 0.8, 11: 0.6}
            mock_ppr_result.iterations = 5
            mock_ppr_result.converged = True
            mock_ppr_result.timing_details = None

            mock_ppr = MagicMock()
            mock_ppr.run_full.return_value = mock_ppr_result
            MockPPR.return_value = mock_ppr

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            # Call retrieve to populate _last_ppr_result
            pipeline.retrieve("test query", top_k=2)

            # Now get_traversal_graph should work
            result = pipeline.get_traversal_graph(score_threshold=0.0001)

            assert result is not None
            assert "passages" in result
            assert "graph_nodes" in result
            assert "graph_edges" in result
            assert "iterations" in result
            assert "converged" in result

    def test_get_traversal_graph_calls_db_get_ppr_neighbors(
        self, mock_db, mock_embedding_client
    ):
        """Should use db.get_ppr_neighbors to build edges."""
        from reverie.db.reverie_db import PPRNeighborEdge

        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({1: 0.9})
            MockSeedSelector.return_value = mock_seed_selector

            mock_ppr_result = MagicMock()
            mock_ppr_result.entity_scores = {1: 0.9}
            mock_ppr_result.passage_scores = {10: 0.8, 11: 0.6}
            mock_ppr_result.iterations = 5
            mock_ppr_result.converged = True
            mock_ppr_result.timing_details = None

            mock_ppr = MagicMock()
            mock_ppr.run_full.return_value = mock_ppr_result
            MockPPR.return_value = mock_ppr

            # Mock edges returned by get_ppr_neighbors
            mock_db.get_ppr_neighbors.return_value = [
                PPRNeighborEdge(edge_type="e2p", source_id=1, target_id=10, out_degree=2),
                PPRNeighborEdge(edge_type="e2p", source_id=1, target_id=11, out_degree=2),
                PPRNeighborEdge(edge_type="p2p_a", source_id=10, target_id=11, out_degree=1),
            ]

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            pipeline.retrieve("test query", top_k=2)
            result = pipeline.get_traversal_graph(score_threshold=0.0001)

            # Verify db.get_ppr_neighbors was called
            mock_db.get_ppr_neighbors.assert_called_once()

            # Should have edges in the result
            assert result is not None
            assert len(result["graph_edges"]) == 3  # 2 e2p + 1 p2p

    def test_get_traversal_graph_filters_by_threshold(
        self, mock_db, mock_embedding_client
    ):
        """Should filter nodes below score_threshold."""
        with patch('reverie.retriever.pipeline_v2.LazySeedSelector') as MockSeedSelector, \
             patch('reverie.retriever.pipeline_v2.LazyBipartitePPR') as MockPPR:

            mock_seed_selector = MagicMock()
            mock_seed_selector.select_seeds.return_value = MockSeedSelectionResult({1: 0.9})
            MockSeedSelector.return_value = mock_seed_selector

            mock_ppr_result = MagicMock()
            mock_ppr_result.entity_scores = {1: 0.9, 2: 0.001}  # 2 is below threshold
            mock_ppr_result.passage_scores = {10: 0.8, 11: 0.0005}  # 11 is below threshold
            mock_ppr_result.iterations = 5
            mock_ppr_result.converged = True
            mock_ppr_result.timing_details = None

            mock_ppr = MagicMock()
            mock_ppr.run_full.return_value = mock_ppr_result
            MockPPR.return_value = mock_ppr

            pipeline = MemoryLinkV2Pipeline(
                db=mock_db,
                embedding_client=mock_embedding_client,
                use_query_ner=False,
            )

            pipeline.retrieve("test query", top_k=2)

            # Use threshold that filters out low scores
            result = pipeline.get_traversal_graph(score_threshold=0.01)

            assert result is not None
            # Only entity 1 and passage 10 should be included
            node_ids = [n["id"] for n in result["graph_nodes"]]
            assert "entity:1" in node_ids
            assert "entity:2" not in node_ids  # Filtered out
            assert "passage:10" in node_ids
            assert "passage:11" not in node_ids  # Filtered out
