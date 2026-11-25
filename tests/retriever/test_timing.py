"""Tests for StageTimer profiling utility."""

import time

import pytest


class TestStageTimer:
    """Tests for StageTimer context manager."""

    def test_single_stage_timing(self):
        """A single stage should record its duration."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()

        with timer.stage("test_stage"):
            time.sleep(0.01)  # 10ms

        timing = timer.get_timing()
        assert "test_stage" in timing.stages
        assert timing.stages["test_stage"] >= 10  # At least 10ms
        assert timing.stages["test_stage"] < 100  # Less than 100ms

    def test_multiple_stages_timing(self):
        """Multiple stages should each record their duration."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()

        with timer.stage("embed_query"):
            time.sleep(0.01)

        with timer.stage("knn_search"):
            time.sleep(0.02)

        timing = timer.get_timing()
        assert len(timing.stages) == 2
        assert timing.stages["embed_query"] >= 10
        assert timing.stages["knn_search"] >= 20

    def test_total_ms_calculation(self):
        """total_ms should sum all stage durations."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()

        with timer.stage("stage1"):
            time.sleep(0.01)

        with timer.stage("stage2"):
            time.sleep(0.01)

        timing = timer.get_timing()
        # Total should be roughly the sum of stages
        expected_min = timing.stages["stage1"] + timing.stages["stage2"]
        assert timing.total_ms >= expected_min * 0.9  # Allow 10% tolerance

    def test_cache_hit_default(self):
        """cache_hit should default to None."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()
        timing = timer.get_timing()
        assert timing.cache_hit is None

    def test_set_cache_hit(self):
        """set_cache_hit should update timing."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()
        timer.set_cache_hit(True)
        timing = timer.get_timing()
        assert timing.cache_hit is True

        timer.set_cache_hit(False)
        timing = timer.get_timing()
        assert timing.cache_hit is False

    def test_graph_build_ms(self):
        """graph_build_ms should track separate graph build duration."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()

        with timer.graph_build():
            time.sleep(0.01)

        timing = timer.get_timing()
        assert timing.graph_build_ms is not None
        assert timing.graph_build_ms >= 10

    def test_graph_build_ms_none_on_cache_hit(self):
        """graph_build_ms should be None when not measured."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()
        timer.set_cache_hit(True)

        timing = timer.get_timing()
        assert timing.graph_build_ms is None

    def test_to_dict(self):
        """StageTiming should serialize to dict properly."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()

        with timer.stage("embed_query"):
            time.sleep(0.005)

        timer.set_cache_hit(True)
        timing = timer.get_timing()
        result = timing.to_dict()

        assert "total_ms" in result
        assert "stages" in result
        assert "cache_hit" in result
        assert "graph_build_ms" in result
        assert result["stages"]["embed_query"] >= 5
        assert result["cache_hit"] is True

    def test_nested_stages_not_allowed(self):
        """Nested stages should raise an error."""
        from reverie.retriever.timing import StageTimer

        timer = StageTimer()

        with pytest.raises(RuntimeError, match="[Nn]ested"):
            with timer.stage("outer"):
                with timer.stage("inner"):
                    pass


class TestStageTiming:
    """Tests for StageTiming dataclass."""

    def test_from_dict(self):
        """StageTiming should deserialize from dict."""
        from reverie.retriever.timing import StageTiming

        data = {
            "total_ms": 47.3,
            "stages": {
                "embed_query": 12.1,
                "knn_search": 8.5,
            },
            "graph_build_ms": 5.2,
            "cache_hit": False,
        }

        timing = StageTiming.from_dict(data)
        assert timing.total_ms == 47.3
        assert timing.stages["embed_query"] == 12.1
        assert timing.graph_build_ms == 5.2
        assert timing.cache_hit is False

    def test_empty_stages(self):
        """StageTiming should handle empty stages."""
        from reverie.retriever.timing import StageTiming

        timing = StageTiming(
            total_ms=0,
            stages={},
            graph_build_ms=None,
            cache_hit=None,
        )

        assert timing.to_dict() == {
            "total_ms": 0,
            "stages": {},
            "graph_build_ms": None,
            "cache_hit": None,
        }
