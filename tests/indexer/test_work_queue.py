"""Tests for WorkQueue with deduplication and priority support."""

import asyncio
import time

import pytest

from reverie.indexer.work_queue import IndexTask, TaskPriority, WorkQueue


@pytest.fixture
def queue():
    """Create a fresh WorkQueue for each test."""
    return WorkQueue()


class TestWorkQueue:
    """Tests for WorkQueue functionality."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, queue):
        """Test basic put and get operations."""
        task = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="session1",
            file_path="/path/to/session.jsonl",
        )

        result = await queue.put(task)
        assert result is True
        assert queue.qsize() == 1

        retrieved = await queue.get()
        assert retrieved.session_id == "session1"
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test that tasks are retrieved in priority order."""
        # Add tasks out of order
        await queue.put(IndexTask(
            priority=TaskPriority.LOW,
            created_at=time.time(),
            profile_id="test",
            session_id="low",
            file_path="/path/low.jsonl",
        ))
        await queue.put(IndexTask(
            priority=TaskPriority.HIGH,
            created_at=time.time(),
            profile_id="test",
            session_id="high",
            file_path="/path/high.jsonl",
        ))
        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="normal",
            file_path="/path/normal.jsonl",
        ))

        # Should get HIGH first
        task1 = await queue.get()
        assert task1.session_id == "high"

        # Then NORMAL
        task2 = await queue.get()
        assert task2.session_id == "normal"

        # Then LOW
        task3 = await queue.get()
        assert task3.session_id == "low"

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self, queue):
        """Test that duplicate tasks are skipped."""
        task1 = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="same_session",
            file_path="/path/to/session.jsonl",
        )
        task2 = IndexTask(
            priority=TaskPriority.HIGH,  # Different priority
            created_at=time.time() + 1,  # Different time
            profile_id="test",
            session_id="same_session",  # Same session
            file_path="/path/to/session.jsonl",
        )

        # First add succeeds
        result1 = await queue.put(task1)
        assert result1 is True
        assert queue.qsize() == 1

        # Second add is skipped (duplicate)
        result2 = await queue.put(task2)
        assert result2 is False
        assert queue.qsize() == 1

        # Check stats
        assert queue.stats["duplicates_skipped"] == 1

    @pytest.mark.asyncio
    async def test_pending_cleared_after_get(self, queue):
        """Test that pending set is cleared after get."""
        task = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="session1",
            file_path="/path/to/session.jsonl",
        )

        await queue.put(task)
        assert queue.is_pending("test", "session1") is True

        retrieved_task = await queue.get()
        # Still pending until mark_done() is called
        assert queue.is_pending("test", "session1") is True

        # Cannot add while still pending
        result = await queue.put(task)
        assert result is False

        # After mark_done, pending is cleared
        await queue.mark_done(retrieved_task)
        assert queue.is_pending("test", "session1") is False

        # Now can add the same session again
        result = await queue.put(task)
        assert result is True

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, queue):
        """Test FIFO ordering for tasks with same priority."""
        base_time = time.time()

        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=base_time,
            profile_id="test",
            session_id="first",
            file_path="/path/first.jsonl",
        ))
        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=base_time + 1,
            profile_id="test",
            session_id="second",
            file_path="/path/second.jsonl",
        ))

        task1 = await queue.get()
        task2 = await queue.get()

        assert task1.session_id == "first"
        assert task2.session_id == "second"

    @pytest.mark.asyncio
    async def test_different_profiles_not_duplicates(self, queue):
        """Test that same session in different profiles are not duplicates."""
        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="agent",
            session_id="session1",
            file_path="/path/session.jsonl",
        ))
        result = await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="shadow",  # Different profile
            session_id="session1",  # Same session ID
            file_path="/path/session.jsonl",
        ))

        assert result is True
        assert queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_empty_and_qsize(self, queue):
        """Test empty and qsize methods."""
        assert queue.empty() is True
        assert queue.qsize() == 0

        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="session1",
            file_path="/path/session.jsonl",
        ))

        assert queue.empty() is False
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_clear(self, queue):
        """Test clearing the queue."""
        for i in range(5):
            await queue.put(IndexTask(
                priority=TaskPriority.NORMAL,
                created_at=time.time(),
                profile_id="test",
                session_id=f"session{i}",
                file_path=f"/path/session{i}.jsonl",
            ))

        assert queue.qsize() == 5

        cleared = queue.clear()
        assert cleared == 5
        assert queue.empty() is True
        assert queue.stats["pending_count"] == 0

    @pytest.mark.asyncio
    async def test_stats(self, queue):
        """Test queue statistics."""
        task1 = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="session1",
            file_path="/path/session.jsonl",
        )
        task2 = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="session2",
            file_path="/path/session.jsonl",
        )

        await queue.put(task1)
        await queue.put(task2)
        await queue.put(task1)  # Duplicate

        retrieved = await queue.get()

        stats = queue.stats
        assert stats["enqueued"] == 2
        assert stats["dequeued"] == 1
        assert stats["duplicates_skipped"] == 1
        assert stats["current_size"] == 1
        # Both tasks still pending (one in queue, one being processed)
        assert stats["pending_count"] == 2

        # After mark_done, pending count decreases
        await queue.mark_done(retrieved)
        stats = queue.stats
        assert stats["pending_count"] == 1

    @pytest.mark.asyncio
    async def test_task_key(self):
        """Test IndexTask key generation (backward compat with default empty path)."""
        task = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="agent",
            session_id="abc123",
            file_path="/path/to/session.jsonl",
        )

        # Key now includes indexing_path (empty by default)
        assert task.key() == "agent:abc123:"

    @pytest.mark.asyncio
    async def test_task_ordering(self):
        """Test that IndexTask ordering works correctly."""
        t1 = time.time()
        t2 = t1 + 1

        high_early = IndexTask(
            priority=TaskPriority.HIGH,
            created_at=t1,
            profile_id="test",
            session_id="a",
            file_path="/a.jsonl",
        )
        high_late = IndexTask(
            priority=TaskPriority.HIGH,
            created_at=t2,
            profile_id="test",
            session_id="b",
            file_path="/b.jsonl",
        )
        low = IndexTask(
            priority=TaskPriority.LOW,
            created_at=t1,
            profile_id="test",
            session_id="c",
            file_path="/c.jsonl",
        )

        # High priority comes before low
        assert high_early < low
        assert high_late < low

        # Earlier time comes before later time (within same priority)
        assert high_early < high_late

    @pytest.mark.asyncio
    async def test_task_key_with_indexing_path(self):
        """Test IndexTask key includes indexing_path for multi-folder support."""
        # Task without indexing_path (default empty string)
        task_default = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="agent",
            session_id="abc123",
            file_path="/path/to/session.jsonl",
        )
        assert task_default.key() == "agent:abc123:"

        # Task with indexing_path
        task_with_path = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="agent",
            session_id="abc123",
            file_path="/path/to/session.jsonl",
            indexing_path="/home/user/.claude/project1",
        )
        assert task_with_path.key() == "agent:abc123:/home/user/.claude/project1"

    @pytest.mark.asyncio
    async def test_same_session_different_paths_not_duplicates(self, queue):
        """Test that same session in different indexing paths are not duplicates."""
        # Same session from different folders
        await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="agent",
            session_id="session1",
            file_path="/project1/sessions/session1.jsonl",
            indexing_path="/project1/sessions",
        ))
        result = await queue.put(IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="agent",
            session_id="session1",  # Same session ID
            file_path="/project2/sessions/session1.jsonl",
            indexing_path="/project2/sessions",  # Different path
        ))

        assert result is True
        assert queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_is_pending_with_indexing_path(self, queue):
        """Test is_pending works with indexing_path parameter."""
        task = IndexTask(
            priority=TaskPriority.NORMAL,
            created_at=time.time(),
            profile_id="test",
            session_id="session1",
            file_path="/path/to/session.jsonl",
            indexing_path="/project1",
        )

        await queue.put(task)
        # Should be pending with correct path
        assert queue.is_pending("test", "session1", "/project1") is True
        # Should NOT be pending with different path
        assert queue.is_pending("test", "session1", "/project2") is False
        # Should NOT be pending with empty path
        assert queue.is_pending("test", "session1") is False
