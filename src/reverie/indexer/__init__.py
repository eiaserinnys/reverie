"""Indexing pipeline for MemoryLink v2.

Uses OpenAI API for embeddings (text-embedding-3-small) and
LLM-based NER for entity extraction.

v2 uses NER only (no triple extraction).

Incremental indexing support:
- IncrementalIndexer: Turn-by-turn indexing with atomic transactions
- WorkQueue: Priority queue with deduplication
- IndexWorkerPool: Multi-worker parallel processing
- StartupScanner: Scans session directories on startup
- SessionFileWatcher: Monitors file changes with debouncing
- IndexScheduler: Coordinates all indexing components
"""

from reverie.indexer.embedding import EmbeddingClient
from reverie.indexer.file_watcher import SessionFileWatcher
from reverie.indexer.incremental import IncrementalIndexer, IndexResult as IncrementalIndexResult
from reverie.indexer.openie import OpenIEClient
from reverie.indexer.pipeline import IndexingPipeline, IndexResult
from reverie.indexer.preprocessor import PreprocessorClient
from reverie.indexer.scheduler import IndexScheduler
from reverie.indexer.startup_scanner import StartupScanner
from reverie.indexer.work_queue import IndexTask, TaskPriority, WorkQueue
from reverie.indexer.worker_pool import IndexWorkerPool

__all__ = [
    "EmbeddingClient",
    "OpenIEClient",
    "IndexingPipeline",
    "IndexResult",
    "PreprocessorClient",
    # Incremental indexing
    "IncrementalIndexer",
    "IncrementalIndexResult",
    "IndexTask",
    "TaskPriority",
    "WorkQueue",
    "IndexWorkerPool",
    # Trigger system
    "StartupScanner",
    "SessionFileWatcher",
    "IndexScheduler",
]
