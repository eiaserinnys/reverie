"""JSONL parser and turn-based chunking.

Parses Claude Code session JSONL files into semantic chunks (Passages).
Each turn = [user-text] -> [tool-use/tool-result] -> [assistant-text] cycle.

For indexing purposes, only user text messages and assistant final responses
are included in passages. Tool calls and results are filtered out to reduce
noise and improve entity extraction quality.
"""

from pathlib import Path
from typing import Iterator

from .jsonl_reader import filter_messages, read_jsonl
from .models import JSONLEntry, Passage
from .serializer import serialize_passage
from .turn_chunker import chunk_by_turns

__all__ = [
    "read_jsonl",
    "filter_messages",
    "chunk_by_turns",
    "serialize_passage",
    "parse_session",
    "Passage",
    "JSONLEntry",
]


def parse_session(
    session_path: Path,
    token_limit: int = 2000,
    tool_result_max_lines: int = 20,
) -> Iterator[Passage]:
    """Parse session JSONL file into passages.

    Convenience function for the full pipeline:
    read -> filter -> chunk by turns.

    Args:
        session_path: Path to session JSONL file.
        token_limit: Max tokens per passage.
        tool_result_max_lines: Max lines for tool results.

    Yields:
        Passage objects.
    """
    entries = read_jsonl(session_path)
    messages = filter_messages(entries)
    yield from chunk_by_turns(messages, token_limit, tool_result_max_lines)
