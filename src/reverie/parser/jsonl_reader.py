"""JSONL file reader for Claude Code session files."""

import json
import logging
from pathlib import Path
from typing import Iterator, Union

from .models import (
    AssistantEntry,
    JSONLEntry,
    QueueOperation,
    UserEntry,
)

logger = logging.getLogger(__name__)


def read_jsonl(file_path: Path) -> Iterator[JSONLEntry]:
    """Read and validate JSONL file line by line.

    Yields validated JSONLEntry objects. Skips lines that
    cannot be parsed (logs warning instead of raising).

    Args:
        file_path: Path to session JSONL file.

    Yields:
        Validated JSONLEntry objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: invalid JSON: %s", line_num, e)
                continue

            entry_type = data.get("type")
            try:
                if entry_type == "queue-operation":
                    yield QueueOperation.model_validate(data)
                elif entry_type == "user":
                    yield UserEntry.model_validate(data)
                elif entry_type == "assistant":
                    yield AssistantEntry.model_validate(data)
                else:
                    logger.debug("Line %d: unknown type %r, skipping", line_num, entry_type)
            except Exception as e:
                logger.warning("Line %d: validation error: %s", line_num, e)
                continue


def filter_messages(
    entries: Iterator[JSONLEntry],
) -> Iterator[Union[UserEntry, AssistantEntry]]:
    """Filter out queue-operation entries, keep only messages.

    Args:
        entries: Iterator of all JSONL entries.

    Yields:
        User and Assistant entries only.
    """
    for entry in entries:
        if isinstance(entry, (UserEntry, AssistantEntry)):
            yield entry
