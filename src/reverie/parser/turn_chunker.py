"""Turn-based chunking for session messages.

Groups messages into semantic turns. A turn starts at each
user text message and includes all subsequent tool interactions
and assistant responses until the next user text message.

For indexing, only user text messages and assistant final text responses
are included in passages. Tool calls and their results are filtered out
as they add noise and degrade entity extraction quality.
"""

from typing import Iterator, Union

from .models import (
    AssistantEntry,
    AssistantMessage,
    Passage,
    TextContent,
    UserEntry,
    UserMessage,
)
from .serializer import serialize_messages
from .token_counter import count_tokens, truncate_tool_result


def _is_user_text(entry: Union[UserEntry, AssistantEntry]) -> bool:
    """Check if entry is a user text message (turn boundary).

    A user text message has content as a plain string,
    not a list of tool_result blocks.
    """
    return (
        isinstance(entry, UserEntry)
        and isinstance(entry.message.content, str)
    )


def _has_assistant_text(entry: AssistantEntry) -> bool:
    """Check if assistant entry has any text content blocks."""
    return any(
        block.type == "text"
        for block in entry.message.content
    )


def _filter_for_indexing(
    messages: list[Union[UserEntry, AssistantEntry]],
) -> list[Union[UserEntry, AssistantEntry]]:
    """Filter messages for indexing: keep only user text and assistant text.

    Removes:
    - Tool use blocks from assistant messages
    - Tool result entries (user messages with tool_result content)

    Returns:
        Filtered list with only meaningful text content.
    """
    filtered: list[Union[UserEntry, AssistantEntry]] = []

    for entry in messages:
        if isinstance(entry, UserEntry):
            # Keep only user text messages, not tool results
            if isinstance(entry.message.content, str):
                filtered.append(entry)
        elif isinstance(entry, AssistantEntry):
            # Filter to keep only text blocks
            text_blocks = [
                block for block in entry.message.content
                if isinstance(block, TextContent) and block.type == "text"
            ]
            if text_blocks:
                # Create new entry with only text blocks
                filtered_entry = AssistantEntry(
                    type=entry.type,
                    message=AssistantMessage(
                        role=entry.message.role,
                        content=text_blocks,
                    ),
                    sessionId=entry.sessionId,
                    timestamp=entry.timestamp,
                    uuid=entry.uuid,
                    parentUuid=entry.parentUuid,
                    cwd=entry.cwd,
                )
                filtered.append(filtered_entry)

    return filtered


def chunk_by_turns(
    messages: Iterator[Union[UserEntry, AssistantEntry]],
    token_limit: int = 2000,
    tool_result_max_lines: int = 20,
) -> Iterator[Passage]:
    """Group messages into turn-based passages.

    Turn boundary = user text message.
    Each turn includes:
    - user text message (the request)
    - tool_use / tool_result exchanges
    - assistant text responses (the conclusion)

    If a turn exceeds token_limit, it is split at
    assistant-text block boundaries.

    Args:
        messages: Iterator of user/assistant entries.
        token_limit: Max tokens per passage (default 2000).
        tool_result_max_lines: Max lines for tool results (default 20).

    Yields:
        Passage objects with metadata and serialized text.
    """
    current_turn: list[Union[UserEntry, AssistantEntry]] = []
    turn_index = 0

    for entry in messages:
        if _is_user_text(entry) and current_turn:
            # Turn boundary — flush current turn
            passages = _finalize_turn(
                current_turn, turn_index, token_limit, tool_result_max_lines
            )
            for p in passages:
                yield p
                turn_index += 1
            current_turn = [entry]
        else:
            current_turn.append(entry)

    # Flush final turn
    if current_turn:
        passages = _finalize_turn(
            current_turn, turn_index, token_limit, tool_result_max_lines
        )
        for p in passages:
            yield p


def _extract_user_text(messages: list[Union[UserEntry, AssistantEntry]]) -> str:
    """Extract user text messages from a list of messages.

    Returns concatenated user text messages.
    """
    user_parts = []
    for entry in messages:
        if isinstance(entry, UserEntry) and isinstance(entry.message.content, str):
            user_parts.append(entry.message.content)
    return "\n".join(user_parts)


def _extract_assistant_text(messages: list[Union[UserEntry, AssistantEntry]]) -> str:
    """Extract assistant text blocks from a list of messages.

    Returns concatenated assistant text responses.
    """
    assistant_parts = []
    for entry in messages:
        if isinstance(entry, AssistantEntry):
            for block in entry.message.content:
                if isinstance(block, TextContent):
                    assistant_parts.append(block.text)
    return "\n".join(assistant_parts)


def _finalize_turn(
    messages: list[Union[UserEntry, AssistantEntry]],
    turn_index: int,
    token_limit: int,
    tool_result_max_lines: int,
) -> list[Passage]:
    """Convert a group of messages into one or more Passages.

    Filters messages to include only user text and assistant text responses,
    excluding tool calls and results.

    If the serialized text exceeds token_limit, splits at
    assistant-text boundaries.
    """
    if not messages:
        return []

    # Filter to keep only user text and assistant text (no tool calls)
    filtered_messages = _filter_for_indexing(messages)
    if not filtered_messages:
        return []

    first = messages[0]  # Use original first for metadata
    session_id = first.sessionId
    timestamp = first.timestamp
    git_branch = getattr(first, "gitBranch", None)

    text = serialize_messages(filtered_messages, tool_result_max_lines)
    tokens = count_tokens(text)

    # Extract separated user/assistant text for dual embedding
    user_text = _extract_user_text(filtered_messages)
    assistant_text = _extract_assistant_text(filtered_messages)

    if tokens <= token_limit:
        return [
            Passage(
                session_id=session_id,
                turn_index=turn_index,
                timestamp=timestamp,
                git_branch=git_branch,
                messages=filtered_messages,
                text=text,
                token_count=tokens,
                user_text=user_text,
                assistant_text=assistant_text,
            )
        ]

    # Split oversized turn at assistant-text boundaries
    return _split_oversized_turn(
        filtered_messages, turn_index, token_limit, tool_result_max_lines
    )


def _split_oversized_turn(
    messages: list[Union[UserEntry, AssistantEntry]],
    base_turn_index: int,
    token_limit: int,
    tool_result_max_lines: int,
) -> list[Passage]:
    """Split oversized turn into multiple passages.

    Strategy:
    1. Find split points at assistant-text boundaries
    2. Greedily group messages until token limit is reached
    3. Create a new passage for each group
    """
    first = messages[0]
    session_id = first.sessionId
    timestamp = first.timestamp
    git_branch = getattr(first, "gitBranch", None)

    # Find split points: indices where assistant text blocks occur
    split_indices = []
    for i, entry in enumerate(messages):
        if isinstance(entry, AssistantEntry):
            for content_block in entry.message.content:
                if content_block.type == "text":
                    split_indices.append(i)
                    break

    if not split_indices:
        # No assistant-text blocks to split on, emit as single passage
        text = serialize_messages(messages, tool_result_max_lines)
        user_text = _extract_user_text(messages)
        assistant_text = _extract_assistant_text(messages)
        return [
            Passage(
                session_id=session_id,
                turn_index=base_turn_index,
                timestamp=timestamp,
                git_branch=git_branch,
                messages=messages,
                text=text,
                token_count=count_tokens(text),
                user_text=user_text,
                assistant_text=assistant_text,
            )
        ]

    # Greedy split
    passages = []
    current_group: list[Union[UserEntry, AssistantEntry]] = []

    for msg in messages:
        candidate = current_group + [msg]
        candidate_text = serialize_messages(candidate, tool_result_max_lines)
        candidate_tokens = count_tokens(candidate_text)

        if candidate_tokens > token_limit and current_group:
            # Flush current group
            text = serialize_messages(current_group, tool_result_max_lines)
            user_text = _extract_user_text(current_group)
            assistant_text = _extract_assistant_text(current_group)
            passages.append(
                Passage(
                    session_id=session_id,
                    turn_index=base_turn_index + len(passages),
                    timestamp=current_group[0].timestamp,
                    git_branch=git_branch,
                    messages=list(current_group),
                    text=text,
                    token_count=count_tokens(text),
                    user_text=user_text,
                    assistant_text=assistant_text,
                )
            )
            current_group = [msg]
        else:
            current_group = candidate

    # Flush remaining
    if current_group:
        text = serialize_messages(current_group, tool_result_max_lines)
        user_text = _extract_user_text(current_group)
        assistant_text = _extract_assistant_text(current_group)
        passages.append(
            Passage(
                session_id=session_id,
                turn_index=base_turn_index + len(passages),
                timestamp=current_group[0].timestamp,
                git_branch=git_branch,
                messages=list(current_group),
                text=text,
                token_count=count_tokens(text),
                user_text=user_text,
                assistant_text=assistant_text,
            )
        )

    return passages
