"""Serialize messages and passages to text format.

Produces human-readable text suitable for embedding and OpenIE extraction.
"""

import json
from typing import Union

from .models import (
    AssistantEntry,
    Passage,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    UserEntry,
)
from .token_counter import truncate_tool_result


def serialize_messages(
    messages: list[Union[UserEntry, AssistantEntry]],
    tool_result_max_lines: int = 20,
) -> str:
    """Serialize a list of messages to text.

    Args:
        messages: List of user/assistant entries.
        tool_result_max_lines: Max lines for tool results.

    Returns:
        Serialized text.
    """
    parts: list[str] = []

    for entry in messages:
        if isinstance(entry, UserEntry):
            parts.append(_format_user_entry(entry, tool_result_max_lines))
        elif isinstance(entry, AssistantEntry):
            parts.append(_format_assistant_entry(entry))

    return "\n".join(parts)


def serialize_passage(
    passage: Passage,
    body_override: str | None = None,
) -> str:
    """Serialize a Passage to text with metadata header.

    Format:
        [Session: {id}] [Turn: {index}] [Branch: {branch}]
        Timestamp: {ts}

        {message text}

    Args:
        passage: Passage to serialize.
        body_override: If provided, use this instead of passage.text.
            Useful for preprocessed/refined text.

    Returns:
        Full text representation with metadata.
    """
    header_parts = [
        f"[Session: {passage.session_id}]",
        f"[Turn: {passage.turn_index}]",
    ]
    if passage.git_branch:
        header_parts.append(f"[Branch: {passage.git_branch}]")

    header = " ".join(header_parts)
    ts = f"Timestamp: {passage.timestamp.isoformat()}"

    body = body_override if body_override is not None else passage.text
    return f"{header}\n{ts}\n\n{body}"


def _format_user_entry(
    entry: UserEntry, tool_result_max_lines: int = 20
) -> str:
    """Format a user entry to text."""
    content = entry.message.content

    if isinstance(content, str):
        return f"User: {content}"

    # tool_result list
    parts: list[str] = []
    for item in content:
        if isinstance(item, ToolResultContent):
            result_text = item.content if isinstance(item.content, str) else json.dumps(item.content)
            truncated = truncate_tool_result(result_text, tool_result_max_lines)
            error_tag = " [ERROR]" if item.is_error else ""
            parts.append(f"Result{error_tag}: {truncated}")
        elif isinstance(item, dict):
            # Handle raw dict for tool_result
            result_text = item.get("content", "")
            if not isinstance(result_text, str):
                result_text = json.dumps(result_text)
            truncated = truncate_tool_result(result_text, tool_result_max_lines)
            error_tag = " [ERROR]" if item.get("is_error", False) else ""
            parts.append(f"Result{error_tag}: {truncated}")

    return "\n".join(parts)


def _format_assistant_entry(entry: AssistantEntry) -> str:
    """Format an assistant entry to text."""
    parts: list[str] = []

    for block in entry.message.content:
        if isinstance(block, TextContent):
            parts.append(f"Assistant: {block.text}")
        elif isinstance(block, ToolUseContent):
            # Format tool input compactly
            input_str = _compact_tool_input(block.input)
            parts.append(f"Tool: {block.name}({input_str})")

    return "\n".join(parts)


def _compact_tool_input(input_data: dict) -> str:
    """Create compact representation of tool input.

    Short inputs are inlined, long ones are summarized.
    """
    if not input_data:
        return ""

    # For simple cases with just one key
    if len(input_data) == 1:
        key, value = next(iter(input_data.items()))
        if isinstance(value, str) and len(value) < 100:
            return f"{key}={value!r}"
        elif isinstance(value, str):
            return f"{key}={value[:80]!r}..."

    # Multi-key: show key=value pairs
    parts = []
    for key, value in input_data.items():
        if isinstance(value, str) and len(value) > 80:
            parts.append(f"{key}={value[:80]!r}...")
        elif isinstance(value, str):
            parts.append(f"{key}={value!r}")
        else:
            val_str = json.dumps(value)
            if len(val_str) > 80:
                parts.append(f"{key}=...")
            else:
                parts.append(f"{key}={val_str}")
    return ", ".join(parts)
