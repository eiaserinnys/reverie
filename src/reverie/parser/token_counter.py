"""Token counting and text truncation utilities."""

import tiktoken

_encoder: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get tiktoken encoder (cl100k_base, cached)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base.

    Args:
        text: Text to count.

    Returns:
        Token count.
    """
    if not text:
        return 0
    return len(get_encoder().encode(text))


def truncate_tool_result(content: str, max_lines: int = 20) -> str:
    """Truncate long tool result to first/last N lines.

    If content has fewer than max_lines * 2, return as-is.
    Otherwise keep first max_lines and last max_lines lines
    with an omission marker in between.

    Args:
        content: Tool result text.
        max_lines: Lines to keep from head and tail.

    Returns:
        Truncated text or original if short enough.
    """
    if not content:
        return content

    lines = content.split("\n")
    total = len(lines)

    if total <= max_lines * 2:
        return content

    head = lines[:max_lines]
    tail = lines[-max_lines:]
    omitted = total - max_lines * 2

    return "\n".join(head) + f"\n... [{omitted} lines omitted] ...\n" + "\n".join(tail)
