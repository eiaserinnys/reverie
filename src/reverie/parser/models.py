"""Data models for JSONL session parsing.

Defines Pydantic models matching the Claude Code session JSONL structure.
"""

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Message content types ---


class TextContent(BaseModel):
    """Text content block in assistant message."""

    type: Literal["text"]
    text: str


class ThinkingContent(BaseModel):
    """Thinking content block in assistant message (extended thinking)."""

    type: Literal["thinking"]

    model_config = {"extra": "ignore"}


class ToolUseContent(BaseModel):
    """Tool use content block in assistant message."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContent(BaseModel):
    """Tool result content block in user message."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, list[dict[str, Any]]] = ""
    is_error: bool = False


# --- Message models ---


class UserMessage(BaseModel):
    """User message payload."""

    role: Literal["user"]
    content: Union[str, list[Union[TextContent, ToolResultContent]]]


class AssistantMessage(BaseModel):
    """Assistant message payload."""

    role: Literal["assistant"]
    content: list[Union[TextContent, ThinkingContent, ToolUseContent]]


# --- JSONL entry types ---


class QueueOperation(BaseModel):
    """Queue operation metadata entry."""

    type: Literal["queue-operation"]
    operation: str
    timestamp: datetime
    sessionId: str

    model_config = {"extra": "ignore"}


class UserEntry(BaseModel):
    """User message JSONL entry."""

    type: Literal["user"]
    message: UserMessage
    sessionId: str
    timestamp: datetime
    uuid: str
    gitBranch: Optional[str] = None
    parentUuid: Optional[str] = None
    cwd: Optional[str] = None

    model_config = {"extra": "ignore"}


class AssistantEntry(BaseModel):
    """Assistant message JSONL entry."""

    type: Literal["assistant"]
    message: AssistantMessage
    sessionId: str
    timestamp: datetime
    uuid: str
    parentUuid: Optional[str] = None
    cwd: Optional[str] = None

    model_config = {"extra": "ignore"}


# --- Discriminated union ---

JSONLEntry = Union[QueueOperation, UserEntry, AssistantEntry]


# --- Output model ---


class Passage(BaseModel):
    """A semantic chunk extracted from a session.

    Corresponds to one "turn" in the conversation:
    user-text -> [tool-use/tool-result]* -> assistant-text

    For indexing, user_text and assistant_text are stored separately
    to enable dual embedding search with harmonic mean scoring.
    """

    session_id: str
    turn_index: int
    timestamp: datetime
    git_branch: Optional[str] = None
    messages: list[Union[UserEntry, AssistantEntry]]
    text: str = ""  # Combined text (deprecated, for backward compatibility)
    token_count: int = 0
    # Separated user/assistant text for dual embedding
    user_text: str = ""
    assistant_text: str = ""
