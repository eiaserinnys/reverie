"""Tests for session preprocessor module.

The preprocessor filters and refines turn-level text before OpenIE extraction,
dropping low-value content (greetings, reference data) and cleaning noise.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from reverie.indexer.preprocessor import PreprocessorClient, PreprocessResult


# =============================================================================
# PreprocessResult tests
# =============================================================================


class TestPreprocessResult:
    """Test the PreprocessResult dataclass."""

    def test_keep_result(self):
        """Keep result should have action='keep' and refined_text."""
        result = PreprocessResult(
            action="keep",
            reason=None,
            refined_text="Refined passage text",
        )
        assert result.action == "keep"
        assert result.reason is None
        assert result.refined_text == "Refined passage text"

    def test_drop_result(self):
        """Drop result should have action='drop' and reason."""
        result = PreprocessResult(
            action="drop",
            reason="Greeting/chitchat only",
            refined_text=None,
        )
        assert result.action == "drop"
        assert result.reason == "Greeting/chitchat only"
        assert result.refined_text is None


# =============================================================================
# Helper functions
# =============================================================================


def _make_chat_response(content: str, finish_reason: str = "stop"):
    """Create a mock chat completion response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client for preprocessor."""
    with patch("reverie.indexer.preprocessor.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


# =============================================================================
# PreprocessorClient tests - keep cases
# =============================================================================


class TestPreprocessorClientKeep:
    """Test cases where the preprocessor keeps the passage."""

    def test_keep_architecture_decision(self, mock_openai):
        """Should keep passages with architecture decisions."""
        response_json = json.dumps({
            "action": "keep",
            "reason": None,
            "refined_text": "Decided to use PostgreSQL instead of SQLite for better concurrency.",
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("Some passage about DB choice")

        assert result.action == "keep"
        assert "PostgreSQL" in result.refined_text
        mock_openai.chat.completions.create.assert_called_once()

    def test_keep_error_discovery(self, mock_openai):
        """Should keep passages with error discovery and fix."""
        response_json = json.dumps({
            "action": "keep",
            "reason": None,
            "refined_text": "Found that the import path was wrong. Fixed by changing 'from foo' to 'from bar'.",
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("Error passage")

        assert result.action == "keep"
        assert "import path" in result.refined_text

    def test_keep_unique_experience(self, mock_openai):
        """Should keep passages with Agent/Shadow-specific experience."""
        response_json = json.dumps({
            "action": "keep",
            "reason": None,
            "refined_text": "Implemented the MCP server's custom_route decorator for REST API support.",
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("MCP implementation passage")

        assert result.action == "keep"
        assert "MCP" in result.refined_text


# =============================================================================
# PreprocessorClient tests - drop cases
# =============================================================================


class TestPreprocessorClientDrop:
    """Test cases where the preprocessor drops the passage."""

    def test_drop_greeting(self, mock_openai):
        """Should drop passages with only greetings."""
        response_json = json.dumps({
            "action": "drop",
            "reason": "Greeting/chitchat only",
            "refined_text": None,
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("[user] 안녕!\n[assistant] 안녕하세요!")

        assert result.action == "drop"
        assert "Greeting" in result.reason
        assert result.refined_text is None

    def test_drop_reference_data_only(self, mock_openai):
        """Should drop passages with only URLs/IDs."""
        response_json = json.dumps({
            "action": "drop",
            "reason": "Reference data only (URLs, IDs)",
            "refined_text": None,
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("Card ID: 00000001... URL: https://trello.com/...")

        assert result.action == "drop"
        assert result.refined_text is None

    def test_drop_simple_tool_call(self, mock_openai):
        """Should drop passages with simple tool calls without insights."""
        response_json = json.dumps({
            "action": "drop",
            "reason": "Simple tool call without meaningful insight",
            "refined_text": None,
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("[tool_use] Bash: git status\n[tool_result] nothing to commit")

        assert result.action == "drop"


# =============================================================================
# PreprocessorClient tests - edge cases
# =============================================================================


class TestPreprocessorClientEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_json_response(self, mock_openai):
        """Should handle non-JSON response gracefully by keeping as-is."""
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            "This is not valid JSON."
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("Some passage")

        # On parse failure, keep original text
        assert result.action == "keep"
        assert result.refined_text == "Some passage"

    def test_malformed_action_defaults_to_keep(self, mock_openai):
        """Should default to 'keep' if action is unrecognized."""
        response_json = json.dumps({
            "action": "unknown_action",
            "reason": None,
            "refined_text": "Some text",
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("Some passage")

        assert result.action == "keep"

    def test_missing_refined_text_uses_original(self, mock_openai):
        """Should use original text if refined_text is missing for keep."""
        response_json = json.dumps({
            "action": "keep",
            "reason": None,
            "refined_text": None,
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("Original passage text")

        assert result.action == "keep"
        assert result.refined_text == "Original passage text"

    def test_empty_passage(self, mock_openai):
        """Empty passage should be dropped without LLM call."""
        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("")

        assert result.action == "drop"
        assert result.reason == "Empty passage"
        mock_openai.chat.completions.create.assert_not_called()

    def test_whitespace_only_passage(self, mock_openai):
        """Whitespace-only passage should be dropped without LLM call."""
        client = PreprocessorClient(api_key="test-key")
        result = client.preprocess_turn("   \n\t  ")

        assert result.action == "drop"
        assert result.reason == "Empty passage"
        mock_openai.chat.completions.create.assert_not_called()


# =============================================================================
# Prompt construction tests
# =============================================================================


class TestPreprocessorPrompts:
    """Test prompt building functions."""

    def test_build_preprocess_messages_structure(self):
        """Preprocess messages should have proper structure."""
        from reverie.indexer.preprocessor import build_preprocess_messages

        messages = build_preprocess_messages("test passage")

        assert len(messages) >= 2  # At least system + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "test passage" in messages[-1]["content"]

    def test_system_prompt_contains_criteria(self):
        """System prompt should contain keep/drop criteria."""
        from reverie.indexer.preprocessor import PREPROCESS_SYSTEM

        # Check keep criteria
        assert "architecture" in PREPROCESS_SYSTEM.lower() or "설계" in PREPROCESS_SYSTEM
        assert "error" in PREPROCESS_SYSTEM.lower() or "실수" in PREPROCESS_SYSTEM

        # Check drop criteria
        assert "greeting" in PREPROCESS_SYSTEM.lower() or "인사" in PREPROCESS_SYSTEM


# =============================================================================
# Cache integration tests
# =============================================================================


class TestPreprocessorCache:
    """Test caching behavior."""

    def test_uses_cache_on_hit(self, mock_openai):
        """Should return cached result without LLM call."""
        with patch("reverie.indexer.preprocessor.ApiCache") as mock_cache_cls:
            mock_cache = MagicMock()
            mock_cache_cls.return_value = mock_cache

            cached_response = {
                "text": json.dumps({
                    "action": "keep",
                    "reason": None,
                    "refined_text": "Cached result",
                }),
                "finish_reason": "stop",
            }
            mock_cache.get.return_value = cached_response

            client = PreprocessorClient(api_key="test-key")
            result = client.preprocess_turn("Some passage")

            assert result.refined_text == "Cached result"
            mock_openai.chat.completions.create.assert_not_called()

    def test_stores_in_cache_on_miss(self, mock_openai):
        """Should store result in cache after LLM call."""
        with patch("reverie.indexer.preprocessor.ApiCache") as mock_cache_cls:
            mock_cache = MagicMock()
            mock_cache_cls.return_value = mock_cache
            mock_cache.get.return_value = None  # Cache miss

            response_json = json.dumps({
                "action": "keep",
                "reason": None,
                "refined_text": "New result",
            })
            mock_openai.chat.completions.create.return_value = _make_chat_response(
                response_json
            )

            client = PreprocessorClient(api_key="test-key")
            client.preprocess_turn("Some passage")

            mock_cache.put.assert_called_once()
