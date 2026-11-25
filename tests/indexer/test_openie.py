"""Tests for OpenIE module (NER + triple extraction)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from reverie.indexer.openie import OpenIEClient, fix_broken_json


# =============================================================================
# fix_broken_json tests
# =============================================================================


class TestFixBrokenJson:
    """Test JSON repair utility."""

    def test_valid_json_unchanged(self):
        """Valid JSON should pass through."""
        valid = '{"named_entities": ["foo", "bar"]}'
        assert fix_broken_json(valid) == valid

    def test_truncated_array(self):
        """Truncated array should be repaired."""
        broken = '{"named_entities": ["foo", "bar", "ba'
        result = fix_broken_json(broken)
        parsed = json.loads(result)
        assert "named_entities" in parsed
        assert parsed["named_entities"] == ["foo", "bar"]

    def test_truncated_triple(self):
        """Truncated triple array should be repaired to valid JSON.

        Note: The repair produces structurally valid JSON. Invalid triples
        (not exactly 3 elements) are filtered by _filter_invalid_triples().
        """
        broken = '{"triples": [["a", "b", "c"], ["d", "e"'
        result = fix_broken_json(broken)
        parsed = json.loads(result)
        # Repair preserves complete elements; the incomplete ["d", "e"]
        # becomes ["d"] after truncation at last comma.
        # Triple filtering happens at a higher level.
        assert ["a", "b", "c"] in parsed["triples"]
        assert "triples" in parsed

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert fix_broken_json("") == ""

    def test_no_json_at_all(self):
        """Non-JSON string should return as-is."""
        assert fix_broken_json("hello world") == "hello world"

    def test_nested_braces_in_strings(self):
        """Should handle braces inside string values."""
        valid = '{"named_entities": ["dict{key}", "arr[0]"]}'
        assert fix_broken_json(valid) == valid


# =============================================================================
# OpenIEClient NER tests
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
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_response.usage = mock_usage
    return mock_response


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client for OpenIE."""
    with patch("reverie.indexer.openie.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


class TestOpenIEClientNER:
    """Test NER extraction."""

    def test_extract_entities_basic(self, mock_openai):
        """Should extract entities from LLM response."""
        response_json = '{"named_entities": ["pytest", "src/config.py", "tiktoken"]}'
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = OpenIEClient(api_key="test-key")
        entities = client.extract_entities("some passage text")

        assert entities == ["pytest", "src/config.py", "tiktoken"]
        mock_openai.chat.completions.create.assert_called_once()

    def test_extract_entities_deduplicates(self, mock_openai):
        """Should remove duplicate entities."""
        response_json = '{"named_entities": ["foo", "bar", "foo", "baz", "bar"]}'
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = OpenIEClient(api_key="test-key")
        entities = client.extract_entities("passage")

        assert entities == ["foo", "bar", "baz"]

    def test_extract_entities_empty_response(self, mock_openai):
        """Empty named_entities should return empty list."""
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            '{"named_entities": []}'
        )

        client = OpenIEClient(api_key="test-key")
        entities = client.extract_entities("passage")
        assert entities == []

    def test_extract_entities_broken_json(self, mock_openai):
        """Should repair broken JSON from truncated response."""
        broken = '{"named_entities": ["foo", "bar", "ba'
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            broken, finish_reason="length"
        )

        client = OpenIEClient(api_key="test-key")
        entities = client.extract_entities("passage")
        assert entities == ["foo", "bar"]

    def test_extract_entities_garbage_response(self, mock_openai):
        """Should return empty list on unparseable response."""
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            "I cannot extract entities from this text."
        )

        client = OpenIEClient(api_key="test-key")
        entities = client.extract_entities("passage")
        assert entities == []

    def test_extract_entities_with_extra_text(self, mock_openai):
        """Should extract JSON from response with surrounding text."""
        response = (
            'Here are the entities:\n'
            '{"named_entities": ["pgvector", "HNSW"]}\n'
            'These are the key entities.'
        )
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response
        )

        client = OpenIEClient(api_key="test-key")
        entities = client.extract_entities("passage")
        assert entities == ["pgvector", "HNSW"]


# =============================================================================
# OpenIEClient triple extraction tests
# =============================================================================


class TestOpenIEClientTriples:
    """Test triple extraction."""

    def test_extract_triples_basic(self, mock_openai):
        """Should extract triples from LLM response."""
        response_json = json.dumps({
            "triples": [
                ["Bash", "executed", "pytest"],
                ["pytest", "tested", "tests/"],
            ]
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = OpenIEClient(api_key="test-key")
        triples = client.extract_triples("passage", ["Bash", "pytest"])

        assert triples == [
            ["Bash", "executed", "pytest"],
            ["pytest", "tested", "tests/"],
        ]

    def test_extract_triples_filters_invalid(self, mock_openai):
        """Should filter out triples with != 3 elements."""
        response_json = json.dumps({
            "triples": [
                ["a", "b", "c"],
                ["d", "e"],          # too short
                ["f", "g", "h", "i"],  # too long
                ["j", "k", "l"],
            ]
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = OpenIEClient(api_key="test-key")
        triples = client.extract_triples("passage", ["a", "j"])

        assert triples == [["a", "b", "c"], ["j", "k", "l"]]

    def test_extract_triples_deduplicates(self, mock_openai):
        """Should remove duplicate triples."""
        response_json = json.dumps({
            "triples": [
                ["a", "b", "c"],
                ["a", "b", "c"],
                ["d", "e", "f"],
            ]
        })
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            response_json
        )

        client = OpenIEClient(api_key="test-key")
        triples = client.extract_triples("passage", ["a"])

        assert triples == [["a", "b", "c"], ["d", "e", "f"]]

    def test_extract_triples_empty(self, mock_openai):
        """Empty triples list should return empty."""
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            '{"triples": []}'
        )

        client = OpenIEClient(api_key="test-key")
        triples = client.extract_triples("passage", ["foo"])
        assert triples == []

    def test_extract_triples_broken_json(self, mock_openai):
        """Should repair broken JSON from truncated response."""
        broken = '{"triples": [["a", "b", "c"], ["d", "e"'
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            broken, finish_reason="length"
        )

        client = OpenIEClient(api_key="test-key")
        triples = client.extract_triples("passage", ["a"])
        assert triples == [["a", "b", "c"]]

    def test_extract_triples_garbage_response(self, mock_openai):
        """Should return empty list on unparseable response."""
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            "I don't know how to extract triples."
        )

        client = OpenIEClient(api_key="test-key")
        triples = client.extract_triples("passage", ["foo"])
        assert triples == []


# =============================================================================
# OpenIEClient process_passage (combined NER + triples)
# =============================================================================


class TestOpenIEClientProcessPassage:
    """Test the combined NER + triple extraction pipeline."""

    def test_process_passage_full_pipeline(self, mock_openai):
        """Should run NER then triple extraction."""
        ner_response = '{"named_entities": ["pytest", "Bash"]}'
        triple_response = json.dumps({
            "triples": [
                ["Bash", "executed", "pytest"],
                ["pytest", "tested", "tests/"],
            ]
        })

        mock_openai.chat.completions.create.side_effect = [
            _make_chat_response(ner_response),
            _make_chat_response(triple_response),
        ]

        client = OpenIEClient(api_key="test-key")
        result = client.process_passage("some passage")

        assert result["entities"] == ["pytest", "Bash"]
        assert result["triples"] == [
            ["Bash", "executed", "pytest"],
            ["pytest", "tested", "tests/"],
        ]
        assert mock_openai.chat.completions.create.call_count == 2

    def test_process_passage_ner_fails_gracefully(self, mock_openai):
        """If NER fails, should still return empty but not crash."""
        mock_openai.chat.completions.create.side_effect = [
            _make_chat_response("garbage"),  # NER fails
            _make_chat_response('{"triples": []}'),  # triples with empty entities
        ]

        client = OpenIEClient(api_key="test-key")
        result = client.process_passage("some passage")

        assert result["entities"] == []
        assert result["triples"] == []

    def test_process_passage_skips_triples_when_no_entities(self, mock_openai):
        """If NER returns no entities, should skip triple extraction."""
        mock_openai.chat.completions.create.return_value = _make_chat_response(
            '{"named_entities": []}'
        )

        client = OpenIEClient(api_key="test-key")
        result = client.process_passage("some passage")

        assert result["entities"] == []
        assert result["triples"] == []
        # Only 1 call (NER), not 2
        assert mock_openai.chat.completions.create.call_count == 1


# =============================================================================
# Prompt construction tests
# =============================================================================


class TestPromptConstruction:
    """Test prompt building functions."""

    def test_build_ner_messages_structure(self):
        """NER messages should have correct structure: system, examples, input."""
        from reverie.indexer.prompts import build_ner_messages, _NER_EXAMPLES

        messages = build_ner_messages("test passage")

        # Structure: 1 system + (N examples * 2) + 1 user input
        expected_len = 1 + len(_NER_EXAMPLES) * 2 + 1
        assert len(messages) == expected_len
        assert messages[0]["role"] == "system"
        # Examples alternate user/assistant
        for i in range(len(_NER_EXAMPLES)):
            assert messages[1 + i * 2]["role"] == "user"
            assert messages[2 + i * 2]["role"] == "assistant"
        # Last message is the actual input
        assert messages[-1]["role"] == "user"
        assert "test passage" in messages[-1]["content"]

    def test_build_triple_messages_structure(self):
        """Triple messages should have 4 turns with entities injected."""
        from reverie.indexer.prompts import build_triple_messages

        messages = build_triple_messages("test passage", ["foo", "bar"])

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[3]["role"] == "user"
        assert "test passage" in messages[3]["content"]
        assert '"foo"' in messages[3]["content"]
        assert '"bar"' in messages[3]["content"]

    def test_build_triple_messages_handles_korean(self):
        """Should handle Korean entity names correctly."""
        from reverie.indexer.prompts import build_triple_messages

        messages = build_triple_messages("한국어 패시지", ["에이전트", "그림자"])

        assert "에이전트" in messages[3]["content"]
        assert "그림자" in messages[3]["content"]
