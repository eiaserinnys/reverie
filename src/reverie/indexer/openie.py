"""OpenIE module for Named Entity Recognition and triple extraction.

Two-stage extraction pipeline:
1. NER: Extract named entities from a session passage
2. Triple extraction: Build RDF triples using the entities

Based on HippoRAG's OpenIE design, adapted for Claude Code session data.
"""

import json
import logging
import re

from openai import OpenAI

from reverie.cache.api_cache import ApiCache
from reverie.config import get_settings
from reverie.indexer.prompts import build_ner_messages, build_query_ner_messages, build_triple_messages

logger = logging.getLogger(__name__)

# Regex patterns for extracting JSON from LLM responses
_NER_PATTERN = re.compile(
    r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
)
_TRIPLE_PATTERN = re.compile(
    r'\{[^{}]*"triples"\s*:\s*\[.*?\]\s*\}', re.DOTALL
)


def fix_broken_json(json_str: str) -> str:
    """Repair truncated JSON strings from LLM responses.

    When finish_reason is 'length', the JSON may be cut mid-generation.
    This function attempts to close unclosed brackets/braces.

    Algorithm:
    1. If valid JSON, return as-is
    2. Remove content after last comma (incomplete element)
    3. Close unclosed brackets/braces in reverse order
    """
    if not json_str:
        return json_str

    # Already valid?
    try:
        json.loads(json_str)
        return json_str
    except (json.JSONDecodeError, ValueError):
        pass

    # Remove trailing incomplete element (after last comma)
    last_comma = json_str.rfind(",")
    if last_comma > 0:
        json_str = json_str[:last_comma]

    # Find and close unclosed brackets/braces
    unclosed = _find_unclosed(json_str)
    closing_map = {"{": "}", "[": "]"}
    suffix = "".join(closing_map[c] for c in reversed(unclosed))
    repaired = json_str + suffix

    # Validate repair
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        # Repair failed, return original
        return json_str + suffix


def _find_unclosed(s: str) -> list[str]:
    """Find unclosed brackets and braces, ignoring those inside strings."""
    stack: list[str] = []
    inside_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if inside_string:
                escape_next = True
            continue
        if ch == '"':
            inside_string = not inside_string
            continue
        if inside_string:
            continue

        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    return stack


def _extract_ner(response_text: str) -> list[str]:
    """Extract named_entities list from LLM response text.

    Uses regex to find JSON block, then parses it.
    Returns deduplicated entity list preserving order.
    """
    match = _NER_PATTERN.search(response_text)
    if not match:
        return []

    try:
        data = json.loads(match.group())
        entities = data.get("named_entities", [])
        # Deduplicate preserving order
        return list(dict.fromkeys(entities))
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _extract_triples(response_text: str) -> list[list[str]]:
    """Extract triples list from LLM response text.

    Uses regex to find JSON block, then parses and validates.
    Filters out invalid triples (not exactly 3 elements) and duplicates.
    """
    match = _TRIPLE_PATTERN.search(response_text)
    if not match:
        return []

    try:
        data = json.loads(match.group())
        raw_triples = data.get("triples", [])
        return _filter_invalid_triples(raw_triples)
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _filter_invalid_triples(
    triples: list[list],
) -> list[list[str]]:
    """Filter out invalid and duplicate triples.

    A valid triple must have exactly 3 string elements.
    """
    unique: set[tuple[str, ...]] = set()
    valid: list[list[str]] = []

    for triple in triples:
        if len(triple) != 3:
            continue
        str_triple = [str(item) for item in triple]
        key = tuple(str_triple)
        if key not in unique:
            unique.add(key)
            valid.append(str_triple)

    return valid


class OpenIEClient:
    """Client for Open Information Extraction using OpenAI chat API.

    Two-stage extraction:
    1. extract_entities() - NER from passage text
    2. extract_triples() - RDF triples using NER results
    3. process_passage() - Combined pipeline (NER → triples)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: ApiCache | None = None,
    ):
        settings = get_settings()
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model or settings.openie_model
        self._query_ner_model = settings.query_ner_model  # Faster model for query NER
        self._temperature = (
            temperature if temperature is not None else settings.openie_temperature
        )
        self._max_tokens = max_tokens or settings.openie_max_tokens
        self._cache = cache or ApiCache(
            cache_dir=settings.cache_dir,
            enabled=settings.cache_enabled,
        )

    def extract_entities(self, passage: str) -> list[str]:
        """Extract named entities from a passage using NER prompt (indexing).

        Returns deduplicated list of entity strings.
        Uses full NER prompt with 15 few-shot examples (~4000 tokens).
        """
        messages = build_ner_messages(passage)
        response_text, finish_reason = self._call_chat(messages)

        if finish_reason == "length":
            response_text = fix_broken_json(response_text)

        entities = _extract_ner(response_text)
        if not entities:
            logger.debug("NER returned no entities for passage: %.80s...", passage)
        return entities

    def extract_query_entities(self, query: str) -> list[str]:
        """Extract named entities from a search query (retrieval).

        Returns deduplicated list of entity strings.
        Uses lightweight prompt with 5 few-shot examples (~400 tokens).
        Uses gpt-4o-mini for ~5x faster response than gpt-5-mini.
        """
        messages = build_query_ner_messages(query)
        response_text, finish_reason = self._call_chat(
            messages, model_override=self._query_ner_model
        )

        if finish_reason == "length":
            response_text = fix_broken_json(response_text)

        entities = _extract_ner(response_text)
        if not entities:
            logger.debug("Query NER returned no entities for: %.50s...", query)
        return entities

    def extract_triples(
        self, passage: str, entities: list[str]
    ) -> list[list[str]]:
        """Extract RDF triples from a passage using entity list.

        Returns list of [subject, predicate, object] triples.
        """
        messages = build_triple_messages(passage, entities)
        response_text, finish_reason = self._call_chat(messages)

        if finish_reason == "length":
            response_text = fix_broken_json(response_text)

        triples = _extract_triples(response_text)
        if not triples:
            logger.debug(
                "Triple extraction returned no triples for passage: %.80s...",
                passage,
            )
        return triples

    def process_passage(self, passage: str) -> dict:
        """Run full OpenIE pipeline: NER → triple extraction.

        Returns dict with 'entities' and 'triples' keys.
        Skips triple extraction if NER returns no entities.
        """
        entities = self.extract_entities(passage)

        if not entities:
            return {"entities": [], "triples": []}

        triples = self.extract_triples(passage, entities)
        return {"entities": entities, "triples": triples}

    def _call_chat(
        self, messages: list[dict], model_override: str | None = None
    ) -> tuple[str, str]:
        """Call OpenAI chat completion API with cache.

        Args:
            messages: Chat messages.
            model_override: Optional model to use instead of default.

        Returns (response_text, finish_reason).
        """
        model = model_override or self._model
        cache_request = {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        cached = self._cache.get("chat", cache_request)
        if cached is not None:
            return cached["text"], cached["finish_reason"]

        # gpt-5+ models use different API parameters:
        # - max_completion_tokens instead of max_tokens
        # - temperature only supports default value (1)
        if model.startswith("gpt-5"):
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=self._max_tokens,
            )
        else:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        choice = response.choices[0]
        text = choice.message.content
        finish_reason = choice.finish_reason

        self._cache.put("chat", cache_request, {
            "text": text,
            "finish_reason": finish_reason,
        })
        return text, finish_reason
