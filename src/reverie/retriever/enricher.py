"""Enricher module for refining retrieval results into focused prompts.

Two-stage enrichment pipeline:
1. Stage 1: Split passages into 5-passage chunks, process with gpt-5-mini in parallel
2. Stage 2: Send all extracted facts to gpt-5.2 for re-ranking and filtering

The enricher:
- Takes query + retrieved passages as input
- Uses gpt-5-mini for fast parallel extraction
- Uses gpt-5.2 for final quality filtering
- Returns structured JSON with scored atomic facts
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openai import OpenAI

from reverie.cache.api_cache import ApiCache
from reverie.config import get_settings

logger = logging.getLogger(__name__)

# Stage 1: Parallel extraction (1 passage per call)
STAGE1_CHUNK_SIZE = 1


# Stage 1 system prompt - extract facts from single passage
STAGE1_SYSTEM = """\
Extract atomic facts from the passage. Return JSON only.

RULES:
1. Score 1-10 (1=barely relevant, 10=directly answers query)
2. One atomic fact per item - no combining
3. Write in passage language (Korean→Korean, English→English)
4. Include specific values: paths, commands, config values, IDs

Output JSON format:
{"facts": [
  {"score": 9, "text": "fact here"},
  ...
]}

If no relevant facts: {"facts": []}"""

STAGE1_USER_TEMPLATE = """\
Query: {query}

Passage:
{passage}

Extract relevant atomic facts as JSON."""


# Stage 2 system prompt - re-rank with full context
STAGE2_SYSTEM = """\
Re-evaluate extracted facts for relevance to the query.

TASK:
1. Score each fact 1-10 based on relevance (importance)
2. Rate confidence 1-10 based on source reliability and recency
3. Filter out facts with score < 5
4. Remove duplicates (keep higher-scored version)
5. Add brief reason explaining the relevance

Output JSON format:
{"facts": [
  {"score": 9, "confidence": 8, "text": "fact text", "passage_id": 3, "reason": "brief explanation"},
  ...
]}

Consider:
- Recent information (newer timestamps) may be more reliable
- Multiple sources increase confidence
- Be strict - only keep truly relevant facts."""

STAGE2_USER_TEMPLATE = """\
Query: {query}

Candidate facts (with passage ID, time, and initial score):
{facts}

Re-evaluate and filter. Return only relevant ones with score, confidence, and reasons."""


@dataclass
class EnrichedFact:
    """A single enriched fact with relevance score."""

    score: int
    text: str


@dataclass
class EnrichResult:
    """Result from the enrich pipeline."""

    query: str
    facts: list[EnrichedFact]
    passage_count: int
    elapsed_ms: float


class Enricher:
    """Two-stage enrichment pipeline.

    Stage 1: gpt-5-mini extracts facts from 5-passage chunks in parallel
    Stage 2: gpt-5.2 re-ranks and filters the combined results
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
        cache: ApiCache | None = None,
    ):
        settings = get_settings()
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._max_tokens = max_tokens
        self._cache = cache or ApiCache(
            cache_dir=settings.cache_dir,
            enabled=settings.cache_enabled,
        )

    def enrich(
        self,
        query: str,
        passages: list[dict],
    ) -> list[dict]:
        """Two-stage enrichment pipeline.

        Stage 1: Extract facts from passages in parallel (gpt-5-mini)
        Stage 2: Re-rank and filter facts (gpt-5.2)

        Args:
            query: The user's original query.
            passages: List of passage dicts with 'text', 'score',
                      'session_seq', 'turn_index' keys.

        Returns:
            List of dicts with 'score' (int 1-10), 'text', 'passage_ids', 'reason'.
        """
        if not passages:
            return []

        # Stage 1: Extract facts in parallel
        logger.info(f"Stage 1: Extracting facts from {len(passages)} passages")
        stage1_facts = self._stage1_extract(query, passages)
        logger.info(f"Stage 1 complete: {len(stage1_facts)} candidate facts")

        if not stage1_facts:
            return []

        # Stage 2: Re-rank and filter
        logger.info(f"Stage 2: Re-ranking {len(stage1_facts)} facts")
        final_facts = self._stage2_rerank(query, stage1_facts)
        logger.info(f"Stage 2 complete: {len(final_facts)} final facts")

        return final_facts

    def _stage1_extract(
        self,
        query: str,
        passages: list[dict],
    ) -> list[dict]:
        """Stage 1: Extract facts from each passage in parallel."""
        all_facts = []
        # Use as many workers as passages - no artificial limit
        with ThreadPoolExecutor(max_workers=len(passages)) as executor:
            futures = {
                executor.submit(self._extract_single, query, idx, p): idx
                for idx, p in enumerate(passages)
            }

            for future in as_completed(futures):
                try:
                    facts = future.result()
                    all_facts.extend(facts)
                except Exception as e:
                    logger.warning(f"Stage 1 extraction failed: {e}")

        return all_facts

    def _extract_single(
        self,
        query: str,
        passage_idx: int,
        passage: dict,
    ) -> list[dict]:
        """Extract facts from a single passage."""
        # Format single passage
        location = ""
        if passage.get("session_seq") is not None:
            location = f"S{passage['session_seq']}-T{passage.get('turn_index', '?')}"
        elif passage.get("session_id"):
            sid = passage["session_id"][:8]
            location = f"{sid}...T{passage.get('turn_index', '?')}"

        score = passage.get("score", 0)
        text = passage.get("text", "").strip()
        formatted = f"(score: {score:.4f}, {location})\n{text}"

        messages = [
            {"role": "system", "content": STAGE1_SYSTEM},
            {
                "role": "user",
                "content": STAGE1_USER_TEMPLATE.format(
                    query=query,
                    passage=formatted,
                ),
            },
        ]

        settings = get_settings()
        response_text = self._call_chat(messages, model=settings.enrich_stage1_model)
        facts = self._parse_stage1_response(response_text)

        # Attach passage metadata to each fact
        # Use actual passage_id from DB if available, otherwise use 1-based index
        actual_passage_id = passage.get("passage_id", passage_idx + 1)
        for f in facts:
            f["passage_id"] = actual_passage_id
            f["passage_time"] = passage.get("timestamp", "")
            f["session_seq"] = passage.get("session_seq")
            f["turn_index"] = passage.get("turn_index")

        return facts

    def _parse_stage1_response(self, response_text: str) -> list[dict]:
        """Parse Stage 1 JSON response."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

            data = json.loads(text)
            facts = data.get("facts", [])

            result = []
            for f in facts:
                if isinstance(f, dict) and "score" in f and "text" in f:
                    score = int(f["score"])
                    score = max(1, min(10, score))
                    result.append({
                        "score": score,
                        "text": str(f["text"]),
                    })
            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Stage 1 parse failed: {e}")
            return []

    def _stage2_rerank(
        self,
        query: str,
        facts: list[dict],
    ) -> list[dict]:
        """Stage 2: Re-rank and filter facts with gpt-5.2."""
        # Format facts with full metadata for re-ranking
        fact_lines = []
        for i, f in enumerate(facts):
            # Build time info
            time_info = ""
            if f.get("session_seq") is not None:
                time_info = f"S{f['session_seq']}-T{f.get('turn_index', '?')}"
            if f.get("passage_time"):
                time_info += f" ({f['passage_time']})"

            line = f"[{i+1}] passage:{f.get('passage_id', '?')} score:{f['score']} time:{time_info or 'unknown'}"
            line += f"\n    {f['text']}"
            fact_lines.append(line)

        facts_text = "\n".join(fact_lines)

        messages = [
            {"role": "system", "content": STAGE2_SYSTEM},
            {
                "role": "user",
                "content": STAGE2_USER_TEMPLATE.format(
                    query=query,
                    facts=facts_text,
                ),
            },
        ]

        settings = get_settings()
        response_text = self._call_chat(messages, model=settings.enrich_stage2_model)
        return self._parse_stage2_response(response_text)

    def _parse_stage2_response(self, response_text: str) -> list[dict]:
        """Parse Stage 2 JSON response with confidence scores."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

            data = json.loads(text)
            facts = data.get("facts", [])

            result = []
            for f in facts:
                if isinstance(f, dict) and "score" in f and "text" in f:
                    score = int(f["score"])
                    score = max(1, min(10, score))

                    confidence = int(f.get("confidence", 5))
                    confidence = max(1, min(10, confidence))

                    passage_id = f.get("passage_id")
                    if isinstance(passage_id, (int, float, str)):
                        passage_id = int(passage_id)
                    else:
                        passage_id = None

                    result.append({
                        "score": score,
                        "confidence": confidence,
                        "text": str(f["text"]),
                        "passage_id": passage_id,
                        "reason": str(f.get("reason", "")),
                    })

            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Stage 2 parse failed: {e}")
            logger.debug(f"Raw response: {response_text}")
            return []

    def _format_passages(self, passages: list[dict]) -> str:
        """Format passages for the enrich prompt."""
        lines = []
        for i, p in enumerate(passages, 1):
            location = ""
            if p.get("session_seq") is not None:
                location = f"S{p['session_seq']}-T{p.get('turn_index', '?')}"
            elif p.get("session_id"):
                sid = p["session_id"][:8]
                location = f"{sid}...T{p.get('turn_index', '?')}"

            score = p.get("score", 0)
            text = p.get("text", "").strip()

            lines.append(f"[{i}] (score: {score:.4f}, {location})")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def _call_chat(self, messages: list[dict], model: str) -> str:
        """Call OpenAI chat completion API with cache."""
        cache_request = {
            "model": model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "type": "enrich",
        }
        cached = self._cache.get("chat", cache_request)
        if cached is not None:
            return cached["text"]

        # gpt-5.x uses max_completion_tokens
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
                temperature=0.0,
                max_tokens=self._max_tokens,
            )

        text = response.choices[0].message.content or ""

        self._cache.put("chat", cache_request, {"text": text})
        return text


class EnricherWithProgress(Enricher):
    """Enricher with progress callback support for SSE streaming."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_callback = None
        self._progress_queue: list[dict] = []
        self._stage1_completed = 0
        self._stage1_total = 0

    def set_progress_callback(self, callback):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, stage: str, current: int, total: int, message: str = ""):
        """Report progress via callback."""
        if self._progress_callback:
            self._progress_callback(stage, current, total, message)

    def _stage1_extract(
        self,
        query: str,
        passages: list[dict],
    ) -> list[dict]:
        """Stage 1 with progress tracking."""
        self._stage1_total = len(passages)
        self._stage1_completed = 0
        self._report_progress("stage1", 0, self._stage1_total, "Starting extraction...")

        all_facts = []
        # Use as many workers as passages - no artificial limit
        with ThreadPoolExecutor(max_workers=len(passages)) as executor:
            futures = {
                executor.submit(self._extract_single_with_progress, query, idx, p): idx
                for idx, p in enumerate(passages)
            }

            for future in as_completed(futures):
                try:
                    facts = future.result()
                    all_facts.extend(facts)
                except Exception as e:
                    logger.warning(f"Stage 1 extraction failed: {e}")

        return all_facts

    def _extract_single_with_progress(
        self,
        query: str,
        passage_idx: int,
        passage: dict,
    ) -> list[dict]:
        """Extract from single passage and report progress."""
        facts = self._extract_single(query, passage_idx, passage)

        # Update progress
        self._stage1_completed += 1
        self._report_progress(
            "stage1",
            self._stage1_completed,
            self._stage1_total,
            f"Extracted from passage {passage_idx + 1}"
        )

        return facts

    def _stage2_rerank(
        self,
        query: str,
        facts: list[dict],
    ) -> list[dict]:
        """Stage 2 with progress tracking."""
        self._report_progress("stage2", 0, 1, "Re-ranking facts...")

        result = super()._stage2_rerank(query, facts)

        self._report_progress("stage2", 1, 1, "Complete")

        return result
