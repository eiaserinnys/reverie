"""Session preprocessor for filtering and refining turn-level text.

Filters out low-value content (greetings, reference data) and refines
valuable passages before OpenIE extraction.

Pipeline:
    raw turn text → preprocess → (keep/drop) → OpenIE

Design decisions:
- Uses Anthropic Claude (Sonnet) for better Korean understanding
- Separate from OpenIE to allow independent tuning
- Caches results via ApiCache for replay without API cost
"""

import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError

from reverie.cache.api_cache import ApiCache
from reverie.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt templates
# =============================================================================

PREPROCESS_SYSTEM = """\
당신은 세션 기록 전처리기입니다. Claude Code 세션의 턴(turn) 단위 기록을 분석하여
지식 그래프에 보존할 가치가 있는지 판단하고, 보존할 경우 노이즈를 제거하여 정제합니다.

## 판단 기준

### 보존 대상 (action: "keep")
- 실수 발견 + 수정 기록 (무엇을, 왜 착각했는지)
- 아키텍처 결정과 설계 의도 (왜 이 방식을 선택했는지)
- 새로운 기능 구현 내역
- 디버깅 과정과 해결책
- 에이전트/그림자만의 유니크한 경험 (특정 프로젝트 맥락)

### 드롭 대상 (action: "drop")
- URL, 보드 ID, 카드 ID 등 참조 데이터만으로 이루어진 턴
- 인사, 잡담 ("안녕", "고마워", "ㅋㅋ" 등)
- 의미 있는 결과 없이 실행만 한 단순 툴 호출 (git status, ls 등)
- LLM이 이미 아는 자명한 개념 설명 (Python 문법, HTTP 메서드 설명 등)
- 에러 로그나 스택 트레이스만 있고 분석/해결이 없는 경우

### 정제 시 (action: "keep"인 경우)
- ID, 해시값, 긴 로그 출력, UUID 등 노이즈 제거
- 핵심 의미만 보존
- 원래 텍스트의 맥락과 의도는 유지
- 한국어와 영어 혼용 허용

## 출력 형식

반드시 다음 JSON 형식으로만 응답하세요:

```json
{
  "action": "keep" 또는 "drop",
  "reason": "드롭 사유 (drop일 때만, keep이면 null)",
  "refined_text": "정제된 텍스트 (keep일 때만, drop이면 null)"
}
```"""

PREPROCESS_EXAMPLE_INPUT = """\
[user] 안녕!
[assistant] 안녕하세요! 무엇을 도와드릴까요?"""

PREPROCESS_EXAMPLE_OUTPUT = """\
{"action": "drop", "reason": "Greeting/chitchat only", "refined_text": null}"""

PREPROCESS_EXAMPLE_INPUT_2 = """\
[user] 왜 PostgreSQL을 선택했어?
[assistant] SQLite 대신 PostgreSQL을 선택한 이유:
1. 동시 접근 안정성 (인덱싱 + 검색이 동시에 발생)
2. pgvector 확장으로 임베딩 KNN 검색을 DB 레벨에서 처리
3. 노트앱와 백업/복구 통합 관리 가능"""

PREPROCESS_EXAMPLE_OUTPUT_2 = """\
{"action": "keep", "reason": null, "refined_text": "SQLite 대신 PostgreSQL 선택 이유: 동시 접근 안정성, pgvector 확장으로 임베딩 KNN 검색 가능, 노트앱와 백업 통합 관리."}"""

PREPROCESS_USER_TEMPLATE = """\
다음 세션 턴을 분석하고 JSON 형식으로 응답하세요:

```
{passage}
```"""


def build_preprocess_messages(passage: str) -> list[dict]:
    """Build chat messages for preprocessing.

    Uses few-shot prompting: system → examples → actual input.
    """
    return [
        {"role": "system", "content": PREPROCESS_SYSTEM},
        {"role": "user", "content": PREPROCESS_EXAMPLE_INPUT},
        {"role": "assistant", "content": PREPROCESS_EXAMPLE_OUTPUT},
        {"role": "user", "content": PREPROCESS_EXAMPLE_INPUT_2},
        {"role": "assistant", "content": PREPROCESS_EXAMPLE_OUTPUT_2},
        {"role": "user", "content": PREPROCESS_USER_TEMPLATE.format(passage=passage)},
    ]


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class PreprocessResult:
    """Result of preprocessing a single turn/passage.

    Attributes:
        action: "keep" to retain, "drop" to discard
        reason: Why it was dropped (only for drop)
        refined_text: Cleaned text (only for keep)
    """

    action: str  # "keep" or "drop"
    reason: str | None
    refined_text: str | None


# =============================================================================
# Preprocessor client
# =============================================================================

# Regex pattern for extracting JSON from LLM response
_PREPROCESS_PATTERN = re.compile(
    r'\{[^{}]*"action"\s*:\s*"(?:keep|drop)"[^{}]*\}', re.DOTALL
)


def _extract_preprocess_result(response_text: str) -> dict | None:
    """Extract preprocessing result JSON from LLM response.

    Returns parsed dict or None if extraction fails.
    """
    match = _PREPROCESS_PATTERN.search(response_text)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None


class PreprocessorClient:
    """Client for session turn preprocessing using LLM.

    Determines whether a turn should be kept or dropped,
    and refines the text if kept.

    Uses Anthropic Claude via OpenAI-compatible API (OpenRouter or direct).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        cache: ApiCache | None = None,
    ):
        settings = get_settings()
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            # For Anthropic models via OpenRouter or similar
            # base_url can be set via OPENAI_BASE_URL env var if needed
        )
        self._model = model or settings.preprocessor_model
        self._temperature = (
            temperature if temperature is not None else settings.preprocessor_temperature
        )
        self._max_tokens = max_tokens or settings.preprocessor_max_tokens
        self._max_retries = max_retries
        self._cache = cache or ApiCache(
            cache_dir=settings.cache_dir,
            enabled=settings.cache_enabled,
        )

    def preprocess_turn(self, text: str) -> PreprocessResult:
        """Preprocess a single turn/passage.

        Args:
            text: Raw turn text to analyze

        Returns:
            PreprocessResult with action, reason, and refined_text
        """
        # Handle empty input
        if not text or not text.strip():
            return PreprocessResult(
                action="drop",
                reason="Empty passage",
                refined_text=None,
            )

        # Call LLM
        messages = build_preprocess_messages(text)
        response_text = self._call_chat(messages)

        # Parse response
        parsed = _extract_preprocess_result(response_text)

        if parsed is None:
            # Parse failure - keep original as fallback
            logger.warning("Failed to parse preprocessor response, keeping original")
            return PreprocessResult(
                action="keep",
                reason=None,
                refined_text=text,
            )

        action = parsed.get("action", "keep")

        # Validate action
        if action not in ("keep", "drop"):
            logger.warning("Unknown action '%s', defaulting to keep", action)
            action = "keep"

        if action == "drop":
            return PreprocessResult(
                action="drop",
                reason=parsed.get("reason"),
                refined_text=None,
            )

        # action == "keep"
        refined = parsed.get("refined_text")
        if not refined:
            # If no refined text provided, use original
            refined = text

        return PreprocessResult(
            action="keep",
            reason=None,
            refined_text=refined,
        )

    def _call_chat(self, messages: list[dict]) -> str:
        """Call chat completion API with cache and retry on rate limit.

        Returns response text.
        """
        cache_request = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        cached = self._cache.get("preprocess", cache_request)
        if cached is not None:
            return cached["text"]

        for attempt in range(self._max_retries + 1):
            try:
                # gpt-5+ models use different API parameters:
                # - max_completion_tokens instead of max_tokens
                # - temperature only supports default value (1)
                if self._model.startswith("gpt-5"):
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_completion_tokens=self._max_tokens,
                    )
                else:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                break
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limited, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, self._max_retries,
                    )
                    time.sleep(wait)
                else:
                    raise

        choice = response.choices[0]
        text = choice.message.content
        finish_reason = choice.finish_reason

        self._cache.put("preprocess", cache_request, {
            "text": text,
            "finish_reason": finish_reason,
        })
        return text
