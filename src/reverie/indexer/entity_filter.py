"""Entity filter for culling low-value entities from the knowledge graph.

Drops noise entities (timestamps, UUIDs, generic roles, XML fragments,
natural-language action descriptions) and normalizes entity names
for deduplication.
"""

import re
from urllib.parse import unquote, urlparse

from reverie.config import get_settings

# Patterns for entities that should be dropped
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:\d{2}|Z)?)?"
)
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# XML/HTML tag pattern
_XML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_-]*[^>]*>")

# URL pattern (http:// or https://)
_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)

# File path patterns: absolute (/...), relative (./...), home (~/..)
_FILE_PATH = re.compile(r"^(?:[/~]|\./)[\w./_~-]+$")

# Known prefixes to strip (longest first to prevent partial matches)
_KNOWN_PREFIXES = [
    "docker container",   # 16
    "systemd service",    # 15
    "discord server",     # 14
    "trello board",       # 12
    "github repo",        # 11
    "pm2 process",        # 11
    "npm package",        # 11
    "pip package",        # 11
]

# Generic role labels that are too vague to be useful entities
_GENERIC_ROLES = frozenset({
    "user", "assistant", "session", "system", "tool",
    "none", "null", "true", "false", "unknown",
    "result", "output", "input", "response", "request",
    "error", "success", "failure", "directory", "url",
})


def _extract_url_slug(text: str) -> str:
    """Extract the last meaningful path segment from a URL.

    Examples:
        https://trello.com/b/abc123/my-board → my-board
        https://github.com/user/my-project → my-project
        https://example.com/ → example.com
    """
    # Remove trailing slash
    url = text.rstrip("/")

    # Try to get last path segment
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path and path != "/":
            segment = path.rsplit("/", 1)[-1]
            # Percent-decode
            segment = unquote(segment)
            return segment
        else:
            # No path, use hostname
            return parsed.hostname or text
    except Exception:
        return text


def _extract_file_segment(text: str) -> str:
    """Extract the last meaningful segment from a file path.

    Examples:
        /home/user/projects/my-api-server/ → my-api-server
        ~/workspace/my-project/ → my-project
        ./src/config.py → config.py
    """
    path = text.rstrip("/")
    # Get the last segment
    segment = path.rsplit("/", 1)[-1]
    return segment if segment else text


def _strip_known_prefix(text: str) -> str:
    """Remove known prefixes like 'trello board', 'github repo', etc."""
    lower = text.lower()
    for prefix in _KNOWN_PREFIXES:
        if lower.startswith(prefix + " "):
            return text[len(prefix) + 1:]
    return text


def _replace_separators(text: str) -> str:
    """Replace hyphens and underscores with spaces."""
    return text.replace("-", " ").replace("_", " ")


def normalize_entity_name(entity_name: str) -> str:
    """Extended normalization pipeline for entity names.

    Steps (in order):
    1. URL slug extraction (if URL detected)
    2. File path extraction (if path detected)
    3. Known prefix removal
    4. Hyphen/underscore → space
    5. Lowercase + strip + collapse spaces

    Args:
        entity_name: Raw entity name string.

    Returns:
        Normalized entity name.
    """
    name = entity_name.strip()

    # Step 1: URL slug extraction
    if _URL.match(name):
        name = _extract_url_slug(name)

    # Step 2: File path extraction
    elif _FILE_PATH.match(name):
        name = _extract_file_segment(name)

    # Step 3: Known prefix removal
    name = _strip_known_prefix(name)

    # Step 4: Replace separators
    name = _replace_separators(name)

    # Step 5: Lowercase, strip, collapse spaces
    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)

    return name


class EntityFilter:
    """Filter and normalize entities for knowledge graph construction.

    Drop rules (in order):
    1. Contains a UUID pattern
    2. Matches an ISO timestamp pattern
    3. Is a generic role label (user, assistant, session)
    4. Is shorter than min_length characters
    5. Is longer than max_length characters
    6. Contains XML/HTML tags

    Normalization:
    - URL slug extraction
    - File path extraction
    - Known prefix removal
    - Hyphen/underscore → space
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    """

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
    ):
        """Initialize EntityFilter with configurable length limits.

        Args:
            min_length: Minimum entity length. Uses config default if None.
            max_length: Maximum entity length. Uses config default if None.
        """
        settings = get_settings()
        self._min_length = min_length if min_length is not None else settings.entity_min_length
        self._max_length = max_length if max_length is not None else settings.entity_max_length

    def should_drop(self, entity_name: str) -> bool:
        """Check if an entity should be excluded from the graph.

        Args:
            entity_name: Raw entity name string.

        Returns:
            True if the entity should be dropped.
        """
        stripped = entity_name.strip()

        # Too short
        if len(stripped) < self._min_length:
            return True

        # Too long (likely a natural-language description, not a name)
        if len(stripped) > self._max_length:
            return True

        # Generic role
        if stripped.lower() in _GENERIC_ROLES:
            return True

        # Contains UUID
        if _UUID.search(stripped):
            return True

        # Is a timestamp (full match after stripping)
        if _ISO_TIMESTAMP.fullmatch(stripped):
            return True

        # Contains XML/HTML tags
        if _XML_TAG.search(stripped):
            return True

        return False

    def normalize(self, entity_name: str) -> str:
        """Normalize an entity name using the extended pipeline.

        Args:
            entity_name: Raw entity name.

        Returns:
            Normalized entity name.
        """
        return normalize_entity_name(entity_name)

    def filter_entities(self, entities: list[str]) -> list[str]:
        """Filter and deduplicate a list of entities.

        1. Drop entities matching drop rules
        2. Normalize remaining entities
        3. Deduplicate (preserve first occurrence order)

        Args:
            entities: Raw entity name list.

        Returns:
            Filtered and normalized entity list.
        """
        seen: set[str] = set()
        result: list[str] = []

        for entity in entities:
            if self.should_drop(entity):
                continue
            normalized = self.normalize(entity)
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return result
