"""Prompt templates for OpenIE (NER + triple extraction).

Based on HippoRAG's prompt design, adapted for Claude Code session data.
Two-stage extraction: NER first, then triples using NER results.

NER prompt is loaded from external file for easier tuning.
"""

import re
from pathlib import Path

# =============================================================================
# NER (Named Entity Recognition) - Load from file
# =============================================================================

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_NER_PROMPT_FILE = _PROMPTS_DIR / "ner_prompt.txt"
_QUERY_NER_PROMPT_FILE = _PROMPTS_DIR / "query_ner_prompt.txt"


def _load_prompt_file(prompt_file: Path) -> tuple[str, list[tuple[str, str]], str]:
    """Load and parse a NER prompt from file.

    Returns:
        Tuple of (system_prompt, examples, user_template)
        where examples is a list of (input, output) tuples.
    """
    content = prompt_file.read_text(encoding="utf-8")

    # Split into sections
    parts = re.split(r"\n# (?=Few-Shot Examples|User Template)", content)

    # Parse system prompt (first section, remove header)
    system_section = parts[0]
    system_lines = system_section.split("\n")
    if system_lines and system_lines[0].startswith("#"):
        system_lines = system_lines[1:]
    system_prompt = "\n".join(system_lines).strip()

    # Parse examples section
    examples: list[tuple[str, str]] = []
    if len(parts) > 1:
        examples_section = parts[1]
        example_blocks = re.split(r"\n## Example \d+:", examples_section)
        for block in example_blocks[1:]:
            input_match = re.search(
                r"Input:\s*```\s*(.*?)\s*```", block, re.DOTALL
            )
            output_match = re.search(
                r"Output:\s*(\{.*?\})", block, re.DOTALL
            )
            if input_match and output_match:
                examples.append((input_match.group(1).strip(), output_match.group(1).strip()))

    # Parse user template section
    user_template = ""
    if len(parts) > 2:
        template_section = parts[2]
        template_lines = template_section.split("\n")
        if template_lines and "User Template" in template_lines[0]:
            template_lines = template_lines[1:]
        user_template = "\n".join(template_lines).strip()

    return system_prompt, examples, user_template


# Load NER prompt components (for indexing - detailed extraction)
_NER_SYSTEM, _NER_EXAMPLES, _NER_USER_TEMPLATE = _load_prompt_file(_NER_PROMPT_FILE)

# Load Query NER prompt (for retrieval - lightweight)
_QUERY_NER_SYSTEM, _QUERY_NER_EXAMPLES, _QUERY_NER_USER_TEMPLATE = _load_prompt_file(
    _QUERY_NER_PROMPT_FILE
)

# Public interface (for backward compatibility)
NER_SYSTEM = _NER_SYSTEM
NER_USER_TEMPLATE = _NER_USER_TEMPLATE if _NER_USER_TEMPLATE else """\
Extract named entities from the following text:

```
{passage}
```"""

QUERY_NER_SYSTEM = _QUERY_NER_SYSTEM
QUERY_NER_USER_TEMPLATE = _QUERY_NER_USER_TEMPLATE if _QUERY_NER_USER_TEMPLATE else """\
Extract named entities from the following search query:

```
{passage}
```"""


# =============================================================================
# Triple Extraction
# =============================================================================

TRIPLE_SYSTEM = (
    "Your task is to construct RDF (Resource Description Framework) triples "
    "from a Claude Code session log passage and its named entity list.\n\n"
    "Each triple is [subject, predicate, object] representing a relationship.\n\n"
    "Requirements:\n"
    "- Both subject and object MUST be named entities from the provided list.\n"
    "- Do NOT use 'user' or 'assistant' as subject or object — use specific entity names instead.\n"
    "- Resolve pronouns and roles to specific entity names.\n"
    "- Use concise, lowercase predicates (e.g., 'modified', 'installed', 'caused', 'depends on').\n"
    "- Focus on relationships between technical entities:\n"
    "  - file modifications: ['Bash', 'modified', 'src/config.py']\n"
    "  - dependencies: ['my-api-server', 'depends on', 'tiktoken']\n"
    "  - error occurrences: ['ImportError', 'occurred in', 'embedding.py']\n"
    "  - tool usage: ['Bash', 'executed', 'pytest']\n"
    "  - configuration: ['my-api-server', 'uses port', '8080']\n"
    "  - containment: ['parse_session', 'defined in', 'parser.py']\n\n"
    "Respond with a JSON object containing a single key 'triples' "
    "with a list of 3-element arrays."
)

TRIPLE_EXAMPLE_INPUT = """\
Passage:
```
[user] EntityFilter 클래스에 normalize 메서드 추가해줘.
[assistant] EntityFilter에 normalize 메서드를 추가하겠습니다.
[tool_use] Edit: ~/projects/my-api-server/src/filters/entity_filter.py
[tool_result] File updated successfully.
[assistant] EntityFilter 클래스에 normalize 메서드를 추가했습니다.
```

Named entities: ["EntityFilter", "my-api-server", "entity_filter.py"]"""

TRIPLE_EXAMPLE_OUTPUT = """\
{"triples": [["EntityFilter", "defined in", "entity_filter.py"], ["entity_filter.py", "part of", "my-api-server"]]}"""

TRIPLE_USER_TEMPLATE = """\
Extract RDF triples from the following passage and named entity list:

Passage:
```
{passage}
```

Named entities: {named_entities}"""


def build_ner_messages(passage: str) -> list[dict]:
    """Build chat messages for NER extraction (indexing).

    Uses few-shot prompting with examples loaded from external prompt file.
    Structure: system → [example_input, example_output]* → actual input.
    """
    messages: list[dict] = [{"role": "system", "content": NER_SYSTEM}]

    # Add few-shot examples from loaded prompt file
    for example_input, example_output in _NER_EXAMPLES:
        messages.append({"role": "user", "content": example_input})
        messages.append({"role": "assistant", "content": example_output})

    # Add the actual user input
    messages.append({"role": "user", "content": NER_USER_TEMPLATE.format(passage=passage)})

    return messages


def build_query_ner_messages(query: str) -> list[dict]:
    """Build chat messages for query NER extraction (retrieval).

    Uses a lightweight prompt optimized for search queries.
    ~10x fewer tokens than full NER prompt.
    """
    messages: list[dict] = [{"role": "system", "content": QUERY_NER_SYSTEM}]

    # Add few-shot examples
    for example_input, example_output in _QUERY_NER_EXAMPLES:
        messages.append({"role": "user", "content": example_input})
        messages.append({"role": "assistant", "content": example_output})

    # Add the actual user input
    messages.append({"role": "user", "content": QUERY_NER_USER_TEMPLATE.format(passage=query)})

    return messages


def build_triple_messages(
    passage: str, named_entities: list[str]
) -> list[dict]:
    """Build chat messages for triple extraction.

    Uses few-shot prompting with NER results injected.
    """
    import json

    entities_json = json.dumps(named_entities, ensure_ascii=False)
    return [
        {"role": "system", "content": TRIPLE_SYSTEM},
        {"role": "user", "content": TRIPLE_EXAMPLE_INPUT},
        {"role": "assistant", "content": TRIPLE_EXAMPLE_OUTPUT},
        {
            "role": "user",
            "content": TRIPLE_USER_TEMPLATE.format(
                passage=passage,
                named_entities=entities_json,
            ),
        },
    ]
