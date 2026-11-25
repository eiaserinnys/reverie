#!/usr/bin/env python3
"""Evaluate NER prompt against testcases.

Usage:
    python evaluate_prompt.py [prompt_file] [--verbose]

Compares a NER prompt against testcases and reports:
- Token count (prompt length)
- Precision/Recall per testcase
- Overall F1 score
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openai import OpenAI
from reverie.config import get_settings


def load_prompt(prompt_file: Path) -> tuple[str, list[tuple[str, str]], str]:
    """Load and parse NER prompt from file.

    Returns: (system_prompt, examples, user_template)
    """
    content = prompt_file.read_text(encoding="utf-8")

    # Split into sections
    parts = re.split(r"\n# (?=Few-Shot Examples|User Template)", content)

    # Parse system prompt
    system_section = parts[0]
    system_lines = system_section.split("\n")
    if system_lines and system_lines[0].startswith("# "):
        system_lines = system_lines[1:]
    system_prompt = "\n".join(system_lines).strip()

    # Parse examples
    examples: list[tuple[str, str]] = []
    if len(parts) > 1:
        examples_section = parts[1]
        example_blocks = re.split(r"\n## Example \d+:", examples_section)
        for block in example_blocks[1:]:
            input_match = re.search(r"Input:\s*```\s*(.*?)\s*```", block, re.DOTALL)
            output_match = re.search(r"Output:\s*(\{.*?\})", block, re.DOTALL)
            if input_match and output_match:
                examples.append((input_match.group(1).strip(), output_match.group(1).strip()))

    # Parse user template
    user_template = "Extract named entities:\n\n```\n{passage}\n```"
    if len(parts) > 2:
        template_section = parts[2]
        template_lines = template_section.split("\n")
        if template_lines and "User Template" in template_lines[0]:
            template_lines = template_lines[1:]
        user_template = "\n".join(template_lines).strip()

    return system_prompt, examples, user_template


def build_messages(system: str, examples: list[tuple[str, str]], user_template: str, query: str) -> list[dict]:
    """Build chat messages for NER."""
    messages = [{"role": "system", "content": system}]

    for ex_input, ex_output in examples:
        messages.append({"role": "user", "content": ex_input})
        messages.append({"role": "assistant", "content": ex_output})

    messages.append({"role": "user", "content": user_template.format(passage=query)})
    return messages


def count_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Estimate token count for messages."""
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        total += len(enc.encode(msg["content"]))
        total += 4  # role overhead
    total += 2  # message framing
    return total


def extract_entities(response_text: str) -> list[str]:
    """Extract entities from LLM response."""
    pattern = re.compile(r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}')
    match = pattern.search(response_text)
    if not match:
        return []

    try:
        data = json.loads(match.group())
        return data.get("named_entities", [])
    except (json.JSONDecodeError, ValueError):
        return []


def compute_metrics(expected: list[str], actual: list[str]) -> dict:
    """Compute precision, recall, F1."""
    expected_set = set(e.lower() for e in expected)
    actual_set = set(a.lower() for a in actual)

    # Handle empty expected case (e.g., greetings)
    if not expected_set and not actual_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hits": 0, "misses": 0, "extra": 0}

    if not actual_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits": 0, "misses": len(expected_set), "extra": 0}

    hits = len(expected_set & actual_set)
    precision = hits / len(actual_set) if actual_set else 0.0
    recall = hits / len(expected_set) if expected_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits": hits,
        "misses": len(expected_set) - hits,
        "extra": len(actual_set) - hits,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER prompt")
    parser.add_argument("prompt_file", nargs="?", default="query_ner_prompt.txt",
                        help="Prompt file to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--dry-run", action="store_true", help="Don't call API, just show token count")
    args = parser.parse_args()

    # Load prompt
    prompt_path = Path(__file__).parent / args.prompt_file
    if not prompt_path.exists():
        print(f"Error: {prompt_path} not found")
        sys.exit(1)

    system, examples, user_template = load_prompt(prompt_path)

    # Load testcases
    testcases_path = Path(__file__).parent / "testcases.json"
    with open(testcases_path) as f:
        testcases = json.load(f)["testcases"]

    # Token count for a sample query
    sample_messages = build_messages(system, examples, user_template, "에이전트")
    token_count = count_tokens(sample_messages)

    print(f"\n=== Prompt: {args.prompt_file} ===")
    print(f"Few-shot examples: {len(examples)}")
    print(f"Estimated tokens (sample query): {token_count}")
    print(f"System prompt length: {len(system)} chars")

    if args.dry_run:
        print("\n[Dry run - skipping API calls]")
        return

    # Initialize OpenAI client
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.openie_model

    print(f"\nModel: {model}")
    print(f"Running {len(testcases)} testcases...\n")

    results = []
    total_f1 = 0.0

    for tc in testcases:
        query = tc["query"]
        expected = tc["expected_entities"]

        messages = build_messages(system, examples, user_template, query)

        try:
            if model.startswith("gpt-5"):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=512,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )

            actual = extract_entities(response.choices[0].message.content)
        except Exception as e:
            print(f"Error on '{query}': {e}")
            actual = []

        metrics = compute_metrics(expected, actual)
        total_f1 += metrics["f1"]
        results.append({"query": query, "expected": expected, "actual": actual, "metrics": metrics})

        if args.verbose:
            status = "✓" if metrics["f1"] >= 0.8 else "✗"
            print(f"{status} [{metrics['f1']:.2f}] {query}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            if metrics["extra"] > 0:
                extra = set(a.lower() for a in actual) - set(e.lower() for e in expected)
                print(f"   Extra:    {list(extra)}")
            print()

    # Summary
    avg_f1 = total_f1 / len(testcases)
    passed = sum(1 for r in results if r["metrics"]["f1"] >= 0.8)

    print(f"\n=== Summary ===")
    print(f"Passed: {passed}/{len(testcases)} (F1 >= 0.8)")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Token count: {token_count}")

    # Find failures
    failures = [r for r in results if r["metrics"]["f1"] < 0.8]
    if failures and not args.verbose:
        print(f"\nFailed cases:")
        for f in failures:
            print(f"  - {f['query']} (F1: {f['metrics']['f1']:.2f})")


if __name__ == "__main__":
    main()
