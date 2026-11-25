"""Tests for token counting and text truncation."""

from reverie.parser.token_counter import count_tokens, truncate_tool_result


class TestCountTokens:
    """Test count_tokens function."""

    def test_basic_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_korean_text(self):
        tokens = count_tokens("안녕하세요. 테스트입니다.")
        assert tokens > 0

    def test_code_text(self):
        code = "def hello():\n    print('hello world')\n"
        tokens = count_tokens(code)
        assert tokens > 0


class TestTruncateToolResult:
    """Test truncate_tool_result function."""

    def test_short_result_unchanged(self):
        content = "line1\nline2\nline3"
        result = truncate_tool_result(content, max_lines=5)
        assert result == content

    def test_exact_limit_unchanged(self):
        """Content at exactly 2*max_lines should not be truncated."""
        lines = [f"line{i}" for i in range(10)]
        content = "\n".join(lines)
        result = truncate_tool_result(content, max_lines=5)
        assert result == content

    def test_long_result_truncated(self):
        lines = [f"line{i}" for i in range(100)]
        content = "\n".join(lines)
        result = truncate_tool_result(content, max_lines=5)

        assert "line0" in result
        assert "line4" in result
        assert "line99" in result
        assert "line95" in result
        assert "lines omitted" in result
        assert "line50" not in result

    def test_omission_count(self):
        lines = [f"line{i}" for i in range(50)]
        content = "\n".join(lines)
        result = truncate_tool_result(content, max_lines=10)
        assert "[30 lines omitted]" in result

    def test_empty_content(self):
        assert truncate_tool_result("") == ""
        assert truncate_tool_result("", max_lines=5) == ""

    def test_single_line(self):
        result = truncate_tool_result("single line", max_lines=5)
        assert result == "single line"
