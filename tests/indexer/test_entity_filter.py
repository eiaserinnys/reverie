"""Tests for entity filter (culling low-value entities)."""

import pytest

from reverie.indexer.entity_filter import EntityFilter, normalize_entity_name


class TestEntityFilterDrop:
    """Test entity drop rules."""

    @pytest.fixture
    def filt(self):
        return EntityFilter()

    # Timestamps
    def test_drop_iso_timestamp(self, filt):
        assert filt.should_drop("2026-01-13T06:12:35.816000+00:00") is True

    def test_drop_iso_timestamp_short(self, filt):
        assert filt.should_drop("2026-02-01T16:30:04") is True

    def test_drop_date_only(self, filt):
        assert filt.should_drop("2026-01-13") is True

    # UUIDs
    def test_drop_uuid(self, filt):
        assert filt.should_drop("a2a04d0e-4a2c-4ba2-87e4-50a43cca6189") is True

    def test_drop_uuid_prefix(self, filt):
        """Session ID with prefix should still be dropped."""
        assert filt.should_drop("session a2a04d0e-4a2c-4ba2-87e4-50a43cca6189") is True

    # Generic roles
    def test_drop_user(self, filt):
        assert filt.should_drop("user") is True

    def test_drop_assistant(self, filt):
        assert filt.should_drop("assistant") is True

    def test_drop_session(self, filt):
        assert filt.should_drop("session") is True

    def test_drop_case_insensitive(self, filt):
        assert filt.should_drop("User") is True
        assert filt.should_drop("ASSISTANT") is True

    # Short entities
    def test_drop_single_char(self, filt):
        assert filt.should_drop("a") is True

    def test_drop_two_chars(self, filt):
        assert filt.should_drop("ab") is True

    # XML/HTML tags
    def test_drop_xml_tag(self, filt):
        assert filt.should_drop("<command-name>") is True

    def test_drop_xml_closing_tag(self, filt):
        assert filt.should_drop("</skill>") is True

    def test_drop_xml_with_content(self, filt):
        assert filt.should_drop("<command-message>clear</command-message>") is True

    # Extended generic roles
    def test_drop_none(self, filt):
        assert filt.should_drop("none") is True
        assert filt.should_drop("None") is True

    def test_drop_error(self, filt):
        assert filt.should_drop("error") is True

    def test_drop_result(self, filt):
        assert filt.should_drop("result") is True

    def test_drop_directory(self, filt):
        assert filt.should_drop("directory") is True

    def test_drop_url(self, filt):
        assert filt.should_drop("URL") is True

    # Too long (descriptions, not names)
    def test_drop_too_long(self, filt):
        assert filt.should_drop("a" * 81) is True

    def test_keep_max_length(self, filt):
        assert filt.should_drop("a" * 80) is False

    # Valid entities - should NOT be dropped
    def test_keep_agent_bot(self, filt):
        assert filt.should_drop("agent_bot.service") is False

    def test_keep_systemctl(self, filt):
        assert filt.should_drop("systemctl") is False

    def test_keep_pgvector(self, filt):
        assert filt.should_drop("pgvector") is False

    def test_keep_cache(self, filt):
        assert filt.should_drop("cache") is False

    def test_keep_moltbook(self, filt):
        assert filt.should_drop("moltbook-social-activity") is False

    def test_keep_sudo_required(self, filt):
        assert filt.should_drop("sudo required") is False

    def test_keep_bash(self, filt):
        assert filt.should_drop("Bash") is False

    def test_keep_three_chars(self, filt):
        assert filt.should_drop("npm") is False

    def test_keep_error_type(self, filt):
        """Specific error types should be kept."""
        assert filt.should_drop("ImportError") is False
        assert filt.should_drop("ConnectionRefusedError") is False


class TestEntityFilterNormalize:
    """Test entity normalization."""

    @pytest.fixture
    def filt(self):
        return EntityFilter()

    def test_lowercase(self, filt):
        assert filt.normalize("Bash") == "bash"

    def test_strip_whitespace(self, filt):
        assert filt.normalize("  hello  ") == "hello"

    def test_collapse_spaces(self, filt):
        assert filt.normalize("sudo  systemctl   restart") == "sudo systemctl restart"

    def test_already_normal(self, filt):
        assert filt.normalize("pgvector") == "pgvector"


class TestEntityFilterBatch:
    """Test batch filtering."""

    @pytest.fixture
    def filt(self):
        return EntityFilter()

    def test_filter_entities(self, filt):
        """Filter a mixed list of entities."""
        entities = [
            "agent_bot.service",
            "2026-01-13T06:12:35.816000+00:00",
            "user",
            "systemctl",
            "a2a04d0e-4a2c-4ba2-87e4-50a43cca6189",
            "sudo required",
            "ab",
        ]
        result = filt.filter_entities(entities)
        # Note: underscores are now converted to spaces by extended normalization
        assert result == ["agent bot.service", "systemctl", "sudo required"]

    def test_filter_deduplicates_after_normalize(self, filt):
        """Normalization may cause duplicates that should be merged."""
        entities = ["Bash", "bash", "BASH"]
        result = filt.filter_entities(entities)
        assert result == ["bash"]


# =============================================================================
# Extended normalization rules (Phase 1)
# =============================================================================


class TestNormalizeUrlSlug:
    """URL slug extraction: URL → last path segment, decoded."""

    def test_trello_url(self):
        result = normalize_entity_name(
            "https://trello.com/b/Ab12CdEf/예시-작업-보드"
        )
        assert result == "예시 작업 보드"

    def test_github_url(self):
        result = normalize_entity_name(
            "https://github.com/example-user/example-repo"
        )
        assert result == "example repo"

    def test_url_with_trailing_slash(self):
        result = normalize_entity_name(
            "https://example.com/my-project/"
        )
        assert result == "my project"

    def test_url_with_percent_encoding(self):
        result = normalize_entity_name(
            "https://example.com/path/%ED%85%8C%EC%8A%A4%ED%8A%B8"
        )
        assert result == "테스트"

    def test_url_root_only(self):
        """URL with no meaningful path segment."""
        result = normalize_entity_name("https://example.com/")
        assert result == "example.com"

    def test_url_root_no_slash(self):
        result = normalize_entity_name("https://example.com")
        assert result == "example.com"


class TestNormalizeHyphenUnderscore:
    """Hyphens and underscores should become spaces."""

    def test_hyphens_to_spaces(self):
        result = normalize_entity_name("예시-작업-보드")
        assert result == "예시 작업 보드"

    def test_underscores_to_spaces(self):
        result = normalize_entity_name("agent_bot_service")
        assert result == "agent bot service"

    def test_mixed_separators(self):
        result = normalize_entity_name("my-cool_project")
        assert result == "my cool project"

    def test_already_spaces(self):
        result = normalize_entity_name("hello world")
        assert result == "hello world"


class TestNormalizePrefixRemoval:
    """Known prefixes should be stripped."""

    def test_trello_board_prefix(self):
        result = normalize_entity_name("trello board 예시 작업 보드")
        assert result == "예시 작업 보드"

    def test_github_repo_prefix(self):
        result = normalize_entity_name("github repo sample-app")
        assert result == "sample app"

    def test_discord_server_prefix(self):
        result = normalize_entity_name("discord server 연구실")
        assert result == "연구실"

    def test_pm2_process_prefix(self):
        result = normalize_entity_name("pm2 process sample-app")
        assert result == "sample app"

    def test_prefix_case_insensitive(self):
        result = normalize_entity_name("Trello Board 내 보드")
        assert result == "내 보드"

    def test_no_prefix_match(self):
        """Should not strip when not a known prefix."""
        result = normalize_entity_name("some random text")
        assert result == "some random text"


class TestNormalizeFilePath:
    """File paths should extract the last meaningful segment."""

    def test_absolute_path(self):
        result = normalize_entity_name("/home/testuser/projects/my-app/")
        assert result == "my app"

    def test_path_no_trailing_slash(self):
        result = normalize_entity_name("/home/testuser/projects/my-app")
        assert result == "my app"

    def test_path_with_extension(self):
        result = normalize_entity_name("/home/testuser/projects/entity_filter.py")
        assert result == "entity filter.py"

    def test_relative_path(self):
        result = normalize_entity_name("./memory/src/config.py")
        assert result == "config.py"

    def test_tilde_path(self):
        result = normalize_entity_name("~/agent-workspace/shadow/")
        assert result == "shadow"


class TestNormalizeCombined:
    """Combined normalization: multiple rules applied together."""

    def test_url_with_hyphens(self):
        """URL slug extraction + hyphen-to-space."""
        result = normalize_entity_name(
            "https://trello.com/b/Ab12CdEf/예시-작업-보드"
        )
        assert result == "예시 작업 보드"

    def test_prefix_with_hyphens(self):
        """Prefix removal + hyphen-to-space."""
        result = normalize_entity_name("github repo sample-app")
        assert result == "sample app"

    def test_entity_filter_dedup_with_normalization(self):
        """EntityFilter should deduplicate after extended normalization."""
        filt = EntityFilter()
        entities = [
            "my-app",
            "my_app",
            "https://github.com/testuser/my-app",
            "/home/testuser/projects/my-app/",
        ]
        result = filt.filter_entities(entities)
        assert result == ["my app"]

    def test_entity_filter_dedup_trello(self):
        """Trello URL and board name should deduplicate."""
        filt = EntityFilter()
        entities = [
            "https://trello.com/b/Ab12CdEf/예시-작업-보드",
            "trello board 예시 작업 보드",
            "예시-작업-보드",
        ]
        result = filt.filter_entities(entities)
        assert result == ["예시 작업 보드"]
