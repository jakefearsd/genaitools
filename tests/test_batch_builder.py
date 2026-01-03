"""Tests for batch_builder.py"""

import json
import pytest
from pathlib import Path

from batch_builder import (
    topic_to_slug,
    topic_to_title_case,
    load_batch_config,
    merge_config,
    get_output_paths,
    build_outline_command,
    build_document_command,
    TopicConfig,
    TOPIC_DEFAULTS,
)


class TestTopicToSlug:
    """Tests for topic_to_slug function."""

    def test_basic_conversion(self):
        assert topic_to_slug("First AWS Deployment") == "first-aws-deployment"

    def test_multiple_spaces(self):
        assert topic_to_slug("AWS  EC2   Instance") == "aws-ec2-instance"

    def test_special_characters(self):
        assert topic_to_slug("What's New in Python 3.12?") == "whats-new-in-python-312"

    def test_underscores(self):
        assert topic_to_slug("docker_compose_tutorial") == "docker-compose-tutorial"

    def test_mixed_case(self):
        assert topic_to_slug("AWS RDS for Java") == "aws-rds-for-java"

    def test_leading_trailing_spaces(self):
        assert topic_to_slug("  Padded Topic  ") == "padded-topic"

    def test_consecutive_hyphens(self):
        assert topic_to_slug("Topic - With - Dashes") == "topic-with-dashes"


class TestTopicToTitleCase:
    """Tests for topic_to_title_case function."""

    def test_basic_conversion(self):
        assert topic_to_title_case("First AWS Deployment") == "FirstAwsDeployment"

    def test_with_hyphens(self):
        assert topic_to_title_case("docker-compose-tutorial") == "DockerComposeTutorial"

    def test_with_underscores(self):
        assert topic_to_title_case("docker_compose_tutorial") == "DockerComposeTutorial"

    def test_lowercase_input(self):
        assert topic_to_title_case("aws rds basics") == "AwsRdsBasics"

    def test_mixed_separators(self):
        assert topic_to_title_case("AWS-EC2 Instance_Types") == "AwsEc2InstanceTypes"


class TestLoadBatchConfig:
    """Tests for load_batch_config function."""

    def test_load_valid_config(self, tmp_path):
        config = {
            "topics": [{"topic": "Test Topic"}],
        }
        config_file = tmp_path / "batch.json"
        config_file.write_text(json.dumps(config))

        result = load_batch_config(str(config_file))
        assert result["topics"][0]["topic"] == "Test Topic"

    def test_missing_topics_raises(self, tmp_path):
        config = {"defaults": {}}
        config_file = tmp_path / "batch.json"
        config_file.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="missing required 'topics'"):
            load_batch_config(str(config_file))

    def test_empty_topics_raises(self, tmp_path):
        config = {"topics": []}
        config_file = tmp_path / "batch.json"
        config_file.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="'topics' array is empty"):
            load_batch_config(str(config_file))

    def test_topic_missing_topic_field_raises(self, tmp_path):
        config = {"topics": [{"context": "Some context"}]}
        config_file = tmp_path / "batch.json"
        config_file.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="missing required 'topic' field"):
            load_batch_config(str(config_file))

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_batch_config("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_path):
        config_file = tmp_path / "batch.json"
        config_file.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_batch_config(str(config_file))


class TestMergeConfig:
    """Tests for merge_config function."""

    def test_topic_only(self):
        result = merge_config({}, {"topic": "Test Topic"})
        assert result.topic == "Test Topic"
        assert result.words == TOPIC_DEFAULTS["words"]
        assert result.sections == TOPIC_DEFAULTS["sections"]

    def test_defaults_override_hardcoded(self):
        defaults = {"words": 10000, "sections": 12}
        result = merge_config(defaults, {"topic": "Test"})
        assert result.words == 10000
        assert result.sections == 12

    def test_topic_overrides_defaults(self):
        defaults = {"words": 10000}
        topic = {"topic": "Test", "words": 5000}
        result = merge_config(defaults, topic)
        assert result.words == 5000

    def test_all_fields_merged(self):
        defaults = {
            "audience": "Developers",
            "content_type": "guide",
            "model": "qwen3:30b",
        }
        topic = {
            "topic": "Test Topic",
            "context": "Focus on security",
            "persona": "Security expert",
        }
        result = merge_config(defaults, topic)

        assert result.topic == "Test Topic"
        assert result.context == "Focus on security"
        assert result.audience == "Developers"
        assert result.content_type == "guide"
        assert result.model == "qwen3:30b"
        assert result.persona == "Security expert"


class TestGetOutputPaths:
    """Tests for get_output_paths function."""

    def test_auto_generated_paths(self, tmp_path):
        config = TopicConfig(topic="First AWS Deployment")
        outline_path, document_path = get_output_paths(config, tmp_path)

        assert outline_path == tmp_path / "first-aws-deployment-outline.yaml"
        assert document_path == tmp_path / "FirstAwsDeployment.md"

    def test_explicit_outline_file(self, tmp_path):
        config = TopicConfig(topic="Test", outline_file="custom-outline.yaml")
        outline_path, document_path = get_output_paths(config, tmp_path)

        assert outline_path == tmp_path / "custom-outline.yaml"
        assert document_path == tmp_path / "Test.md"

    def test_explicit_document_file(self, tmp_path):
        config = TopicConfig(topic="Test", document_file="CustomDoc.md")
        outline_path, document_path = get_output_paths(config, tmp_path)

        assert outline_path == tmp_path / "test-outline.yaml"
        assert document_path == tmp_path / "CustomDoc.md"


class TestBuildOutlineCommand:
    """Tests for build_outline_command function."""

    def test_minimal_config(self, tmp_path):
        config = TopicConfig(topic="Test Topic")
        output_path = tmp_path / "test-outline.yaml"
        cmd = build_outline_command(config, output_path)

        assert "-t" in cmd
        assert "Test Topic" in cmd
        assert "-o" in cmd
        assert str(output_path) in cmd

    def test_with_persona(self, tmp_path):
        config = TopicConfig(topic="Test", persona="An expert")
        cmd = build_outline_command(config, tmp_path / "test.yaml")

        assert "-p" in cmd
        assert "An expert" in cmd

    def test_with_context(self, tmp_path):
        config = TopicConfig(topic="Test", context="Focus on X")
        cmd = build_outline_command(config, tmp_path / "test.yaml")

        assert "-c" in cmd
        assert "Focus on X" in cmd

    def test_deep_research_flag(self, tmp_path):
        config = TopicConfig(topic="Test", deep_research=True)
        cmd = build_outline_command(config, tmp_path / "test.yaml")

        assert "--deep-research" in cmd

    def test_no_think_flag(self, tmp_path):
        config = TopicConfig(topic="Test", think=False)
        cmd = build_outline_command(config, tmp_path / "test.yaml")

        assert "--no-think" in cmd
        assert "--think" not in cmd

    def test_num_gpu_flag(self, tmp_path):
        config = TopicConfig(topic="Test", num_gpu=20)
        cmd = build_outline_command(config, tmp_path / "test.yaml")

        assert "--num-gpu" in cmd
        assert "20" in cmd


class TestBuildDocumentCommand:
    """Tests for build_document_command function."""

    def test_minimal_config(self, tmp_path):
        config = TopicConfig(topic="Test")
        outline_path = tmp_path / "outline.yaml"
        output_path = tmp_path / "doc.md"
        cmd = build_document_command(config, outline_path, output_path)

        assert "-i" in cmd
        assert str(outline_path) in cmd
        assert "-o" in cmd
        assert str(output_path) in cmd

    def test_with_instructions(self, tmp_path):
        config = TopicConfig(topic="Test", document_instructions="Use formal tone")
        cmd = build_document_command(config, tmp_path / "o.yaml", tmp_path / "d.md")

        assert "--instructions" in cmd
        assert "Use formal tone" in cmd

    def test_verbose_flag(self, tmp_path):
        config = TopicConfig(topic="Test", verbose=True)
        cmd = build_document_command(config, tmp_path / "o.yaml", tmp_path / "d.md")

        assert "--verbose" in cmd

    def test_no_verbose_flag(self, tmp_path):
        config = TopicConfig(topic="Test", verbose=False)
        cmd = build_document_command(config, tmp_path / "o.yaml", tmp_path / "d.md")

        assert "--verbose" not in cmd
