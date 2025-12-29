"""Tests for document_builder.py"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import yaml

from document_builder import (
    load_outline,
    build_section_prompt,
    build_key_points_prompt,
    extract_key_points,
    generate_section,
    assemble_document,
    build_smoothing_prompt,
    extract_smoothed_parts,
    smooth_transitions,
    SectionResult,
)


class TestLoadOutline:
    """Tests for the load_outline function."""

    def test_load_valid_outline(self, temp_outline_file):
        """Test loading a valid outline file."""
        outline = load_outline(str(temp_outline_file))

        assert "metadata" in outline
        assert "sections" in outline
        assert len(outline["sections"]) == 3

    def test_load_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_outline("/nonexistent/path/outline.yaml")

    def test_load_missing_metadata(self, tmp_path):
        """Test loading outline without metadata raises error."""
        outline_path = tmp_path / "bad_outline.yaml"
        with open(outline_path, "w") as f:
            yaml.dump({"sections": []}, f)

        with pytest.raises(ValueError, match="missing required key.*metadata"):
            load_outline(str(outline_path))

    def test_load_missing_sections(self, tmp_path):
        """Test loading outline without sections raises error."""
        outline_path = tmp_path / "bad_outline.yaml"
        with open(outline_path, "w") as f:
            yaml.dump({"metadata": {}}, f)

        with pytest.raises(ValueError, match="missing required key.*sections"):
            load_outline(str(outline_path))

    def test_load_empty_sections(self, tmp_path):
        """Test loading outline with empty sections raises error."""
        outline_path = tmp_path / "bad_outline.yaml"
        with open(outline_path, "w") as f:
            yaml.dump({"metadata": {}, "sections": []}, f)

        with pytest.raises(ValueError, match="no sections"):
            load_outline(str(outline_path))

    def test_load_section_missing_fields(self, tmp_path):
        """Test loading outline with incomplete section raises error."""
        outline_path = tmp_path / "bad_outline.yaml"
        with open(outline_path, "w") as f:
            yaml.dump({
                "metadata": {},
                "sections": [{"id": "test"}]  # Missing title, order, word_count
            }, f)

        with pytest.raises(ValueError, match="Section missing required key"):
            load_outline(str(outline_path))


class TestBuildSectionPrompt:
    """Tests for the build_section_prompt function."""

    def test_basic_section_prompt(self, sample_outline):
        """Test basic section prompt generation."""
        section = sample_outline["sections"][0]
        metadata = sample_outline["metadata"]

        prompt = build_section_prompt(
            section=section,
            metadata=metadata,
            research_context="Some research",
            previous_key_points={},
        )

        assert section["title"] in prompt
        assert metadata["topic"] in prompt
        assert str(section["word_count"]) in prompt

    def test_section_prompt_with_dependencies(self, sample_outline):
        """Test section prompt includes key points from dependencies."""
        section = sample_outline["sections"][2]  # conclusion with dependencies
        metadata = sample_outline["metadata"]

        previous_key_points = {
            "introduction": ["Key point 1 from intro", "Key point 2 from intro"],
            "main-content": ["Key point from main"],
        }

        prompt = build_section_prompt(
            section=section,
            metadata=metadata,
            research_context="",
            previous_key_points=previous_key_points,
        )

        assert "Key point 1 from intro" in prompt
        assert "Key point from main" in prompt
        assert "PRIOR SECTION KEY POINTS" in prompt

    def test_section_prompt_with_persona(self, sample_outline):
        """Test section prompt includes persona."""
        section = sample_outline["sections"][0]
        metadata = sample_outline["metadata"].copy()
        metadata["persona"] = "a Python expert with 10 years experience"

        prompt = build_section_prompt(
            section=section,
            metadata=metadata,
            research_context="",
            previous_key_points={},
        )

        assert "Python expert" in prompt

    def test_section_prompt_with_guidance(self, sample_outline):
        """Test section prompt includes guidance."""
        section = sample_outline["sections"][0]
        section = section.copy()
        section["guidance"] = "Focus on practical examples"
        section["content_hints"] = ["Include code samples", "Show real-world usage"]

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context="",
            previous_key_points={},
        )

        assert "practical examples" in prompt
        assert "code samples" in prompt

    def test_section_prompt_with_research(self, sample_outline):
        """Test section prompt includes research context."""
        section = sample_outline["sections"][0]
        research = "[1] Research Title\nResearch content\nSource: https://example.com"

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context=research,
            previous_key_points={},
        )

        assert "Research Title" in prompt
        assert "RESEARCH CONTEXT" in prompt

    def test_section_prompt_with_instructions(self, sample_outline):
        """Test section prompt includes additional instructions."""
        section = sample_outline["sections"][0]

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context="",
            previous_key_points={},
            instructions="Do not include code examples. Focus on high-level concepts only.",
        )

        assert "OUTPUT INSTRUCTIONS" in prompt
        assert "Do not include code examples" in prompt
        assert "high-level concepts" in prompt

    def test_section_prompt_without_instructions(self, sample_outline):
        """Test section prompt omits OUTPUT INSTRUCTIONS when not provided."""
        section = sample_outline["sections"][0]

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context="",
            previous_key_points={},
        )

        assert "OUTPUT INSTRUCTIONS" not in prompt


class TestBuildKeyPointsPrompt:
    """Tests for the build_key_points_prompt function."""

    def test_key_points_prompt_structure(self):
        """Test key points prompt has correct structure."""
        prompt = build_key_points_prompt(
            section_title="Introduction",
            section_content="This is the section content about testing.",
        )

        assert "Introduction" in prompt
        assert "section content about testing" in prompt
        assert "5-10 key points" in prompt
        assert "bullet" in prompt.lower()

    def test_key_points_prompt_includes_examples(self):
        """Test key points prompt includes good/bad examples."""
        prompt = build_key_points_prompt(
            section_title="Test",
            section_content="Content",
        )

        assert "WRONG:" in prompt
        assert "RIGHT:" in prompt


class TestExtractKeyPoints:
    """Tests for the extract_key_points function."""

    def test_extract_key_points_mocked(self, use_ollama, mock_key_points_response):
        """Test key points extraction with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": mock_key_points_response}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response):
            points = extract_key_points(
                section_title="Test Section",
                section_content="Test content",
                ollama_url="http://localhost:11434",
                model="test-model",
            )

            assert len(points) > 0
            assert any("Topic X" in point for point in points)

    def test_extract_key_points_handles_error(self, use_ollama):
        """Test key points extraction handles errors gracefully."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Error: Connection refused"}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response):
            points = extract_key_points(
                section_title="Test",
                section_content="Content",
                ollama_url="http://localhost:11434",
                model="test-model",
            )

            # Should return empty list on error, not raise
            assert points == []

    def test_extract_key_points_parses_bullets(self, use_ollama):
        """Test key points extraction parses different bullet formats."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        bullet_response = """- Point one with dash
- Point two with dash
* Point three with asterisk
* Point four with asterisk"""

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": bullet_response}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response):
            points = extract_key_points(
                section_title="Test",
                section_content="Content",
                ollama_url="http://localhost:11434",
                model="test-model",
            )

            assert len(points) == 4
            assert "Point one with dash" in points
            assert "Point three with asterisk" in points


class TestGenerateSection:
    """Tests for the generate_section function."""

    def test_generate_section_mocked(self, use_ollama, sample_outline):
        """Test section generation with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        section_content = """## Introduction

This is the introduction section with some content.

### Overview

Here is an overview of the topic.
"""
        key_points_response = """- Introduction covers the basics
- Overview section included
- Topic is well explained"""

        call_count = [0]

        def mock_generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            if call_count[0] == 1:
                mock_resp.json.return_value = {"response": section_content}
            else:
                mock_resp.json.return_value = {"response": key_points_response}
            return mock_resp

        with patch("genaitools.ollama_client.requests.post", side_effect=mock_generate_side_effect):
            section = sample_outline["sections"][0]
            metadata = sample_outline["metadata"]

            result = generate_section(
                section=section,
                metadata=metadata,
                research_context="",
                previous_key_points={},
                ollama_url="http://localhost:11434",
                model="test-model",
                temperature=0.7,
                think=False,
                verbose=False,
            )

            assert isinstance(result, SectionResult)
            assert result.id == "introduction"
            assert "Introduction" in result.content
            assert result.word_count > 0
            assert len(result.key_points) > 0


class TestAssembleDocument:
    """Tests for the assemble_document function."""

    def test_assemble_single_section(self):
        """Test assembling document with single section."""
        results = [
            SectionResult(
                id="intro",
                title="Introduction",
                content="## Introduction\n\nThis is the intro.",
                word_count=5,
                key_points=["Point 1"],
            )
        ]
        metadata = {"title": "Test Document"}

        document = assemble_document(results, metadata)

        assert "# Test Document" in document
        assert "## Introduction" in document
        assert "This is the intro" in document

    def test_assemble_multiple_sections(self):
        """Test assembling document with multiple sections."""
        results = [
            SectionResult(
                id="intro",
                title="Introduction",
                content="## Introduction\n\nIntro content.",
                word_count=2,
                key_points=[],
            ),
            SectionResult(
                id="main",
                title="Main Content",
                content="## Main Content\n\nMain content here.",
                word_count=3,
                key_points=[],
            ),
            SectionResult(
                id="conclusion",
                title="Conclusion",
                content="## Conclusion\n\nConclusion content.",
                word_count=2,
                key_points=[],
            ),
        ]
        metadata = {"title": "Multi-Section Document"}

        document = assemble_document(results, metadata)

        assert "# Multi-Section Document" in document
        assert "## Introduction" in document
        assert "## Main Content" in document
        assert "## Conclusion" in document
        # Sections should appear in order
        intro_pos = document.find("Introduction")
        main_pos = document.find("Main Content")
        conclusion_pos = document.find("Conclusion")
        assert intro_pos < main_pos < conclusion_pos

    def test_assemble_uses_metadata_title(self):
        """Test document uses title from metadata."""
        results = [
            SectionResult(
                id="test",
                title="Test",
                content="Content",
                word_count=1,
                key_points=[],
            )
        ]
        metadata = {"title": "Custom Document Title"}

        document = assemble_document(results, metadata)

        assert "# Custom Document Title" in document


class TestDocumentBuilderIntegration:
    """Integration tests for document_builder."""

    def test_full_document_generation_mocked(self, use_ollama, temp_outline_file):
        """Test full document generation with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        section_responses = [
            "## Introduction\n\nThis is the introduction.",
            "- Key point from intro",
            "## Main Content\n\nThis is the main content.",
            "- Key point from main",
            "## Conclusion\n\nThis is the conclusion.",
            "- Key point from conclusion",
        ]
        response_idx = [0]

        def mock_post(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"response": section_responses[response_idx[0]]}
            response_idx[0] = min(response_idx[0] + 1, len(section_responses) - 1)
            return mock_resp

        with patch("genaitools.ollama_client.requests.post", side_effect=mock_post):
            outline = load_outline(str(temp_outline_file))
            metadata = outline["metadata"]
            sections = sorted(outline["sections"], key=lambda s: s["order"])
            research_context = outline.get("research", {}).get("context", "")

            results = []
            key_points = {}

            for section in sections:
                result = generate_section(
                    section=section,
                    metadata=metadata,
                    research_context=research_context,
                    previous_key_points=key_points,
                    ollama_url="http://localhost:11434",
                    model="test-model",
                    temperature=0.7,
                    think=False,
                    verbose=False,
                )
                results.append(result)
                key_points[result.id] = result.key_points

            document = assemble_document(results, metadata)

            assert "# Test Document" in document
            assert "Introduction" in document
            assert "Main Content" in document
            assert "Conclusion" in document


class TestBuildSmoothingPrompt:
    """Tests for the build_smoothing_prompt function."""

    def test_smoothing_prompt_structure(self):
        """Test smoothing prompt has correct structure."""
        prompt = build_smoothing_prompt(
            section_a_title="Introduction",
            section_a_ending="This concludes our introduction to the topic.",
            section_b_title="Core Concepts",
            section_b_beginning="In this section, we will explore the core concepts.",
        )

        assert "Introduction" in prompt
        assert "Core Concepts" in prompt
        assert "concludes our introduction" in prompt
        assert "explore the core concepts" in prompt
        assert "===SECTION_A_END===" in prompt
        assert "===SECTION_B_START===" in prompt

    def test_smoothing_prompt_includes_instructions(self):
        """Test smoothing prompt includes rewriting instructions."""
        prompt = build_smoothing_prompt(
            section_a_title="Section A",
            section_a_ending="Ending text.",
            section_b_title="Section B",
            section_b_beginning="Beginning text.",
        )

        assert "transition" in prompt.lower()
        assert "flow" in prompt.lower()


class TestExtractSmoothedParts:
    """Tests for the extract_smoothed_parts function."""

    def test_extract_valid_format(self):
        """Test extraction from properly formatted response."""
        response = """Here is the improved transition:

===SECTION_A_END===
The improved ending text that flows better.
===SECTION_B_START===
The improved beginning that connects naturally.
===END==="""

        result = extract_smoothed_parts(response)

        assert result is not None
        a_end, b_start = result
        assert "improved ending" in a_end
        assert "connects naturally" in b_start

    def test_extract_missing_markers(self):
        """Test extraction returns None when markers are missing."""
        response = "This response doesn't have the required markers."

        result = extract_smoothed_parts(response)

        assert result is None

    def test_extract_partial_markers(self):
        """Test extraction returns None with only one marker."""
        response = """===SECTION_A_END===
Some text here but no section B marker."""

        result = extract_smoothed_parts(response)

        assert result is None

    def test_extract_strips_whitespace(self):
        """Test that extracted parts have whitespace stripped."""
        response = """===SECTION_A_END===

   Text with extra whitespace

===SECTION_B_START===

   More whitespace here

===END==="""

        result = extract_smoothed_parts(response)

        assert result is not None
        a_end, b_start = result
        assert a_end == "Text with extra whitespace"
        assert b_start == "More whitespace here"


class TestSmoothTransitions:
    """Tests for the smooth_transitions function."""

    def test_smooth_transitions_single_section(self, use_ollama):
        """Test smoothing with single section returns unchanged."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        results = [
            SectionResult(
                id="intro",
                title="Introduction",
                content="## Introduction\n\nThis is the intro content.",
                word_count=5,
                key_points=["Point 1"],
            )
        ]

        # Single section should return unchanged (no transitions to smooth)
        smoothed = smooth_transitions(
            results=results,
            ollama_url="http://localhost:11434",
            model="test-model",
            verbose=False,
        )

        assert len(smoothed) == 1
        assert smoothed[0].content == results[0].content

    def test_smooth_transitions_mocked(self, use_ollama):
        """Test smoothing with mocked Ollama response."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        results = [
            SectionResult(
                id="intro",
                title="Introduction",
                content="## Introduction\n\nIntro content here. In conclusion, this section covered the basics.",
                word_count=10,
                key_points=[],
            ),
            SectionResult(
                id="main",
                title="Main Content",
                content="## Main Content\n\nNow let's dive into the main content.",
                word_count=8,
                key_points=[],
            ),
        ]

        smoothed_response = """===SECTION_A_END===
Intro content here. Now that we understand the basics, let's explore further.
===SECTION_B_START===
Building on our introduction, let's dive into the main content.
===END==="""

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": smoothed_response}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response):
            smoothed = smooth_transitions(
                results=results,
                ollama_url="http://localhost:11434",
                model="test-model",
                verbose=False,
            )

            assert len(smoothed) == 2
            # Content should be modified by smoothing
            assert "In conclusion" not in smoothed[0].content
            assert "Building on" in smoothed[1].content


class TestPositionAwarePrompts:
    """Tests for position-aware prompt generation."""

    def test_first_section_prompt(self, sample_outline):
        """Test first section gets introduction instructions."""
        section = sample_outline["sections"][0].copy()
        section["position"] = "first"
        section["section_role"] = "introduce"

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context="",
            previous_key_points={},
        )

        assert "INTRODUCTION" in prompt
        assert "Do NOT include a conclusion" in prompt

    def test_middle_section_prompt(self, sample_outline):
        """Test middle section gets anti-conclusion instructions."""
        section = sample_outline["sections"][1].copy()
        section["position"] = "middle"
        section["section_role"] = "develop"
        section["transition_to"] = "conclusion"

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context="",
            previous_key_points={},
        )

        assert "MIDDLE section" in prompt
        assert "Do NOT include a conclusion" in prompt
        assert "Do NOT use phrases like" in prompt
        assert "BRIDGE" in prompt
        assert "'conclusion'" in prompt  # transition_to hint

    def test_last_section_prompt(self, sample_outline):
        """Test last section gets conclusion permission."""
        section = sample_outline["sections"][2].copy()
        section["position"] = "last"
        section["section_role"] = "conclude"

        prompt = build_section_prompt(
            section=section,
            metadata=sample_outline["metadata"],
            research_context="",
            previous_key_points={},
        )

        assert "FINAL section" in prompt
        assert "SHOULD include a conclusion" in prompt
        assert "Tie together themes" in prompt
