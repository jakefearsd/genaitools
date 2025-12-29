"""Tests for outline_builder.py"""

import json
import pytest
from unittest.mock import patch, MagicMock

from outline_builder import (
    build_outline_prompt,
    extract_json,
    build_yaml_outline,
    ensure_section_positions,
)


class TestBuildOutlinePrompt:
    """Tests for the build_outline_prompt function."""

    def test_basic_prompt(self):
        """Test basic prompt generation."""
        prompt = build_outline_prompt(
            topic="Python Basics",
            audience="beginners",
            content_type="tutorial",
            total_words=3000,
            num_sections=5,
        )

        assert "Python Basics" in prompt
        assert "beginners" in prompt
        assert "tutorial" in prompt
        assert "3000" in prompt
        assert "5" in prompt

    def test_prompt_with_persona(self):
        """Test prompt includes persona when provided."""
        prompt = build_outline_prompt(
            topic="Test Topic",
            audience="developers",
            content_type="guide",
            total_words=2000,
            num_sections=4,
            persona="a senior software architect",
        )

        assert "senior software architect" in prompt

    def test_prompt_with_guidance(self):
        """Test prompt includes guidance when provided."""
        prompt = build_outline_prompt(
            topic="Test Topic",
            audience="developers",
            content_type="guide",
            total_words=2000,
            num_sections=4,
            guidance="focus on security best practices",
        )

        assert "security best practices" in prompt

    def test_prompt_with_research(self):
        """Test prompt includes research context when provided."""
        research = "[1] Test Research\nSome content\nSource: https://example.com"
        prompt = build_outline_prompt(
            topic="Test Topic",
            audience="developers",
            content_type="guide",
            total_words=2000,
            num_sections=4,
            research_context=research,
        )

        assert "Test Research" in prompt
        assert "https://example.com" in prompt

    def test_prompt_requests_json_output(self):
        """Test prompt requests JSON output format."""
        prompt = build_outline_prompt(
            topic="Test",
            audience="test",
            content_type="tutorial",
            total_words=1000,
            num_sections=3,
        )

        assert "JSON" in prompt
        assert '"sections"' in prompt
        assert '"id"' in prompt

    def test_prompt_includes_position_fields(self):
        """Test prompt requests position and role fields."""
        prompt = build_outline_prompt(
            topic="Test",
            audience="test",
            content_type="tutorial",
            total_words=1000,
            num_sections=3,
        )

        assert "position" in prompt
        assert "section_role" in prompt
        assert "transition_to" in prompt
        assert "CRITICAL DOCUMENT STRUCTURE RULES" in prompt


class TestEnsureSectionPositions:
    """Tests for the ensure_section_positions function."""

    def test_empty_sections(self):
        """Empty sections list returns empty."""
        assert ensure_section_positions([]) == []

    def test_single_section(self):
        """Single section gets first position."""
        sections = [{"id": "intro", "order": 1}]
        result = ensure_section_positions(sections)

        assert result[0]["position"] == "first"
        assert result[0]["section_role"] == "introduce"
        assert result[0]["transition_to"] is None

    def test_two_sections(self):
        """Two sections: first and last positions."""
        sections = [
            {"id": "intro", "order": 1},
            {"id": "conclusion", "order": 2},
        ]
        result = ensure_section_positions(sections)

        assert result[0]["position"] == "first"
        assert result[0]["transition_to"] == "conclusion"
        assert result[1]["position"] == "last"
        assert result[1]["section_role"] == "conclude"
        assert result[1]["transition_to"] is None

    def test_multiple_sections(self):
        """Multiple sections get correct positions."""
        sections = [
            {"id": "intro", "order": 1},
            {"id": "middle-1", "order": 2},
            {"id": "middle-2", "order": 3},
            {"id": "conclusion", "order": 4},
        ]
        result = ensure_section_positions(sections)

        assert result[0]["position"] == "first"
        assert result[1]["position"] == "middle"
        assert result[2]["position"] == "middle"
        assert result[3]["position"] == "last"

    def test_middle_sections_cannot_conclude(self):
        """Middle sections with role=conclude get corrected to develop."""
        sections = [
            {"id": "intro", "order": 1, "section_role": "introduce"},
            {"id": "middle", "order": 2, "section_role": "conclude"},  # Wrong!
            {"id": "conclusion", "order": 3, "section_role": "conclude"},
        ]
        result = ensure_section_positions(sections)

        assert result[1]["section_role"] == "develop"  # Corrected
        assert result[2]["section_role"] == "conclude"  # Kept

    def test_transition_chain(self):
        """Transition_to forms a chain through all sections."""
        sections = [
            {"id": "a", "order": 1},
            {"id": "b", "order": 2},
            {"id": "c", "order": 3},
        ]
        result = ensure_section_positions(sections)

        assert result[0]["transition_to"] == "b"
        assert result[1]["transition_to"] == "c"
        assert result[2]["transition_to"] is None

    def test_sorts_by_order(self):
        """Sections are sorted by order before processing."""
        sections = [
            {"id": "c", "order": 3},
            {"id": "a", "order": 1},
            {"id": "b", "order": 2},
        ]
        result = ensure_section_positions(sections)

        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"
        assert result[2]["id"] == "c"


class TestExtractJson:
    """Tests for the extract_json function."""

    def test_extract_plain_json(self):
        """Test extracting plain JSON."""
        text = '{"title": "Test", "sections": []}'
        result = extract_json(text)

        assert result["title"] == "Test"
        assert result["sections"] == []

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the outline:

```json
{"title": "Test", "sections": [{"id": "intro"}]}
```

That's the outline.'''
        result = extract_json(text)

        assert result["title"] == "Test"
        assert len(result["sections"]) == 1

    def test_extract_json_from_code_block_no_lang(self):
        """Test extracting JSON from code block without language tag."""
        text = '''```
{"title": "Test"}
```'''
        result = extract_json(text)

        assert result["title"] == "Test"

    def test_extract_json_with_surrounding_text(self):
        """Test extracting JSON with surrounding text."""
        text = 'Some prefix text {"title": "Test"} some suffix text'
        result = extract_json(text)

        assert result["title"] == "Test"

    def test_extract_json_invalid_raises(self):
        """Test invalid JSON raises ValueError."""
        text = '{"title": "Test", sections: invalid}'

        with pytest.raises(ValueError, match="Invalid JSON"):
            extract_json(text)

    def test_extract_json_no_json_raises(self):
        """Test missing JSON raises ValueError."""
        text = "This is just plain text with no JSON"

        with pytest.raises(ValueError, match="No JSON found"):
            extract_json(text)


class TestBuildYamlOutline:
    """Tests for the build_yaml_outline function."""

    def test_basic_outline(self):
        """Test basic YAML outline structure."""
        json_data = {
            "title": "Test Document",
            "sections": [
                {"id": "intro", "title": "Introduction", "order": 1, "word_count": 400}
            ],
        }
        outline = build_yaml_outline(
            json_data=json_data,
            topic="Test Topic",
            audience="developers",
            content_type="tutorial",
            total_words=2000,
        )

        assert "metadata" in outline
        assert "research" in outline
        assert "sections" in outline
        assert outline["metadata"]["title"] == "Test Document"
        assert outline["metadata"]["topic"] == "Test Topic"

    def test_outline_includes_persona(self):
        """Test outline includes persona when provided."""
        json_data = {"title": "Test", "sections": []}
        outline = build_yaml_outline(
            json_data=json_data,
            topic="Test",
            audience="test",
            content_type="tutorial",
            total_words=1000,
            persona="a Python expert",
        )

        assert outline["metadata"]["persona"] == "a Python expert"

    def test_outline_includes_guidance(self):
        """Test outline includes guidance when provided."""
        json_data = {"title": "Test", "sections": []}
        outline = build_yaml_outline(
            json_data=json_data,
            topic="Test",
            audience="test",
            content_type="tutorial",
            total_words=1000,
            guidance="focus on examples",
        )

        assert outline["metadata"]["guidance"] == "focus on examples"

    def test_outline_includes_research(self):
        """Test outline includes research context."""
        json_data = {"title": "Test", "sections": []}
        research = "[1] Research result\nContent\nSource: https://example.com"
        outline = build_yaml_outline(
            json_data=json_data,
            topic="Test",
            audience="test",
            content_type="tutorial",
            total_words=1000,
            research_context=research,
        )

        assert outline["research"]["context"] == research

    def test_outline_has_timestamp(self):
        """Test outline includes generated_at timestamp."""
        json_data = {"title": "Test", "sections": []}
        outline = build_yaml_outline(
            json_data=json_data,
            topic="Test",
            audience="test",
            content_type="tutorial",
            total_words=1000,
        )

        assert "generated_at" in outline["metadata"]
        # Should be ISO format
        assert "T" in outline["metadata"]["generated_at"]


class TestOutlineBuilderIntegration:
    """Integration tests for outline_builder."""

    def test_full_outline_generation_mocked(self, use_ollama, mock_outline_json):
        """Test full outline generation with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": mock_outline_json}
        mock_response.raise_for_status = MagicMock()

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response):
            with patch("genaitools.research.DDGS", return_value=mock_ddgs):
                from outline_builder import build_outline_prompt, extract_json, build_yaml_outline
                from genaitools import generate, search_duckduckgo, build_research_context

                # Simulate the outline generation flow
                results = search_duckduckgo("Test Topic")
                research = build_research_context(results)

                prompt = build_outline_prompt(
                    topic="Test Topic",
                    audience="developers",
                    content_type="tutorial",
                    total_words=3000,
                    num_sections=3,
                    research_context=research,
                )

                response = generate(prompt)
                json_data = extract_json(response)
                outline = build_yaml_outline(
                    json_data=json_data,
                    topic="Test Topic",
                    audience="developers",
                    content_type="tutorial",
                    total_words=3000,
                    research_context=research,
                )

                assert len(outline["sections"]) == 3
                assert outline["sections"][0]["id"] == "introduction"
