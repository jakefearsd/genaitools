"""Tests for link_builder.py and genaitools/embeddings.py"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from genaitools.embeddings import (
    cosine_similarity,
    find_similar_pairs,
    get_embedding,
)

from link_builder import (
    extract_title,
    extract_headings,
    extract_existing_links,
    load_documents,
    build_link_prompt,
    parse_link_response,
    insert_link,
    is_inside_link,
    is_inside_heading,
    compute_relative_path,
    Document,
    LinkSuggestion,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors have high similarity."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.1, 2.1, 3.1]
        sim = cosine_similarity(vec_a, vec_b)
        assert sim > 0.99

    def test_different_lengths_raises(self):
        """Vectors of different lengths raise ValueError."""
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity(vec_a, vec_b)

    def test_empty_vectors_raises(self):
        """Empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="empty vectors"):
            cosine_similarity([], [])

    def test_zero_vector(self):
        """Zero vector returns 0.0 similarity."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0


class TestFindSimilarPairs:
    """Tests for find_similar_pairs function."""

    def test_empty_list(self):
        """Empty list returns empty pairs."""
        assert find_similar_pairs([]) == []

    def test_single_document(self):
        """Single document returns empty pairs."""
        docs = [{"embedding": [1.0, 2.0, 3.0]}]
        assert find_similar_pairs(docs) == []

    def test_similar_pair_above_threshold(self):
        """Similar pair above threshold is returned."""
        docs = [
            {"embedding": [1.0, 0.0, 0.0]},
            {"embedding": [0.9, 0.1, 0.0]},  # Very similar
        ]
        pairs = find_similar_pairs(docs, threshold=0.9)
        assert len(pairs) == 1
        assert pairs[0][0] == 0
        assert pairs[0][1] == 1
        assert pairs[0][2] > 0.9

    def test_dissimilar_pair_below_threshold(self):
        """Dissimilar pair below threshold is not returned."""
        docs = [
            {"embedding": [1.0, 0.0, 0.0]},
            {"embedding": [0.0, 1.0, 0.0]},  # Orthogonal
        ]
        pairs = find_similar_pairs(docs, threshold=0.5)
        assert len(pairs) == 0

    def test_sorted_by_similarity(self):
        """Pairs are sorted by similarity descending."""
        docs = [
            {"embedding": [1.0, 0.0, 0.0]},
            {"embedding": [0.9, 0.1, 0.0]},  # Most similar to 0
            {"embedding": [0.5, 0.5, 0.0]},  # Less similar
        ]
        pairs = find_similar_pairs(docs, threshold=0.0)
        assert len(pairs) == 3
        # First pair should have highest similarity
        assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]


class TestGetEmbedding:
    """Tests for get_embedding function."""

    def test_get_embedding_mocked(self, use_ollama):
        """Test embedding generation with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": mock_embedding}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.embeddings.requests.post", return_value=mock_response):
            result = get_embedding("test text", model="test-model")
            assert result == mock_embedding

    def test_get_embedding_error_handling(self, use_ollama):
        """Test embedding handles errors gracefully."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {}  # No embedding key
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.embeddings.requests.post", return_value=mock_response):
            with pytest.raises(ValueError, match="No embedding"):
                get_embedding("test text")


class TestExtractTitle:
    """Tests for extract_title function."""

    def test_extract_h1_title(self):
        """Extract title from H1 heading."""
        content = "# My Document Title\n\nSome content here."
        assert extract_title(content, "file.md") == "My Document Title"

    def test_extract_h1_with_leading_content(self):
        """Extract first H1 even with leading content."""
        content = "Some intro\n\n# The Real Title\n\nContent."
        assert extract_title(content, "file.md") == "The Real Title"

    def test_fallback_to_filename(self):
        """Use filename when no H1 found."""
        content = "## Not an H1\n\nSome content."
        assert extract_title(content, "my-document.md") == "My Document"

    def test_filename_with_underscores(self):
        """Handle filename with underscores."""
        content = "No headings here"
        assert extract_title(content, "my_cool_doc.md") == "My Cool Doc"


class TestExtractHeadings:
    """Tests for extract_headings function."""

    def test_extract_all_heading_levels(self):
        """Extract headings at all levels."""
        content = """# H1
## H2
### H3
#### H4
##### H5
###### H6"""
        headings = extract_headings(content)
        assert headings == ["H1", "H2", "H3", "H4", "H5", "H6"]

    def test_no_headings(self):
        """No headings returns empty list."""
        content = "Just plain text here."
        assert extract_headings(content) == []

    def test_headings_with_content(self):
        """Extract headings mixed with content."""
        content = """# Introduction

Some intro text.

## Background

More text here.

## Conclusion

Final thoughts."""
        headings = extract_headings(content)
        assert headings == ["Introduction", "Background", "Conclusion"]


class TestExtractExistingLinks:
    """Tests for extract_existing_links function."""

    def test_extract_relative_links(self):
        """Extract relative path links."""
        content = "See [this doc](other.md) and [another](../folder/doc.md)"
        links = extract_existing_links(content)
        assert "other.md" in links
        assert "doc.md" in links

    def test_ignore_http_links(self):
        """Ignore HTTP(S) links."""
        content = "Visit [Google](https://google.com) or [site](http://example.com)"
        links = extract_existing_links(content)
        assert len(links) == 0

    def test_ignore_anchor_links(self):
        """Ignore internal anchor links."""
        content = "Jump to [section](#my-section)"
        links = extract_existing_links(content)
        assert len(links) == 0

    def test_no_links(self):
        """No links returns empty set."""
        content = "Plain text without any links."
        links = extract_existing_links(content)
        assert len(links) == 0


class TestLoadDocuments:
    """Tests for load_documents function."""

    def test_load_single_file(self, tmp_path):
        """Load a single markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nContent here.")

        docs = load_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0].title == "Test"
        assert "Content here" in docs[0].content

    def test_load_multiple_files(self, tmp_path):
        """Load multiple markdown files."""
        (tmp_path / "doc1.md").write_text("# Doc One\n\nFirst doc.")
        (tmp_path / "doc2.md").write_text("# Doc Two\n\nSecond doc.")

        docs = load_documents(tmp_path)
        assert len(docs) == 2
        titles = {d.title for d in docs}
        assert titles == {"Doc One", "Doc Two"}

    def test_ignore_non_markdown(self, tmp_path):
        """Ignore non-markdown files."""
        (tmp_path / "doc.md").write_text("# Markdown")
        (tmp_path / "readme.txt").write_text("Not markdown")

        docs = load_documents(tmp_path)
        assert len(docs) == 1

    def test_recursive_loading(self, tmp_path):
        """Load recursively when flag set."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.md").write_text("# Root")
        (subdir / "nested.md").write_text("# Nested")

        docs = load_documents(tmp_path, recursive=False)
        assert len(docs) == 1

        docs = load_documents(tmp_path, recursive=True)
        assert len(docs) == 2


class TestBuildLinkPrompt:
    """Tests for build_link_prompt function."""

    def test_prompt_contains_source_info(self):
        """Prompt includes source document info."""
        source = Document(
            path=Path("source.md"),
            title="Source Doc",
            content="Source content here.",
            headings=["Heading 1"],
        )
        target = Document(
            path=Path("target.md"),
            title="Target Doc",
            content="Target content here.",
            headings=["Target Heading"],
        )

        prompt = build_link_prompt(source, target)

        assert "Source Doc" in prompt
        assert "Source content here" in prompt
        assert "Target Doc" in prompt
        assert "JSON" in prompt


class TestParseLinkResponse:
    """Tests for parse_link_response function."""

    def test_parse_clean_json(self):
        """Parse clean JSON response."""
        response = '{"should_link": true, "anchor_text": "test", "confidence": 0.8}'
        result = parse_link_response(response)

        assert result["should_link"] is True
        assert result["anchor_text"] == "test"
        assert result["confidence"] == 0.8

    def test_parse_json_in_code_block(self):
        """Parse JSON in markdown code block."""
        response = """Here's my analysis:

```json
{"should_link": true, "anchor_text": "test"}
```"""
        result = parse_link_response(response)

        assert result["should_link"] is True

    def test_parse_json_with_text(self):
        """Parse JSON embedded in text."""
        response = 'I think {"should_link": false, "reasoning": "no match"} is right.'
        result = parse_link_response(response)

        assert result["should_link"] is False

    def test_invalid_json_returns_none(self):
        """Invalid JSON returns None."""
        response = "This is not JSON at all."
        result = parse_link_response(response)

        assert result is None


class TestInsertLink:
    """Tests for insert_link function."""

    def test_simple_insertion(self):
        """Insert link for simple text."""
        content = "Learn about Docker containers here."
        result = insert_link(content, "Docker containers", "docker.md")

        assert result == "Learn about [Docker containers](docker.md) here."

    def test_anchor_not_found(self):
        """Return None if anchor text not found."""
        content = "No matching text here."
        result = insert_link(content, "missing text", "target.md")

        assert result is None

    def test_skip_already_linked(self):
        """Skip text already inside a link."""
        content = "See [Docker containers](other.md) for more."
        result = insert_link(content, "Docker containers", "target.md")

        assert result is None


class TestIsInsideLink:
    """Tests for is_inside_link function."""

    def test_inside_link_text(self):
        """Detect position inside link text."""
        content = "See [this link](url.md) here."
        pos = content.index("this link")
        assert is_inside_link(content, pos) is True

    def test_outside_link(self):
        """Detect position outside link."""
        content = "Normal text [link](url.md) more text."
        pos = content.index("Normal")
        assert is_inside_link(content, pos) is False


class TestIsInsideHeading:
    """Tests for is_inside_heading function."""

    def test_inside_heading(self):
        """Detect position inside heading."""
        content = "# Heading Text\n\nParagraph."
        pos = content.index("Heading")
        assert is_inside_heading(content, pos) is True

    def test_outside_heading(self):
        """Detect position outside heading."""
        content = "# Heading\n\nParagraph text."
        pos = content.index("Paragraph")
        assert is_inside_heading(content, pos) is False


class TestComputeRelativePath:
    """Tests for compute_relative_path function."""

    def test_same_directory(self):
        """Files in same directory."""
        source = Path("/docs/a.md")
        target = Path("/docs/b.md")
        assert compute_relative_path(source, target) == "b.md"

    def test_subdirectory(self):
        """Target in subdirectory."""
        source = Path("/docs/a.md")
        target = Path("/docs/sub/b.md")
        assert compute_relative_path(source, target) == "sub/b.md"

    def test_parent_directory(self):
        """Target in parent directory."""
        source = Path("/docs/sub/a.md")
        target = Path("/docs/b.md")
        assert compute_relative_path(source, target) == "../b.md"


class TestLinkBuilderIntegration:
    """Integration tests for link_builder."""

    def test_full_workflow_mocked(self, use_ollama, tmp_path):
        """Test full workflow with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        # Create test files
        (tmp_path / "docker.md").write_text(
            "# Docker Basics\n\nLearn about containers and images."
        )
        (tmp_path / "kubernetes.md").write_text(
            "# Kubernetes Guide\n\nOrchestrate your Docker containers at scale."
        )

        docs = load_documents(tmp_path)
        assert len(docs) == 2

        # Verify documents loaded correctly
        titles = {d.title for d in docs}
        assert "Docker Basics" in titles
        assert "Kubernetes Guide" in titles
