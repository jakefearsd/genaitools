"""Tests for genaitools/deep_research.py"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from genaitools.deep_research import (
    fetch_page,
    html_to_text,
    extract_title,
    build_summarize_prompt,
    summarize_page,
    deep_research,
    get_cache_path,
    load_from_cache,
    save_to_cache,
    CACHE_DIR,
)


class TestFetchPage:
    """Tests for fetch_page function."""

    def test_fetch_success(self, use_ollama):
        """Test successful page fetch."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.deep_research.requests.get", return_value=mock_response):
            result = fetch_page("https://example.com")
            assert result == "<html><body>Test content</body></html>"

    def test_fetch_failure(self, use_ollama):
        """Test fetch returns None on failure."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        import requests
        with patch("genaitools.deep_research.requests.get", side_effect=requests.exceptions.Timeout):
            result = fetch_page("https://example.com")
            assert result is None


class TestHtmlToText:
    """Tests for html_to_text function."""

    def test_basic_html(self):
        """Extract text from basic HTML."""
        html = "<html><body><p>Hello world</p></body></html>"
        title, text = html_to_text(html)
        assert "Hello world" in text

    def test_removes_scripts(self):
        """Scripts are removed from output."""
        html = "<html><body><script>alert('bad')</script><p>Good content</p></body></html>"
        title, text = html_to_text(html)
        assert "alert" not in text
        assert "Good content" in text

    def test_removes_styles(self):
        """Styles are removed from output."""
        html = "<html><body><style>body{color:red}</style><p>Content</p></body></html>"
        title, text = html_to_text(html)
        assert "color" not in text
        assert "Content" in text

    def test_removes_nav(self):
        """Navigation elements are removed."""
        html = "<html><body><nav>Menu items</nav><p>Main content</p></body></html>"
        title, text = html_to_text(html)
        assert "Menu" not in text
        assert "Main content" in text

    def test_extracts_title_from_tag(self):
        """Title is extracted from <title> tag."""
        html = "<html><head><title>Page Title</title></head><body>Content</body></html>"
        title, text = html_to_text(html)
        assert title == "Page Title"

    def test_extracts_title_from_h1(self):
        """Title falls back to H1 if no <title>."""
        html = "<html><body><h1>Heading Title</h1><p>Content</p></body></html>"
        title, text = html_to_text(html)
        assert title == "Heading Title"

    def test_max_chars(self):
        """Text is limited to max_chars."""
        html = "<html><body>" + "x" * 10000 + "</body></html>"
        title, text = html_to_text(html, max_chars=100)
        assert len(text) <= 103  # 100 + "..."

    def test_collapses_whitespace(self):
        """Multiple newlines are collapsed."""
        html = "<html><body><p>Line 1</p>\n\n\n\n<p>Line 2</p></body></html>"
        title, text = html_to_text(html)
        assert "\n\n\n" not in text


class TestExtractTitle:
    """Tests for extract_title function."""

    def test_from_title_tag(self):
        """Extract from <title> tag."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<html><head><title>My Title</title></head></html>", "html.parser")
        assert extract_title(soup) == "My Title"

    def test_from_h1_fallback(self):
        """Fall back to H1 when no title tag."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<html><body><h1>H1 Title</h1></body></html>", "html.parser")
        assert extract_title(soup) == "H1 Title"

    def test_untitled_fallback(self):
        """Return 'Untitled' when no title or h1."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<html><body><p>Just text</p></body></html>", "html.parser")
        assert extract_title(soup) == "Untitled"


class TestBuildSummarizePrompt:
    """Tests for build_summarize_prompt function."""

    def test_includes_topic(self):
        """Prompt includes the research topic."""
        prompt = build_summarize_prompt("content", "https://example.com", "Docker")
        assert "Docker" in prompt

    def test_includes_url(self):
        """Prompt includes the source URL."""
        prompt = build_summarize_prompt("content", "https://example.com/article", "topic")
        assert "https://example.com/article" in prompt

    def test_includes_content(self):
        """Prompt includes the page content."""
        prompt = build_summarize_prompt("This is the page content", "https://example.com", "topic")
        assert "This is the page content" in prompt


class TestSummarizePage:
    """Tests for summarize_page function."""

    def test_summarize_mocked(self, use_ollama):
        """Test summarization with mocked Ollama."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        with patch("genaitools.deep_research.generate") as mock_gen:
            mock_gen.return_value = "This is a summary of the page."
            result = summarize_page(
                text="Page content here",
                url="https://example.com",
                topic="Docker",
                model="test-model",
                ollama_url="http://localhost:11434",
            )
            assert result == "This is a summary of the page."
            mock_gen.assert_called_once()

    def test_summarize_handles_error(self, use_ollama):
        """Test error handling in summarization."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        with patch("genaitools.deep_research.generate") as mock_gen:
            mock_gen.return_value = "Error: Connection refused"
            result = summarize_page(
                text="Page content",
                url="https://example.com",
                topic="Docker",
                model="test-model",
                ollama_url="http://localhost:11434",
            )
            assert "[Summary failed:" in result


class TestCaching:
    """Tests for caching functions."""

    def test_get_cache_path(self):
        """Cache path is based on URL hash."""
        path = get_cache_path("https://example.com/article")
        assert path.parent == CACHE_DIR
        assert path.suffix == ".json"

    def test_cache_round_trip(self, tmp_path):
        """Save and load from cache."""
        with patch("genaitools.deep_research.CACHE_DIR", tmp_path):
            url = "https://example.com/test"
            save_to_cache(url, "Test Title", "Test summary content")

            cached = load_from_cache(url)
            assert cached is not None
            assert cached["url"] == url
            assert cached["title"] == "Test Title"
            assert cached["summary"] == "Test summary content"
            assert "fetched_at" in cached

    def test_load_missing_cache(self, tmp_path):
        """Load returns None for missing cache."""
        with patch("genaitools.deep_research.CACHE_DIR", tmp_path):
            result = load_from_cache("https://example.com/nonexistent")
            assert result is None


class TestDeepResearch:
    """Tests for deep_research main function."""

    def test_deep_research_mocked(self, use_ollama, tmp_path):
        """Test full pipeline with mocks."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_search_results = [
            {"title": "Article 1", "body": "Snippet 1", "href": "https://example.com/1"},
            {"title": "Article 2", "body": "Snippet 2", "href": "https://example.com/2"},
        ]

        mock_html = "<html><head><title>Test Page</title></head><body><p>" + "Lots of content here for testing. " * 20 + "</p></body></html>"

        with patch("genaitools.deep_research.search_duckduckgo", return_value=mock_search_results), \
             patch("genaitools.deep_research.fetch_page", return_value=mock_html), \
             patch("genaitools.deep_research.generate", return_value="Mocked summary."), \
             patch("genaitools.deep_research.CACHE_DIR", tmp_path):

            result = deep_research(
                topic="Docker",
                max_pages=2,
                model="test-model",
                ollama_url="http://localhost:11434",
                use_cache=False,
            )

            assert "[1]" in result
            assert "Test Page" in result
            assert "Mocked summary" in result

    def test_no_results(self, use_ollama):
        """Test handling when search returns no results."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        with patch("genaitools.deep_research.search_duckduckgo", return_value=[]):
            result = deep_research(topic="nonexistent topic xyz")
            assert "No research results found" in result

    def test_uses_cache(self, use_ollama, tmp_path):
        """Test that cached results are used."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_search_results = [
            {"title": "Cached Article", "body": "Snippet", "href": "https://example.com/cached"},
        ]

        # Pre-populate cache
        with patch("genaitools.deep_research.CACHE_DIR", tmp_path):
            save_to_cache(
                "https://example.com/cached",
                "Cached Title",
                "Cached summary from previous run.",
            )

            with patch("genaitools.deep_research.search_duckduckgo", return_value=mock_search_results), \
                 patch("genaitools.deep_research.fetch_page") as mock_fetch:

                result = deep_research(
                    topic="Docker",
                    max_pages=1,
                    use_cache=True,
                )

                # fetch_page should not be called - using cache
                mock_fetch.assert_not_called()
                assert "Cached summary from previous run" in result
