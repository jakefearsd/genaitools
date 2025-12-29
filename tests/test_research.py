"""Tests for genaitools/research.py"""

import pytest
from unittest.mock import patch, MagicMock

from genaitools.research import search_duckduckgo, build_research_context


class TestBuildResearchContext:
    """Tests for the build_research_context function."""

    def test_empty_results(self):
        """Empty results return fallback message."""
        result = build_research_context([])
        assert "No research available" in result
        assert "training knowledge" in result

    def test_single_result(self):
        """Single result formatted correctly."""
        results = [
            {
                "title": "Test Title",
                "body": "Test body content",
                "href": "https://example.com/test",
            }
        ]
        context = build_research_context(results)

        assert "[1]" in context
        assert "Test Title" in context
        assert "Test body content" in context
        assert "https://example.com/test" in context

    def test_multiple_results(self):
        """Multiple results numbered correctly."""
        results = [
            {"title": "First", "body": "Body 1", "href": "https://example.com/1"},
            {"title": "Second", "body": "Body 2", "href": "https://example.com/2"},
            {"title": "Third", "body": "Body 3", "href": "https://example.com/3"},
        ]
        context = build_research_context(results)

        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "First" in context
        assert "Second" in context
        assert "Third" in context

    def test_missing_fields(self):
        """Missing fields handled gracefully."""
        results = [
            {"title": "Has Title"},  # Missing body and href
            {"body": "Has Body"},    # Missing title and href
            {},                       # Missing everything
        ]
        context = build_research_context(results)

        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "Has Title" in context
        assert "Has Body" in context
        assert "Untitled" in context  # Default for missing title

    def test_results_separated(self):
        """Results are separated by double newlines."""
        results = [
            {"title": "First", "body": "Body 1", "href": "https://example.com/1"},
            {"title": "Second", "body": "Body 2", "href": "https://example.com/2"},
        ]
        context = build_research_context(results)

        # Results should be separated by double newlines
        assert "\n\n" in context


class TestSearchDuckDuckGo:
    """Tests for the search_duckduckgo function."""

    def test_search_with_mock(self, use_search, mock_search_results):
        """Test search with mocked DuckDuckGo."""
        if use_search:
            pytest.skip("Skipping mock test when --use-search is set")

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = mock_search_results

        with patch("genaitools.research.DDGS", return_value=mock_ddgs):
            results = search_duckduckgo("test query")

            assert len(results) == len(mock_search_results)
            assert results[0]["title"] == mock_search_results[0]["title"]

    def test_search_max_results(self, use_search, mock_search_results):
        """Test search respects max_results parameter."""
        if use_search:
            pytest.skip("Skipping mock test when --use-search is set")

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = mock_search_results[:2]

        with patch("genaitools.research.DDGS", return_value=mock_ddgs):
            results = search_duckduckgo("test query", max_results=2)

            mock_ddgs.text.assert_called_once_with("test query", max_results=2)

    def test_search_handles_exception(self, use_search):
        """Test search handles exceptions gracefully."""
        if use_search:
            pytest.skip("Skipping mock test when --use-search is set")

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.side_effect = Exception("Network error")

        with patch("genaitools.research.DDGS", return_value=mock_ddgs):
            results = search_duckduckgo("test query")

            # Should return empty list on error, not raise
            assert results == []

    @pytest.mark.skipif(True, reason="Requires --use-search flag")
    def test_search_real_duckduckgo(self, use_search):
        """Integration test with real DuckDuckGo (only runs with --use-search)."""
        if not use_search:
            pytest.skip("Use --use-search to run this test")

        results = search_duckduckgo("Python programming", max_results=3)

        assert len(results) > 0
        assert "title" in results[0]
        assert "body" in results[0]
        assert "href" in results[0]
