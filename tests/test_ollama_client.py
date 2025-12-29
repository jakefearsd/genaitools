"""Tests for genaitools/ollama_client.py"""

import pytest
from unittest.mock import patch, MagicMock

from genaitools.ollama_client import generate, count_words
from genaitools.config import DEFAULTS


class TestCountWords:
    """Tests for the count_words function."""

    def test_count_words_empty(self):
        """Empty string returns 0."""
        assert count_words("") == 0

    def test_count_words_single(self):
        """Single word returns 1."""
        assert count_words("hello") == 1

    def test_count_words_multiple(self):
        """Multiple words counted correctly."""
        assert count_words("hello world") == 2
        assert count_words("one two three four five") == 5

    def test_count_words_with_newlines(self):
        """Words across newlines counted correctly."""
        assert count_words("hello\nworld") == 2
        assert count_words("one\ntwo\nthree") == 3

    def test_count_words_with_extra_spaces(self):
        """Extra spaces handled correctly."""
        assert count_words("hello   world") == 2
        assert count_words("  hello  world  ") == 2


class TestGenerate:
    """Tests for the generate function."""

    def test_generate_with_mock(self, use_ollama):
        """Test generate with mocked Ollama API."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response) as mock_post:
            result = generate("Test prompt")

            assert result == "Generated text"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert "json" in call_kwargs.kwargs
            assert call_kwargs.kwargs["json"]["prompt"] == "Test prompt"

    def test_generate_uses_defaults(self, use_ollama):
        """Test that generate uses DEFAULTS when no args provided."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test"}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response) as mock_post:
            generate("Test prompt")

            call_kwargs = mock_post.call_args.kwargs["json"]
            assert call_kwargs["model"] == DEFAULTS["model"]
            assert call_kwargs["think"] == DEFAULTS["think"]
            assert call_kwargs["options"]["temperature"] == DEFAULTS["temperature"]

    def test_generate_custom_params(self, use_ollama):
        """Test generate with custom parameters."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Custom response"}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response) as mock_post:
            result = generate(
                "Test prompt",
                model="custom-model",
                temperature=0.5,
                think=False,
            )

            assert result == "Custom response"
            call_kwargs = mock_post.call_args.kwargs["json"]
            assert call_kwargs["model"] == "custom-model"
            assert call_kwargs["think"] is False
            assert call_kwargs["options"]["temperature"] == 0.5

    def test_generate_timeout_error(self, use_ollama):
        """Test generate handles timeout errors."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        import requests

        with patch("genaitools.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()
            result = generate("Test prompt", timeout=30)

            assert result.startswith("Error:")
            assert "timed out" in result.lower()

    def test_generate_request_error(self, use_ollama):
        """Test generate handles request errors."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        import requests

        with patch("genaitools.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
            result = generate("Test prompt")

            assert result.startswith("Error:")

    def test_generate_empty_response(self, use_ollama):
        """Test generate handles empty response."""
        if use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("genaitools.ollama_client.requests.post", return_value=mock_response):
            result = generate("Test prompt")
            assert result == ""

    @pytest.mark.skipif(True, reason="Requires --use-ollama flag")
    def test_generate_real_ollama(self, use_ollama):
        """Integration test with real Ollama (only runs with --use-ollama)."""
        if not use_ollama:
            pytest.skip("Use --use-ollama to run this test")

        result = generate(
            "Say 'Hello, test!' and nothing else.",
            num_predict=50,
            think=False,
        )
        assert not result.startswith("Error:")
        assert len(result) > 0
