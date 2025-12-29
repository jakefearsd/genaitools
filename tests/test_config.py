"""Tests for genaitools/config.py"""

from genaitools.config import DEFAULTS


class TestDefaults:
    """Tests for the DEFAULTS configuration."""

    def test_defaults_has_required_keys(self):
        """Verify all required keys are present in DEFAULTS."""
        required_keys = [
            "ollama_url",
            "model",
            "num_predict",
            "num_ctx",
            "repeat_penalty",
            "temperature",
            "num_gpu",
            "think",
        ]
        for key in required_keys:
            assert key in DEFAULTS, f"Missing required key: {key}"

    def test_defaults_types(self):
        """Verify DEFAULTS values have correct types."""
        assert isinstance(DEFAULTS["ollama_url"], str)
        assert isinstance(DEFAULTS["model"], str)
        assert isinstance(DEFAULTS["num_predict"], int)
        assert isinstance(DEFAULTS["num_ctx"], int)
        assert isinstance(DEFAULTS["repeat_penalty"], (int, float))
        assert isinstance(DEFAULTS["temperature"], (int, float))
        assert isinstance(DEFAULTS["num_gpu"], int)
        assert isinstance(DEFAULTS["think"], bool)

    def test_defaults_reasonable_values(self):
        """Verify DEFAULTS values are within reasonable ranges."""
        assert DEFAULTS["num_predict"] > 0
        assert DEFAULTS["num_ctx"] > 0
        assert 0 <= DEFAULTS["temperature"] <= 2.0
        assert DEFAULTS["repeat_penalty"] >= 1.0
        assert DEFAULTS["ollama_url"].startswith("http")
