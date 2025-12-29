"""
Pytest configuration and fixtures for genaitools tests.

Usage:
    pytest                     # Run with mocks (default)
    pytest --use-ollama        # Run with real Ollama calls
    pytest --use-search        # Run with real DuckDuckGo calls
    pytest --use-ollama --use-search  # Run with all real calls
"""

import pytest
from unittest.mock import patch, MagicMock


def pytest_addoption(parser):
    """Add command-line options for test modes."""
    parser.addoption(
        "--use-ollama",
        action="store_true",
        default=False,
        help="Use real Ollama API calls instead of mocks",
    )
    parser.addoption(
        "--use-search",
        action="store_true",
        default=False,
        help="Use real DuckDuckGo search instead of mocks",
    )


@pytest.fixture
def use_ollama(request):
    """Return True if real Ollama calls should be used."""
    return request.config.getoption("--use-ollama")


@pytest.fixture
def use_search(request):
    """Return True if real DuckDuckGo search should be used."""
    return request.config.getoption("--use-search")


# Sample mock responses for consistent testing
MOCK_OLLAMA_RESPONSE = """## Introduction

This is a sample generated section about the topic. It covers the key concepts
and provides practical examples for the reader.

### Key Concepts

Here are the main points to understand:

1. First concept explanation
2. Second concept with code example

```python
def example():
    return "Hello, World!"
```

### Conclusion

This section introduced the fundamentals needed for the next sections.
"""

MOCK_OUTLINE_JSON = """{
  "title": "Test Document Title",
  "sections": [
    {
      "id": "introduction",
      "title": "Introduction",
      "order": 1,
      "word_count": 400,
      "dependencies": [],
      "keywords": ["overview", "basics"],
      "guidance": "Set up the topic and explain why it matters",
      "content_hints": ["Hook the reader", "Preview main sections"]
    },
    {
      "id": "core-concepts",
      "title": "Core Concepts",
      "order": 2,
      "word_count": 800,
      "dependencies": ["introduction"],
      "keywords": ["fundamentals", "key terms"],
      "guidance": "Explain the foundational concepts",
      "content_hints": ["Define terminology", "Use examples"]
    },
    {
      "id": "conclusion",
      "title": "Conclusion",
      "order": 3,
      "word_count": 300,
      "dependencies": ["introduction", "core-concepts"],
      "keywords": ["summary", "next steps"],
      "guidance": "Wrap up and provide next steps",
      "content_hints": ["Recap key points", "Suggest further reading"]
    }
  ]
}"""

MOCK_KEY_POINTS = """- Topic X is defined as a method for achieving Y
- The main components are: component A, component B, and component C
- Code examples use Python 3.10+ syntax with type hints
- Best practice is to start with the simplest implementation
- Advanced usage requires understanding of async/await patterns"""

MOCK_SEARCH_RESULTS = [
    {
        "title": "Introduction to Topic X",
        "body": "Topic X is a fundamental concept in software development that enables...",
        "href": "https://example.com/topic-x-intro",
    },
    {
        "title": "Topic X Best Practices",
        "body": "When working with Topic X, it's important to follow these guidelines...",
        "href": "https://example.com/topic-x-best-practices",
    },
    {
        "title": "Advanced Topic X Techniques",
        "body": "For advanced users, Topic X offers several powerful features including...",
        "href": "https://example.com/topic-x-advanced",
    },
]


@pytest.fixture
def mock_ollama_generate():
    """Fixture that provides a mock for ollama_client.generate()."""
    def _mock_generate(response_text=MOCK_OLLAMA_RESPONSE):
        """Create a mock that returns the specified response."""
        mock = MagicMock(return_value=response_text)
        return mock
    return _mock_generate


@pytest.fixture
def mock_search_results():
    """Fixture that provides mock DuckDuckGo search results."""
    return MOCK_SEARCH_RESULTS.copy()


@pytest.fixture
def mock_outline_json():
    """Fixture that provides mock outline JSON response."""
    return MOCK_OUTLINE_JSON


@pytest.fixture
def mock_key_points_response():
    """Fixture that provides mock key points extraction response."""
    return MOCK_KEY_POINTS


@pytest.fixture
def sample_outline():
    """Fixture that provides a complete sample outline dict."""
    return {
        "metadata": {
            "title": "Test Document",
            "topic": "Test Topic",
            "audience": "developers",
            "content_type": "tutorial",
            "total_word_count": 1500,
            "generated_at": "2024-01-15T10:00:00Z",
        },
        "research": {
            "context": "[1] Test Result\nSome research content\nSource: https://example.com",
        },
        "sections": [
            {
                "id": "introduction",
                "title": "Introduction",
                "order": 1,
                "word_count": 400,
                "dependencies": [],
                "keywords": ["overview", "basics"],
                "guidance": "Introduce the topic",
                "content_hints": ["Hook reader"],
            },
            {
                "id": "main-content",
                "title": "Main Content",
                "order": 2,
                "word_count": 800,
                "dependencies": ["introduction"],
                "keywords": ["details", "examples"],
                "guidance": "Cover the main points",
                "content_hints": ["Include code"],
            },
            {
                "id": "conclusion",
                "title": "Conclusion",
                "order": 3,
                "word_count": 300,
                "dependencies": ["introduction", "main-content"],
                "keywords": ["summary"],
                "guidance": "Wrap up",
                "content_hints": ["Next steps"],
            },
        ],
    }


@pytest.fixture
def temp_outline_file(tmp_path, sample_outline):
    """Fixture that creates a temporary outline YAML file."""
    import yaml
    outline_path = tmp_path / "test_outline.yaml"
    with open(outline_path, "w") as f:
        yaml.dump(sample_outline, f)
    return outline_path


# =============================================================================
# Mock Factory Fixtures - eliminate repeated mock setup across test files
# =============================================================================


@pytest.fixture
def mock_requests_response():
    """Factory fixture for creating mock requests.post responses.

    Usage:
        def test_something(mock_requests_response):
            mock_resp = mock_requests_response({"response": "Generated text"})
            with patch("requests.post", return_value=mock_resp):
                # test code
    """
    def _create_response(json_data: dict = None, raise_for_status=None):
        mock_resp = MagicMock()
        if json_data is not None:
            mock_resp.json.return_value = json_data
        else:
            mock_resp.json.return_value = {"response": MOCK_OLLAMA_RESPONSE}
        mock_resp.raise_for_status = MagicMock(side_effect=raise_for_status)
        return mock_resp
    return _create_response


@pytest.fixture
def mock_ddgs_context(mock_search_results):
    """Create a mock DDGS context manager for DuckDuckGo searches.

    Usage:
        def test_search(mock_ddgs_context):
            with patch("genaitools.research.DDGS", return_value=mock_ddgs_context):
                results = search_duckduckgo("topic")
    """
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)
    mock_ddgs.text.return_value = mock_search_results
    return mock_ddgs


@pytest.fixture
def skip_if_real_api(use_ollama, use_search):
    """Skip mock tests when real API flags are set.

    Usage:
        def test_with_mock(skip_if_real_api):
            skip_if_real_api("ollama")  # Skip if --use-ollama is set
            # mock test code
    """
    def _skip(api_name: str):
        if api_name == "ollama" and use_ollama:
            pytest.skip("Skipping mock test when --use-ollama is set")
        elif api_name == "search" and use_search:
            pytest.skip("Skipping mock test when --use-search is set")
    return _skip
