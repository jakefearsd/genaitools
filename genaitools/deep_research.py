"""Deep research utilities - fetch, strip, summarize web pages."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from .config import DEFAULTS
from .ollama_client import generate
from .output import tprint
from .research import search_duckduckgo

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_PAGES = 3
DEFAULT_FETCH_TIMEOUT = 15
DEFAULT_MAX_TEXT_CHARS = 8000
SUMMARIZE_TEMPERATURE = 0.5
CACHE_DIR = Path.home() / ".cache" / "genaitools" / "research"

# Elements to remove from HTML before text extraction
REMOVE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript"]


# =============================================================================
# Caching
# =============================================================================


def get_cache_path(url: str) -> Path:
    """Get cache file path for a URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{url_hash}.json"


def load_from_cache(url: str) -> dict[str, Any] | None:
    """Load cached summary for URL, or None if not cached."""
    cache_path = get_cache_path(url)
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_to_cache(url: str, title: str, summary: str) -> None:
    """Save summary to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(url)
    cache_entry = {
        "url": url,
        "title": title,
        "summary": summary,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_path, "w") as f:
        json.dump(cache_entry, f, indent=2)


# =============================================================================
# Page Fetching
# =============================================================================


def fetch_page(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT) -> str | None:
    """
    Fetch URL and return raw HTML.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        HTML content as string, or None on failure
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GenAITools/1.0; research bot)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        return None


def extract_title(soup: BeautifulSoup) -> str:
    """Extract page title from BeautifulSoup object."""
    # Try <title> tag first
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # Fall back to first H1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Untitled"


def html_to_text(html: str, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> tuple[str, str]:
    """
    Strip HTML to plain text.

    Args:
        html: Raw HTML content
        max_chars: Maximum characters to return

    Returns:
        Tuple of (title, text_content)
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract title before removing elements
    title = extract_title(soup)

    # Remove unwanted elements
    for tag in REMOVE_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    # Get text with newline separators
    text = soup.get_text(separator="\n")

    # Collapse multiple whitespace/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    # Limit length
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return title, text


# =============================================================================
# Summarization
# =============================================================================


def build_summarize_prompt(text: str, url: str, topic: str) -> str:
    """Build prompt for page summarization."""
    return f"""You are summarizing a web page for research on: {topic}

SOURCE URL: {url}

PAGE CONTENT:
{text}

TASK: Write a 200-400 word summary focusing on information relevant to the topic.
Include specific facts, figures, and insights that would be useful for writing about this topic.
Focus on concrete information, not general statements.

SUMMARY:"""


def summarize_page(
    text: str,
    url: str,
    topic: str,
    model: str,
    ollama_url: str,
) -> str:
    """
    Generate focused summary of page content.

    Args:
        text: Plain text content of page
        url: Source URL for context
        topic: Research topic to focus on
        model: Ollama model name
        ollama_url: Ollama API URL

    Returns:
        Summary text (200-400 words)
    """
    prompt = build_summarize_prompt(text, url, topic)

    response = generate(
        prompt=prompt,
        ollama_url=ollama_url,
        model=model,
        temperature=SUMMARIZE_TEMPERATURE,
        think=False,
        num_predict=1024,
        num_ctx=DEFAULTS["num_ctx"],
    )

    if response.startswith("Error:"):
        return f"[Summary failed: {response}]"

    return response.strip()


# =============================================================================
# Main Pipeline
# =============================================================================


def deep_research(
    topic: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    model: str | None = None,
    ollama_url: str | None = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> str:
    """
    Full research pipeline: search, fetch, strip, summarize, format.

    Args:
        topic: Research topic/query
        max_pages: Maximum pages to fetch and summarize
        model: Ollama model (default: from DEFAULTS)
        ollama_url: Ollama API URL (default: from DEFAULTS)
        use_cache: Whether to use cached summaries
        verbose: Print progress

    Returns:
        Formatted research context string with numbered summaries
    """
    if model is None:
        model = DEFAULTS["model"]
    if ollama_url is None:
        ollama_url = DEFAULTS["ollama_url"]

    # Step 1: Search
    if verbose:
        tprint(f"Searching for: {topic}")
    results = search_duckduckgo(topic, max_results=max_pages + 2)  # Extra in case some fail

    if not results:
        return "No research results found. Using training knowledge only."

    # Step 2-4: Fetch, strip, summarize each page
    summaries = []
    for result in results:
        if len(summaries) >= max_pages:
            break

        url = result.get("href", "")
        if not url:
            continue

        # Check cache first
        if use_cache:
            cached = load_from_cache(url)
            if cached:
                if verbose:
                    tprint(f"  [cached] {cached['title'][:50]}")
                summaries.append(cached)
                continue

        # Fetch page
        if verbose:
            tprint(f"  Fetching: {url[:60]}...")
        html = fetch_page(url)
        if not html:
            if verbose:
                tprint(f"    Failed to fetch")
            continue

        # Strip HTML
        title, text = html_to_text(html)
        if len(text) < 200:
            if verbose:
                tprint(f"    Too short ({len(text)} chars), skipping")
            continue

        # Summarize
        if verbose:
            tprint(f"    Summarizing: {title[:50]}...")
        summary = summarize_page(text, url, topic, model, ollama_url)

        # Cache result
        if use_cache:
            save_to_cache(url, title, summary)

        summaries.append({
            "url": url,
            "title": title,
            "summary": summary,
        })

    # Format output
    if not summaries:
        return "No pages could be fetched. Using training knowledge only."

    context_parts = []
    for i, s in enumerate(summaries, 1):
        context_parts.append(
            f"[{i}] {s['title']}\n"
            f"URL: {s['url']}\n"
            f"Summary: {s['summary']}"
        )

    return "\n\n".join(context_parts)
