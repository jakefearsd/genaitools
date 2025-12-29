"""DuckDuckGo research utilities."""

# Handle package rename: 'ddgs' is the newer package name,
# 'duckduckgo_search' is the older name. Support both for compatibility.
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def search_duckduckgo(topic: str, max_results: int = 8) -> list[dict]:
    """
    Search DuckDuckGo for research content.

    Args:
        topic: Search query
        max_results: Maximum number of results to return

    Returns:
        List of result dicts with 'title', 'body', and 'href' keys
    """
    print(f"Searching DuckDuckGo for: {topic}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=max_results))
        print(f"Found {len(results)} search results")
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []


def build_research_context(results: list[dict]) -> str:
    """
    Format search results into research context for prompts.

    Args:
        results: List of search result dicts

    Returns:
        Formatted string with numbered citations
    """
    if not results:
        return "No research available. Use your training knowledge."

    context_parts = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        body = result.get("body", "")
        href = result.get("href", "")
        context_parts.append(f"[{i}] {title}\n{body}\nSource: {href}")

    return "\n\n".join(context_parts)
