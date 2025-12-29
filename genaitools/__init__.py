"""GenAI Tools - shared utilities for LLM-powered content generation."""

from .config import DEFAULTS
from .ollama_client import generate, count_words
from .research import search_duckduckgo, build_research_context
from .embeddings import (
    get_embedding,
    cosine_similarity,
    find_similar_pairs,
    batch_get_embeddings,
    DEFAULT_EMBED_MODEL,
)

__all__ = [
    "DEFAULTS",
    "generate",
    "count_words",
    "search_duckduckgo",
    "build_research_context",
    "get_embedding",
    "cosine_similarity",
    "find_similar_pairs",
    "batch_get_embeddings",
    "DEFAULT_EMBED_MODEL",
]
