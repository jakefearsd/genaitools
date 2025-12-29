"""Embedding utilities for semantic similarity via Ollama."""

import math
import requests
from typing import Any

from .config import DEFAULTS

# Default embedding model
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    ollama_url: str | None = None,
    timeout: int = 60,
) -> list[float]:
    """
    Get embedding vector for text via Ollama.

    Args:
        text: Text to embed
        model: Embedding model name (default: nomic-embed-text)
        ollama_url: Ollama API URL (default: from DEFAULTS)
        timeout: Request timeout in seconds

    Returns:
        List of floats representing the embedding vector

    Raises:
        ValueError: If embedding generation fails
    """
    if ollama_url is None:
        ollama_url = DEFAULTS["ollama_url"]

    url = f"{ollama_url}/api/embeddings"
    payload = {
        "model": model,
        "prompt": text,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        embedding = result.get("embedding")
        if embedding is None:
            raise ValueError(f"No embedding in response: {result.keys()}")
        return embedding
    except requests.exceptions.Timeout:
        raise ValueError(f"Embedding request timed out after {timeout}s")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Embedding request failed: {e}")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1

    Raises:
        ValueError: If vectors have different lengths or are zero-length
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")
    if len(vec_a) == 0:
        raise ValueError("Cannot compute similarity of empty vectors")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def find_similar_pairs(
    documents: list[dict[str, Any]],
    threshold: float = 0.6,
) -> list[tuple[int, int, float]]:
    """
    Find document pairs with similarity above threshold.

    Args:
        documents: List of dicts with 'embedding' key containing vector
        threshold: Minimum similarity score (0-1) to include pair

    Returns:
        List of (index_a, index_b, similarity) tuples, sorted by similarity descending.
        Only includes pairs where index_a < index_b to avoid duplicates.

    Raises:
        KeyError: If any document is missing 'embedding' key
    """
    pairs = []
    n = len(documents)

    for i in range(n):
        for j in range(i + 1, n):
            emb_a = documents[i]["embedding"]
            emb_b = documents[j]["embedding"]
            sim = cosine_similarity(emb_a, emb_b)

            if sim >= threshold:
                pairs.append((i, j, sim))

    # Sort by similarity descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def batch_get_embeddings(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    ollama_url: str | None = None,
    timeout: int = 60,
    verbose: bool = False,
) -> list[list[float]]:
    """
    Get embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model: Embedding model name
        ollama_url: Ollama API URL
        timeout: Request timeout per embedding
        verbose: Print progress

    Returns:
        List of embedding vectors in same order as input texts
    """
    embeddings = []
    for i, text in enumerate(texts):
        if verbose:
            print(f"  Embedding {i + 1}/{len(texts)}...")
        emb = get_embedding(text, model=model, ollama_url=ollama_url, timeout=timeout)
        embeddings.append(emb)
    return embeddings
