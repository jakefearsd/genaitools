"""Ollama API client for text generation."""

import requests

from .config import DEFAULTS


def generate(
    prompt: str,
    ollama_url: str = DEFAULTS["ollama_url"],
    model: str = DEFAULTS["model"],
    num_predict: int = DEFAULTS["num_predict"],
    num_ctx: int = DEFAULTS["num_ctx"],
    repeat_penalty: float = DEFAULTS["repeat_penalty"],
    temperature: float = DEFAULTS["temperature"],
    num_gpu: int | None = None,
    think: bool = DEFAULTS["think"],
    timeout: int = 600,
) -> str:
    """
    Call Ollama API to generate text.

    Args:
        prompt: The prompt to send to the model
        ollama_url: Ollama API URL
        model: Model name
        num_predict: Max tokens to generate
        num_ctx: Context window size
        repeat_penalty: Repetition penalty (1.0 = none)
        temperature: Sampling temperature
        num_gpu: Number of GPU layers (None = let Ollama auto-detect)
        think: Enable chain-of-thought for supported models (qwen3, deepseek-r1)
        timeout: Request timeout in seconds

    Returns:
        Generated text, or error message string on failure
    """
    url = f"{ollama_url}/api/generate"

    options = {
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "repeat_penalty": repeat_penalty,
        "temperature": temperature,
    }
    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": options,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout // 60} minutes"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())
