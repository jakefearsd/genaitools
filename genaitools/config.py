"""Shared configuration and defaults for GenAI Tools."""

# Defaults based on tuning with qwen3:14b on 16GB GPU
# Model supports up to 40960 context, but 16384 is safe for 16GB VRAM
# (model uses ~10GB, leaving ~6GB for KV cache at ~160KB/token)
DEFAULTS = {
    "ollama_url": "http://inference.jakefear.com:11434",
    "model": "qwen3:14b",
    "num_predict": 16384,
    "num_ctx": 16384,
    "repeat_penalty": 1.1,
    "temperature": 0.7,
    "think": True,  # Enable chain-of-thought reasoning for qwen3
    # num_gpu intentionally omitted - let Ollama auto-detect based on VRAM
}
