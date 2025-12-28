# GenAI Tools

A collection of Python tools for working with local LLMs via Ollama.

## Tools

### simple_publisher.py

One-shot article generator using Ollama and DuckDuckGo research.

**Features:**
- DuckDuckGo web search for research context
- Configurable persona and guidance
- Support for thinking/reasoning models (qwen3, deepseek-r1)
- Tuned defaults for 16GB GPU (qwen3:14b)

**Usage:**
```bash
python simple_publisher.py -t "Your Topic" -o article.md

# With persona and guidance
python simple_publisher.py \
  -t "Docker Multi-Stage Builds" \
  -p "a senior DevOps engineer" \
  -c "focus on security best practices" \
  -o article.md

# Different model
python simple_publisher.py -t "Topic" --model gemma3:12b --no-think -o article.md
```

**Options:**
| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--topic` | `-t` | *required* | Article topic |
| `--audience` | `-a` | `college educated general audience` | Target audience |
| `--type` | | `tutorial` | Content type: tutorial, concept, guide, reference |
| `--words` | `-w` | `2500` | Target word count |
| `--persona` | `-p` | None | Writer persona/voice |
| `--context` | `-c` | None | Additional guidance |
| `--output` | `-o` | stdout | Output file path |
| `--model` | | `qwen3:14b` | Ollama model |
| `--think` | | `True` | Enable chain-of-thought (qwen3/deepseek-r1 only) |
| `--no-think` | | | Disable thinking |
| `--no-search` | | | Skip DuckDuckGo research |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- Ollama running locally or on a remote server
- 16GB+ GPU recommended for 14B models
