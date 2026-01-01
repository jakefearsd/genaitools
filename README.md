# GenAI Tools

A collection of Python tools for content generation and document management using local LLMs via Ollama.

## Overview

This toolkit provides three main capabilities:

| Tool | Purpose |
|------|---------|
| `simple_publisher.py` | One-shot article generation with web research |
| `outline_builder.py` + `document_builder.py` | Multi-phase structured document generation |
| `link_builder.py` | Automated cross-reference linking for markdown files |

All tools use Ollama for LLM inference and are optimized for models like `qwen3:14b` running on 16GB GPUs.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify Ollama is accessible
curl http://localhost:11434/api/tags
```

### Requirements

- Python 3.10+
- Ollama running locally or remotely
- 16GB+ GPU recommended for 14B parameter models
- Models: `qwen3:14b` (generation), `nomic-embed-text` (embeddings)

---

## Tool 1: Simple Publisher

**One-shot article generation** with integrated DuckDuckGo research.

### Usage

```bash
# Basic usage
python simple_publisher.py -t "Your Topic" -o article.md

# With persona and guidance
python simple_publisher.py \
  -t "Docker Multi-Stage Builds" \
  -p "a senior DevOps engineer" \
  -c "focus on security best practices" \
  -w 2000 \
  -o article.md
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --topic` | Article topic (required) | - |
| `-w, --words` | Target word count | 1500 |
| `-p, --persona` | Writer persona (e.g., "a data scientist") | None |
| `-c, --context` | Additional guidance for the writer | None |
| `-o, --output` | Output file (or stdout) | stdout |
| `--no-search` | Skip DuckDuckGo research | False |
| `--deep-research` | Fetch full pages and summarize (richer context) | False |
| `--no-cache` | Skip research cache (use with --deep-research) | False |
| `--think` | Enable chain-of-thought reasoning | True |
| `--model` | Ollama model | qwen3:14b |

### How It Works

1. **Research**: Searches DuckDuckGo for relevant content (snippets or full pages with `--deep-research`)
2. **Prompt Construction**: Builds a structured prompt with persona, task, guidance, research, and requirements
3. **Generation**: Single LLM call generates the complete article
4. **Output**: Writes markdown to file or stdout

---

## Tool 2: Multi-Phase Document Builder

**Structured document generation** using YAML outlines for comprehensive tutorials, guides, and references.

### Workflow

```bash
# Step 1: Generate structured outline
python outline_builder.py \
  -t "Building REST APIs with FastAPI" \
  -a "Python developers new to async" \
  --type tutorial \
  -w 6000 \
  -s 7 \
  -o fastapi-outline.yaml

# Step 2: (Optional) Edit the outline
vim fastapi-outline.yaml

# Step 3: Build document from outline
python document_builder.py \
  -i fastapi-outline.yaml \
  -o fastapi-tutorial.md \
  --verbose \
  --smooth
```

### Outline Builder Options

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --topic` | Document topic (required) | - |
| `-w, --words` | Total target word count | 5000 |
| `-s, --sections` | Number of sections | 5 |
| `-a, --audience` | Target audience | General |
| `--type` | Content type (tutorial/concept/guide/reference) | tutorial |
| `-p, --persona` | Writer persona | None |
| `-c, --context` | Additional guidance | None |
| `-o, --output` | Output YAML file | stdout |
| `--deep-research` | Fetch full pages and summarize | False |
| `--no-cache` | Skip research cache | False |

### Document Builder Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input YAML outline (required) | - |
| `-o, --output` | Output markdown file | stdout |
| `--section` | Generate only one section (for testing) | None |
| `--smooth` | Post-process to smooth transitions | False |
| `--instructions` | Content constraints (e.g., "no code examples") | None |
| `--verbose` | Show progress and key points | False |
| `--dry-run` | Validate outline without generating | False |

### YAML Outline Schema

```yaml
metadata:
  title: "Document Title"
  topic: "The main topic"
  audience: "Target readers"
  content_type: "tutorial"
  total_word_count: 6000
  persona: "a senior engineer"
  guidance: "Focus on practical examples"

research:
  context: "DuckDuckGo search results..."

sections:
  - id: "introduction"
    title: "Getting Started"
    order: 1
    word_count: 800
    position: "intro"
    dependencies: []
    keywords: ["setup", "installation"]
    guidance: "Cover prerequisites and basic concepts"
    content_hints:
      - "Explain why this topic matters"
      - "List required tools"
```

### How It Works

1. **Outline Generation**: LLM generates JSON structure, converted to YAML for editing
2. **Section Ordering**: Sections sorted by `order` field with dependency tracking
3. **Key Points Extraction**: After generating each section, 5-10 key facts are extracted
4. **Context Passing**: Key points from dependencies are included in subsequent section prompts
5. **Position Awareness**: Sections know if they're intro/middle/conclusion for appropriate tone
6. **Transition Smoothing**: Optional post-processing to improve flow between sections

### Safety Features

- **Output file check**: Fails immediately if output file already exists
- **Dry-run mode**: Validate outline structure without making API calls

---

## Tool 3: Link Builder

**Automated cross-reference linking** for markdown document collections using semantic similarity.

### Usage

```bash
# Preview mode - see what links would be created
python link_builder.py --dir ./docs --dry-run --verbose

# Apply changes with custom threshold
python link_builder.py --dir ./docs --similarity 0.7 --max-links 3

# Recursive with report
python link_builder.py --dir ./docs --recursive --output-report links.json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-d, --dir` | Directory containing markdown files (required) | - |
| `--recursive` | Include subdirectories | False |
| `--similarity` | Minimum similarity threshold (0-1) | 0.6 |
| `--max-links` | Maximum links per file | 5 |
| `--dry-run` | Preview without writing | False |
| `--verbose` | Show detailed progress | False |
| `--embed-model` | Embedding model | nomic-embed-text |
| `--model` | LLM for link generation | qwen3:14b |
| `--output-report` | Write JSON report | None |

### How It Works

The link builder uses a **hybrid approach** combining embeddings and LLM analysis:

#### Phase 1: Document Indexing
- Scan directory for `*.md` files
- Extract title (first H1 or filename) and all headings
- Generate embedding from: title + headings + first 8000 chars of content
- Store document metadata for comparison

#### Phase 2: Candidate Discovery
- Compute pairwise cosine similarity between all document embeddings
- Filter pairs above the similarity threshold (default: 0.6)
- Exclude self-links and already-linked document pairs
- Sort by similarity score (highest first)

#### Phase 3: Link Generation
For each candidate pair, the LLM analyzes both documents and returns:
```json
{
  "should_link": true,
  "confidence": 0.85,
  "anchor_text": "Docker containers",
  "context_before": "When working with",
  "reasoning": "The source discusses container basics..."
}
```

#### Phase 4: File Update
- Insert markdown links at identified positions
- Create **bidirectional links** (A→B and B→A)
- Skip text inside existing links or headings
- Idempotent: running twice won't create duplicates

### Example Transformation

**Before:**
```markdown
When working with Docker containers, you need to understand networking.
```

**After:**
```markdown
When working with [Docker containers](./docker-basics.md), you need to understand networking.
```

### Understanding Embeddings

Embeddings are 768-dimensional vectors that capture semantic meaning:

```python
# Similar concepts have high cosine similarity
"Docker containers" ↔ "Container orchestration"  → 0.85
"Docker containers" ↔ "Medieval poetry"          → 0.40
```

The `--similarity` threshold controls how related documents must be before the LLM analyzes them for linking opportunities.

---

## Shared Module: genaitools/

Common utilities used across all tools:

| Module | Functions |
|--------|-----------|
| `config.py` | `DEFAULTS` dict with Ollama URL, model params |
| `ollama_client.py` | `generate()`, `count_words()` |
| `research.py` | `search_duckduckgo()`, `build_research_context()` |
| `embeddings.py` | `get_embedding()`, `cosine_similarity()`, `find_similar_pairs()` |
| `deep_research.py` | `deep_research()` - RAG pipeline for richer context |

### Deep Research (RAG)

The `--deep-research` flag enables a richer research pipeline:

1. **Search**: DuckDuckGo search for 3 relevant pages
2. **Fetch**: Download full HTML pages (15s timeout)
3. **Strip**: Remove scripts, styles, nav elements; extract plain text
4. **Summarize**: LLM generates 200-400 word focused summary per page
5. **Cache**: Results cached by URL hash in `~/.cache/genaitools/research/`

```bash
# With deep research
python simple_publisher.py -t "Docker Security" --deep-research

# Force fresh fetch (skip cache)
python outline_builder.py -t "Kubernetes" --deep-research --no-cache
```

**Trade-offs**:
- ~20-30 seconds vs ~2 seconds for regular search
- 3 additional LLM calls for summarization
- Much richer context (~1500-2500 chars of focused summaries)

### Configuration

Default settings in `genaitools/config.py`:

```python
DEFAULTS = {
    "ollama_url": "http://localhost:11434",
    "model": "qwen3:14b",
    "temperature": 0.7,
    "num_ctx": 16384,
    "num_predict": 4096,
    "think": True,
}
```

Override via command-line flags or environment.

---

## Testing

```bash
# Run all tests with mocks (no external API calls)
pytest tests/ -v

# Run with real Ollama API
pytest tests/ --use-ollama

# Run with real DuckDuckGo search
pytest tests/ --use-search

# Run with both
pytest tests/ --use-ollama --use-search
```

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | DEFAULTS validation |
| `test_ollama_client.py` | generate(), count_words() |
| `test_research.py` | DuckDuckGo search, context formatting |
| `test_outline_builder.py` | Prompt building, JSON extraction, YAML output |
| `test_document_builder.py` | Outline loading, section prompts, key points |
| `test_link_builder.py` | Embeddings, similarity, link insertion |
| `test_deep_research.py` | Page fetching, HTML stripping, summarization, caching |

---

## Model Support

### Chain-of-Thought Reasoning

- `--think` enables extended reasoning (works with qwen3, deepseek-r1)
- `--no-think` disables for faster generation or unsupported models
- Key points extraction always uses `think=False` for consistency

### Recommended Models

| Task | Model | Notes |
|------|-------|-------|
| Content generation | qwen3:14b | Good balance of quality and speed |
| Embeddings | nomic-embed-text | 768 dimensions, fast, MIT licensed |
| Alternative | mxbai-embed-large | Higher quality embeddings (1024 dim) |

---

## License

MIT
