# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenAI Tools is a collection of Python tools for working with local LLMs via Ollama:
- `simple_publisher.py` - One-shot article generator
- `outline_builder.py` + `document_builder.py` - Multi-phase document generation
- `batch_builder.py` - Batch processing multiple topics from JSON config
- `link_builder.py` - Cross-reference link generator for markdown files

## Setup and Running

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# One-shot article
python simple_publisher.py -t "Topic" -o article.md

# Multi-phase document
python outline_builder.py -t "Topic" -w 6000 -o outline.yaml
python document_builder.py -i outline.yaml -o document.md --verbose

# With content constraints (e.g., no code examples)
python document_builder.py -i outline.yaml -o doc.md --instructions "Do not include code examples"

# Batch processing multiple topics
python batch_builder.py -i batch.json --dry-run
python batch_builder.py -i batch.json

# Cross-reference linking (preview mode)
python link_builder.py --dir ./docs --dry-run --verbose

# Cross-reference linking (apply changes)
python link_builder.py --dir ./docs --recursive --max-links 5
```

## Architecture

### Shared Module: genaitools/

Common utilities extracted for reuse across tools:
- `config.py` - DEFAULTS dict (Ollama URL, model params tuned for qwen3:14b on 16GB GPU)
- `ollama_client.py` - `generate()` wrapper with timeout handling, `count_words()`
- `research.py` - `search_duckduckgo()`, `build_research_context()`
- `embeddings.py` - `get_embedding()`, `cosine_similarity()`, `find_similar_pairs()`, `batch_get_embeddings()`
- `deep_research.py` - `deep_research()` for RAG: fetch pages, strip HTML, LLM summarize, cache

### simple_publisher.py

Single-file one-shot generator. Prompt structure (lines 61-120):
1. PERSONA first - colors interpretation of everything after
2. TASK - content type, topic, audience
3. GUIDANCE - placed before research to frame interpretation
4. RESEARCH - web search content
5. REQUIREMENTS last - mechanical formatting rules

**Deep Research:** Use `--deep-research` for richer context (fetches full pages, summarizes with LLM).

### outline_builder.py

Generates YAML outlines by prompting LLM for JSON structure, then converting.

**Deep Research:** Use `--deep-research` for richer context. Cached by URL hash in `~/.cache/genaitools/research/`.

**YAML Schema:**
```yaml
metadata:
  title, topic, audience, content_type, total_word_count, persona, guidance
research:
  context: "formatted DuckDuckGo results"
sections:
  - id: "url-safe-slug"
    title: "Section Title"
    order: 1
    word_count: 400
    dependencies: ["prior-section-id"]
    keywords: ["term1", "term2"]
    guidance: "what to cover"
    content_hints: ["specific point"]
```

### document_builder.py

Orchestrates section-by-section generation:
1. Load YAML outline
2. For each section (sorted by order):
   - Build prompt with persona, task, document context, key points from dependencies, guidance, research
   - Generate section via Ollama
   - Extract 5-10 key points (separate LLM call, temp=0.3)
   - Store key points for subsequent sections
3. Optionally smooth transitions between sections (`--smooth`)
4. Assemble final Markdown

**Prompt Structure:**
1. PERSONA - Writer identity
2. TASK - Section-specific assignment
3. DOCUMENT CONTEXT - Where this section fits
4. SECTION POSITION - first/middle/last role in document
5. CONTINUITY - Key points from dependencies
6. GUIDANCE - Section-specific direction
7. OUTPUT INSTRUCTIONS - Additional constraints via `--instructions`
8. RESEARCH - Web search context
9. REQUIREMENTS - Format constraints (position-aware)

**Key Flags:**
- `--smooth` - Post-process to smooth transitions between sections
- `--instructions "..."` - Add constraints to all section prompts (e.g., "Do not include code examples")

**Key Design:** Key points extraction uses a focused prompt asking for specific facts/definitions, not vague summaries. This enables continuity without repeating full sections.

### batch_builder.py

Batch execution tool that runs `outline_builder.py` + `document_builder.py` for each topic in a JSON config.

**JSON Schema:**
```json
{
  "defaults": { /* shared params: audience, words, sections, model, etc. */ },
  "output_dir": "./output",
  "cooldown_seconds": 10,
  "topics": [
    {"topic": "Topic 1", "context": "guidance for this topic"},
    {"topic": "Topic 2", "words": 6000}  /* per-topic overrides */
  ]
}
```

**Config Inheritance:** TOPIC_DEFAULTS → JSON defaults → per-topic values

**File Naming:**
- Outline: `{slug}-outline.yaml` (e.g., `first-aws-deployment-outline.yaml`)
- Document: `{TitleCase}.md` (e.g., `FirstAwsDeployment.md`)

**Key Features:**
- Auto-resume: Skips topics with existing output files
- Subprocess execution: Runs outline/document builders as subprocesses
- Signal handling: Graceful Ctrl+C saves partial batch report
- Batch report: `{output_dir}/batch-report.json` with status/timing

**Key Flags:**
- `-i, --input` - JSON config file (required)
- `--dry-run` - Show plan without executing
- `--start-at N` - Start at topic N (1-indexed)
- `--only N` - Run only topic N
- `--skip-outlines` - Use existing outlines, only build documents
- `--cooldown SECONDS` - GPU cooldown between topics

### link_builder.py

Discovers semantic linking opportunities between markdown files and inserts cross-references.

**Hybrid Approach:**
1. **Phase 1 - Indexing**: Scan directory, extract titles/headings, generate embeddings via Ollama
2. **Phase 2 - Candidate Discovery**: Compute pairwise cosine similarity, filter pairs above threshold
3. **Phase 3 - Link Generation**: For each candidate pair, ask LLM for natural anchor text
4. **Phase 4 - File Update**: Insert bidirectional links, respecting existing links and headings

**Key Flags:**
- `--dir` - Directory containing markdown files (required)
- `--recursive` - Include subdirectories
- `--similarity` - Minimum similarity threshold (default: 0.6)
- `--max-links` - Maximum links to add per file (default: 5)
- `--dry-run` - Preview changes without writing
- `--embed-model` - Embedding model (default: nomic-embed-text)
- `--model` - LLM for link text generation (default: qwen3:14b)
- `--output-report` - Write JSON report of changes

**Design Decisions:**
- Uses embedding similarity to reduce LLM calls (only promising pairs sent to LLM)
- Always creates bidirectional links (A→B implies B→A)
- Skips text already inside links or headings
- Idempotent: running twice won't create duplicate links

## Key Dependencies

- `ddgs` - DuckDuckGo search (no API key)
- `requests` - Ollama API calls
- `PyYAML` - Outline parsing
- `beautifulsoup4` - HTML parsing for deep research
- Ollama instance (default: http://inference.jakefear.com:11434)

## Model Support

- Chain-of-thought via `--think` works with qwen3 and deepseek-r1
- Use `--no-think` for models without reasoning support
- Key points extraction always uses `think=False` and lower temperature (0.3)

## Testing

```bash
# Run all tests with mocks (default, no external calls)
pytest tests/ -v

# Run with real Ollama API calls
pytest tests/ --use-ollama

# Run with real DuckDuckGo search
pytest tests/ --use-search

# Run with both real services
pytest tests/ --use-ollama --use-search
```

Test structure:
- `tests/conftest.py` - Fixtures, mock data, and `--use-ollama`/`--use-search` flags
- `tests/test_config.py` - DEFAULTS validation
- `tests/test_ollama_client.py` - generate(), count_words()
- `tests/test_research.py` - search_duckduckgo(), build_research_context()
- `tests/test_outline_builder.py` - Prompt building, JSON extraction, YAML generation
- `tests/test_document_builder.py` - Outline loading, section prompts, key points, assembly
- `tests/test_batch_builder.py` - JSON loading, config merging, command building, skip detection
- `tests/test_link_builder.py` - Embeddings, similarity, link insertion, markdown parsing
- `tests/test_deep_research.py` - Page fetching, HTML stripping, summarization, caching
