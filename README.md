# GenAI Tools

A collection of Python tools for content generation and document management using local LLMs via Ollama.

## Overview

| Tool | Purpose |
|------|---------|
| `simple_publisher.py` | One-shot article generation with web research |
| `outline_builder.py` + `document_builder.py` | Multi-phase structured document generation |
| `batch_builder.py` | Batch processing multiple topics from JSON config |
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

### Quick Start

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

# With deep research (fetches and summarizes full web pages)
python simple_publisher.py \
  -t "Kubernetes Security Best Practices" \
  --deep-research \
  -o k8s-security.md
```

### Command Line Reference

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --topic` | Article topic (required) | - |
| `-a, --audience` | Target audience description | "college educated general audience" |
| `--type` | Content type: tutorial, concept, guide, reference | tutorial |
| `-w, --words` | Target word count | 1500 |
| `-p, --persona` | Writer persona (e.g., "a data scientist with 10 years experience") | None |
| `-c, --context` | Additional guidance for the writer | None |
| `-o, --output` | Output file path (omit for stdout) | stdout |

**Research Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--no-search` | Skip DuckDuckGo research entirely | False |
| `--deep-research` | Fetch full web pages and generate LLM summaries | False |
| `--no-cache` | Force fresh fetch, skip research cache | False |

**Model Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--ollama-url` | Ollama API URL | http://localhost:11434 |
| `--model` | Ollama model name | qwen3:14b |
| `--num-predict` | Maximum tokens to generate | 16384 |
| `--num-ctx` | Context window size | 16384 |
| `--temperature` | Sampling temperature (0.0-2.0) | 0.7 |
| `--repeat-penalty` | Repetition penalty (1.0 = none) | 1.1 |
| `--think` | Enable chain-of-thought reasoning | True |
| `--no-think` | Disable chain-of-thought reasoning | False |
| `--num-gpu` | GPU layers to offload (None = auto-detect) | None |

### Recommended Usage Patterns

**Quick draft with minimal research:**
```bash
python simple_publisher.py -t "Topic" --no-search -w 1000
```

**High-quality article with deep research:**
```bash
python simple_publisher.py \
  -t "Topic" \
  --deep-research \
  -p "an expert in the field" \
  -w 2500 \
  -o article.md
```

**Using a larger model (e.g., 30B):**
```bash
python simple_publisher.py \
  -t "Topic" \
  --model qwen3:30b \
  --num-gpu 20 \
  -o article.md
```

---

## Tool 2: Multi-Phase Document Builder

**Structured document generation** using YAML outlines for comprehensive tutorials, guides, and references. This two-step process allows you to review and edit the outline before generating content.

### Quick Start

```bash
# Step 1: Generate structured outline
python outline_builder.py \
  -t "Building REST APIs with FastAPI" \
  -w 6000 \
  -s 7 \
  -o fastapi-outline.yaml

# Step 2: (Optional) Edit the outline
# Review sections, adjust word counts, refine guidance

# Step 3: Build document from outline
python document_builder.py \
  -i fastapi-outline.yaml \
  -o fastapi-tutorial.md \
  --verbose
```

### Outline Builder

Generates a structured YAML outline that defines document sections, dependencies, and guidance.

#### Command Line Reference

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --topic` | Document topic (required) | - |
| `-a, --audience` | Target audience description | "college educated general audience" |
| `--type` | Content type: tutorial, concept, guide, reference | tutorial |
| `-w, --words` | Total target word count | 5000 |
| `-s, --sections` | Number of sections to generate | 5 |
| `-p, --persona` | Writer persona | None |
| `-c, --context` | Additional guidance for outline generation | None |
| `-o, --output` | Output YAML file (omit for stdout) | stdout |

**Research Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--no-search` | Skip DuckDuckGo research | False |
| `--deep-research` | Fetch full pages and generate summaries | False |
| `--no-cache` | Skip research cache | False |

**Model Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--ollama-url` | Ollama API URL | http://localhost:11434 |
| `--model` | Ollama model name | qwen3:14b |
| `--temperature` | Sampling temperature | 0.7 |
| `--num-predict` | Max tokens for outline generation | 8192 |
| `--think` | Enable chain-of-thought reasoning | True |
| `--no-think` | Disable chain-of-thought reasoning | False |
| `--num-gpu` | GPU layers (None = auto-detect) | None |

### Document Builder

Generates the full document from a YAML outline, processing sections sequentially with key point extraction for continuity.

#### Command Line Reference

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input YAML outline file (required) | - |
| `-o, --output` | Output markdown file (omit for stdout) | stdout |
| `--verbose` | Show detailed progress and key points | False |
| `--dry-run` | Validate outline without generating | False |
| `--section` | Generate only this section ID (for testing) | None |
| `--smooth` | Post-process to smooth section transitions | False |
| `--instructions` | Additional content constraints | None |

**Model Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--ollama-url` | Ollama API URL | http://localhost:11434 |
| `--model` | Ollama model name | qwen3:14b |
| `--temperature` | Sampling temperature | 0.7 |
| `--think` | Enable chain-of-thought reasoning | True |
| `--no-think` | Disable chain-of-thought reasoning | False |
| `--num-gpu` | GPU layers (None = auto-detect) | None |

### YAML Outline Schema

```yaml
metadata:
  title: "Document Title"
  topic: "The main topic"
  audience: "Target readers"
  content_type: "tutorial"
  total_word_count: 6000
  generated_at: "2024-12-30T10:00:00Z"
  persona: "a senior engineer who values simplicity"
  guidance: "Focus on practical examples over theory"

research:
  context: "Formatted DuckDuckGo search results or deep research summaries..."

sections:
  - id: "introduction"
    title: "Getting Started with FastAPI"
    order: 1
    word_count: 800
    position: "first"           # first, middle, or last
    section_role: "introduce"   # introduce, develop, or conclude
    transition_to: "core-concepts"
    dependencies: []
    keywords: ["FastAPI", "async", "Python"]
    guidance: "Hook the reader and explain why FastAPI matters"
    content_hints:
      - "Compare briefly to Flask/Django"
      - "Mention automatic OpenAPI docs"

  - id: "core-concepts"
    title: "Core Concepts and Architecture"
    order: 2
    word_count: 1200
    position: "middle"
    section_role: "develop"
    transition_to: "building-endpoints"
    dependencies: ["introduction"]
    keywords: ["Pydantic", "dependency injection", "async/await"]
    guidance: "Explain the foundational concepts needed for the rest"
    content_hints:
      - "Show Pydantic model examples"
      - "Explain the request lifecycle"
```

### Recommended Usage Patterns

**Validate outline before generating:**
```bash
python document_builder.py -i outline.yaml --dry-run
```

**Generate with verbose progress (recommended):**
```bash
python document_builder.py -i outline.yaml -o document.md --verbose
```

**Test a single section:**
```bash
python document_builder.py -i outline.yaml --section core-concepts --verbose
```

**Add content constraints:**
```bash
python document_builder.py \
  -i outline.yaml \
  -o document.md \
  --instructions "Use formal academic tone. Include citations where appropriate."
```

**Use a larger model for higher quality:**
```bash
python document_builder.py \
  -i outline.yaml \
  -o document.md \
  --model qwen3:30b \
  --verbose
```

### How It Works

1. **Outline Generation**: LLM generates JSON structure, converted to YAML for human editing
2. **Section Ordering**: Sections sorted by `order` field with dependency tracking
3. **Key Points Extraction**: After each section, 5-10 key facts are extracted
4. **Context Passing**: Key points from dependencies inform subsequent sections
5. **Position Awareness**: Intro/middle/conclusion sections get appropriate instructions
6. **Transition Smoothing**: Optional `--smooth` flag improves flow between sections

### Safety Features

- **Output file check**: Fails if output file already exists (prevents accidental overwrite)
- **Dry-run mode**: Validate outline structure without API calls
- **Timestamped output**: All progress messages include `[HH:MM:SS]` timestamps

---

## Tool 3: Batch Builder

**Batch processing** for generating multiple documents from a JSON configuration file. Runs `outline_builder.py` + `document_builder.py` for each topic with configurable cooldown between runs.

### Quick Start

```bash
# Preview what would be executed
python batch_builder.py -i batch.json --dry-run

# Run batch processing
python batch_builder.py -i batch.json

# Start from a specific topic (1-indexed)
python batch_builder.py -i batch.json --start-at 3

# Run only a single topic
python batch_builder.py -i batch.json --only 2
```

### JSON Configuration

```json
{
  "defaults": {
    "audience": "A software engineer learning AWS",
    "content_type": "tutorial",
    "words": 8000,
    "sections": 9,
    "persona": "A sys admin with AWS experience",
    "model": "qwen3:30b",
    "ollama_url": "http://localhost:11434",
    "temperature": 0.7,
    "deep_research": false,
    "think": true,
    "verbose": true,
    "document_instructions": "Use an academic and direct tone."
  },
  "output_dir": "./output",
  "cooldown_seconds": 10,
  "topics": [
    {
      "topic": "First AWS Deployment",
      "context": "Focus on differences from on-premises deployment"
    },
    {
      "topic": "AWS EC2 Instance Types",
      "context": "Cost optimization for Java applications",
      "words": 6000,
      "sections": 7
    }
  ]
}
```

### Command Line Reference

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input JSON batch file (required) | - |
| `-o, --output-dir` | Override output directory from JSON | from JSON |
| `--dry-run` | Show execution plan without running | False |
| `--start-at N` | Start at topic N (1-indexed) | 1 |
| `--only N` | Run only topic N (1-indexed) | None |
| `--skip-outlines` | Use existing outlines, only build documents | False |
| `--cooldown SECONDS` | Override cooldown between topics | from JSON or 10 |
| `--no-cooldown` | Disable cooldown between topics | False |

### JSON Schema

#### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `defaults` | object | No | Default values applied to all topics |
| `output_dir` | string | No | Output directory (default: `./output`) |
| `cooldown_seconds` | integer | No | GPU cooldown between topics (default: 10) |
| `topics` | array | **Yes** | Array of topic objects |

#### Defaults Object (all optional)

| Field | Type | Maps To | Default |
|-------|------|---------|---------|
| `audience` | string | `-a` | "college educated general audience" |
| `content_type` | string | `--type` | "tutorial" |
| `words` | integer | `-w` | 5000 |
| `sections` | integer | `-s` | 5 |
| `persona` | string | `-p` | None |
| `model` | string | `--model` | "qwen3:14b" |
| `ollama_url` | string | `--ollama-url` | from DEFAULTS |
| `temperature` | float | `--temperature` | 0.7 |
| `deep_research` | boolean | `--deep-research` | false |
| `no_cache` | boolean | `--no-cache` | false |
| `think` | boolean | `--think/--no-think` | true |
| `verbose` | boolean | `--verbose` | true |
| `document_instructions` | string | `--instructions` | None |
| `num_gpu` | integer | `--num-gpu` | None (auto-detect) |

#### Topic Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | **Yes** | The topic for generation |
| `context` | string | No | Outline guidance for `-c` |
| `outline_file` | string | No | Override auto-generated outline filename |
| `document_file` | string | No | Override auto-generated document filename |
| Any default field | varies | No | Overrides the default for this topic only |

### Output File Naming

Auto-generated from topic if not specified:

| Topic | Outline File | Document File |
|-------|--------------|---------------|
| "First AWS Deployment" | `first-aws-deployment-outline.yaml` | `FirstAwsDeployment.md` |
| "AWS EC2 Instance Types" | `aws-ec2-instance-types-outline.yaml` | `AwsEc2InstanceTypes.md` |

### Auto-Resume Behavior

Before each phase, the tool checks if output exists:
- If outline exists → skip outline generation
- If document exists → skip document generation
- If both exist → skip entire topic

This allows interrupted batches to resume automatically.

### Batch Report

On completion, a report is saved to `{output_dir}/batch-report.json`:

```json
{
  "input_file": "batch.json",
  "started_at": "2024-12-30T14:30:00Z",
  "completed_at": "2024-12-30T16:30:00Z",
  "duration_seconds": 7200,
  "summary": {
    "total": 5,
    "successful": 4,
    "skipped": 1,
    "failed": 0
  },
  "topics": [
    {
      "topic": "First AWS Deployment",
      "status": "success",
      "outline_file": "first-aws-deployment-outline.yaml",
      "document_file": "FirstAwsDeployment.md",
      "word_count": 8234
    }
  ]
}
```

---

## Tool 4: Link Builder

**Automated cross-reference linking** for markdown document collections using semantic similarity.

### Quick Start

```bash
# Preview mode - see what links would be created
python link_builder.py --dir ./docs --dry-run --verbose

# Apply changes with custom threshold
python link_builder.py --dir ./docs --similarity 0.7 --max-links 3

# Generate report
python link_builder.py --dir ./docs --output-report links.json
```

### Command Line Reference

| Flag | Description | Default |
|------|-------------|---------|
| `-d, --dir` | Directory containing markdown files (required) | - |
| `--recursive` | Include subdirectories | False |
| `--similarity` | Minimum similarity threshold (0.0-1.0) | 0.6 |
| `--max-links` | Maximum links to add per file | 5 |
| `--dry-run` | Preview changes without writing files | False |
| `--verbose` | Show detailed progress | False |
| `--exclude` | Glob patterns to exclude (e.g., "drafts/*") | [] |
| `--output-report` | Write JSON report of changes | None |

**Model Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--ollama-url` | Ollama API URL | http://localhost:11434 |
| `--model` | LLM for link text generation | qwen3:14b |
| `--embed-model` | Model for embeddings | nomic-embed-text |

### Recommended Usage Patterns

**Always preview first:**
```bash
python link_builder.py --dir ./docs --dry-run --verbose
```

**Conservative linking (high confidence only):**
```bash
python link_builder.py --dir ./docs --similarity 0.75 --max-links 2
```

**Aggressive linking (more connections):**
```bash
python link_builder.py --dir ./docs --similarity 0.5 --max-links 8
```

**Exclude certain directories:**
```bash
python link_builder.py --dir ./docs --recursive --exclude "drafts/*" "archive/*"
```

### How It Works

**Phase 1: Document Indexing**
- Scan directory for `*.md` files
- Extract title (first H1 or filename) and all headings
- Generate embedding from: title + headings + first 8000 chars
- Store document metadata for comparison

**Phase 2: Candidate Discovery**
- Compute pairwise cosine similarity between embeddings
- Filter pairs above threshold (default: 0.6)
- Exclude self-links and already-linked pairs
- Sort by similarity (highest first)

**Phase 3: Link Generation**
- LLM analyzes each candidate pair
- Returns anchor text and insertion point
- Validates link placement (not in headings or existing links)

**Phase 4: File Update**
- Insert markdown links at identified positions
- Create bidirectional links (A→B and B→A)
- Idempotent: running twice won't create duplicates

---

## Shared Configuration

### Default Settings

All tools share defaults defined in `genaitools/config.py`:

```python
DEFAULTS = {
    "ollama_url": "http://localhost:11434",
    "model": "qwen3:14b",
    "num_predict": 16384,
    "num_ctx": 16384,
    "repeat_penalty": 1.1,
    "temperature": 0.7,
    "think": True,
}
```

Override any setting via command-line flags.

### GPU Memory Management

The `--num-gpu` flag controls GPU layer offloading:

| Value | Behavior |
|-------|----------|
| (omitted) | Ollama auto-detects based on available VRAM |
| `99` | Force all layers to GPU (may OOM on large models) |
| `20` | Offload 20 layers to GPU, rest to CPU |
| `0` | CPU-only inference |

**Recommendation**: Omit `--num-gpu` unless you encounter issues. Ollama handles this well automatically.

### Deep Research (RAG)

The `--deep-research` flag enables a richer research pipeline:

1. **Search**: DuckDuckGo for 3 relevant pages
2. **Fetch**: Download full HTML (15s timeout)
3. **Strip**: Remove scripts, styles, navigation
4. **Summarize**: LLM generates 200-400 word focused summary per page
5. **Cache**: Results cached in `~/.cache/genaitools/research/`

```bash
# Force fresh fetch (skip cache)
python outline_builder.py -t "Topic" --deep-research --no-cache
```

**Trade-offs**:
- ~20-30 seconds vs ~2 seconds for regular search
- 3 additional LLM calls for summarization
- Much richer context (~1500-2500 chars of focused summaries)

### Timestamped Output

All tools display timestamps on status messages:

```
[10:15:32] Loading outline: fastapi-outline.yaml
[10:15:32]   Title: Building REST APIs with FastAPI
[10:15:32]   Sections: 7
[10:15:32] [1/7] Generating: Getting Started (800 words)
[10:16:45]   Generated 823 words
[10:16:47]   Extracting key points...
[10:16:52] [2/7] Generating: Core Concepts (1200 words)
```

This helps track timing of long-running operations.

---

## Model Recommendations

### Generation Models

| Model | VRAM | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| qwen3:14b | ~10GB | Fast | Good | Recommended default |
| qwen3:30b | ~20GB | Medium | Better | Use `--num-gpu` for partial offload |
| qwen3:8b | ~6GB | Very Fast | Acceptable | For quick drafts |

### Embedding Models

| Model | Dimensions | Notes |
|-------|------------|-------|
| nomic-embed-text | 768 | Fast, MIT licensed, recommended |
| mxbai-embed-large | 1024 | Higher quality, slower |

### Chain-of-Thought

- `--think` enables extended reasoning (works with qwen3, deepseek-r1)
- `--no-think` disables for faster generation
- Key points extraction always uses `think=False` for consistency

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
| `test_batch_builder.py` | JSON loading, config merging, command building |
| `test_link_builder.py` | Embeddings, similarity, link insertion |
| `test_deep_research.py` | Page fetching, HTML stripping, summarization |
| `test_output.py` | Timestamped printing |

---

## Troubleshooting

### "500 Server Error" with large models

**Cause**: Model too large for available GPU VRAM.

**Solutions**:
1. Let Ollama auto-detect: remove any `--num-gpu` flag
2. Force partial offload: `--num-gpu 20` (adjust based on VRAM)
3. Use a smaller model: `--model qwen3:14b`

### Output file already exists

**Cause**: Safety feature to prevent accidental overwrite.

**Solution**: Delete or rename the existing file, or use a different output path.

### Slow generation

**Possible causes**:
1. Large context window filling up
2. Chain-of-thought enabled
3. Deep research fetching slow pages

**Solutions**:
1. Reduce word count targets
2. Use `--no-think` for faster generation
3. Use regular search instead of `--deep-research`

### Research returns no results

**Cause**: DuckDuckGo rate limiting or network issues.

**Solutions**:
1. Wait a few minutes and retry
2. Use `--no-search` and rely on model knowledge
3. Use cached results if available

---

## License

MIT
