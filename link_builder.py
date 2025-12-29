#!/usr/bin/env python3
"""
Generate cross-reference links between markdown files in a directory.
Uses embeddings to find semantically similar documents, then LLM to suggest link placement.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genaitools import (
    DEFAULTS,
    generate,
    get_embedding,
    cosine_similarity,
    find_similar_pairs,
    DEFAULT_EMBED_MODEL,
)

# =============================================================================
# Module Constants
# =============================================================================

# Link generation parameters
LINK_GENERATION_TEMPERATURE = 0.3  # Low for consistency
LINK_GENERATION_NUM_PREDICT = 2048


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Document:
    """Represents a markdown document for linking."""
    path: Path
    title: str
    content: str
    headings: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    existing_links: set[str] = field(default_factory=set)


@dataclass
class LinkSuggestion:
    """A suggested link from source to target."""
    source_path: Path
    target_path: Path
    anchor_text: str
    context_before: str
    confidence: float
    reasoning: str


# =============================================================================
# Document Indexing (Phase 1)
# =============================================================================


def extract_title(content: str, filename: str) -> str:
    """Extract title from markdown content (first H1) or use filename."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return Path(filename).stem.replace("-", " ").replace("_", " ").title()


def extract_headings(content: str) -> list[str]:
    """Extract all headings from markdown content."""
    headings = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE):
        headings.append(match.group(2).strip())
    return headings


def extract_existing_links(content: str) -> set[str]:
    """Extract all existing link targets from markdown."""
    links = set()
    # Match [text](target) pattern
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
        target = match.group(2)
        # Normalize to just filename for comparison
        if not target.startswith(("http://", "https://", "#")):
            links.add(Path(target).name)
    return links


def load_documents(
    directory: Path,
    recursive: bool = False,
    exclude_patterns: list[str] | None = None,
) -> list[Document]:
    """
    Load all markdown documents from directory.

    Args:
        directory: Directory to scan
        recursive: Include subdirectories
        exclude_patterns: Glob patterns to exclude

    Returns:
        List of Document objects
    """
    exclude_patterns = exclude_patterns or []
    documents = []

    if recursive:
        md_files = list(directory.rglob("*.md"))
    else:
        md_files = list(directory.glob("*.md"))

    for path in md_files:
        # Check exclusions
        skip = False
        for pattern in exclude_patterns:
            if path.match(pattern):
                skip = True
                break
        if skip:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
            continue

        doc = Document(
            path=path,
            title=extract_title(content, path.name),
            content=content,
            headings=extract_headings(content),
            existing_links=extract_existing_links(content),
        )
        documents.append(doc)

    return documents


def index_documents(
    documents: list[Document],
    embed_model: str,
    ollama_url: str,
    verbose: bool = False,
) -> None:
    """
    Generate embeddings for all documents (modifies in place).

    Args:
        documents: List of documents to index
        embed_model: Ollama embedding model name
        ollama_url: Ollama API URL
        verbose: Print progress
    """
    if verbose:
        print(f"Generating embeddings with {embed_model}...")

    for i, doc in enumerate(documents):
        if verbose:
            print(f"  [{i + 1}/{len(documents)}] {doc.title}")

        # Build embedding text: title + headings + content (up to 8000 chars)
        heading_text = "\n".join(doc.headings) if doc.headings else ""
        prefix = f"{doc.title}\n\n{heading_text}\n\n" if heading_text else f"{doc.title}\n\n"
        remaining_chars = 8000 - len(prefix)
        text_for_embedding = prefix + doc.content[:remaining_chars]
        try:
            doc.embedding = get_embedding(
                text_for_embedding,
                model=embed_model,
                ollama_url=ollama_url,
            )
        except ValueError as e:
            print(f"Warning: Could not embed {doc.path}: {e}", file=sys.stderr)
            doc.embedding = []


# =============================================================================
# Candidate Discovery (Phase 2)
# =============================================================================


def find_link_candidates(
    documents: list[Document],
    similarity_threshold: float = 0.6,
    verbose: bool = False,
) -> list[tuple[int, int, float]]:
    """
    Find document pairs that might need cross-links.

    Args:
        documents: Indexed documents with embeddings
        similarity_threshold: Minimum similarity to consider
        verbose: Print progress

    Returns:
        List of (source_idx, target_idx, similarity) tuples
    """
    # Filter to only documents with embeddings
    doc_dicts = [{"embedding": doc.embedding} for doc in documents if doc.embedding]

    if len(doc_dicts) < 2:
        return []

    pairs = find_similar_pairs(doc_dicts, threshold=similarity_threshold)

    if verbose:
        print(f"Found {len(pairs)} candidate pairs above threshold {similarity_threshold}")

    return pairs


def filter_already_linked(
    documents: list[Document],
    candidates: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Remove pairs where source already links to target."""
    filtered = []
    for src_idx, tgt_idx, sim in candidates:
        src_doc = documents[src_idx]
        tgt_doc = documents[tgt_idx]

        # Check if source already links to target
        if tgt_doc.path.name not in src_doc.existing_links:
            filtered.append((src_idx, tgt_idx, sim))

    return filtered


# =============================================================================
# Link Generation (Phase 3)
# =============================================================================


def build_link_prompt(source: Document, target: Document) -> str:
    """Build prompt for LLM to suggest link placement."""
    # Truncate content for prompt
    source_excerpt = source.content[:2000]
    target_summary = target.content[:500]

    return f"""You are analyzing two related documents to suggest a cross-reference link.

SOURCE DOCUMENT: "{source.title}"
---
{source_excerpt}
---

TARGET DOCUMENT: "{target.title}"
Summary: {target_summary}
Headings: {', '.join(target.headings[:5]) if target.headings else 'None'}

TASK: Identify where in the SOURCE document we should add a link to the TARGET document.

Requirements:
1. Find a phrase in the source that semantically relates to the target
2. The link should feel natural, not forced
3. Prefer linking from conceptual mentions, not just keyword matches
4. The anchor_text must be EXACT text that appears in the source document
5. Do NOT suggest linking text that is already inside a markdown link [...](...)
6. Do NOT suggest linking heading text (lines starting with #)

OUTPUT FORMAT: Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "should_link": true,
  "confidence": 0.8,
  "anchor_text": "exact text from source to make into link",
  "context_before": "few words before anchor for identification",
  "reasoning": "brief explanation of why this link makes sense"
}}

If no natural link opportunity exists, return:
{{
  "should_link": false,
  "confidence": 0.0,
  "anchor_text": "",
  "context_before": "",
  "reasoning": "explanation of why no link fits"
}}

Respond with JSON only:"""


def parse_link_response(response: str) -> dict[str, Any] | None:
    """Parse LLM response for link suggestion."""
    # Try to find JSON in response
    try:
        # First try direct parse
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def generate_link_suggestions(
    documents: list[Document],
    candidates: list[tuple[int, int, float]],
    ollama_url: str,
    model: str,
    max_links_per_file: int = 5,
    verbose: bool = False,
) -> list[LinkSuggestion]:
    """
    Use LLM to generate specific link suggestions for candidate pairs.

    Args:
        documents: List of documents
        candidates: Candidate pairs from similarity analysis
        ollama_url: Ollama API URL
        model: LLM model for link generation
        max_links_per_file: Maximum links to add per source file
        verbose: Print progress

    Returns:
        List of LinkSuggestion objects
    """
    suggestions = []
    links_per_source: dict[Path, int] = {}

    if verbose:
        print(f"Generating link suggestions with {model}...")

    for i, (src_idx, tgt_idx, sim) in enumerate(candidates):
        src_doc = documents[src_idx]
        tgt_doc = documents[tgt_idx]

        # Check max links per file
        current_links = links_per_source.get(src_doc.path, 0)
        if current_links >= max_links_per_file:
            continue

        if verbose:
            print(f"  [{i + 1}/{len(candidates)}] {src_doc.title} → {tgt_doc.title} (sim={sim:.2f})")

        prompt = build_link_prompt(src_doc, tgt_doc)

        response = generate(
            prompt=prompt,
            ollama_url=ollama_url,
            model=model,
            temperature=LINK_GENERATION_TEMPERATURE,
            num_predict=LINK_GENERATION_NUM_PREDICT,
            think=False,
        )

        if response.startswith("Error:"):
            if verbose:
                print(f"    Error: {response}")
            continue

        result = parse_link_response(response)
        if result is None:
            if verbose:
                print("    Could not parse response")
            continue

        if result.get("should_link") and result.get("anchor_text"):
            # Verify anchor text exists in source
            anchor = result["anchor_text"]
            if anchor in src_doc.content:
                suggestion = LinkSuggestion(
                    source_path=src_doc.path,
                    target_path=tgt_doc.path,
                    anchor_text=anchor,
                    context_before=result.get("context_before", ""),
                    confidence=result.get("confidence", 0.5),
                    reasoning=result.get("reasoning", ""),
                )
                suggestions.append(suggestion)
                links_per_source[src_doc.path] = current_links + 1

                if verbose:
                    print(f"    ✓ Link: '{anchor}' → {tgt_doc.title}")
            else:
                if verbose:
                    print(f"    Anchor text not found in source: '{anchor[:50]}...'")
        else:
            if verbose:
                reason = result.get("reasoning", "No reason given")
                print(f"    No link suggested: {reason[:60]}...")

    return suggestions


# =============================================================================
# File Update (Phase 4)
# =============================================================================


def compute_relative_path(source: Path, target: Path) -> str:
    """Compute relative path from source to target."""
    try:
        return os.path.relpath(target, source.parent)
    except ValueError:
        # Different drives on Windows
        return str(target)


def is_inside_link(content: str, position: int) -> bool:
    """Check if position is inside an existing markdown link."""
    # Look backwards for [ and check if it's an unclosed link
    before = content[:position]
    after = content[position:]

    # Find last [ before position
    last_open = before.rfind("[")
    if last_open == -1:
        return False

    # Check if there's a ] between [ and position
    between = content[last_open:position]
    if "]" in between:
        return False

    # Check if there's a ] followed by ( after position (completing the link)
    # This is a simplified check
    close_bracket = after.find("]")
    if close_bracket != -1:
        rest = after[close_bracket:]
        if rest.startswith("]("):
            return True

    return False


def is_inside_heading(content: str, position: int) -> bool:
    """Check if position is inside a heading line."""
    # Find the start of the line containing this position
    line_start = content.rfind("\n", 0, position) + 1
    line = content[line_start:position + 100].split("\n")[0]
    return line.lstrip().startswith("#")


def insert_link(content: str, anchor_text: str, target_path: str) -> str | None:
    """
    Insert a markdown link into content.

    Args:
        content: Original markdown content
        anchor_text: Text to convert to link
        target_path: Relative path to target file

    Returns:
        Modified content with link, or None if insertion failed
    """
    # Find first occurrence of anchor text
    position = content.find(anchor_text)
    if position == -1:
        return None

    # Check if it's safe to insert here
    if is_inside_link(content, position):
        # Try to find another occurrence
        next_pos = content.find(anchor_text, position + 1)
        if next_pos == -1 or is_inside_link(content, next_pos):
            return None
        position = next_pos

    if is_inside_heading(content, position):
        # Try to find another occurrence
        next_pos = content.find(anchor_text, position + 1)
        if next_pos == -1 or is_inside_heading(content, next_pos):
            return None
        position = next_pos

    # Build the link
    link = f"[{anchor_text}]({target_path})"

    # Insert the link
    new_content = (
        content[:position] +
        link +
        content[position + len(anchor_text):]
    )

    return new_content


def apply_suggestions(
    documents: list[Document],
    suggestions: list[LinkSuggestion],
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[Path, list[str]]:
    """
    Apply link suggestions to files.

    Args:
        documents: List of documents
        suggestions: Link suggestions to apply
        dry_run: Preview changes without writing
        verbose: Print progress

    Returns:
        Dict mapping file paths to list of changes made
    """
    changes: dict[Path, list[str]] = {}

    # Group suggestions by source file
    by_source: dict[Path, list[LinkSuggestion]] = {}
    for sug in suggestions:
        by_source.setdefault(sug.source_path, []).append(sug)

    for src_path, src_suggestions in by_source.items():
        # Find the document
        doc = next((d for d in documents if d.path == src_path), None)
        if doc is None:
            continue

        content = doc.content
        file_changes = []

        for sug in src_suggestions:
            rel_path = compute_relative_path(sug.source_path, sug.target_path)
            new_content = insert_link(content, sug.anchor_text, rel_path)

            if new_content is not None:
                content = new_content
                change_desc = f"'{sug.anchor_text}' → {sug.target_path.name}"
                file_changes.append(change_desc)
                if verbose:
                    print(f"  {src_path.name}: {change_desc}")

        if file_changes:
            changes[src_path] = file_changes
            if not dry_run:
                src_path.write_text(content, encoding="utf-8")

    return changes


def make_bidirectional(
    suggestions: list[LinkSuggestion],
    documents: list[Document],
    ollama_url: str,
    model: str,
    verbose: bool = False,
) -> list[LinkSuggestion]:
    """
    For each A→B link, also generate B→A link.

    Args:
        suggestions: Original link suggestions
        documents: List of documents
        ollama_url: Ollama API URL
        model: LLM model for link generation
        verbose: Print progress

    Returns:
        Extended list including reverse links
    """
    all_suggestions = list(suggestions)

    # Create reverse candidates
    reverse_candidates = []
    for sug in suggestions:
        # Find document indices
        src_idx = next((i for i, d in enumerate(documents) if d.path == sug.source_path), None)
        tgt_idx = next((i for i, d in enumerate(documents) if d.path == sug.target_path), None)

        if src_idx is not None and tgt_idx is not None:
            # Check if reverse link already exists
            tgt_doc = documents[tgt_idx]
            src_doc = documents[src_idx]
            if src_doc.path.name not in tgt_doc.existing_links:
                reverse_candidates.append((tgt_idx, src_idx, 1.0))  # Use 1.0 since we know they're related

    if reverse_candidates:
        if verbose:
            print(f"\nGenerating {len(reverse_candidates)} reverse links...")

        reverse_suggestions = generate_link_suggestions(
            documents=documents,
            candidates=reverse_candidates,
            ollama_url=ollama_url,
            model=model,
            max_links_per_file=1,  # Just one reverse link per file
            verbose=verbose,
        )
        all_suggestions.extend(reverse_suggestions)

    return all_suggestions


# =============================================================================
# Main CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-reference links between markdown files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--dir",
        required=True,
        type=Path,
        help="Directory containing markdown files"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include subdirectories"
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.6,
        help="Minimum similarity threshold (0-1)"
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=5,
        help="Maximum links to add per file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Glob patterns to exclude"
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULTS["ollama_url"],
        help="Ollama API URL"
    )
    parser.add_argument(
        "--model",
        default=DEFAULTS["model"],
        help="LLM model for link text generation"
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Model for embeddings"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Write JSON report of changes"
    )

    args = parser.parse_args()

    # Validate directory
    if not args.dir.is_dir():
        print(f"Error: {args.dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Phase 1: Load and index documents
    print(f"Loading documents from {args.dir}...")
    documents = load_documents(
        args.dir,
        recursive=args.recursive,
        exclude_patterns=args.exclude,
    )

    if len(documents) < 2:
        print(f"Found only {len(documents)} document(s). Need at least 2 for linking.")
        sys.exit(0)

    print(f"Found {len(documents)} markdown files")

    # Generate embeddings
    index_documents(
        documents,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        verbose=args.verbose,
    )

    # Phase 2: Find candidates
    print("\nFinding similar document pairs...")
    candidates = find_link_candidates(
        documents,
        similarity_threshold=args.similarity,
        verbose=args.verbose,
    )

    # Filter already-linked pairs
    candidates = filter_already_linked(documents, candidates)

    if not candidates:
        print("No new link opportunities found.")
        sys.exit(0)

    print(f"Found {len(candidates)} potential link opportunities")

    # Phase 3: Generate link suggestions
    print("\nAnalyzing documents for link placement...")
    suggestions = generate_link_suggestions(
        documents=documents,
        candidates=candidates,
        ollama_url=args.ollama_url,
        model=args.model,
        max_links_per_file=args.max_links,
        verbose=args.verbose,
    )

    if not suggestions:
        print("No suitable links found after analysis.")
        sys.exit(0)

    # Make bidirectional
    print("\nGenerating bidirectional links...")
    suggestions = make_bidirectional(
        suggestions=suggestions,
        documents=documents,
        ollama_url=args.ollama_url,
        model=args.model,
        verbose=args.verbose,
    )

    print(f"\nTotal link suggestions: {len(suggestions)}")

    # Phase 4: Apply changes
    if args.dry_run:
        print("\n[DRY RUN] Would make these changes:")
    else:
        print("\nApplying links...")

    changes = apply_suggestions(
        documents=documents,
        suggestions=suggestions,
        dry_run=args.dry_run,
        verbose=True,  # Always show what changed
    )

    # Summary
    total_links = sum(len(c) for c in changes.values())
    files_modified = len(changes)

    if args.dry_run:
        print(f"\n[DRY RUN] Would add {total_links} links across {files_modified} files")
    else:
        print(f"\nAdded {total_links} links across {files_modified} files")

    # Write report if requested
    if args.output_report:
        report = {
            "total_links": total_links,
            "files_modified": files_modified,
            "changes": {str(k): v for k, v in changes.items()},
            "suggestions": [
                {
                    "source": str(s.source_path),
                    "target": str(s.target_path),
                    "anchor_text": s.anchor_text,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                }
                for s in suggestions
            ],
        }
        args.output_report.write_text(json.dumps(report, indent=2))
        print(f"Report written to: {args.output_report}")


if __name__ == "__main__":
    main()
