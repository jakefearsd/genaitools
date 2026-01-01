#!/usr/bin/env python3
"""
Build multi-section documents from YAML outlines.
Generates each section sequentially, extracting key points to pass as context.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from genaitools import DEFAULTS, generate, count_words, tprint

# =============================================================================
# Module Constants
# =============================================================================

# Generation parameters for key points extraction
KEY_POINTS_TEMPERATURE = 0.3  # Lower for consistency
KEY_POINTS_NUM_PREDICT = 1024  # Concise extraction

# Generation parameters for transition smoothing
SMOOTHING_TEMPERATURE = 0.3  # Lower for consistency
SMOOTHING_NUM_PREDICT = 2048  # More space for rewriting

# Paragraph extraction for smoothing
SMOOTHING_CONTEXT_PARAGRAPHS = 3  # ~2-3 paragraphs at boundaries
SMOOTHING_MIN_PARAGRAPHS = 2

# Display settings
KEY_POINTS_DISPLAY_LENGTH = 80  # Truncation for verbose output


# =============================================================================
# Utility Functions
# =============================================================================


def _parse_bullet_list(text: str) -> list[str]:
    """
    Parse bullet points (- or *) from text.

    Args:
        text: Text containing bullet-pointed list

    Returns:
        List of bullet point contents with prefixes stripped
    """
    points = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            points.append(line[2:].strip())
        elif line.startswith("* "):
            points.append(line[2:].strip())
    return points


@dataclass
class SectionResult:
    """Result of generating a single section."""
    id: str
    title: str
    content: str
    word_count: int
    key_points: list[str]


def load_outline(path: str) -> dict[str, Any]:
    """Load and validate YAML outline."""
    with open(path) as f:
        outline = yaml.safe_load(f)

    # Basic validation
    required = ["metadata", "sections"]
    for key in required:
        if key not in outline:
            raise ValueError(f"Outline missing required key: {key}")

    if not outline["sections"]:
        raise ValueError("Outline has no sections")

    # Validate sections
    for section in outline["sections"]:
        for key in ["id", "title", "order", "word_count"]:
            if key not in section:
                raise ValueError(f"Section missing required key: {key}")

    return outline


def build_section_prompt(
    section: dict[str, Any],
    metadata: dict[str, Any],
    research_context: str,
    previous_key_points: dict[str, list[str]],
    instructions: str | None = None,
) -> str:
    """
    Build prompt for generating a single section.

    Follows simple_publisher.py prompt ordering:
    1. PERSONA - Writer identity
    2. TASK - Section-specific assignment
    3. DOCUMENT CONTEXT - Where this section fits
    4. SECTION POSITION - first/middle/last role in document
    5. CONTINUITY - Key points from dependencies
    6. GUIDANCE - Section-specific direction
    7. OUTPUT INSTRUCTIONS - Additional constraints on content style
    8. RESEARCH - Web search context
    9. REQUIREMENTS - Format constraints
    """
    parts = []

    # 1. PERSONA
    persona = metadata.get("persona", "an expert technical writer")
    parts.append(f"You are {persona}.")

    # 2. TASK
    content_type = metadata.get("content_type", "tutorial")
    topic = metadata.get("topic", "the topic")
    audience = metadata.get("audience", "a general audience")
    parts.append(
        f'Write the "{section["title"]}" section of a {content_type} '
        f'about "{topic}" for {audience}.\n\n'
        f'TARGET: Approximately {section["word_count"]} words.'
    )

    # 3. DOCUMENT CONTEXT
    title = metadata.get("title", topic)
    keywords = section.get("keywords", [])
    parts.append(
        f"DOCUMENT CONTEXT:\n"
        f'This is section {section["order"]} of the document titled "{title}".'
        + (f'\nKeywords to cover: {", ".join(keywords)}' if keywords else "")
    )

    # 4. SECTION POSITION AND ROLE - Critical for document flow
    position = section.get("position", "middle")
    transition_to = section.get("transition_to")

    if position == "first":
        parts.append(
            "SECTION POSITION: This is the INTRODUCTION of the document.\n"
            "- Set up the topic and preview what's to come\n"
            "- Do NOT include a conclusion - this is just the beginning\n"
            "- End by transitioning to the next topic"
        )
    elif position == "last":
        parts.append(
            "SECTION POSITION: This is the FINAL section of the document.\n"
            "- You SHOULD include a conclusion and summary\n"
            "- Tie together themes from ALL previous sections\n"
            "- Provide actionable next steps for the reader"
        )
    else:  # middle
        next_section_hint = f"'{transition_to}'" if transition_to else "the following section"
        parts.append(
            "SECTION POSITION: This is a MIDDLE section of a larger document.\n"
            "CRITICAL RULES FOR MIDDLE SECTIONS:\n"
            "- Do NOT include a conclusion, summary, or 'final thoughts' subsection\n"
            "- Do NOT use phrases like 'In conclusion', 'To summarize', 'To wrap up', 'Finally'\n"
            "- Do NOT end with a summary of what was covered\n"
            "- Instead, END with a BRIDGE sentence that transitions to the next topic\n"
            "- The reader will continue to the next section immediately\n"
            f"- Next section: {next_section_hint}"
        )

    # 5. CONTINUITY - Key points from dependencies
    dependencies = section.get("dependencies", [])
    if dependencies and previous_key_points:
        continuity_parts = []
        for dep_id in dependencies:
            if dep_id in previous_key_points:
                dep_points = previous_key_points[dep_id]
                if dep_points:
                    continuity_parts.append(
                        f"From '{dep_id}':\n" +
                        "\n".join(f"- {point}" for point in dep_points)
                    )
        if continuity_parts:
            parts.append(
                "PRIOR SECTION KEY POINTS (build on these, don't repeat verbatim):\n" +
                "\n\n".join(continuity_parts)
            )

    # 6. SECTION GUIDANCE
    guidance_text = section.get("guidance", "")
    content_hints = section.get("content_hints", [])
    overall_guidance = metadata.get("guidance", "")

    guidance_parts = []
    if guidance_text:
        guidance_parts.append(guidance_text)
    if content_hints:
        guidance_parts.append("Content hints:\n" + "\n".join(f"- {h}" for h in content_hints))
    if overall_guidance:
        guidance_parts.append(f"Overall document guidance: {overall_guidance}")

    if guidance_parts:
        parts.append("SECTION GUIDANCE:\n" + "\n\n".join(guidance_parts))

    # 7. OUTPUT INSTRUCTIONS - Additional constraints on content style
    if instructions:
        parts.append(f"OUTPUT INSTRUCTIONS:\n{instructions}")

    # 8. RESEARCH
    if research_context:
        parts.append(f"RESEARCH CONTEXT:\n{research_context}")

    # 9. REQUIREMENTS (position-aware)
    if position == "middle":
        parts.append(
            "REQUIREMENTS:\n"
            f'1. Start with the section heading: ## {section["title"]}\n'
            "2. Use Markdown syntax (### for subsections, `code` for inline, ```lang for blocks)\n"
            "3. Include practical examples with code where relevant\n"
            "4. Reference concepts from prior sections naturally\n"
            "5. Write for the target audience level\n"
            "6. Be comprehensive within your word target\n"
            "7. END with a transition sentence to the next topic - NOT a conclusion\n"
            "8. Do NOT include any 'Conclusion', 'Summary', or 'Final Thoughts' subsection\n\n"
            "Write this section now. Start with the heading."
        )
    elif position == "last":
        parts.append(
            "REQUIREMENTS:\n"
            f'1. Start with the section heading: ## {section["title"]}\n'
            "2. Use Markdown syntax (### for subsections, `code` for inline, ```lang for blocks)\n"
            "3. Summarize key themes from the entire document\n"
            "4. Provide actionable next steps\n"
            "5. End on a strong, memorable note\n\n"
            "Write this final section now. Start with the heading."
        )
    else:  # first
        parts.append(
            "REQUIREMENTS:\n"
            f'1. Start with the section heading: ## {section["title"]}\n'
            "2. Use Markdown syntax (### for subsections, `code` for inline, ```lang for blocks)\n"
            "3. Hook the reader and establish relevance\n"
            "4. Preview what the document will cover\n"
            "5. End by transitioning to the next section\n"
            "6. Do NOT include a conclusion - this is the introduction\n\n"
            "Write this introduction now. Start with the heading."
        )

    return "\n\n".join(parts)


def build_key_points_prompt(section_title: str, section_content: str) -> str:
    """Build prompt for extracting key points from a generated section."""
    return f"""You are analyzing a section of a larger document to extract key points for continuity.

SECTION TITLE: {section_title}
SECTION CONTENT:
{section_content}

Extract 5-10 key points from this section that subsequent sections might need to reference. Focus on:
1. Specific facts, definitions, or concepts introduced
2. Examples or code patterns established
3. Terminology defined (term: definition format)
4. Decisions or recommendations made
5. Open questions or topics deferred to later sections

Format as a simple bulleted list with one point per line starting with "- ". Be specific and factual, not vague.

WRONG: "- Discussed the importance of configuration"
RIGHT: "- Configuration uses YAML format with three required fields: host, port, timeout"

WRONG: "- Covered error handling"
RIGHT: "- Errors are wrapped in ApiError class with code, message, and retry_after fields"

KEY POINTS:"""


def extract_key_points(
    section_title: str,
    section_content: str,
    ollama_url: str,
    model: str,
) -> list[str]:
    """Extract key points from a generated section."""
    prompt = build_key_points_prompt(section_title, section_content)

    response = generate(
        prompt=prompt,
        ollama_url=ollama_url,
        model=model,
        temperature=KEY_POINTS_TEMPERATURE,
        num_predict=KEY_POINTS_NUM_PREDICT,
        think=False,  # No chain-of-thought for extraction
    )

    if response.startswith("Error:"):
        return []

    return _parse_bullet_list(response)


def generate_section(
    section: dict[str, Any],
    metadata: dict[str, Any],
    research_context: str,
    previous_key_points: dict[str, list[str]],
    ollama_url: str,
    model: str,
    temperature: float,
    think: bool,
    verbose: bool,
    instructions: str | None = None,
    num_gpu: int | None = None,
) -> SectionResult:
    """Generate a single section and extract its key points."""
    section_id = section["id"]
    section_title = section["title"]

    if verbose:
        deps = section.get("dependencies", [])
        dep_str = ", ".join(deps) if deps else "none"
        tprint(f"  Dependencies: {dep_str}")
        if deps:
            total_points = sum(len(previous_key_points.get(d, [])) for d in deps)
            tprint(f"  Passing {total_points} key points from dependencies")

    # Build and send prompt
    prompt = build_section_prompt(
        section=section,
        metadata=metadata,
        research_context=research_context,
        previous_key_points=previous_key_points,
        instructions=instructions,
    )

    if verbose:
        tprint(f"  Calling Ollama ({model})...")

    content = generate(
        prompt=prompt,
        ollama_url=ollama_url,
        model=model,
        temperature=temperature,
        think=think,
        num_gpu=num_gpu,
    )

    if content.startswith("Error:"):
        tprint(f"  ERROR: {content}", file=sys.stderr)
        return SectionResult(
            id=section_id,
            title=section_title,
            content=f"## {section_title}\n\n*Generation failed: {content}*\n",
            word_count=0,
            key_points=[],
        )

    words = count_words(content)
    if verbose:
        tprint(f"  Generated {words} words")

    # Extract key points
    if verbose:
        tprint("  Extracting key points...")

    key_points = extract_key_points(
        section_title=section_title,
        section_content=content,
        ollama_url=ollama_url,
        model=model,
    )

    if verbose and key_points:
        tprint(f"  Key points ({len(key_points)}):")
        for point in key_points[:5]:  # Show first 5
            tprint(f"    - {point[:KEY_POINTS_DISPLAY_LENGTH]}{'...' if len(point) > KEY_POINTS_DISPLAY_LENGTH else ''}")
        if len(key_points) > 5:
            tprint(f"    ... and {len(key_points) - 5} more")

    return SectionResult(
        id=section_id,
        title=section_title,
        content=content,
        word_count=words,
        key_points=key_points,
    )


def build_smoothing_prompt(
    section_a_title: str,
    section_a_ending: str,
    section_b_title: str,
    section_b_beginning: str,
) -> str:
    """Build prompt for smoothing the transition between two sections."""
    return f"""You are editing a document for flow and continuity between sections.

END OF SECTION "{section_a_title}":
{section_a_ending}

START OF SECTION "{section_b_title}":
{section_b_beginning}

TASK: Rewrite ONLY the transition between these sections to improve flow.
- If section A ends with a conclusion, summary, or "final thoughts", REMOVE it
- Remove phrases like "In conclusion", "To summarize", "To wrap up"
- Ensure section A ends with content or a bridge to section B
- Ensure section B begins naturally, connecting to section A
- Preserve ALL substantive content - only change transitional language
- Keep approximately the same word count

OUTPUT FORMAT (use these exact markers):
===SECTION_A_END===
[rewritten ending of section A - last 2-3 paragraphs]
===SECTION_B_START===
[rewritten beginning of section B - first 2-3 paragraphs]
===END==="""


def extract_smoothed_parts(response: str) -> tuple[str, str] | None:
    """Extract the smoothed section parts from LLM response."""
    # Find section A ending
    a_match = re.search(r"===SECTION_A_END===\s*(.*?)\s*===SECTION_B_START===", response, re.DOTALL)
    b_match = re.search(r"===SECTION_B_START===\s*(.*?)\s*===END===", response, re.DOTALL)

    if a_match and b_match:
        return a_match.group(1).strip(), b_match.group(1).strip()
    return None


def smooth_transitions(
    results: list[SectionResult],
    ollama_url: str,
    model: str,
    verbose: bool,
) -> list[SectionResult]:
    """
    Smooth transitions between adjacent sections using LLM.

    For each pair of adjacent sections:
    1. Extract the ending of section N (2-3 paragraphs)
    2. Extract the beginning of section N+1 (2-3 paragraphs)
    3. Ask LLM to rewrite the transition for better flow
    4. Replace the affected portions in the results

    Paragraph Extraction Strategy:
        - Paragraphs are split by double newlines (\\n\\n)
        - Takes last SMOOTHING_CONTEXT_PARAGRAPHS from ending section
        - Takes first SMOOTHING_CONTEXT_PARAGRAPHS from beginning section
        - Falls back to SMOOTHING_MIN_PARAGRAPHS if section is short

    Args:
        results: List of generated section results
        ollama_url: Ollama API endpoint
        model: Model name for generation
        verbose: Whether to print progress

    Returns:
        Modified SectionResult list with smoothed transitions
    """
    if len(results) < 2:
        return results

    smoothed = [r for r in results]  # Copy list

    for i in range(len(results) - 1):
        section_a = smoothed[i]
        section_b = smoothed[i + 1]

        if verbose:
            tprint(f"  Smoothing: '{section_a.title}' → '{section_b.title}'")

        # Extract ~last 500 chars of section A and ~first 500 chars of section B
        # This roughly corresponds to 2-3 paragraphs
        content_a = section_a.content
        content_b = section_b.content

        # Find good split points (paragraph boundaries)
        a_paragraphs = content_a.split("\n\n")
        b_paragraphs = content_b.split("\n\n")

        # Take last 2-3 paragraphs of A (but not if it's just the heading)
        a_ending_paras = a_paragraphs[-SMOOTHING_CONTEXT_PARAGRAPHS:] if len(a_paragraphs) > SMOOTHING_CONTEXT_PARAGRAPHS else a_paragraphs[-SMOOTHING_MIN_PARAGRAPHS:]
        a_ending = "\n\n".join(a_ending_paras)

        # Take first 2-3 paragraphs of B
        b_beginning_paras = b_paragraphs[:SMOOTHING_CONTEXT_PARAGRAPHS] if len(b_paragraphs) > SMOOTHING_CONTEXT_PARAGRAPHS else b_paragraphs[:SMOOTHING_MIN_PARAGRAPHS]
        b_beginning = "\n\n".join(b_beginning_paras)

        # Build and send prompt
        prompt = build_smoothing_prompt(
            section_a_title=section_a.title,
            section_a_ending=a_ending,
            section_b_title=section_b.title,
            section_b_beginning=b_beginning,
        )

        response = generate(
            prompt=prompt,
            ollama_url=ollama_url,
            model=model,
            temperature=SMOOTHING_TEMPERATURE,
            num_predict=SMOOTHING_NUM_PREDICT,
            think=False,
        )

        if response.startswith("Error:"):
            if verbose:
                tprint(f"    Smoothing failed: {response}")
            continue

        # Extract smoothed parts
        smoothed_parts = extract_smoothed_parts(response)
        if smoothed_parts is None:
            if verbose:
                tprint("    Could not parse smoothed response")
            continue

        new_a_ending, new_b_beginning = smoothed_parts

        # Rebuild section A content
        a_prefix_paras = a_paragraphs[:-len(a_ending_paras)]
        new_a_content = "\n\n".join(a_prefix_paras + [new_a_ending])

        # Rebuild section B content
        b_suffix_paras = b_paragraphs[len(b_beginning_paras):]
        new_b_content = "\n\n".join([new_b_beginning] + b_suffix_paras)

        # Update results
        smoothed[i] = SectionResult(
            id=section_a.id,
            title=section_a.title,
            content=new_a_content,
            word_count=count_words(new_a_content),
            key_points=section_a.key_points,
        )
        smoothed[i + 1] = SectionResult(
            id=section_b.id,
            title=section_b.title,
            content=new_b_content,
            word_count=count_words(new_b_content),
            key_points=section_b.key_points,
        )

        if verbose:
            tprint("    Transition smoothed")

    return smoothed


def assemble_document(results: list[SectionResult], metadata: dict[str, Any]) -> str:
    """Assemble all sections into final Markdown document."""
    parts = []

    # Document title
    title = metadata.get("title", "Document")
    parts.append(f"# {title}\n")

    # All section content
    for result in results:
        parts.append(result.content)

    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-section document from YAML outline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input YAML outline file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Markdown file (default: stdout)"
    )
    parser.add_argument(
        "--section",
        help="Generate only this section ID (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress and key points extraction"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate outline and show plan without generating"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Post-process to smooth transitions between sections"
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULTS["ollama_url"],
        help="Ollama API URL"
    )
    parser.add_argument(
        "--model",
        default=DEFAULTS["model"],
        help="Ollama model"
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="Number of GPU layers (default: let Ollama auto-detect)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULTS["temperature"],
        help="Temperature for generation"
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=DEFAULTS["think"],
        help="Enable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--instructions",
        help="Additional instructions for content generation (e.g., 'Do not include code examples')"
    )

    args = parser.parse_args()

    # Handle --no-think override
    think_enabled = args.think and not args.no_think

    # Check if output file already exists
    if args.output and Path(args.output).exists():
        tprint(f"Error: Output file already exists: {args.output}", file=sys.stderr)
        sys.exit(1)

    # Load outline
    try:
        outline = load_outline(args.input)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        tprint(f"Failed to load outline: {e}", file=sys.stderr)
        sys.exit(1)

    metadata = outline["metadata"]
    sections = sorted(outline["sections"], key=lambda s: s["order"])
    research_context = outline.get("research", {}).get("context", "")

    # Print summary
    tprint(f"Loading outline: {args.input}")
    tprint(f"  Title: {metadata.get('title', 'Untitled')}")
    tprint(f"  Sections: {len(sections)}")
    tprint(f"  Total target: {metadata.get('total_word_count', 'unspecified')} words")

    # Filter to single section if requested
    if args.section:
        sections = [s for s in sections if s["id"] == args.section]
        if not sections:
            tprint(f"Section not found: {args.section}", file=sys.stderr)
            sys.exit(1)
        tprint(f"  Generating only: {args.section}")

    # Dry run - just show plan
    if args.dry_run:
        tprint("Dry run - sections to generate:")
        for section in sections:
            deps = section.get("dependencies", [])
            dep_str = f" (deps: {', '.join(deps)})" if deps else ""
            tprint(f"  {section['order']}. {section['title']} ({section['word_count']} words){dep_str}")
        tprint("Outline is valid. Remove --dry-run to generate.")
        return

    # Generate sections
    tprint("")
    results: list[SectionResult] = []
    key_points: dict[str, list[str]] = {}

    for i, section in enumerate(sections, 1):
        tprint(f"[{i}/{len(sections)}] Generating: {section['title']} ({section['word_count']} words)")

        result = generate_section(
            section=section,
            metadata=metadata,
            research_context=research_context,
            previous_key_points=key_points,
            ollama_url=args.ollama_url,
            model=args.model,
            temperature=args.temperature,
            think=think_enabled,
            verbose=args.verbose,
            instructions=args.instructions,
            num_gpu=args.num_gpu,
        )

        results.append(result)
        key_points[result.id] = result.key_points
        tprint("")

    # Optional smoothing pass
    if args.smooth and len(results) > 1:
        tprint("Smoothing transitions between sections...")
        results = smooth_transitions(
            results=results,
            ollama_url=args.ollama_url,
            model=args.model,
            verbose=args.verbose,
        )
        tprint("")

    # Assemble document
    document = assemble_document(results, metadata)
    total_words = sum(r.word_count for r in results)
    target_words = metadata.get("total_word_count", 0)

    if target_words:
        tprint(f"Document complete! Total words: {total_words} ({total_words * 100 // target_words}% of target)")
    else:
        tprint(f"Document complete! Total words: {total_words}")

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(document)
        tprint(f"  Saved to: {args.output}")
    else:
        tprint("=" * 60)
        print(document)
        tprint("=" * 60)


if __name__ == "__main__":
    main()
