#!/usr/bin/env python3
"""
Generate structured YAML outlines for multi-section document generation.
Works with document_builder.py to create comprehensive documents.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from genaitools import DEFAULTS, generate, search_duckduckgo, build_research_context, deep_research, tprint

# =============================================================================
# Module Constants
# =============================================================================

# Default outline parameters
DEFAULT_AUDIENCE = "college educated general audience"
DEFAULT_CONTENT_TYPE = "tutorial"
DEFAULT_TOTAL_WORDS = 5000
DEFAULT_NUM_SECTIONS = 5

# Content type choices for validation
CONTENT_TYPE_CHOICES = ["tutorial", "concept", "guide", "reference"]

# Generation parameters
OUTLINE_TEMPERATURE = 0.7
OUTLINE_NUM_PREDICT = 8192  # Increase for many sections


def build_outline_prompt(
    topic: str,
    audience: str,
    content_type: str,
    total_words: int,
    num_sections: int,
    persona: str | None = None,
    guidance: str | None = None,
    research_context: str | None = None,
) -> str:
    """
    Build prompt for outline generation.

    The LLM will output JSON which we convert to YAML for human editing.
    """
    sections = []

    # Identity and task
    sections.append(
        "You are an expert content strategist and technical writer. "
        f"Create a detailed outline for a {content_type} about \"{topic}\" "
        f"targeting {audience}."
    )

    # Constraints
    sections.append(
        f"CONSTRAINTS:\n"
        f"- Total document: approximately {total_words} words\n"
        f"- Number of sections: {num_sections} (including introduction and conclusion)\n"
        f"- Content type: {content_type}"
    )

    # Optional persona and guidance
    if persona:
        sections.append(f"WRITER PERSONA: The document will be written as {persona}")
    if guidance:
        sections.append(f"ADDITIONAL GUIDANCE: {guidance}")

    # Research context
    if research_context:
        sections.append(f"RESEARCH CONTEXT (use to inform section topics):\n{research_context}")

    # Output format specification
    sections.append(
        'OUTPUT FORMAT: Respond with ONLY valid JSON (no markdown, no explanation). '
        'Use this exact structure:\n'
        '{\n'
        '  "title": "Document Title",\n'
        '  "sections": [\n'
        '    {\n'
        '      "id": "url-safe-slug",\n'
        '      "title": "Section Title",\n'
        '      "order": 1,\n'
        '      "word_count": 400,\n'
        '      "position": "first",\n'
        '      "section_role": "introduce",\n'
        '      "transition_to": "next-section-id",\n'
        '      "dependencies": [],\n'
        '      "keywords": ["key term 1", "key term 2"],\n'
        '      "guidance": "What this section should cover",\n'
        '      "content_hints": ["Specific point to include", "Another point"]\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "REQUIREMENTS:\n"
        "- id: lowercase, hyphens only (e.g., 'getting-started', 'core-concepts')\n"
        "- order: sequential integers starting at 1\n"
        "- word_count: distribute total words across sections (intro/conclusion shorter)\n"
        "- position: 'first' for introduction, 'last' for conclusion, 'middle' for all others\n"
        "- section_role: 'introduce' (sets up topic), 'develop' (main content), 'conclude' (wraps up ENTIRE document)\n"
        "- transition_to: the id of the next section (null for the last section)\n"
        "- dependencies: list section IDs that must come before (empty for first section)\n"
        "- keywords: 3-5 key terms this section must cover\n"
        "- guidance: 1-2 sentences on what to cover\n"
        "- content_hints: 2-4 specific points or examples to include\n\n"
        "CRITICAL DOCUMENT STRUCTURE RULES:\n"
        "- ONLY the FINAL section should have section_role='conclude'\n"
        "- Middle sections (position='middle') must NOT include conclusions or summaries\n"
        "- Middle sections should END with a bridge/transition to the next topic\n"
        "- The document flows as one cohesive piece, not separate articles\n\n"
        "Generate the outline JSON now:"
    )

    return "\n\n".join(sections)


def extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON from LLM response, handling common formats.

    Attempts to extract JSON in this order:
    1. From ```json code block (with or without 'json' tag)
    2. As raw JSON object anywhere in the text

    Args:
        text: Raw LLM response text

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValueError: If no valid JSON found or JSON is malformed
    """
    # Try to find JSON in code block first
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    # Try to find JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

    raise ValueError("No JSON found in response")


def ensure_section_positions(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure all sections have correct position, section_role, and transition_to fields.

    This validates/corrects the LLM output to guarantee proper document structure.

    Validation Rules:
        - First section: position='first', section_role='introduce' (default)
        - Last section: position='last', section_role='conclude' (forced)
        - Middle sections: position='middle', section_role cannot be 'conclude'
        - All sections except last: transition_to = next section's ID
        - Last section: transition_to = None

    Args:
        sections: List of section dictionaries from LLM output

    Returns:
        Sorted list of sections with corrected position metadata
    """
    if not sections:
        return sections

    # Sort by order
    sorted_sections = sorted(sections, key=lambda s: s.get("order", 0))
    num_sections = len(sorted_sections)

    for i, section in enumerate(sorted_sections):
        # Set position based on order
        if i == 0:
            section["position"] = "first"
            section["section_role"] = section.get("section_role", "introduce")
        elif i == num_sections - 1:
            section["position"] = "last"
            section["section_role"] = "conclude"  # Always conclude for last
        else:
            section["position"] = "middle"
            # Middle sections should develop, not conclude
            if section.get("section_role") == "conclude":
                section["section_role"] = "develop"
            else:
                section["section_role"] = section.get("section_role", "develop")

        # Set transition_to
        if i < num_sections - 1:
            section["transition_to"] = sorted_sections[i + 1].get("id")
        else:
            section["transition_to"] = None

    return sorted_sections


def build_yaml_outline(
    json_data: dict[str, Any],
    topic: str,
    audience: str,
    content_type: str,
    total_words: int,
    persona: str | None = None,
    guidance: str | None = None,
    research_context: str | None = None,
) -> dict[str, Any]:
    """
    Convert LLM JSON output to full YAML outline structure.
    """
    # Get sections and ensure positions are correct
    sections = json_data.get("sections", [])
    sections = ensure_section_positions(sections)

    return {
        "metadata": {
            "title": json_data.get("title", topic),
            "topic": topic,
            "audience": audience,
            "content_type": content_type,
            "total_word_count": total_words,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **({"persona": persona} if persona else {}),
            **({"guidance": guidance} if guidance else {}),
        },
        "research": {
            "context": research_context or "No research gathered.",
        },
        "sections": sections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML outline for multi-section document generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--topic", required=True, help="Document topic")
    parser.add_argument(
        "-a", "--audience",
        default=DEFAULT_AUDIENCE,
        help="Target audience"
    )
    parser.add_argument(
        "--type",
        default=DEFAULT_CONTENT_TYPE,
        choices=CONTENT_TYPE_CHOICES,
        help="Content type"
    )
    parser.add_argument(
        "-w", "--words",
        type=int,
        default=DEFAULT_TOTAL_WORDS,
        help="Total target word count"
    )
    parser.add_argument(
        "-s", "--sections",
        type=int,
        default=DEFAULT_NUM_SECTIONS,
        help="Number of sections"
    )
    parser.add_argument(
        "-p", "--persona",
        help="Writer persona, e.g. 'a senior DevOps engineer'"
    )
    parser.add_argument(
        "-c", "--context",
        help="Additional guidance for outline generation"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output YAML file (default: stdout)"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip DuckDuckGo research"
    )
    parser.add_argument(
        "--deep-research",
        action="store_true",
        help="Fetch and summarize full web pages (slower, richer context)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip research cache (use with --deep-research)"
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
        "--temperature",
        type=float,
        default=OUTLINE_TEMPERATURE,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=OUTLINE_NUM_PREDICT,
        help="Max tokens for outline generation (increase for many sections)"
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
        "--num-gpu",
        type=int,
        default=None,
        help="Number of GPU layers (default: let Ollama auto-detect)"
    )

    args = parser.parse_args()

    # Handle --no-think override
    think_enabled = args.think and not args.no_think

    # Check if output file already exists
    if args.output and Path(args.output).exists():
        tprint(f"Error: Output file already exists: {args.output}", file=sys.stderr)
        sys.exit(1)

    # Research phase
    research_context = None
    if not args.no_search:
        if args.deep_research:
            research_context = deep_research(
                topic=args.topic,
                model=args.model,
                ollama_url=args.ollama_url,
                use_cache=not args.no_cache,
                verbose=True,
            )
        else:
            results = search_duckduckgo(args.topic)
            research_context = build_research_context(results)

    # Build prompt
    prompt = build_outline_prompt(
        topic=args.topic,
        audience=args.audience,
        content_type=args.type,
        total_words=args.words,
        num_sections=args.sections,
        persona=args.persona,
        guidance=args.context,
        research_context=research_context,
    )

    tprint(f"Generating outline with {args.model}...")

    # Generate outline
    response = generate(
        prompt=prompt,
        ollama_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        think=think_enabled,
        num_predict=args.num_predict,
        num_ctx=DEFAULTS["num_ctx"],
        num_gpu=args.num_gpu,
    )

    if response.startswith("Error:"):
        tprint(response, file=sys.stderr)
        sys.exit(1)

    # Parse JSON from response
    try:
        json_data = extract_json(response)
    except ValueError as e:
        tprint(f"Failed to parse outline: {e}", file=sys.stderr)
        tprint(f"Raw response:\n{response}", file=sys.stderr)
        sys.exit(1)

    # Build full YAML structure
    outline = build_yaml_outline(
        json_data=json_data,
        topic=args.topic,
        audience=args.audience,
        content_type=args.type,
        total_words=args.words,
        persona=args.persona,
        guidance=args.context,
        research_context=research_context,
    )

    # Output
    yaml_output = yaml.dump(outline, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.output:
        with open(args.output, "w") as f:
            f.write(yaml_output)
        tprint(f"Outline saved to: {args.output}")
        tprint(f"Sections: {len(outline['sections'])}")
    else:
        print(yaml_output)


if __name__ == "__main__":
    main()
