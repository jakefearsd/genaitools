#!/usr/bin/env python3
"""
Batch document builder - process multiple topics from a JSON configuration file.
Runs outline_builder.py and document_builder.py for each topic with GPU cooldown.
"""

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genaitools import tprint, DEFAULTS

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_COOLDOWN_SECONDS = 10

# Default values for topic parameters (used when not in JSON defaults or topic)
TOPIC_DEFAULTS = {
    "audience": "college educated general audience",
    "content_type": "tutorial",
    "words": 5000,
    "sections": 5,
    "persona": None,
    "model": DEFAULTS["model"],
    "ollama_url": DEFAULTS["ollama_url"],
    "temperature": DEFAULTS["temperature"],
    "deep_research": False,
    "no_cache": False,
    "think": DEFAULTS["think"],
    "verbose": True,
    "document_instructions": None,
    "num_gpu": None,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TopicConfig:
    """Configuration for a single topic."""

    topic: str
    context: str | None = None
    audience: str = "college educated general audience"
    content_type: str = "tutorial"
    words: int = 5000
    sections: int = 5
    persona: str | None = None
    model: str = "qwen3:14b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.7
    deep_research: bool = False
    no_cache: bool = False
    think: bool = True
    verbose: bool = True
    document_instructions: str | None = None
    num_gpu: int | None = None
    outline_file: str | None = None
    document_file: str | None = None


@dataclass
class TopicResult:
    """Result of processing a single topic."""

    topic: str
    status: str  # "success", "skipped", "failed"
    outline_file: str | None = None
    document_file: str | None = None
    word_count: int | None = None
    outline_duration_seconds: float | None = None
    document_duration_seconds: float | None = None
    error: str | None = None
    reason: str | None = None


@dataclass
class BatchReport:
    """Report of the entire batch run."""

    input_file: str
    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    summary: dict = field(default_factory=lambda: {"total": 0, "successful": 0, "skipped": 0, "failed": 0})
    topics: list[dict] = field(default_factory=list)


# =============================================================================
# Utility Functions
# =============================================================================


def topic_to_slug(topic: str) -> str:
    """
    Convert a topic string to a URL-safe slug.

    Example: "First AWS Deployment" -> "first-aws-deployment"
    """
    # Convert to lowercase
    slug = topic.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove non-alphanumeric characters except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


def topic_to_title_case(topic: str) -> str:
    """
    Convert a topic string to TitleCase for document filenames.

    Example: "First AWS Deployment" -> "FirstAWSDeployment"
    """
    # Split on spaces and other separators
    words = re.split(r"[\s_-]+", topic)
    # Capitalize each word and join
    return "".join(word.capitalize() for word in words if word)


def load_batch_config(path: str) -> dict[str, Any]:
    """Load and validate batch configuration from JSON file."""
    with open(path) as f:
        config = json.load(f)

    # Validate required fields
    if "topics" not in config:
        raise ValueError("Batch config missing required 'topics' array")

    if not isinstance(config["topics"], list):
        raise ValueError("'topics' must be an array")

    if len(config["topics"]) == 0:
        raise ValueError("'topics' array is empty")

    for i, topic in enumerate(config["topics"]):
        if "topic" not in topic:
            raise ValueError(f"Topic at index {i} missing required 'topic' field")

    return config


def merge_config(defaults: dict, topic: dict) -> TopicConfig:
    """Merge defaults and topic-specific config into a TopicConfig."""
    merged = {}

    # Start with hardcoded defaults
    for key, value in TOPIC_DEFAULTS.items():
        merged[key] = value

    # Override with JSON defaults
    for key, value in defaults.items():
        if key in TOPIC_DEFAULTS:
            merged[key] = value

    # Override with topic-specific values
    for key, value in topic.items():
        merged[key] = value

    return TopicConfig(**merged)


def get_output_paths(config: TopicConfig, output_dir: Path) -> tuple[Path, Path]:
    """Get the outline and document output paths for a topic."""
    slug = topic_to_slug(config.topic)
    title = topic_to_title_case(config.topic)

    outline_file = config.outline_file or f"{slug}-outline.yaml"
    document_file = config.document_file or f"{title}.md"

    return output_dir / outline_file, output_dir / document_file


# =============================================================================
# Execution Functions
# =============================================================================


def build_outline_command(config: TopicConfig, output_path: Path) -> list[str]:
    """Build the command line for outline_builder.py."""
    cmd = [
        sys.executable,
        "outline_builder.py",
        "-t",
        config.topic,
        "-a",
        config.audience,
        "--type",
        config.content_type,
        "-w",
        str(config.words),
        "-s",
        str(config.sections),
        "-o",
        str(output_path),
        "--ollama-url",
        config.ollama_url,
        "--model",
        config.model,
        "--temperature",
        str(config.temperature),
    ]

    if config.persona:
        cmd.extend(["-p", config.persona])

    if config.context:
        cmd.extend(["-c", config.context])

    if config.deep_research:
        cmd.append("--deep-research")

    if config.no_cache:
        cmd.append("--no-cache")

    if config.think:
        cmd.append("--think")
    else:
        cmd.append("--no-think")

    if config.num_gpu is not None:
        cmd.extend(["--num-gpu", str(config.num_gpu)])

    return cmd


def build_document_command(config: TopicConfig, outline_path: Path, output_path: Path) -> list[str]:
    """Build the command line for document_builder.py."""
    cmd = [
        sys.executable,
        "document_builder.py",
        "-i",
        str(outline_path),
        "-o",
        str(output_path),
        "--ollama-url",
        config.ollama_url,
        "--model",
        config.model,
        "--temperature",
        str(config.temperature),
    ]

    if config.verbose:
        cmd.append("--verbose")

    if config.think:
        cmd.append("--think")
    else:
        cmd.append("--no-think")

    if config.document_instructions:
        cmd.extend(["--instructions", config.document_instructions])

    if config.num_gpu is not None:
        cmd.extend(["--num-gpu", str(config.num_gpu)])

    return cmd


def run_command(cmd: list[str], description: str) -> tuple[bool, str | None]:
    """
    Run a command and return (success, error_message).

    Output is streamed to stdout in real-time.
    """
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output line by line
        output_lines = []
        for line in process.stdout:
            print(line, end="")  # Already has newline
            output_lines.append(line)

        process.wait()

        if process.returncode != 0:
            return False, f"Exit code {process.returncode}"

        return True, None

    except Exception as e:
        return False, str(e)


def extract_word_count(document_path: Path) -> int | None:
    """Extract word count from a generated document."""
    try:
        content = document_path.read_text()
        return len(content.split())
    except Exception:
        return None


def process_topic(
    config: TopicConfig,
    output_dir: Path,
    index: int,
    total: int,
    skip_outlines: bool = False,
) -> TopicResult:
    """Process a single topic, generating outline and document."""
    outline_path, document_path = get_output_paths(config, output_dir)

    result = TopicResult(
        topic=config.topic,
        status="success",
        outline_file=outline_path.name,
        document_file=document_path.name,
    )

    # Check if already complete
    outline_exists = outline_path.exists()
    document_exists = document_path.exists()

    if outline_exists and document_exists:
        result.status = "skipped"
        result.reason = "output files exist"
        result.word_count = extract_word_count(document_path)
        tprint(f"Outline: Skipping (file exists)")
        tprint(f"Document: Skipping (file exists)")
        return result

    # Phase 1: Generate outline
    if outline_exists or skip_outlines:
        tprint(f"Outline: Skipping (file exists)" if outline_exists else "Outline: Skipping (--skip-outlines)")
    else:
        tprint(f"Outline: Generating...")
        start_time = time.time()

        cmd = build_outline_command(config, outline_path)
        success, error = run_command(cmd, "outline generation")

        result.outline_duration_seconds = time.time() - start_time

        if not success:
            result.status = "failed"
            result.error = f"Outline generation failed: {error}"
            tprint(f"  ERROR: {error}", file=sys.stderr)
            return result

        tprint(f"  -> {outline_path.name}")

    # Phase 2: Generate document
    if document_exists:
        tprint(f"Document: Skipping (file exists)")
        result.word_count = extract_word_count(document_path)
    else:
        if not outline_path.exists():
            result.status = "failed"
            result.error = "Outline file not found"
            tprint(f"  ERROR: Outline file not found", file=sys.stderr)
            return result

        tprint(f"Document: Building...")
        start_time = time.time()

        cmd = build_document_command(config, outline_path, document_path)
        success, error = run_command(cmd, "document generation")

        result.document_duration_seconds = time.time() - start_time

        if not success:
            result.status = "failed"
            result.error = f"Document generation failed: {error}"
            tprint(f"  ERROR: {error}", file=sys.stderr)
            return result

        result.word_count = extract_word_count(document_path)
        tprint(f"  -> {document_path.name} ({result.word_count:,} words)")

    return result


# =============================================================================
# Report Functions
# =============================================================================


def save_report(report: BatchReport, output_dir: Path) -> None:
    """Save the batch report to JSON."""
    report_path = output_dir / "batch-report.json"

    report_dict = {
        "input_file": report.input_file,
        "started_at": report.started_at,
        "completed_at": report.completed_at,
        "duration_seconds": report.duration_seconds,
        "summary": report.summary,
        "topics": report.topics,
    }

    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    tprint(f"Report: {report_path}")


def result_to_dict(result: TopicResult) -> dict:
    """Convert a TopicResult to a dictionary for the report."""
    d = {"topic": result.topic, "status": result.status}

    if result.status == "success":
        d["outline_file"] = result.outline_file
        d["document_file"] = result.document_file
        if result.word_count:
            d["word_count"] = result.word_count
        if result.outline_duration_seconds:
            d["outline_duration_seconds"] = round(result.outline_duration_seconds, 1)
        if result.document_duration_seconds:
            d["document_duration_seconds"] = round(result.document_duration_seconds, 1)
    elif result.status == "skipped":
        d["reason"] = result.reason
        if result.word_count:
            d["word_count"] = result.word_count
    elif result.status == "failed":
        d["error"] = result.error

    return d


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch document builder - process multiple topics from JSON config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSON batch file")
    parser.add_argument("-o", "--output-dir", help="Override output directory from JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show execution plan without running")
    parser.add_argument("--start-at", type=int, help="Start at topic N (1-indexed)")
    parser.add_argument("--only", type=int, help="Run only topic N (1-indexed)")
    parser.add_argument("--skip-outlines", action="store_true", help="Use existing outline files, only build documents")
    parser.add_argument("--cooldown", type=int, help="Override cooldown between topics (seconds)")
    parser.add_argument("--no-cooldown", action="store_true", help="Disable cooldown between topics")

    args = parser.parse_args()

    # Load config
    try:
        config = load_batch_config(args.input)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        tprint(f"Error loading batch config: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract settings
    defaults = config.get("defaults", {})
    output_dir = Path(args.output_dir or config.get("output_dir", DEFAULT_OUTPUT_DIR))
    cooldown = 0 if args.no_cooldown else (args.cooldown or config.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))
    topics = config["topics"]

    # Filter topics based on --start-at and --only
    if args.only:
        if args.only < 1 or args.only > len(topics):
            tprint(f"Error: --only {args.only} out of range (1-{len(topics)})", file=sys.stderr)
            sys.exit(1)
        topics = [topics[args.only - 1]]
        start_index = args.only - 1
    elif args.start_at:
        if args.start_at < 1 or args.start_at > len(topics):
            tprint(f"Error: --start-at {args.start_at} out of range (1-{len(topics)})", file=sys.stderr)
            sys.exit(1)
        topics = topics[args.start_at - 1 :]
        start_index = args.start_at - 1
    else:
        start_index = 0

    total_topics = len(config["topics"])

    # Print header
    tprint("=" * 60)
    tprint("Batch Builder")
    tprint("=" * 60)
    tprint(f"Input: {args.input}")
    tprint(f"Output: {output_dir}/")
    tprint(f"Topics: {len(topics)}" + (f" (of {total_topics})" if len(topics) != total_topics else ""))
    tprint(f"Cooldown: {cooldown}s")

    # Dry run - show plan and exit
    if args.dry_run:
        tprint("")
        tprint("Execution Plan (dry run):")
        for i, topic_data in enumerate(topics):
            topic_config = merge_config(defaults, topic_data)
            outline_path, document_path = get_output_paths(topic_config, output_dir)
            idx = start_index + i + 1
            tprint(f"  [{idx}/{total_topics}] {topic_config.topic}")
            tprint(f"       Outline: {outline_path.name}")
            tprint(f"       Document: {document_path.name}")
            tprint(f"       Model: {topic_config.model}, Words: {topic_config.words}, Sections: {topic_config.sections}")
        tprint("")
        tprint("Remove --dry-run to execute.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = BatchReport(
        input_file=args.input,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    report.summary["total"] = len(topics)

    # Set up signal handler for graceful Ctrl+C
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            tprint("\nForce quit.", file=sys.stderr)
            sys.exit(1)
        interrupted = True
        tprint("\nInterrupted. Saving partial report...", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_interrupt)

    # Process topics
    batch_start_time = time.time()

    for i, topic_data in enumerate(topics):
        if interrupted:
            break

        topic_config = merge_config(defaults, topic_data)
        idx = start_index + i + 1

        tprint("")
        tprint("-" * 60)
        tprint(f"[{idx}/{total_topics}] {topic_config.topic}")
        tprint("-" * 60)

        result = process_topic(
            config=topic_config,
            output_dir=output_dir,
            index=idx,
            total=total_topics,
            skip_outlines=args.skip_outlines,
        )

        # Update report
        report.topics.append(result_to_dict(result))
        if result.status == "success":
            report.summary["successful"] += 1
        elif result.status == "skipped":
            report.summary["skipped"] += 1
        else:
            report.summary["failed"] += 1

        # Cooldown between topics (not after last one)
        if cooldown > 0 and i < len(topics) - 1 and not interrupted:
            tprint(f"Cooling down ({cooldown}s)...")
            time.sleep(cooldown)

    # Finalize report
    report.completed_at = datetime.now(timezone.utc).isoformat()
    report.duration_seconds = round(time.time() - batch_start_time, 1)

    # Print summary
    tprint("")
    tprint("=" * 60)
    tprint("Batch Complete" + (" (interrupted)" if interrupted else ""))
    tprint("=" * 60)
    tprint(f"Successful: {report.summary['successful']}")
    tprint(f"Skipped: {report.summary['skipped']}")
    tprint(f"Failed: {report.summary['failed']}")

    # Format duration
    duration = report.duration_seconds
    if duration >= 3600:
        tprint(f"Total time: {int(duration // 3600)}h {int((duration % 3600) // 60)}m")
    elif duration >= 60:
        tprint(f"Total time: {int(duration // 60)}m {int(duration % 60)}s")
    else:
        tprint(f"Total time: {int(duration)}s")

    # Save report
    save_report(report, output_dir)


if __name__ == "__main__":
    main()
