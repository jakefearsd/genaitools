#!/usr/bin/env python3
"""
Simple one-shot article generator using Ollama and DuckDuckGo research.
Uses optimal parameters discovered through tuning sessions.
"""

import argparse
import json
import requests

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


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
    "num_gpu": 99,
    "word_count": 2500,
    "think": True,  # Enable chain-of-thought reasoning for qwen3
}


def search_duckduckgo(topic: str, max_results: int = 8) -> list[dict]:
    """Scrape DuckDuckGo for research content."""
    print(f"Searching DuckDuckGo for: {topic}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=max_results))
        print(f"Found {len(results)} search results")
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []


def build_research_context(results: list[dict]) -> str:
    """Format search results into research context."""
    if not results:
        return "No research available. Use your training knowledge."

    context_parts = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        body = result.get("body", "")
        href = result.get("href", "")
        context_parts.append(f"[{i}] {title}\n{body}\nSource: {href}")

    return "\n\n".join(context_parts)


def build_prompt(
    topic: str,
    research: str,
    word_count: int,
    audience: str,
    content_type: str,
    persona: str = None,
    context: str = None,
) -> str:
    """
    Build a one-shot prompt for comprehensive article generation.

    Prompt structure (ordered by influence on output):
    1. PERSONA - Establishes voice, expertise, and writing style. Placed first
       because it colors interpretation of everything that follows.
    2. TASK - Content type, topic, audience. Core assignment.
    3. TARGET - Word count constraint.
    4. GUIDANCE - Optional requestor context. Placed BEFORE research so it
       frames how the model should interpret and prioritize research facts.
    5. RESEARCH - Factual content from web search. Source material.
    6. REQUIREMENTS - Format and structure constraints. Placed last as
       mechanical rules that don't need to influence content interpretation.
    """
    # Persona: determines voice and expertise level
    # Default is generic; custom persona enables specialized voices like
    # "a senior DevOps engineer" or "a patient instructor who uses analogies"
    writer_identity = persona if persona else "an expert technical writer"

    # Build prompt as sections, joined with double newlines
    sections = []

    # Section 1: Identity and task definition
    sections.append(
        f'You are {writer_identity}. Write a comprehensive {content_type} '
        f'about "{topic}" for {audience}.\n\n'
        f'TARGET: {word_count} words minimum. Write substantially and thoroughly.'
    )

    # Section 2: Optional guidance from requestor
    # Placed before research so it frames interpretation of facts
    # Examples: "focus on security implications", "compare with alternatives",
    #           "this is for a team migrating from VMs to containers"
    if context:
        sections.append(f"GUIDANCE FROM REQUESTOR:\n{context}")

    # Section 3: Research context from web search
    sections.append(f"RESEARCH CONTEXT:\n{research}")

    # Section 4: Format requirements (mechanical, so placed last)
    sections.append(
        "REQUIREMENTS:\n"
        "1. Use Markdown syntax (# for h1, ## for h2, ### for h3, `code` for inline, ```lang for code blocks)\n"
        "2. Include practical examples with code where relevant\n"
        "3. Structure with clear sections: Introduction, Main Content (multiple sections), Conclusion\n"
        "4. Be comprehensive - cover the topic thoroughly\n"
        "5. Write for the target audience level\n\n"
        "Write the complete article now. Do not stop early. Continue until you have covered all aspects thoroughly."
    )

    return "\n\n".join(sections)


def generate_article(
    prompt: str,
    ollama_url: str = DEFAULTS["ollama_url"],
    model: str = DEFAULTS["model"],
    num_predict: int = DEFAULTS["num_predict"],
    num_ctx: int = DEFAULTS["num_ctx"],
    repeat_penalty: float = DEFAULTS["repeat_penalty"],
    temperature: float = DEFAULTS["temperature"],
    num_gpu: int = DEFAULTS["num_gpu"],
    think: bool = DEFAULTS["think"],
) -> str:
    """Call Ollama API to generate the article."""
    url = f"{ollama_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,  # Enable chain-of-thought for supported models (qwen3)
        "options": {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "repeat_penalty": repeat_penalty,
            "temperature": temperature,
            "num_gpu": num_gpu,
        },
    }

    print(f"Generating with {model} (num_predict={num_predict}, num_ctx={num_ctx}, repeat_penalty={repeat_penalty}, think={think})...")

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 10 minutes"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(
        description="Simple one-shot article generator using Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--topic", required=True, help="Article topic")
    parser.add_argument("-a", "--audience", default="college educated general audience", help="Target audience")
    parser.add_argument("--type", default="tutorial", choices=["tutorial", "concept", "guide", "reference"],
                        help="Content type")
    parser.add_argument("-w", "--words", type=int, default=DEFAULTS["word_count"], help="Target word count")
    parser.add_argument("-p", "--persona", help="Writer persona, e.g. 'a senior DevOps engineer with 15 years experience'")
    parser.add_argument("-c", "--context", help="Additional context/guidance, e.g. 'focus on security best practices'")
    parser.add_argument("-o", "--output", help="Output file, e.g. article.md (default: stdout)")
    parser.add_argument("--ollama-url", default=DEFAULTS["ollama_url"], help="Ollama API URL")
    parser.add_argument("--model", default=DEFAULTS["model"], help="Ollama model")
    parser.add_argument("--num-predict", type=int, default=DEFAULTS["num_predict"], help="Max tokens")
    parser.add_argument("--num-ctx", type=int, default=DEFAULTS["num_ctx"], help="Context window")
    parser.add_argument("--repeat-penalty", type=float, default=DEFAULTS["repeat_penalty"], help="Repetition penalty")
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"], help="Temperature")
    parser.add_argument("--think", action="store_true", default=DEFAULTS["think"], help="Enable chain-of-thought reasoning")
    parser.add_argument("--no-think", action="store_true", help="Disable chain-of-thought reasoning")
    parser.add_argument("--no-search", action="store_true", help="Skip DuckDuckGo search")

    args = parser.parse_args()

    # Handle --no-think override
    think_enabled = args.think and not args.no_think

    # Research phase
    if args.no_search:
        research = "Use your training knowledge to write about this topic."
    else:
        results = search_duckduckgo(args.topic)
        research = build_research_context(results)

    # Build prompt
    prompt = build_prompt(
        topic=args.topic,
        research=research,
        word_count=args.words,
        audience=args.audience,
        content_type=args.type,
        persona=args.persona,
        context=args.context,
    )

    # Show prompt to user
    print("\n" + "=" * 60)
    print("PROMPT BEING SENT TO MODEL:")
    print("=" * 60)
    print(prompt)
    print("=" * 60 + "\n")

    # Generate
    article = generate_article(
        prompt=prompt,
        ollama_url=args.ollama_url,
        model=args.model,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        repeat_penalty=args.repeat_penalty,
        temperature=args.temperature,
        think=think_enabled,
    )

    # Output
    word_count = count_words(article)
    print(f"\nGenerated {word_count} words ({word_count * 100 // args.words}% of target)")

    if args.output:
        with open(args.output, "w") as f:
            f.write(article)
        print(f"Saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(article)
        print("=" * 60)


if __name__ == "__main__":
    main()
