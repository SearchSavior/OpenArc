#!/usr/bin/env python3
"""
Context Overflow Test Script

Tests increasing context sizes with OpenAI API-compatible endpoint.
Progressively increases prompt size by 512 tokens each iteration and accumulates context.
"""

import argparse
import os
import sys
from openai import OpenAI


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (approximately 4 characters per token)."""
    return len(text) // 4


def generate_filler_text(token_count: int) -> str:
    """Generate filler text for approximately the given token count."""
    # Rough estimate: 4 chars per token, so multiply by 4
    target_chars = token_count * 4
    filler = "The quick brown fox jumps over the lazy dog. " * ((target_chars // 45) + 1)
    return filler[:target_chars]


def main():
    parser = argparse.ArgumentParser(
        description="Test context overflow with increasing prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for testing"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run (default: 10)"
    )
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get("OPENARC_API_KEY")
    if not api_key:
        print("ERROR: OPENARC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Initialize OpenAI client pointing to localhost:8000
    client = OpenAI(
        api_key=api_key,
        base_url="http://localhost:8000/v1"
    )

    # Configuration
    token_increment = 2048
    accumulated_context = ""
    base_prompt = "You are a helpful assistant. Answer the following:\n\n"


    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Token increment per iteration: {token_increment}")

    print(f"{'='*40}\n")

    for iteration in range(1, args.iterations + 1):
        # Add more context each iteration
        filler = generate_filler_text(token_increment)
        accumulated_context += filler

        # Build the full prompt
        full_prompt = base_prompt + accumulated_context + "\nWhat have you learned from this context?"

        estimated_tokens = estimate_tokens(full_prompt)

        print(f"Iteration {iteration}/{args.iterations}")
        print(f"  Estimated prompt tokens: ~{estimated_tokens}")
        print(f"  Accumulated context size: ~{estimate_tokens(accumulated_context)} tokens")

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=150
            )

            # Print response details
            if response.choices:
                content = response.choices[0].message.content
                print(f"  ✓ Success")
                print(f"  Response preview: {content[:80]}...")

            if response.usage:
                print(f"  Prompt tokens (actual): {response.usage.prompt_tokens}")
                print(f"  Completion tokens: {response.usage.completion_tokens}")
                print(f"  Total tokens: {response.usage.total_tokens}")

        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {str(e)}")
            break

        print()

    print("=== Test Complete ===")


if __name__ == "__main__":
    main()
