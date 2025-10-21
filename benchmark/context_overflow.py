#!/usr/bin/env python3
"""
Context Overflow Test Script

Tests increasing context sizes with OpenAI API-compatible endpoint.
Progressively increases prompt size each iteration and accumulates context.
Relies on server-reported metrics for accurate token counts.
"""

import argparse
import os
import sys
from openai import OpenAI


def generate_filler_text(char_count: int) -> str:
    """Generate filler text for approximately the given character count."""
    filler = "The quick brown fox jumps over the lazy dog. " * ((char_count // 45) + 1)
    return filler[:char_count]


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
    char_increment = 8192  # Add roughly ~2048 tokens worth of characters per iteration
    accumulated_context = ""
    base_prompt = "You are a helpful assistant. Answer the following:\n\n"

    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Character increment per iteration: {char_increment}")
    print(f"{'='*40}\n")

    for iteration in range(1, args.iterations + 1):
        # Add more context each iteration
        filler = generate_filler_text(char_increment)
        accumulated_context += filler

        # Build the full prompt
        full_prompt = base_prompt + accumulated_context + "\nWhat have you learned from this context?"

        print(f"Iteration {iteration}/{args.iterations}")

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
                print("  ✓ Success")
                print(f"  Response preview: {content[:80]}...")

            # Extract and display OpenArc internal metrics
            metrics = getattr(response, 'metrics', None)
            if metrics:
                print("\n  === Performance Metrics ===")
                if 'ttft (s)' in metrics:
                    print(f"  TTFT: {metrics['ttft (s)']:.2f}s")
                if 'tpot (ms)' in metrics:
                    print(f"  TPOT: {metrics['tpot (ms)']:.2f}ms")
                if 'prefill_throughput (tokens/s)' in metrics:
                    print(f"  Prefill throughput: {metrics['prefill_throughput (tokens/s)']:.2f} tokens/s")
                if 'decode_throughput (tokens/s)' in metrics:
                    print(f"  Decode throughput: {metrics['decode_throughput (tokens/s)']:.2f} tokens/s")
                if 'decode_duration (s)' in metrics:
                    print(f"  Decode duration: {metrics['decode_duration (s)']:.2f}s")

            # Use server-reported token counts
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                print(f"\n  Prompt tokens: {prompt_tokens:,}")
                print(f"  Completion tokens: {completion_tokens:,}")
                print(f"  Total tokens: {total_tokens:,}")

        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {str(e)}")
            break

        print()

    print("=== Test Complete ===")


if __name__ == "__main__":
    main()
