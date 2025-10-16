import os
from openai import OpenAI


def completions_non_streaming_example():
    """Run a simple non-streaming completions request against localhost:8000."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    try:
        model_name = "qwen25-14b"
        
        resp = client.completions.create(
            model=model_name,
            prompt="The future of artificial intelligence is",
            max_tokens=64,
            temperature=0.7,
        )

        print("Non-streaming completions response:")
        print(resp)
        if resp and resp.choices:
            print("Generated text:", resp.choices[0].text)
            print(f"Tokens used - Prompt: {resp.usage.prompt_tokens}, Completion: {resp.usage.completion_tokens}")
    except Exception as e:
        print(f"completions (non-streaming) error: {e}")


def completions_streaming_example():
    """Run a streaming completions request against localhost:8000."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    try:
        model_name = "qwen25-14b"
        
        stream = client.completions.create(
            model=model_name,
            prompt="Explain quantum computing in simple terms:",
            max_tokens=128,
            temperature=0.8,
            stream=True,
        )

        print("Streaming completions response:")
        collected_text = ""
        try:
            for chunk in stream:
                if not chunk or not chunk.choices:
                    continue
                choice = chunk.choices[0]
                text = choice.text
                if text:
                    collected_text += text
                    print(text, end="", flush=True)
        finally:
            print()
            print(f"Total collected text: {len(collected_text)} characters")
    except Exception as e:
        print(f"completions (streaming) error: {e}")


def completions_with_parameters():
    """Test completions with various generation parameters."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    try:
        model_name = "qwen25-14b"
        
        resp = client.completions.create(
            model=model_name,
            prompt="Translate this to French: Hello, how are you?",
            max_tokens=64,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
        )

        print("Completions with parameters response:")
        print(f"Model: {resp.model}")
        print(f"Generated text: {resp.choices[0].text}")
        print(f"Finish reason: {resp.choices[0].finish_reason}")
        print(f"Usage: {resp.usage}")
    except Exception as e:
        print(f"completions (with parameters) error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing OpenAI-compatible /v1/completions endpoint")
    print("Model: qwen25-14b")
    print("=" * 60)
    print()
    
    print("Test 1: Non-streaming completion")
    print("-" * 60)
    completions_non_streaming_example()
    
    print()
    print("Test 2: Streaming completion")
    print("-" * 60)
    completions_streaming_example()
    
    print()
    print("Test 3: Completions with parameters")
    print("-" * 60)
    completions_with_parameters()
    
    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
