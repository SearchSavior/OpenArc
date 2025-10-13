import os
from openai import OpenAI


def chat_completion_non_streaming_example():
    """Run a simple non-streaming chat.completions request against localhost:8000."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    try:
        models = client.models.list().data
        if not models:
            print("No models available. Load a model before running this test.")
            return
        model_name = models[0].id

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello briefly."},
            ],
        )

        print("Non-streaming chat completion response:")
        print(resp)
        if resp and resp.choices:
            print("Assistant:", resp.choices[0].message.content)
    except Exception as e:
        print(f"chat.completions (non-streaming) error: {e}")


def chat_completion_streaming_example():
    """Run a streaming chat.completions request against localhost:8000."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    try:
        models = client.models.list().data
        if not models:
            print("No models available. Load a model before running this test.")
            return
        model_name = models[0].id

        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello briefly."},
            ],
            stream=True,
        )

        print("Streaming chat completion response:")
        try:
            for part in stream:
                if not part or not part.choices:
                    continue
                delta = part.choices[0].delta
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    print(content, end="", flush=True)
        finally:
            print()
    except Exception as e:
        print(f"chat.completions (streaming) error: {e}")


if __name__ == "__main__":
    chat_completion_non_streaming_example()
    chat_completion_streaming_example()
