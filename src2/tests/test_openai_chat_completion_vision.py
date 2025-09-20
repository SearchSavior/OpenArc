import os
import base64
from openai import OpenAI


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode image: {e}")
        return None


def chat_completion_vision_non_streaming_example():
    """Run a simple non-streaming vision chat.completions request against localhost:8000."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    # Encode the image
    image_path = "/home/echo/Projects/OpenArc/src2/tests/dedication.png"
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        print("Failed to encode image. Exiting.")
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
                {"role": "system", "content": "You are a helpful assistant that can analyze images."},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe what you see in this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                },
            ],
        )

        print("Non-streaming vision chat completion response:")
        print(resp)
        if resp and resp.choices:
            print("Assistant:", resp.choices[0].message.content)
    except Exception as e:
        print(f"chat.completions (vision non-streaming) error: {e}")


def chat_completion_vision_streaming_example():
    """Run a streaming vision chat.completions request against localhost:8000."""
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    # Encode the image
    image_path = "/home/echo/Projects/OpenArc/src2/tests/dedication.png"
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        print("Failed to encode image. Exiting.")
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
                {"role": "system", "content": "You are a helpful assistant that can analyze images."},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe what you see in this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                },
            ],
            stream=True,
        )

        print("Streaming vision chat completion response:")
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
        print(f"chat.completions (vision streaming) error: {e}")


if __name__ == "__main__":
    chat_completion_vision_non_streaming_example()
    chat_completion_vision_streaming_example()
