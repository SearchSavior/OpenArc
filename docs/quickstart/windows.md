# Windows

1. OpenVINO requires **device specific drivers**.

    Visit [OpenVINO System Requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) to get the latest information on drivers.

2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

3. Clone OpenArc, enter the directory and run:

    ```
    uv sync
    ```

4. Activate your environment with:

    ```
    .venv\Scripts\activate
    ```

    Build latest optimum:

    ```
    uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
    ```

    Build latest OpenVINO and OpenVINO GenAI from nightly wheels:

    ```
    uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    ```

5. Set your API key as an environment variable:

    ```
    setx OPENARC_API_KEY openarc-api-key
    ```

6. To get started, run:

    ```
    openarc --help
    ```
