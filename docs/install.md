---
icon: lucide/rocket
---



=== "Linux"

    1. OpenVINO requires **device specifc drivers**.
    
        - Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for the latest information on drivers.

    2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

    3. After cloning use:

        ```
        uv sync
        ```

    4. Activate your environment with:

        ```
        source .venv/bin/activate
        ```

        Build latest optimum
        ```
        uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
        ```

        Build latest OpenVINO and OpenVINO GenAI from nightly wheels
        ```
        uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        ```

    5. Set your API key as an environment variable:
        ```
        export OPENARC_API_KEY=api-key
        ```

    6. To get started, run:

        ```
        openarc --help
        ```

=== "Windows"

    1. OpenVINO requires **device specifc drivers**.
    
        - Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) to get the latest information on drivers.

    2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

    3. Clone OpenArc, enter the directory and run:
    ```
    uv sync
    ```

    4. Activate your environment with:

        ```
        .venv\Scripts\activate
        ```

        **Build latest optimum**
        ```
        uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
        ```

        **Build latest OpenVINO and OpenVINO GenAI from nightly wheels**
        ```
        uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
        ```

    5. **Set your API key as an environment variable:**
        ```
        setx OPENARC_API_KEY openarc-api-key
        ```

    6. To get started, run:

        ```
        openarc --help
        ```

=== "Docker"

    Instead of fighting with Intel's own docker images, we built our own which is as close to boilerplate as possible. For a primer on docker [check out this video](https://www.youtube.com/watch?v=DQdB7wFEygo).


    **Build and run the container:**
    ```bash
    docker-compose up --build -d
    ```

    **Run the container:**
    ```bash
    docker run -d -p 8000:8000 openarc:latest
    ```
    **Enter the container:**
    ```bash
    docker exec -it openarc /bin/bash
    ```

    ## Environment Variables

    ```bash
    export OPENARC_API_KEY="openarc-api-key" # default, set it to whatever you want
    export OPENARC_AUTOLOAD_MODEL="model_name" # model_name to load on startup
    export MODEL_PATH="/path/to/your/models" # mount your models to `/models` inside the container
    docker-compose up --build -d
    ```


    Take a look at the [Dockerfile](Dockerfile) and [docker-compose](docker-compose.yaml) for more details.