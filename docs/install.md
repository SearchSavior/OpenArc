---
icon: lucide/cog
---

Use the instructions here for your operating system or deployment strategy. They will walk you through building the project as a python environment for your OS. 

OpenArc supports *most* OpenVINO devices (including AMD CPUs). You can use CPUs, NPUs, and GPUs; however these require different drivers.

Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for the latest information on drivers for your device and OS.

=== "Linux"

    1. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

    2. After cloning use:

        ```
        uv sync
        ```

    3. Activate your environment with:

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

    4. Optionally, set an API key to authenticate clients connecting to the server:
        ```
        export OPENARC_API_KEY=api-key
        ```

        Pass `--use-api-key` to `openarc serve start` to enforce authentication. See [serve](commands.md#serve) for details.

    5. To get started, run:

        ```
        openarc --help
        ```

=== "Windows"

    1. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

    2. Clone OpenArc, enter the directory and run:
    ```
    uv sync
    ```

    3. Activate your environment with:

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

    4. **Optionally, set an API key to authenticate clients connecting to the server:**
        ```
        setx OPENARC_API_KEY openarc-api-key
        ```

        Pass `--use-api-key` to `openarc serve start` to enforce authentication. See [serve](commands.md#serve) for details.

    5. To get started, run:

        ```
        openarc --help
        ```

=== "Docker"

    Instead of fighting with Intel's own docker images, we built our own which is as close to boilerplate as possible. For a primer on docker [check out this video](https://www.youtube.com/watch?v=DQdB7wFEygo).

    The Dockerfiles and Compose file live in [`.devops/`](https://github.com/SearchSavior/OpenArc/tree/main/.devops). Run all commands below from the repository root.

    **Build and start the container:**
    ```bash
    docker compose -f .devops/docker-compose.yaml up --build -d
    ```

    **Stop and remove the container:**
    ```bash
    docker compose -f .devops/docker-compose.yaml down
    ```

    **View logs:**
    ```bash
    docker compose -f .devops/docker-compose.yaml logs -f
    ```

    **Enter the container:**
    ```bash
    docker compose -f .devops/docker-compose.yaml exec openarc /bin/bash
    ```

    **Battlemage (or newer) GPUs:** the default `Dockerfile` targets older Intel GPUs. For Battlemage, build with `Battlemage.Dockerfile` instead:
    ```bash
    docker build -f .devops/Battlemage.Dockerfile -t openarc:latest .
    docker compose -f .devops/docker-compose.yaml up -d
    ```

    Environment Variables

    ```bash
    export OPENARC_API_KEY="openarc-api-key" # optional — pass --use-api-key to openarc serve start to enforce
    export OPENARC_AUTOLOAD_MODEL="model_name" # model_name to load on startup
    export MODEL_PATH="/path/to/your/models" # mount your models to `/models` inside the container
    docker compose -f .devops/docker-compose.yaml up --build -d
    ```

    Pass `--use-api-key` to `openarc serve start` to require clients to authenticate. See [serve](commands.md#serve) for details.


    Take a look at the [Dockerfile](https://github.com/SearchSavior/OpenArc/blob/main/.devops/Dockerfile) and [docker-compose](https://github.com/SearchSavior/OpenArc/blob/main/.devops/docker-compose.yaml) for more details.



    