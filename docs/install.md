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
    
    Prebuilt Images are available on [ghcr.io](https://github.com/SearchSavior?tab=packages&repo_name=OpenArc).

    **Running OpenArc with compose:**

    download the `docker-compose.yaml` file from the repository (or clone the repository) and run:
    ```bash
    docker-compose up -d
    ```
    > _To run the battlemage-optimized image, edit the `docker-compose.yaml` file and switch to the alternative `image:` entry._

    **Optional Environment Variables for compose:**

    _(you can also place these environment variables in a `.env` file in the root of the repository)_

    ```bash
    export OPENARC_API_KEY="openarc-api-key" # optional — pass --use-api-key to openarc serve start to enforce
    export OPENARC_AUTOLOAD_MODEL="model_name" # model_name to load on startup
    export MODEL_PATH="/path/to/your/models" # mount your models to `/models` inside the container
    docker-compose up -d
    ```

    **Run OpenArc with docker (no compose):**
    ```
    docker run --name openarc \
        --device /dev/dri:/dev/dri \
        -v <path-to-your-models-folder>:/models \
        -v <path-to-config-folder>:/persist \
        -p 8000:8000 \ 
        ghcr.io/searchsavior/openarc:latest
    ```
    _(replace `openarc:latest` with `openarc-battlemage:latest` for the battlemage image)_ 

    **Building the OpenArc docker images yourself**
    
    If you want to build the docker image yourself, clone the repository and build from the [Dockerfile](https://github.com/SearchSavior/OpenArc/blob/main/Dockerfile):

    ```bash
    docker build --target standard -t ghcr.io/searchsavior/openarc:latest .
    ```
    _or, for battlemage:_
    ```bash
    docker build --target battlemage -t ghcr.io/searchsavior/openarc-battlemage:latest .
    ``` 

    **Enter the container:**
    ```bash
    docker exec -it openarc /bin/bash
    ```

    

    Pass `--use-api-key` to `openarc serve start` to require clients to authenticate. See [serve](commands.md#serve) for details.
 
    Take a look at the [Dockerfile](https://github.com/SearchSavior/OpenArc/blob/main/Dockerfile) and [docker-compose](https://github.com/SearchSavior/OpenArc/blob/main/docker-compose.yaml) for more details.



    