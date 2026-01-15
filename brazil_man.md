ARG ONEAPI_VERSION=2025.3.2-0-devel-ubuntu24.04

FROM docker.io/intel/deep-learning-essentials:$ONEAPI_VERSION

RUN apt update \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt install -y \
    python3.11 \
    python3.11-dev \
    git

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN git clone https://github.com/SearchSavior/OpenArc
WORKDIR OpenArc
ARG BRANCH=2.0.2
RUN git checkout $BRANCH

RUN bash -c "source $HOME/.local/bin/env && uv sync && uv pip install optimum-intel[openvino]@git+https://github.com/huggingface/optimum-intel && uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly"

ENV OPENARC_API_KEY=key
ENV OPENARC_HOST=0.0.0.0
ENV OPENARC_PORT=1234

CMD ["sh", "-c", "/OpenArc/.venv/bin/openarc serve start --host $OPENARC_HOST --openarc-port $OPENARC_PORT"]