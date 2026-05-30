# ============================================================================
# OpenARC multi-target image build
#
# Targets:
#   - standard   : Intel GPU path (level-zero packages)
#   - battlemage : Battlemage Intel GPU path
#
# Build examples:
#   docker build --target standard   -t {imagename}:dev .
#   docker build --target battlemage -t {imagename}-battlemage:dev .
# ============================================================================

FROM ubuntu:24.04 AS common-base

ENV DEBIAN_FRONTEND=noninteractive

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG VCS_DESCRIBE=unknown
ARG OPENARC_SOURCE=local-working-tree

# ============================================================================
# Common System Dependencies
# ============================================================================


RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gpg \
    gpg-agent \
    wget \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    cmake \
    build-essential \
    libudev-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Intel GPU Drivers
# ============================================================================
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > \
      /etc/apt/sources.list.d/intel-gpu-noble.list

RUN git clone https://github.com/intel/linux-npu-driver.git /tmp/npu-driver && \
    cd /tmp/npu-driver && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/npu-driver

# ============================================================================
# Install uv package manager
# ============================================================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ============================================================================
# Setup OpenArc; Assume we're running from am local repo, versus cloning.
#   (This allows local image generation from the local code repository)
# ============================================================================
WORKDIR /app

# Copy dependency metadata first so app code changes do not always invalidate
# the Python dependency layers.
COPY pyproject.toml ./
COPY uv.lock* ./
COPY README* ./

RUN uv sync && \
    uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel" && \
    uv pip install --pre -U openvino-genai openvino-tokenizers \
      --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

# Copy the local checked-out repository into the image.
COPY . /app

# Add venv to PATH so openarc command works
ENV PATH="/app/.venv/bin:$PATH"

# ============================================================================
# Precompile Python bytecode to avoid slow first-start imports.
# NB: If ever code path / dependencies change between standard and battlemage,
#       this should move down into the image-specific sections.
#       For now, compiled output does not differ between images.
# ============================================================================
RUN python -m compileall -q /app/src /app/.venv/lib/python3.12/site-packages

# ============================================================================
# Runtime Configuration
# ============================================================================
ENV NEOReadDebugKeys=1 \
    OverrideGpuAddressSpace=48 \
    EnableImplicitScaling=1 \
    OPENARC_API_KEY=key \
    OPENARC_AUTOLOAD_MODEL="" \
    OPENARC_MODELS_DIR=/models

# Create persistent config directory and symlink
RUN mkdir -p /persist /models && \
    ln -sf /persist/openarc_config.json /app/openarc_config.json

# ============================================================================
# Startup Script
# ============================================================================
RUN cat > /usr/local/bin/start-openarc.sh <<'SCRIPT'
#!/usr/bin/env bash
set -e

echo "================================================"
echo "=== Starting OpenArc Server ==="
echo "================================================"

if [ -f /app/BUILD_INFO.txt ]; then
  cat /app/BUILD_INFO.txt
  echo ""
fi

echo "=== Runtime Configuration ==="
echo "Port: 8000"
echo "API Key: ${OPENARC_API_KEY:0:10}..."
echo "Auto-load Model: ${OPENARC_AUTOLOAD_MODEL:-none}"
echo ""
echo "================================================"

# Start server in background
openarc serve start --host 0.0.0.0 --port 8000 &
SERVER_PID=$!


# Auto-load model if specified
if [ -n "$OPENARC_AUTOLOAD_MODEL" ]; then
  echo "Waiting for server to start..."
  for i in {1..24}; do
    if curl -s -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "Server ready after $(( (i - 1) * 5)) seconds"
      echo "Auto-loading model: $OPENARC_AUTOLOAD_MODEL"
      openarc load "$OPENARC_AUTOLOAD_MODEL" || echo "Failed to auto-load model"
      openarc status || true
      break
    fi
    sleep 5
  done
fi

# Wait for server
wait $SERVER_PID
SCRIPT

RUN chmod +x /usr/local/bin/start-openarc.sh

# ============================================================================
# Build Standard Version
# ============================================================================
FROM common-base AS standard
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    level-zero-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Standard Version Build Info Logging
# ============================================================================
RUN printf '%s\n' \
    '=== Build Information ===' \
    "Build Date: ${BUILD_DATE}" \
    "OpenARC Version: ${VCS_DESCRIBE}" \
    "Git Ref: ${VCS_REF}" \
    "Source: ${OPENARC_SOURCE}" \
    '' \
    '=== Intel Package Versions ===' \
    > /app/BUILD_INFO.txt && \
    (uv pip list | grep -E '(openvino|optimum|torch)' >> /app/BUILD_INFO.txt || true) && \
    printf '\n=== System Package Versions ===\n' >> /app/BUILD_INFO.txt && \
    (dpkg -l | grep -E 'intel-opencl|level-zero|libze' | awk '{print $2 " " $3}' >> /app/BUILD_INFO.txt || true)

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models || exit 1
CMD ["/usr/local/bin/start-openarc.sh"]

# ============================================================================
# Build Battlemage Version
# ============================================================================
FROM common-base AS battlemage
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:kobuk-team/intel-graphics \
    && apt-get update && apt-get install -y --no-install-recommends \
    libze-intel-gpu1 \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Battlemage Version Build Info Logging
# ============================================================================
RUN printf '%s\n' \
    '=== Build Information ===' \
    "Build Date: ${BUILD_DATE}" \
    "OpenARC Version: ${VCS_DESCRIBE}" \
    "Git Ref: ${VCS_REF}" \
    "Source: ${OPENARC_SOURCE}" \
    '' \
    '=== Intel Package Versions ===' \
    > /app/BUILD_INFO.txt && \
    (uv pip list | grep -E '(openvino|optimum|torch)' >> /app/BUILD_INFO.txt || true) && \
    printf '\n=== System Package Versions ===\n' >> /app/BUILD_INFO.txt && \
    (dpkg -l | grep -E 'intel-opencl|level-zero|libze' | awk '{print $2 " " $3}' >> /app/BUILD_INFO.txt || true)

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models || exit 1

CMD ["/usr/local/bin/start-openarc.sh"]