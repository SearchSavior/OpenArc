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

# ============================================================================
# Get tags / metadata info from git for use in both targets
# ============================================================================
FROM ubuntu:24.04 AS metadata
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY .git .git

RUN BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" && \
    VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo unknown)" && \
    VCS_DESCRIBE="$(git describe --tags --always --dirty 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)" && \
    OPENARC_SOURCE="$(git remote get-url origin 2>/dev/null || echo local-working-tree)" && \
    printf '%s\n' "$BUILD_DATE" > /build-date && \
    printf '%s\n' "$VCS_REF" > /git-vcs-ref && \
    printf '%s\n' "$VCS_DESCRIBE" > /git-vcs-describe && \
    printf '%s\n' "$OPENARC_SOURCE" > /git-openarc-source


# ============================================================================
# Build Common image including common dependencies
# ============================================================================
FROM ubuntu:24.04 AS common-base

ENV DEBIAN_FRONTEND=noninteractive

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

# Add venv to PATH so openarc command works
ENV PATH="/app/.venv/bin:$PATH"

# ============================================================================
# Precompile Python bytecode from dependencies to avoid slow first-start imports.
# ============================================================================
RUN python -m compileall -q /app/.venv/lib/python3.12/site-packages

# Copy the local checked-out repository into the image.
COPY --exclude=.git/ . /app

# Install OpenARC
RUN uv pip install --no-deps -e .

# ============================================================================
# Precompile Python bytecode for OpenARC to speed up server start
# This is done in a second step so we don't have to recompile EVERYTHING
#   if only local code changes (docker build cache keeps the previous compilation)
# ============================================================================
RUN python -m compileall -q /app/src

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
#!/bin/bash
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

# Fix ^M / CRLF characters if building on windows
RUN sed -i 's/\r$//' /usr/local/bin/start-openarc.sh && \
    chmod +x /usr/local/bin/start-openarc.sh

# ============================================================================
# Build Standard Version
# ============================================================================
FROM common-base AS standard
ARG OPENARC_VARIANT=standard

RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    level-zero-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Standard Version Build Info Logging
# ============================================================================

# copy git/build metadata forward from the metadata stage.
COPY --from=metadata /build-date /tmp/build-date
COPY --from=metadata /git-vcs-ref /tmp/git-vcs-ref
COPY --from=metadata /git-vcs-describe /tmp/git-vcs-describe
COPY --from=metadata /git-openarc-source /tmp/git-openarc-source

RUN BUILD_DATE="$(cat /tmp/build-date)" && \
    VCS_REF="$(cat /tmp/git-vcs-ref)" && \
    VCS_DESCRIBE="$(cat /tmp/git-vcs-describe)" && \
    OPENARC_SOURCE="$(cat /tmp/git-openarc-source)" && \
    echo "=== Build Information ===" > /app/BUILD_INFO.txt && \
    echo "Docker Image Variant: ${OPENARC_VARIANT}" >> /app/BUILD_INFO.txt && \
    echo "Build Date: ${BUILD_DATE}" >> /app/BUILD_INFO.txt && \
    echo "OpenARC Version: ${VCS_DESCRIBE}" >> /app/BUILD_INFO.txt && \
    echo "Git Ref: ${VCS_REF}" >> /app/BUILD_INFO.txt && \
    echo "Source: ${OPENARC_SOURCE}" >> /app/BUILD_INFO.txt && \
    echo "" >> /app/BUILD_INFO.txt && \
    echo "=== Intel Package Versions ===" >> /app/BUILD_INFO.txt && \
    uv pip list | grep -E "(openvino|optimum|torch)" >> /app/BUILD_INFO.txt || true && \
    echo "" >> /app/BUILD_INFO.txt && \
    echo "=== System Package Versions ===" >> /app/BUILD_INFO.txt && \
    dpkg -l | grep -E "intel-opencl|level-zero|libze" | awk '{print $2 " " $3}' >> /app/BUILD_INFO.txt || true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models || exit 1
CMD ["/usr/local/bin/start-openarc.sh"]

# ============================================================================
# Build Battlemage Version
# ============================================================================
FROM common-base AS battlemage
ARG OPENARC_VARIANT=battlemage

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:kobuk-team/intel-graphics \
    && apt-get update && apt-get install -y --no-install-recommends \
    libze-intel-gpu1 \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Bring git/build metadata forward from the git-metadata stage.
COPY --from=metadata /build-date /tmp/build-date
COPY --from=metadata /git-vcs-ref /tmp/git-vcs-ref
COPY --from=metadata /git-vcs-describe /tmp/git-vcs-describe
COPY --from=metadata /git-openarc-source /tmp/git-openarc-source

# ============================================================================
# Battlemage Version Build Info Logging
# ============================================================================
RUN BUILD_DATE="$(cat /tmp/build-date)" && \
    VCS_REF="$(cat /tmp/git-vcs-ref)" && \
    VCS_DESCRIBE="$(cat /tmp/git-vcs-describe)" && \
    OPENARC_SOURCE="$(cat /tmp/git-openarc-source)" && \
    echo "=== Build Information ===" > /app/BUILD_INFO.txt && \
    echo "Docker Image Variant: ${OPENARC_VARIANT}" >> /app/BUILD_INFO.txt && \
    echo "Build Date: ${BUILD_DATE}" >> /app/BUILD_INFO.txt && \
    echo "OpenARC Version: ${VCS_DESCRIBE}" >> /app/BUILD_INFO.txt && \
    echo "Git Ref: ${VCS_REF}" >> /app/BUILD_INFO.txt && \
    echo "Source: ${OPENARC_SOURCE}" >> /app/BUILD_INFO.txt && \
    echo "" >> /app/BUILD_INFO.txt && \
    echo "=== Intel Package Versions ===" >> /app/BUILD_INFO.txt && \
    uv pip list | grep -E "(openvino|optimum|torch)" >> /app/BUILD_INFO.txt || true && \
    echo "" >> /app/BUILD_INFO.txt && \
    echo "=== System Package Versions ===" >> /app/BUILD_INFO.txt && \
    dpkg -l | grep -E "intel-opencl|level-zero|libze" | awk '{print $2 " " $3}' >> /app/BUILD_INFO.txt || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f -H "Authorization: Bearer ${OPENARC_API_KEY}" http://localhost:8000/v1/models || exit 1

CMD ["/usr/local/bin/start-openarc.sh"]