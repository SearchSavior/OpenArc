#!/usr/bin/env bash
set -e

echo OpenArc setup script for Linux

if ! command -v uv &> /dev/null; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "ERROR: Intel oneAPI not found."
    echo "install from https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
    exit 1
fi

source /opt/intel/oneapi/setvars.sh intel64 --force

uv sync
source .venv/bin/activate

uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

read -p "set OPENARC_API_KEY? (y/N): " set_key
if [[ "$set_key" =~ ^[Yy]$ ]]; then
    read -p "key (default: openarc-api-key): " api_key
    export OPENARC_API_KEY="${api_key:-openarc-api-key}"
    echo "OPENARC_API_KEY set for this session."
fi

openarc --help
