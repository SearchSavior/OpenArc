FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# Copy the project files
COPY . /app

# Run uv sync to install dependencies
RUN uv sync

# Create models directory with placeholder file
RUN mkdir -p /models && echo "" > /models/place_models_here.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]

