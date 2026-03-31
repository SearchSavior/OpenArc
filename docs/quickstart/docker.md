# Docker

Instead of fighting with Intel's own docker images, we built our own which is as close to boilerplate as possible. For a primer on docker [check out this video](https://www.youtube.com/watch?v=DQdB7wFEygo).

## Build and run the container

```bash
docker-compose up --build -d
```

## Run the container

```bash
docker run -d -p 8000:8000 openarc:latest
```

## Enter the container

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

Take a look at the [Dockerfile](https://github.com/SearchSavior/OpenArc/blob/main/Dockerfile) and [docker-compose.yaml](https://github.com/SearchSavior/OpenArc/blob/main/docker-compose.yaml) for more details.
