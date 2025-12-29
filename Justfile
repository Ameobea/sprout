set dotenv-load := true

run:
  bun run dev --port 3080

yarn-build:
  yarn run build

build:
  bun run build

preview:
  bun run preview --port 3080 --host 0.0.0.0

docker-build:
  docker build -t anime-atlas:latest .

deploy-gcs:
  just build
  just docker-build
  docker tag anime-atlas:latest gcr.io/free-tier-164405/anime-atlas:latest
  docker push gcr.io/free-tier-164405/anime-atlas:latest
  gcloud run deploy anime-atlas \
    --image=gcr.io/free-tier-164405/anime-atlas \
    --platform=managed \
    --region=us-west1 \
    --project=free-tier-164405

deploy:
  #!/bin/bash

  just build
  just docker-build
  docker save anime-atlas:latest | bzip2 > anime-atlas.tar.bz2
  ameotrack upload -e 1 anime-atlas.tar.bz2

launch-jupyter:
  #!/usr/bin/env zsh

  id="$(docker run -d --net host -v "${PWD}":/home/jovyan/work --user root --memory-swap -1 -e GRANT_SUDO=yes -e MYSQL_HOST="$MYSQL_HOST" -e MYSQL_USER="$MYSQL_USER" -e MYSQL_PASSWORD="$MYSQL_PASSWORD" -e MYSQL_DATABASE="$MYSQL_DATABASE" jupyter/tensorflow-notebook:latest)"
  echo "Launched docker container with id=${id}"
  sleep 1
  echo "$((docker logs $id) 2>&1 | grep token | head -n 1)"

launch-jax:
  docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 32G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/jax_dir --name rocm_jax rocm/jax:latest /bin/bash

model-server-build:
  docker build -f Dockerfile.model_server -t anime-model-server:latest .

model-server-run:
  docker run --rm -p 8000:8000 anime-model-server:latest

build-and-deploy-model-server:
  #!/bin/bash

  just model-server-build
  docker save anime-model-server:latest | bzip2 > anime-model-server.tar.bz2
  scp anime-model-server.tar.bz2 debian@ameo.dev:/tmp/anime-model-server.tar.bz2

  ssh debian@ameo.dev -t 'cat /tmp/anime-model-server.tar.bz2 | bunzip2 | docker load && docker kill anime-model-server  && docker container rm anime-model-server && docker run  --name anime-model-server  --restart=always   -d   --net host   anime-model-server:latest   uvicorn model_server:app --host 0.0.0.0 --port 5708 --timeout-keep-alive 120'
