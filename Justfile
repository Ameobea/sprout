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

  id="$(docker run -d --rm --net host -v "${PWD}":/home/jovyan/work --user root --memory-swap -1 -e GRANT_SUDO=yes -e MYSQL_HOST="$MYSQL_HOST" -e MYSQL_USER="$MYSQL_USER" -e MYSQL_PASSWORD="$MYSQL_PASSWORD" -e MYSQL_DATABASE="$MYSQL_DATABASE" jupyter/tensorflow-notebook:latest)"
  echo "Launched docker container with id=${id}"
  sleep 1
  echo "$((docker logs $id) 2>&1 | grep token | head -n 1)"
