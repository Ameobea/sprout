set dotenv-load := true

run:
  yarn dev --port 3080

build:
  yarn build

preview:
  yarn preview --port 3080 --host 0.0.0.0

docker-build:
  docker build -t anime-atlas:latest .

deploy:
  just build
  just docker-build
  docker tag anime-atlas:latest gcr.io/free-tier-164405/anime-atlas:latest
  docker push gcr.io/free-tier-164405/anime-atlas:latest
  gcloud run deploy anime-atlas \
    --image=gcr.io/free-tier-164405/anime-atlas \
    --platform=managed \
    --region=us-west1 \
    --project=free-tier-164405
