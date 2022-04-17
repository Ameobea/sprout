set dotenv-load := true

run:
  yarn dev --port 3080

build:
  yarn build

preview:
  yarn preview --port 3080 --host 0.0.0.0
