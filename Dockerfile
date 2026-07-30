FROM node:20.9-slim

WORKDIR /app

COPY package.json yarn.lock ./
RUN corepack enable && COREPACK_ENABLE_DOWNLOAD_PROMPT=0 yarn install --frozen-lockfile

COPY data/processed-metadata.csv /opt/data/processed-metadata.csv
COPY data/projected_model_embedding.json /opt/data/projected_model_embedding.json
COPY build ./build

CMD ["node", "./build/index.js"]
