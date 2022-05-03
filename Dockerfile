FROM node:17.3-slim

ADD . /app
ADD ./data/processed-metadata.csv /opt/data/processed-metadata.csv
ADD ./data/projected_embedding.json /opt/data/projected_embedding.json
ADD ./data/projected_embedding_pymde.json /opt/data/projected_embedding_pymde.json
WORKDIR /app

RUN npm install

CMD node ./build/index.js

