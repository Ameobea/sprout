FROM node:20.9-slim

ADD . /app
ADD ./data/processed-metadata.csv /opt/data/processed-metadata.csv
# ADD ./data/projected_embedding_pymde.json /opt/data/projected_embedding_pymde.json
ADD ./data/projected_model_embedding.json /opt/data/projected_model_embedding.json

WORKDIR /app

RUN npm install

CMD node ./build/index.js
