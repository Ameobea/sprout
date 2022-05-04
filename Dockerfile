FROM node:17.3-slim

ADD . /app
ADD ./data/processed-metadata.csv /opt/data/processed-metadata.csv
ADD ./data/projected_embedding.json /opt/data/projected_embedding.json
ADD ./data/projected_embedding_pymde.json /opt/data/projected_embedding_pymde.json
ADD ./data/projected_embedding_pymde_3d_40n.json /opt/data/projected_embedding_pymde_3d_40n.json
ADD ./data/projected_embedding_pymde_4d_40n.json /opt/data/projected_embedding_pymde_4d_40n.json
ADD ./data/projected_embedding_pymde_4d_100n.json /opt/data/projected_embedding_pymde_4d_100n.json
WORKDIR /app

RUN npm install

CMD node ./build/index.js

