FROM node:17.3-slim

ADD . /app
ADD ./data /opt/data
WORKDIR /app

RUN npm install

CMD node ./build/index.js
