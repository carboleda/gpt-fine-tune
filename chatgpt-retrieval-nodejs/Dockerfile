FROM oven/bun:1.0

WORKDIR /usr/app

COPY *.ts .
COPY .env .env
COPY data/ ./data
COPY package*.json .
COPY tsconfig.json .

RUN bun install \
    bun install hnswlib-node

ENTRYPOINT [ "bun", "chatgpt.ts" ]