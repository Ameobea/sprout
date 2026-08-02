#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TABLE=${1:-mal-user-animelists}
OUT=${2:-../data/${TABLE}.tsv.zst}
LIMIT=${3:-}
U=$(grep '^MYSQL_USER' ../.env | cut -d= -f2-)
P=$(grep '^MYSQL_PASSWORD' ../.env | cut -d= -f2-)
STAGE="dumps/${TABLE}.tsv.zst"
QUERY="SELECT * FROM \`$TABLE\`${LIMIT:+ LIMIT $LIMIT}"

# --quick streams rows instead of buffering the full result set in the client (OOMs the box otherwise)
ssh debian@ameo.dev "set -o pipefail; mkdir -p dumps && docker exec mysql_main_mariadb mariadb -u'$U' -p'$P' --batch --raw --quick anime-atlas -e '$QUERY' | zstd -T0 -3 -f -o $STAGE"
rsync -P "debian@ameo.dev:$STAGE" "$OUT"
echo "done: $OUT ($(du -h "$OUT" | cut -f1)); decompress-stream with: zstd -dc $OUT"
