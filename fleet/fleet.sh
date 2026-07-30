#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

source fleet.env
COMPUTE="https://compute.${OS_REGION}.cloud.ovh.us/v2.1/${OS_PROJECT_ID}"
NETWORK="https://network.${OS_REGION}.cloud.ovh.us"
KEYPAIR=mal-fleet
PREFIX=mal-collector
MAIN_SERVER=ameo.dev
SSH_OPTS=(-i fleet_key -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ConnectTimeout=10 -o BatchMode=yes)

die() { echo "error: $*" >&2; exit 1; }

auth() {
  local body
  body=$(jq -n --arg u "$OS_USERNAME" --arg p "$OS_PASSWORD" --arg proj "$OS_PROJECT_ID" \
    '{auth:{identity:{methods:["password"],password:{user:{name:$u,domain:{name:"Default"},password:$p}}},scope:{project:{id:$proj}}}}')
  TOK=$(curl -sf -D - -o /dev/null -X POST "$OS_AUTH_URL/auth/tokens" -H 'Content-Type: application/json' -d "$body" \
    | grep -i '^x-subject-token' | tr -d '\r' | cut -d' ' -f2)
  [[ -n $TOK ]] || die "openstack auth failed"
}

os() {
  local method=$1 path=$2
  if [[ $# -ge 3 ]]; then
    curl -sf -X "$method" -H "X-Auth-Token: $TOK" -H 'Content-Type: application/json' -d "$3" "$COMPUTE$path"
  else
    curl -sf -X "$method" -H "X-Auth-Token: $TOK" "$COMPUTE$path"
  fi
}

env_val() { grep "^$1=" ../.env | cut -d= -f2-; }

gen_agent_env() {
  mkdir -p agent/dist
  cat > agent/dist/agent.env <<EOF
MAL_CLIENT_ID=$MAL_COLLECTOR_CLIENT_ID
MAL_CLIENT_SECRET=$MAL_COLLECTOR_CLIENT_SECRET
ADMIN_API_TOKEN=$(env_val ADMIN_API_TOKEN)
MAIN_SERVER_URL=https://anime.ameo.dev
COLLECTION_TYPE=${COLLECTION_TYPE:-anime}
PACE_MS=${PACE_MS:-1200}
MYSQL_HOST=unused
MYSQL_USER=unused
MYSQL_PASSWORD=unused
MYSQL_DATABASE=unused
EOF
}

ensure_key() {
  [[ -f fleet_key ]] || ssh-keygen -t ed25519 -N '' -f fleet_key -C mal-fleet >/dev/null
  if ! os GET /os-keypairs | jq -e --arg k "$KEYPAIR" '.keypairs[] | select(.keypair.name == $k)' >/dev/null; then
    os POST /os-keypairs "$(jq -n --arg k "$KEYPAIR" --arg pub "$(cat fleet_key.pub)" '{keypair:{name:$k,public_key:$pub}}')" >/dev/null
    echo "created keypair $KEYPAIR"
  fi
}

user_data() {
  cat <<'EOF'
#cloud-config
package_update: true
packages: [nodejs]
write_files:
  - path: /etc/systemd/system/mal-collector.service
    content: |
      [Unit]
      Description=MAL profile collector agent
      After=network-online.target

      [Service]
      User=debian
      ExecStart=/usr/bin/node /opt/mal-collector/agent.cjs
      EnvironmentFile=/opt/mal-collector/agent.env
      Restart=always
      RestartSec=5

      [Install]
      WantedBy=multi-user.target
runcmd:
  - install -d -o debian -g debian /opt/mal-collector
  - systemctl daemon-reload
EOF
}

fleet_servers() { os GET /servers/detail | jq -c --arg p "$PREFIX" '[.servers[] | select(.name | startswith($p))]'; }

refresh_hosts() {
  fleet_servers \
    | jq -r '.[] | .name + " " + ((.addresses["Ext-Net"] // []) | map(select(.version == 4)) | (.[0].addr // "-")) + " " + .id' \
    | sort -V > hosts
  cat hosts
}

provision_host() {
  local ip=$1
  echo "provisioning $ip..."
  for _ in $(seq 40); do
    ssh "${SSH_OPTS[@]}" "debian@$ip" true 2>/dev/null && break
    sleep 5
  done
  ssh "${SSH_OPTS[@]}" "debian@$ip" 'cloud-init status --wait >/dev/null || true'
  scp "${SSH_OPTS[@]}" agent/dist/agent.cjs agent/dist/agent.env "debian@$ip:/opt/mal-collector/"
  ssh "${SSH_OPTS[@]}" "debian@$ip" 'sudo systemctl enable --now mal-collector && systemctl is-active mal-collector'
  echo "$ip: collector running"
}

cmd_spawn() {
  local n=${1:-1}
  auth
  ensure_key
  gen_agent_env
  [[ -f agent/dist/agent.cjs ]] || (cd .. && just build-agent)
  local image flavor net
  image=$(os GET /images/detail | jq -r '[.images[] | select(.name == "Debian 12")][0].id')
  flavor=$(os GET /flavors/detail | jq -r '[.flavors[] | select(.name == "d2-2")][0].id')
  net=$(curl -sf -H "X-Auth-Token: $TOK" "$NETWORK/v2.0/networks?name=Ext-Net" | jq -r '.networks[0].id')
  [[ -n $image && $image != null && -n $flavor && $flavor != null && -n $net && $net != null ]] \
    || die "failed to resolve image/flavor/network"
  local ud existing
  ud=$(user_data | base64 -w0)
  existing=$(fleet_servers | jq -r --arg p "$PREFIX-" '[.[].name | ltrimstr($p) | tonumber] | max // 0')
  echo "spawning $n d2-2 instance(s) in $OS_REGION starting at $PREFIX-$((existing + 1))"
  local ids=()
  for i in $(seq $((existing + 1)) $((existing + n))); do
    local body id
    body=$(jq -n --arg name "$PREFIX-$i" --arg img "$image" --arg fl "$flavor" --arg net "$net" --arg key "$KEYPAIR" --arg ud "$ud" \
      '{server:{name:$name,imageRef:$img,flavorRef:$fl,key_name:$key,user_data:$ud,networks:[{uuid:$net}]}}')
    id=$(os POST /servers "$body" | jq -r '.server.id')
    echo "  $PREFIX-$i: $id"
    ids+=("$id")
  done
  local id detail status name ip
  for id in "${ids[@]}"; do
    for _ in $(seq 60); do
      detail=$(os GET "/servers/$id")
      status=$(jq -r '.server.status' <<<"$detail")
      [[ $status == ACTIVE || $status == ERROR ]] && break
      sleep 5
    done
    name=$(jq -r '.server.name' <<<"$detail")
    [[ $status == ACTIVE ]] || die "$name failed to become ACTIVE (status=$status)"
    ip=$(jq -r '(.server.addresses["Ext-Net"] // []) | map(select(.version == 4)) | .[0].addr' <<<"$detail")
    echo "$name is ACTIVE at $ip"
    provision_host "$ip"
  done
  refresh_hosts
}

cmd_status() {
  local token
  token=$(env_val ADMIN_API_TOKEN)
  echo "== queue =="
  curl -s "https://anime.ameo.dev/metrics?token=$token" | grep -v '^#' || echo "(metrics unavailable)"
  echo "== agents =="
  local name ip id
  while read -r name ip id; do
    printf '%s (%s): ' "$name" "$ip"
    ssh -n "${SSH_OPTS[@]}" "debian@$ip" \
      'echo "$(systemctl is-active mal-collector)  $(journalctl -u mal-collector -n1 --no-pager -o cat 2>/dev/null | tail -c 200)"' \
      2>/dev/null || echo unreachable
  done < hosts
}

cmd_each() {
  local verb=$1 ip
  for ip in $(awk '{print $2}' hosts); do
    ssh "${SSH_OPTS[@]}" "debian@$ip" "sudo systemctl $verb mal-collector" &
  done
  wait
  echo "$verb done"
}

cmd_destroy() {
  auth
  local servers count
  servers=$(fleet_servers)
  count=$(jq length <<<"$servers")
  [[ $count -gt 0 ]] || { echo "no $PREFIX instances"; return; }
  jq -r '.[] | "\(.name)  \(.id)"' <<<"$servers"
  read -rp "delete these $count instance(s)? [y/N] " yn
  [[ $yn == y* ]] || die aborted
  local id
  for id in $(jq -r '.[].id' <<<"$servers"); do
    os DELETE "/servers/$id"
    echo "deleted $id"
  done
  rm -f hosts
}

cmd_requeue() {
  local type=${1:-anime} statuses=${2:-200} col
  case $type in
    anime) col='collected' ;;
    manga) col='`collected-manga`' ;;
    *) die "type must be anime or manga" ;;
  esac
  local user pw
  user=$(env_val MYSQL_USER)
  pw=$(env_val MYSQL_PASSWORD)
  run_sql() { ssh "debian@$MAIN_SERVER" "docker exec -i mysql_main_mariadb mariadb -u'$user' -p'$pw' -t anime-atlas"; }
  echo "current status counts:"
  echo "SELECT $col AS status, COUNT(*) AS count FROM \`usernames-to-collect\` GROUP BY $col;" | run_sql
  read -rp "requeue $type rows with status IN ($statuses) back to 0? [y/N] " yn
  [[ $yn == y* ]] || die aborted
  echo "UPDATE \`usernames-to-collect\` SET $col = 0 WHERE $col IN ($statuses); SELECT ROW_COUNT() AS requeued;" | run_sql
}

case ${1:-} in
  spawn) cmd_spawn "${2:-1}" ;;
  provision) gen_agent_env; provision_host "$2" ;;
  ls) auth; refresh_hosts ;;
  status) cmd_status ;;
  start | stop | restart) cmd_each "$1" ;;
  destroy) cmd_destroy ;;
  requeue) cmd_requeue "${2:-anime}" "${3:-200}" ;;
  *)
    cat <<'EOF'
usage: fleet.sh <command>
  spawn [n]             boot n d2-2 instances (default 1), provision, start collectors
  provision <ip>        re-push agent.cjs + agent.env to a host and (re)start its collector
  ls                    list fleet instances (refreshes ./hosts)
  status                queue metrics + per-agent status
  start|stop|restart    systemctl verb for the collector on all hosts
  destroy               delete ALL mal-collector instances (prompts)
  requeue [type] [st]   flip usernames with status in list back to 0 (default: anime 200)
EOF
    ;;
esac
