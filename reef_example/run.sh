#!/bin/bash
# Serve + run. Setup (once): `uv sync --group reef` in the repo root (installs
# the reef checkout at ../reef editable, plus reef-client) and `pi` on PATH.
# State and logs go to ./work. Usage: ./run.sh [--limit N] (passed to run.py).
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
mkdir -p work/recipes work/results

# 1. Environment.
#    - the venv first on PATH so `python3` (reef.service) and `restore` resolve
#      there; bin/ before it so pi episodes get the RESTORE_* wrapper
#    - upstream key from ~/.dsh/.env (AIGW_KEY) -> REEF_UPSTREAM_API_KEY; the
#      key never enters serve.yaml or the repo
#    - the local proxy hijacks aigw.meshy.team (known trap: every call 4xx/5xx
#      or hangs); reef (urllib/aiohttp honour *_proxy) and pi both run from
#      this shell, so unset all of it here
export PATH="$PWD/bin:$ROOT/.venv/bin:$PATH"
if [ -z "$REEF_UPSTREAM_API_KEY" ]; then
    REEF_UPSTREAM_API_KEY="$(grep -E '^AIGW_KEY=' "$HOME/.dsh/.env" | head -1 | cut -d= -f2- | tr -d '"'"'")"
    export REEF_UPSTREAM_API_KEY
fi
[ -n "$REEF_UPSTREAM_API_KEY" ] || { echo "no REEF_UPSTREAM_API_KEY and no AIGW_KEY in ~/.dsh/.env" >&2; exit 1; }
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="aigw.meshy.team,127.0.0.1,localhost" no_proxy="$NO_PROXY"
export RESTORE_RESULTS_DIR="$PWD/work/results" RESTORE_CACHE_DIR="$PWD/work/cache" RESTORE_TRACE="$PWD/work/results/trace.jsonl"

# 2. Copy serve.yaml's recipe sections where the recipe registry reads them,
#    with the task prompts generated from tasks/data/manifest.json.
export REEF_RECIPE_CONFIG_DIR="$PWD/work/recipes"
python3 materialize.py

# 3. Start Reef, stop it again when this script exits. -c is absolute: reef
#    resolves a relative config path against its own repo root.
python3 -m reef serve -c "$PWD/serve.yaml" > work/reef.log 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT  # SIGTERM: reef stops its services

while ! curl -sf --noproxy '*' http://127.0.0.1:8900/healthz > /dev/null; do
    kill -0 "$SERVE_PID" 2>/dev/null || { cat work/reef.log >&2; exit 1; }
    sleep 1
done

# 4. The loop.
python3 run.py "$@"
