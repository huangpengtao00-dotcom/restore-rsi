#!/bin/bash
# Serve + run. Setup (once): `uv sync --group reef` in the repo root (installs
# the reef checkout at ../reef editable, plus reef-client) and `pi` on PATH.
# Usage: ./run.sh [--limit N] (passed to run.py).
#
# **一台机器上可以同时跑多个配置。** 消融那几个臂之间互不相干,而 reef 的 evaluate
# 是串行跑 episode 的(cordis_backend 里 `tuple(... for task in self._tasks)`),
# 改不动也不该改。所以并行的粒度放在**配置之间**而不是 episode 之间:
#
#   RESTORE_ABLATION=measurement_failure=zero REEF_PORT=8901 RESTORE_WORK=work/abl-mf-zero ./run.sh &
#   RESTORE_ABLATION=gate=wins_over_losses    REEF_PORT=8902 RESTORE_WORK=work/abl-gate    ./run.sh &
#
# 总时间于是等于最慢的那个配置,而不是它们的和。实测网关在并发 16 下零失败、
# 本机 CPU 67%,几个配置各占一个在途请求毫无压力。
#
# 每个配置必须有**自己的** work 目录和端口:共用 work 会让 trace、缓存、
# task_refs 互相覆盖,而那种污染事后完全看不出来。
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
WORK="${RESTORE_WORK:-work}"
PORT="${REEF_PORT:-8900}"
mkdir -p "$WORK/recipes" "$WORK/results"
echo "== work=$WORK port=$PORT ablation=${RESTORE_ABLATION:-baseline}"

# 1. Environment.
#    - the venv first on PATH so `python3` (reef.service) and `restore` resolve
#      there; bin/ before it so pi episodes get the RESTORE_* wrapper
#    - upstream endpoint and key come from the environment, or from
#      ~/.dsh/.env (RESTORE_UPSTREAM_URL / RESTORE_UPSTREAM_KEY); neither ever
#      enters serve.yaml or the repo. Any OpenAI-compatible endpoint works.
#    - a local proxy will hijack the upstream host (known trap: every call
#      4xx/5xx or hangs); reef (urllib/aiohttp honour *_proxy) and pi both run
#      from this shell, so unset all of it here and pin NO_PROXY to the host
export PATH="$PWD/bin:$ROOT/.venv/bin:$PATH"
_from_env_file() {  # $1=key in ~/.dsh/.env
    [ -f "$HOME/.dsh/.env" ] && grep -E "^$1=" "$HOME/.dsh/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"
}
if [ -z "$REEF_UPSTREAM_API_KEY" ]; then
    # RESTORE_UPSTREAM_KEY 优先;AIGW_KEY 是既有配置的键名,保留兼容 ——
    # 改读新键名时没同步 ~/.dsh/.env,run.sh 于是直接报错退出(2026-09-06 引入,当天修)。
    REEF_UPSTREAM_API_KEY="$(_from_env_file RESTORE_UPSTREAM_KEY)"
    [ -n "$REEF_UPSTREAM_API_KEY" ] || REEF_UPSTREAM_API_KEY="$(_from_env_file AIGW_KEY)"
    export REEF_UPSTREAM_API_KEY
fi
if [ -z "$REEF_UPSTREAM_URL" ]; then
    REEF_UPSTREAM_URL="$(_from_env_file RESTORE_UPSTREAM_URL)"
    export REEF_UPSTREAM_URL
fi
[ -n "$REEF_UPSTREAM_API_KEY" ] || { echo "set REEF_UPSTREAM_API_KEY, or RESTORE_UPSTREAM_KEY= in ~/.dsh/.env" >&2; exit 1; }
[ -n "$REEF_UPSTREAM_URL" ] || { echo "set REEF_UPSTREAM_URL, or RESTORE_UPSTREAM_URL= in ~/.dsh/.env (any OpenAI-compatible endpoint)" >&2; exit 1; }
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# Pin NO_PROXY to whatever host the endpoint actually is, so this stays correct
# for any upstream rather than naming one.
_upstream_host="$(printf '%s' "$REEF_UPSTREAM_URL" | sed -E 's#^[a-z]+://##; s#/.*##; s#:[0-9]+$##')"
export NO_PROXY="$_upstream_host,127.0.0.1,localhost" no_proxy="$NO_PROXY"
export RESTORE_RESULTS_DIR="$PWD/$WORK/results" RESTORE_CACHE_DIR="$PWD/$WORK/cache" RESTORE_TRACE="$PWD/$WORK/results/trace.jsonl"

#    The trace is the only record of what actually ran, and a reset between
#    runs already cost one (the leaky run's 1653 events, gone before they could
#    be checked against the ceiling). Keep each run's under its own name.
if [ -s "$WORK/results/trace.jsonl" ]; then
    mv "$WORK/results/trace.jsonl" "$WORK/results/trace.$(date +%Y%m%d-%H%M%S).jsonl"
fi

# 2. Copy serve.yaml's recipe sections where the recipe registry reads them,
#    with the task prompts generated from tasks/data/manifest.json.
export REEF_RECIPE_CONFIG_DIR="$PWD/$WORK/recipes"
export RESTORE_WORK="$WORK"
python3 materialize.py

# 3. Start Reef, stop it again when this script exits. -c is absolute: reef
#    resolves a relative config path against its own repo root.
# reef's artifact client runs `git ls-remote <path>`; a relative path is
# resolved by git against the *enclosing* worktree root (this dir sits inside
# the restore-rsi repo), so every work/ path must be absolute. Resolve them
# into a generated copy instead of hard-coding this machine's path in serve.yaml.
python3 - "$PWD" "$WORK" "$PORT" <<'PY'
import re, sys, pathlib
here, work, port = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
src = (here / "serve.yaml").read_text()
# 路径必须绝对:reef 的 artifact client 跑 `git ls-remote <path>`,相对路径会被 git
# 按外层 worktree 解析(这个目录在 restore-rsi 仓里面)
resolved = re.sub(r"^(\s*(?:agent_record_dir|artifact_repository|artifact_work_dir|artifact_cache_dir|run_dir):\s*)work/", lambda m: m.group(1) + str(here / work) + "/", src, flags=re.M)
# 端口跟着 REEF_PORT 走,几个配置才能同时起服务
resolved = re.sub(r"^(\s*port:\s*)8900\s*$", lambda m: m.group(1) + port, resolved, flags=re.M)
(here / work / "serve.resolved.yaml").write_text(resolved)
PY
python3 -m reef serve -c "$PWD/$WORK/serve.resolved.yaml" > "$WORK/reef.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT  # SIGTERM: reef stops its services

while ! curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" > /dev/null; do
    kill -0 "$SERVE_PID" 2>/dev/null || { cat "$WORK/reef.log" >&2; exit 1; }
    sleep 1
done

# 4. The loop.
python3 run.py "$@"
