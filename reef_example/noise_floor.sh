#!/bin/bash
# 噪声地板:同一棵树(种子 skill)重复评估 N 次,量分数抖动。
#
# 为什么要单独跑:闸门判胜负靠的是"候选比现版高多少",而这个差只有在**大于同配置
# 重掷的抖动**时才有信息量。2026-09-03 第一版里这个数是白捡的 —— 三步闸门全部拒绝、
# 零发布,于是每步的 current 树都是同一棵,那些 current 分数天然就是重掷样本。修好
# 观测与刻度之后闸门开始发布,树会变,重掷样本就没了,必须显式跑。
#
# 做法:`run.sh --limit 1` 只记录第一个任务 → 只触发一个 evolve step → 该 step 的
# current_scores 就是种子树在**全部三个任务**上的一次 episode 评估。重复 N 次,每次
# 之前清状态(agent-record 不清会 409),只取 step 1 的 current_scores。
#
# 用法: ./noise_floor.sh [N]   (默认 4);结果落 work/noise_floor.jsonl
set -e
cd "$(dirname "$0")"
N="${1:-4}"
OUT="$PWD/work/noise_floor.jsonl"
: > "$OUT"

for i in $(seq 1 "$N"); do
    echo "=== 重掷 $i/$N ==="
    rm -rf work/agent-record work/artifacts.git work/artifact-work work/artifact-cache work/stack
    # 缓存留着:工具确定性,缓存命中不改分数,只省时间。
    PULL_TIMEOUT_S=2400 ./run.sh --limit 1 > "work/noise_run_$i.log" 2>&1 || true
    ../.venv/bin/python - "$i" "$OUT" <<'PY'
import glob, json, sys
run, out = sys.argv[1], sys.argv[2]
files = glob.glob("work/agent-record/*.commits.jsonl")
scores = None
if files:
    lines = open(files[0]).read().splitlines()
    if lines:
        first = json.loads(lines[0])
        scores = first["metrics"]["selection"]["evaluation"]["metrics"].get("current_scores")
with open(out, "a") as f:
    f.write(json.dumps({"run": int(run), "current_scores": scores}) + "\n")
print(f"  重掷 {run}: current_scores = {scores}")
PY
done

echo
echo "=== 汇总 ==="
../.venv/bin/python - "$OUT" <<'PY'
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r["current_scores"]]
if not rows:
    sys.exit("没有拿到任何 current_scores,查 work/noise_run_*.log")
ids = list(json.load(open("work/task_map.json")))
names = json.load(open("work/task_map.json"))
ceil = json.load(open("../work/oracle_v1_d4.json"))["tasks"]
print(f"{'任务':22s}" + "".join(f"{'掷'+str(r['run']):>9s}" for r in rows) + f"{'极差':>9s}{'天花板':>9s}{'极差/天花板':>12s}")
for i, pid in enumerate(ids):
    vals = [r["current_scores"][i] for r in rows]
    rng = max(vals) - min(vals)
    c = ceil[names[pid]]["best_score"]
    print(f"{names[pid]:22s}" + "".join(f"{v:9.4f}" for v in vals) + f"{rng:9.4f}{c:9.4f}{rng/c*100:11.0f}%")
PY
