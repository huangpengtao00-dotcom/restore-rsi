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
[ "${KEEP:-}" = "1" ] || : > "$OUT"

# 一次失败不能静默变成"少一个样本":重掷失败就重试,连续失败到上限才记为缺失。
# (第一版没有重试,一次上游瞬时 503 就丢了一个样本,而且只在汇总时才发现。)
ATTEMPTS=3
for i in $(seq 1 "$N"); do
    for attempt in $(seq 1 "$ATTEMPTS"); do
        echo "=== 重掷 $i/$N (第 $attempt 次尝试) ==="
        rm -rf work/agent-record work/artifacts.git work/artifact-work work/artifact-cache work/stack
        # 缓存留着:工具确定性,缓存命中不改分数,只省时间。
        RESTORE_MAX_SCORE=1.0 PULL_TIMEOUT_S=2400 ./run.sh --limit 1 > "work/noise_run_${i}_${attempt}.log" 2>&1 || true
        if ../.venv/bin/python - "$i" "$attempt" "$OUT" <<'PY'
import glob, json, sys
run, attempt, out = sys.argv[1], sys.argv[2], sys.argv[3]
scores = None
for path in glob.glob("work/agent-record/*.commits.jsonl"):
    lines = open(path).read().splitlines()
    if lines:
        scores = json.loads(lines[0])["metrics"]["selection"]["evaluation"]["metrics"].get("current_scores")
        break
if scores is None:
    print(f"  重掷 {run} 第 {attempt} 次:没拿到 current_scores(见 work/noise_run_{run}_{attempt}.log)")
    sys.exit(1)
with open(out, "a") as f:
    f.write(json.dumps({"run": int(run), "attempt": int(attempt), "current_scores": scores}) + "\n")
print(f"  重掷 {run}: current_scores = {scores}")
PY
        then
            break
        fi
        [ "$attempt" = "$ATTEMPTS" ] && echo "  ！重掷 $i 连续失败 $ATTEMPTS 次,记为缺失"
    done
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
