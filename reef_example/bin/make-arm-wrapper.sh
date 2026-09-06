#!/bin/bash
# 为一个消融臂生成它专属的 `restore` wrapper。
#
#   make-arm-wrapper.sh <reef_example 绝对路径> <臂目录,相对 reef_example>
#
# 为什么需要它:reef 把 episode 的环境剥到只剩 PATH/TMPDIR(见 bin/restore 的注释),
# 所以 RESTORE_RESULTS_DIR 这些**传不进 pi episode**,wrapper 里的 `${VAR:-default}`
# 总是取到默认值 `work/`。并行跑多个臂时那个默认值是错的。
#
# 2026-09-06 实测代价:baseline 跑在 work/baseline 下,pi 的 376 条 trace 全写进了
# 共用的 work/,judge 在本臂目录里找不到任何成功的 `restore run`,把每个 episode
# 都判 0 分。48 分钟跑完,candidate_score 和 current_score 都是 0.0,日志一切正常。
#
# 传不进去的东西就别指望环境变量 —— 把路径烧进一份只属于这个臂的 wrapper。
set -e
HERE="$1"
WORK="$2"
[ -n "$HERE" ] && [ -n "$WORK" ] || { echo "usage: $0 <reef_example_abs> <arm_work_dir>" >&2; exit 2; }
mkdir -p "$HERE/$WORK/bin"
cat > "$HERE/$WORK/bin/restore" <<WRAPPER
#!/bin/bash
# 由 make-arm-wrapper.sh 生成 —— 只属于 $WORK 这个臂,不要手改。
export RESTORE_RESULTS_DIR="$HERE/$WORK/results"
export RESTORE_CACHE_DIR="$HERE/$WORK/cache"
export RESTORE_TRACE="$HERE/$WORK/results/trace.jsonl"
exec "$HERE/bin/restore" "\$@"
WRAPPER
chmod +x "$HERE/$WORK/bin/restore"
echo "$HERE/$WORK/bin/restore"
