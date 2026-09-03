# restore-rsi as a reef `harness_evolve` example

The restoration toolbox (`restore diagnose / run / score`) wired into reef's
harness-evolution loop, mirroring `reef/tutorials/harness_evolve/`: the agent
harness is a composition tree (here: one `restore-strategy` skill), the served
model proposes one skill mutation over its own failing tasks, real headless pi
episodes score current vs candidate, a win publishes a versioned tree that
`GET /reef/harness` returns. Zero GPU: the tools are classical (DCP dehaze,
gamma/CLAHE, bilateral/median, unsharp) and the model is a remote text LLM.

## Layout

```text
reef_example/
  serve.yaml       deployment (reef: section) + the harness_evolve recipe sections;
                   `restore.task_ids` picks the tasks, `evolution.tasks` is generated
  materialize.py   copies the recipe sections to work/recipes/ with the task prompts
                   built from ../tasks/data/manifest.json (+ work/tasks.json, task_refs.json)
  harness/evolution.py
                   propose: served model writes one SKILL.md over the failures
                   evaluate: REEF_SCORE parsed from the `restore score` tool result;
                   shortcut episodes (reference/manifest/own numpy) -> 0.0
  run.py           record -> execute plan + score -> report -> pull
  run.sh           env (key, no proxy, PATH), materialize, reef serve, run.py
  bin/restore      wrapper pi sees inside an episode: pins RESTORE_* to work/,
                   derives RESTORE_EPISODE_ID from the episode root name
  pyproject.toml   makes harness/ an installable method package
  work/            all state (gitignored): recipes/, tasks.json, reef.log, stack/,
                   agent-record/*.commits.jsonl, results/<episode>.verdict.json, ...
```

## Run

```bash
# once, repo root: reef checkout expected at ../reef (editable), pi on PATH
uv sync --group reef
uv run python tasks/make_tasks.py          # if tasks/data/ is missing

cd reef_example
./run.sh --limit 1      # 1 task through the whole chain (record -> evolve -> pull)
./run.sh                # the 3 configured tasks
```

`run.sh` needs `AIGW_KEY=` in `~/.dsh/.env` (or `REEF_UPSTREAM_API_KEY` already
exported). The upstream is `https://aigw.meshy.team` (no `/v1`: reef appends it)
with `litellm/DeepSeek-V4-Flash`; both live in `serve.yaml`, the key never does.

## What one run does

1. `materialize.py` builds one prompt per task id: `[task_id]`, input path,
   the `restore` commands, the tool catalog from `toolbox/registry.yaml`, the
   reference path for `--ref`, the reply format (`REEF_SCORE=...` alone on the
   last line) and the rules (only `restore run` touches images; never open the
   reference / manifest / degradation code).
2. `run.py` sends each prompt once through reef inference (plain chat, no
   shell). The model can only answer with the `restore run ...` chain it would
   issue; `run.py` executes that chain for real through `bin/restore`, scores
   it with `restore score --ref`, and reports the real `REEF_SCORE` against the
   receipt. Reports with score <= `data.max_score` (0.5) batch; `batch_size: 1`
   makes every failing report one gated evolve step.
3. Per step, reef calls `harness.evolution:propose` (the failing prompts +
   scores + current skills go to the served model, which returns one skill
   JSON), renders current and candidate trees into throwaway pi episodes (one
   per task each), runs them headless with the model bound to the upstream, and
   scores each with `harness.evolution:evaluate`. `ScoreComparisonSelector`
   publishes only if the candidate wins.
4. `run.py` polls `GET /reef/harness`, prints gate metrics and the evolved
   `SKILL.md`, and saves them under `work/evolved_skills/`.

## Environment variables

| variable | set by | purpose |
|---|---|---|
| `REEF_UPSTREAM_API_KEY` | run.sh from `~/.dsh/.env` `AIGW_KEY` | interpolated into serve.yaml `reef.upstream_api_key` |
| `HTTP(S)_PROXY / ALL_PROXY / *_proxy` | run.sh **unsets** | the local proxy (127.0.0.1:7897) hijacks aigw calls; reef's urllib/aiohttp honour them |
| `NO_PROXY` | run.sh | `aigw.meshy.team,127.0.0.1,localhost` |
| `PATH` | run.sh | `bin/` (restore wrapper) then `../.venv/bin` (python3 with reef, restore) |
| `REEF_RECIPE_CONFIG_DIR` | run.sh | `work/recipes` (where the registry finds `harness_evolve.yaml`) |
| `RESTORE_RESULTS_DIR / RESTORE_CACHE_DIR / RESTORE_TRACE` | run.sh (run.py's record pass) and `bin/restore` (episodes) | `work/results`, `work/cache`, `work/results/trace.jsonl` |
| `RESTORE_EPISODE_ID` | run.py: `record-<task>-<ts>`; `bin/restore`: `pi-XXXXXX` from the episode root | keys `work/results/<id>.verdict.json` / `.output.png` and trace lines |
| `REEF_SCENARIO`, `PULL_TIMEOUT_S` | optional, run.py | scenario lane (default `restore-evolve`), publish wait (default 3600 s) |

reef strips the episode environment to `PATH`/`TMPDIR` plus pi's own vars
(`reef/harness/episode.py:_inherited_env`), which is why the `RESTORE_*` knobs
travel through the `bin/restore` wrapper rather than through serve.yaml. This
also means episodes never see the proxy variables.

## Known traps

- **Local proxy**: without the unset, every call to aigw fails or hangs; the
  first symptom is all episodes scoring 0 and every gate tying.
- **Reward hacking is real**: the first manual pi episode (2026-09-02) found
  `tasks/data/manifest.json`, read the exact degradation parameters, inverted
  them in numpy and scored 1.0 through the official scorer. `evaluate` now
  zeroes any episode whose tool calls touch `clean_*.png`, `manifest.json`, the
  degradation code, or python/PIL/numpy/imagemagick. The score is also taken
  from the `restore score` *tool result*, never from the agent's claim alone.
- **Package shadowing**: the repo root used to hold an empty `harness/`; with
  the editable install it shadowed `reef_example/harness`. Removed; do not
  recreate a root-level `harness` package.
- **DeepSeek-V4-Flash reasoning** shares `max_tokens` with the reply; the
  proposer does not cap tokens and allows 240 s per call.
- **Root run scripts and `cat`**: the user shell aliases `cat` to `bat`; the
  scripts here only use `cat` for logs.
- **pi version**: descriptor pins 0.84.2 for the install script only; local
  0.84.3 runs episodes fine with `PI_SKIP_VERSION_CHECK=1` (set by reef).
- No reef source was patched.

## Results

See the "Run log" section below (filled from an actual run).

## 已知坑(2026-09-03 实测)

1. **artifact 路径必须绝对**。reef 用 `git ls-remote <artifact_repository>` 探测已注册 scenario;git 对相对路径会先 chdir 到*外层* git 仓的顶层再解析——而本目录就在 restore-rsi 仓里,于是 `work/artifacts.git` 永远"不是 git 仓",服务起不来。`run.sh` 现在把 serve.yaml 里五个 `work/` 路径解析成绝对路径写进 `work/serve.resolved.yaml` 再启动。
2. **evolve 期间服务不响应 HTTP**。reef 同步跑 evaluate 的 pi episode,期间 `/reef/harness`、`/reef/status` 都会 socket 超时。`run.py` 的轮询原本只接 `ReefClientError`(404),裸 `TimeoutError` 直接把 run.py 崩掉;`run.sh` 的 `trap ... EXIT` 随即杀掉 reef,正在跑的 episode 一起死——表现为"记录成功、evolve 永远等不到"。现已把超时当"还在进化"处理。
3. **重跑前清状态**。`agent_record_id` 按任务 id 生成,残留的 `work/agent-record` sqlite 会让第二次报告 409 冲突;`work/artifacts.git` 若在上次 SIGKILL 时没建完会是坏仓。重跑:`rm -rf work/agent-record work/artifact-* work/results work/artifacts.git && git init --bare work/artifacts.git`。
4. 机器别睡:episode 几分钟,睡眠会连服务一起掐(`caffeinate -i`)。

## 实测记录

### 2026-09-03 · 1 任务,完整 evolve 循环首次跑通
- 记录:`haze_low_light_00`,agent(DeepSeek-V4-Flash 经 pi)选链 `lowlight_gamma → dehaze_dcp → denoise_bilateral`,report score 0.0448(identity=0),低于 0.5 阈值 → 进 batch。
- propose:模型对 `restore-strategy` skill 提了一次 `update`(文本当时未落盘,已加 `proposals.jsonl` 记录)。
- evaluate:现版 vs 候选版各跑 3 个任务(6 个 pi episode,每个 4–7 分钟):
  - current_scores = [0.425, 0.137, 0.296],合计 0.858
  - candidate_scores = [0.290, 0.133, 0.000],合计 0.423
- 门:`score_comparison`,候选 0 胜 3 负 → **reject,未发布**(`GET /reef/harness` → "serves no files")。机制按设计工作:一条让第三个任务归零的经验被拦住了。
- 证据:`work/agent-record/*.commits.jsonl`、`work/results/trace.jsonl`(1653 条工具/打分事件)。

### 发现:episode 里存在参考答案泄露
任务 prompt 把参考图路径给了 agent,agent 在 episode 内反复调 `restore score --ref` 对着 GT 爬山(单 episode 几十次打分,分数 0 → 0.42)。这意味着 gate 分数衡量的是"agent 会不会用答案刷分",不是"会不会复原"。**下一版必须只给无参考诊断,参考图仅供 evaluate 使用**;本轮数据保留为"泄题对照"。
