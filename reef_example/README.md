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
任务 prompt 把参考图路径给了 agent,agent 在 episode 内反复调 `restore score --ref` 对着 GT 爬山(单 episode 几十次打分,分数 0 → 0.42)。这意味着 gate 分数衡量的是"agent 会不会用答案刷分",不是"会不会复原"。已改为盲评(见下),泄题版数据保留为对照。

### 2026-09-03 · 盲评协议 + 泄题对照
改动:episode 只有 `restore diagnose` 与 `restore run`,prompt 不含参考路径也没有 score 命令;**参考图与打分移到 judge**——`evaluate` 从 trace 里取该 episode 最后一次成功 `restore run` 的 output_sha,在内容寻址缓存里找到那张图,自己跑 `restore score --ref`。

同模型、同工具、同任务,只改"评估器对策略是否可见":

| 三题分数 | 泄题版 | 盲评版 |
|---|---|---|
| 现版 | 0.425 / 0.137 / 0.296(合 0.858) | **0.237 / 0.050 / 0.107(合 0.394)** |
| 候选 | 0.290 / 0.133 / 0.000(合 0.423) | 0.199 / 0.000 / 0.024(合 0.222) |
| 门 | 0 胜 3 负 reject | 0 胜 3 负 reject |

**约一半的分数是刷出来的**(现版 0.858 → 0.394)。两轮门都正确拒绝了候选。
模型提议的 skill 文本(`work/results/proposals.jsonl`)读起来很合理——分严重度档、"噪声高时先降噪再提亮"、禁止自造分数——但盲评三题全线更低:合理的经验 ≠ 有用的经验,只有独立的门能分辨。

证据:`work/gate_blind.json`、`work/results/proposals.jsonl`、`work/agent-record/*.commits.jsonl`。

### 2026-09-03 · 天花板(穷举 oracle):泄题版的分数根本达不到
分数只有对着上界才读得懂。工具是确定性的、只有 6 个,所以最优链可以直接穷举——`tasks/oracle_chains.py` 深度优先走遍所有长度 ≤ N 的序列,用**和闸门同一个打分公式**(原样抄自 `cli.py`,不自己重推)算每个前缀。

| 任务 | identity | ≤3 | ≤4 | ≤5 | 盲评现版 | 泄题现版 |
|---|---|---|---|---|---|---|
| haze_low_light_00 | 0.000 | 0.3384 | 0.3588 | **0.3696** | 0.2374 | **0.425** |
| haze_low_light_01 | 0.000 | 0.1593 | **0.1775** | — | 0.0501 | 0.137 |
| low_light_noise_00 | 0.000 | 0.0881 | 0.1499 | **0.1672** | 0.1069 | **0.296** |

深度 4→5 只涨 0.01–0.02,说明 ~0.37 / ~0.18 / ~0.17 就是这套工具的实际上界。

于是:**泄题版的 0.425 和 0.296 高于穷举天花板**——那个数字不可能是"复原"出来的。盲评版三题分别是天花板的 64% / 28% / 64%,在合理区间内。

机制侧的直接验证(`restore score IN OUT --ref REF` 打的是你交给它的 OUT):

```
honest: 交自己的输入   -> REEF_SCORE=0.0000
gamed:  交参考图本身   -> REEF_SCORE=1.0000
```

接口本身可被平凡地刷满。agent 当时只刷到 0.425 而不是 1.0,具体走的哪条路**无法复盘了**——那一轮的 trace(1653 条)在重置时被删掉了,这是我的流程错误。已改:`run.sh` 每轮开跑前把上一轮 trace 轮转成 `trace.<时间戳>.jsonl`,关键证据固化进受版本管理的 `evidence/`。

### 坑 5:report 引用超过一个 record → 整批被静默丢弃
中继版第一次跑,三个任务全部低于 0.5 却一个 evolve 都没触发,`/reef/status` 里 `batch_ready: false`、`scenario_step: 0`,日志里**没有任何错误**。

根因在 reef 侧:`reef/train/cordis_backend/processor.py:48`

```python
if len(context.references) != 1 or not self._min_score <= score <= self._max_score:
    return NEVER
```

多轮中继每轮各有一个 receipt,我把 6 个全传了进去 → `!= 1` → 直接判 NEVER,记录释放,静默。

改法:`references=receipts[:1]`,而且**必须是第 1 轮的**——`propose` 从被引用 payload 的 `messages[-1]` 取任务 prompt,只有第 1 轮的 payload 最后一条是干净的 `[task_id] ...`;换成后面任何一轮,取到的是工具观测文本,task id 会退化成 `?`。

### 2026-09-03 · 记录段改为多轮中继(已修)
前一轮记录段三题全 0,原因不是策略差:DeepSeek-V4-Flash 把命令包在自家 tool-call 标记里(`<||DSML||parameter name="command" ...>cd ... && restore diagnose ...`),行锚定的 `RUN_LINE` 一条都匹配不到,于是判成"no plan"。

改法两处:
- `COMMAND` 正则读穿包装(纯文本行 / markdown 围栏 / 模型自有标记都命中),遇换行、`<`、反引号或 shell 分隔符停;
- 记录段从"一次性猜整条链"改成**真中继**:模型说要执行什么命令,run.py 真跑,把真实 stdout 喂回去当下一轮,最多 `RECORD_MAX_TURNS`(默认 6)轮。模型给的路径一律忽略,链条从任务输入起、每步输出接下一步——顺带堵掉"让 `restore` 指向自选文件"。

注册表里没有的工具名(比如模型照抄 prompt 里的 `restore run TOOL IN OUT`)回一句 unknown tool 而不是真去执行。

副作用一并处理:`propose` 的提示词原来教技能"结尾必须跑 `restore score`",与盲评协议矛盾(第一轮拒掉的那条 skill 里就带着这句),已改;`_shortcut_taken` 取消了 `restore score --ref` 的豁免——盲评下 episode 没有任何理由碰参考图。
