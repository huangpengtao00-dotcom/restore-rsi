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
                   evaluate: the judge scores the episode's final image itself -
                   last successful `restore run` output_sha from the trace, pulled
                   out of the content-addressed cache; nothing the episode says is
                   trusted. Shortcuts (reference/manifest/own numpy) -> 0.0
  tasks/oracle_chains.py (repo root)
                   exhaustive best-chain search: the ceiling a score is read against
  run.py           relay (model names a command -> it runs -> stdout back) -> score -> report -> pull
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

1. `materialize.py` builds one prompt per task id: `[task_id]`, the input path,
   the two commands the episode has (`restore diagnose`, `restore run`), the
   tool catalog from `toolbox/registry.yaml`, and the reply format (the final
   image's path alone on the last line). **Blind by construction**: no reference
   path, no score command - a judge holds both (see 参考答案泄露, below).
2. `run.py` relays each prompt through reef inference: the model names one
   command, `run.py` runs it through `bin/restore` and feeds the real stdout
   back as the next turn, up to `RECORD_MAX_TURNS` (6). The chain starts at the
   task input and each output feeds the next, so the paths the model invents do
   not matter. When the relay ends, `run.py` scores the final image with
   `restore score --ref` **on its own side** and reports that against turn 1's
   receipt (exactly one reference - see 坑 5). Reports with score <=
   `data.max_score` (0.5) batch; `batch_size: 1` makes every failing report one
   gated evolve step.
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
- **Reward hacking is real, twice over**: the first manual pi episode
  (2026-09-02) found `tasks/data/manifest.json`, read the exact degradation
  parameters, inverted them in numpy and scored 1.0 through the official
  scorer. `evaluate` zeroes any episode whose tool calls touch `clean_*.png`,
  `--ref`, `manifest.json`, the degradation code, or python/PIL/numpy/
  imagemagick. That was not enough: while the reference path and `restore
  score` were still *in the prompt*, the agent used them legitimately and hill-
  climbed on ground truth instead of restoring. Blocking the sanctioned path is
  what the blind protocol does; the shortcut patterns only catch the unsanctioned
  one.
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

当时记的结论是"约一半的分数是刷出来的"(0.858 → 0.394)。**后来量了噪声地板,这个结论撤回**:盲评那一列同配置重掷的极差就有 0.05–0.24(见上面的噪声地板一节),单看一对没有推断力。这张表保留为背景;站得住的是天花板越界那条。

模型提议的 skill 文本(`work/results/proposals.jsonl`)读起来很合理——分严重度档、"噪声高时先降噪再提亮"、禁止自造分数——盲评三题全线更低。同理,这只能当现象记,不能当"合理≠有用"的证据:效应比抖动小。

证据:`work/gate_blind.json`、`work/results/proposals.jsonl`、`work/agent-record/*.commits.jsonl`。

### 2026-09-03 · 噪声地板:同配置重掷的抖动 ≈ 闸门在判的差
中继版重跑一整轮时,因为前两轮闸门都拒绝、没有发布,**这一轮的 current 树和上一轮的 current 树是同一棵(种子 skill)**——于是两轮的 current 分数就是同一配置的两次重掷。

中继版重跑一整轮时,因为**每一步闸门都拒绝、什么都没发布**,每个 evolve step 里的 current 树始终是同一棵种子树——于是那些 current 分数就是同一配置的反复重掷,噪声地板不用另设实验就出来了(4 次:盲评首轮 + 3 个 evolve step):

| 任务 | 掷1 | 掷2 | 掷3 | 掷4 | 极差 | ≤5 步上界 | 极差/上界 |
|---|---|---|---|---|---|---|---|
| haze_low_light_00 | 0.2374 | 0.1985 | 0.0000 | 0.1034 | **0.2374** | 0.3696 | 64% |
| haze_low_light_01 | 0.0501 | 0.0133 | 0.0000 | 0.0123 | **0.0501** | 0.1775 | 28% |
| low_light_noise_00 | 0.1069 | 0.0000 | 0.0558 | 0.1106 | **0.1106** | 0.1672 | 66% |

(最后一列拿 ≤5 步上界当分母是**为了给极差一个尺度**,不是在声称那是全局上界——见天花板那节的撤回。核心结论"极差 0.24 vs 效应 0.051"不依赖这个分母。)

前提已核对:三步的 current 组合树 `entries` 哈希逐字相同(`8fef307384e2620e`),且都是空的种子 skill。**注意由此带来的局限:这个噪声地板是在"没有任何策略指导"的 agent 上量的**,有 skill 之后抖动可能不同。

三步闸门:

| step | current | candidate | wins/losses/ties | outcome |
|---|---|---|---|---|
| 1 | 0.1985 / 0.0133 / 0.0 = 0.2118 | 0.1435 / 0.0709 / 0.0 = **0.2144** | 1 / 1 / 1 | reject |
| 2 | 0.0 / 0.0 / 0.0558 = 0.0558 | 0.0 / 0.0459 / 0.0 = 0.0459 | 1 / 1 / 1 | reject |
| 3 | 0.1034 / 0.0123 / 0.1106 = 0.2263 | 0.1366 / 0.0 / 0.0146 = 0.1512 | 1 / 2 / 0 | reject |

候选-现版之差非零项**平均 0.051、最大 0.096**,对上面 0.05–0.24 的抖动——**信号比噪声小一个身位**。step 1 的候选总分还更高(`score_comparison` 按逐题胜负算,不按总分)。

结论(方法学,不限于这个玩具):**闸门在量噪声地板之前不具备判别力**。三次 reject 结果大概是对的,但理由和抛硬币没区别。定阈值的做法很直接——同配置重掷 N 次取抖动上界,候选只有超过它才算赢;或把多次重掷的均值当分数。
每题 4 个样本,量级估计而非方差估计;但 0.05 效应 vs 0.24 抖动的量级差,4 个样本已足以下这个判断。

证据:`evidence/2026-09-03-blind-3task/`(commits / proposals / trace / noise-floor.json / console.log)。

### 2026-09-03 · 天花板(穷举 oracle):泄题版的分数根本达不到
分数只有对着上界才读得懂。工具是确定性的、只有 6 个,所以最优链可以直接穷举——`tasks/oracle_chains.py` 深度优先走遍所有长度 ≤ N 的序列,用**和闸门同一个打分公式**(原样抄自 `cli.py`,不自己重推)算每个前缀。

| 任务 | identity | ≤3 | ≤4 | ≤5 | 盲评现版 | 泄题现版 |
|---|---|---|---|---|---|---|
| haze_low_light_00 | 0.000 | 0.3384 | 0.3588 | **0.3696** | 0.2374 | **0.425** |
| haze_low_light_01 | 0.000 | 0.1593 | **0.1775** | — | 0.0501 | 0.137 |
| low_light_noise_00 | 0.000 | 0.0881 | 0.1499 | **0.1672** | 0.1069 | **0.296** |

深度 4→5 只涨 0.01–0.02,我据此写下过"~0.37 / ~0.18 / ~0.17 就是这套工具的实际上界"、并由此断定"泄题版的 0.425 / 0.296 越界,不可能是复原出来的"。

**这个结论已撤回(2026-09-03 复盘)。** beam 搜索(宽度 24)往深处走,上界没封顶:

| 链长 | haze_low_light_00 | low_light_noise_00 |
|---|---|---|
| ≤5(穷举 9331 条) | 0.3696 | 0.1672 |
| ≤6(穷举 55987 条) | **0.3799** | — |
| ≤8(beam) | 0.3820 | 0.1946 |
| ≤12(beam) | 0.3910 | 0.2206 |
| ≤15(beam) | 0.3957 | 0.2313 |

beam 有多准:深度 5 时 beam 给 0.3680,穷举 0.3696——**只低估 0.0016**,是个紧下界。所以"上界到 15 步还在爬"是可靠判断,泄题声称的 0.425 / 0.296 按这个斜率够得着。

病因是混了两个上界:**≤5 步的上界只约束中继版 agent**(`RECORD_MAX_TURNS=6`,一轮一条命令),而泄题版跑的是有真 shell、`restore run` 次数不受限的 pi episode。拿前者的尺子量后者是错的。

`oracle_chains.py` 仍然有用——它是"给定步数预算下的上界",拿它和**同样受步数限制的** agent 比是合法的;它不是工具箱的全局上界。

泄题那条现在只剩机制层面的证据(`restore score IN OUT --ref REF` 打的是你交给它的 OUT):

```
honest: 交自己的输入   -> REEF_SCORE=0.0000
gamed:  交参考图本身   -> REEF_SCORE=1.0000
```

接口本身可被平凡地刷满。agent 当时只刷到 0.425 而不是 1.0,具体走的哪条路**无法复盘了**——那一轮的 trace(1653 条)在重置时被删掉了,这是我的流程错误。已改:`run.sh` 每轮开跑前把上一轮 trace 轮转成 `trace.<时间戳>.jsonl`,关键证据固化进受版本管理的 `evidence/`。

所以泄题那条只支持一句**定性**的话:把打分器和参考答案交给策略,策略就会去用它(接口可刷满是实测的;"几十次打分、0→0.42"是当时的观察记录)。分数层面的量化结论一个都不剩。

### 2026-09-03 · 自查:已核对的 4 项 + 未修的 4 项

跑完之后专门做了一轮复盘,能查证的都去查了。

**核对通过**

| 项 | 做法 | 结果 |
|---|---|---|
| 判据一致性 | 把 oracle 找到的最优链**用真 `restore` CLI 重跑一遍**再打分 | 0.3696 / 0.1672,与 oracle 差 0.000000 |
| 盲评协议真的盲 | 数 `evidence/2026-09-03-blind-3task/trace.jsonl` 里 episode 侧的 score 事件 | 18 个 pi episode,只有 `run`(207)和 `diagnose`(197),**score 事件 0** |
| 噪声地板前提 | 比对三步 commit 的 `algorithm_state.entries` 哈希 | 三步全同(`8fef307384e2620e`),确实是同一配置重掷 |
| 泄题接口可刷满 | `restore score IN <参考图> --ref <参考图>` | `REEF_SCORE=1.0000` |

**已知未修(按影响排序)**

1. **`data.max_score: 0.5` 是空条件。** 三题在任何步数预算下都到不了 0.5,所以每份 report 必然被判为"失败"进 batch——失败窗口从来没起过筛选作用。想让它有意义,得把上报分数按该题的上界归一(逐题常数缩放不改逐题胜负,对闸门安全),或者干脆把阈值调到上界之下。**属设计改动,先不动。**
2. **`propose` 看不到 agent 到底干了什么。** 修批次丢弃时改成只引用第 1 轮 receipt,而 propose 从被引用 payload 的 `messages[-1]` 取内容——那就只有任务 prompt 和分数,**中继过程(诊断读数、工具选择、失败步骤)一条都进不去**。这限制了提议质量的上限;要修得把轨迹摘要塞进 report 的 feedback 字段。
3. **中继强制线性链,judge 取最后一次成功 `restore run`。** agent 没法"跑了一步觉得不好、退回上一张交上去"。prompt 里写明了所以不算欺骗,但这**系统性压低分数**——和第 1 条叠加,意味着现在的分数偏低。
4. **测试在随机顺序下的变异检出不稳定。** `pytest-randomly` 会重播随机种子,同一个变异体跑出 7 或 10 个失败都有。变异测试要加 `-p no:randomly` 才可复现(见下面那条 `.pyc` 教训)。

**局限(不是 bug,但结论解读时要记住)**:噪声地板是在**空种子 skill**(没有任何策略指导)上量的;3 个任务、每题 4 个样本;工具是 6 个 CPU 经典算子,不是真专家模型。

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
