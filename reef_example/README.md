# restore-rsi as a reef `harness_evolve` example

The restoration toolbox (`restore diagnose / run / score`) wired into reef's
harness-evolution loop, mirroring `reef/tutorials/harness_evolve/`: the agent
harness is a composition tree (here: one `restore-strategy` skill), the served
model proposes one skill mutation over its own failing tasks, real headless pi
episodes score current vs candidate, a win publishes a versioned tree that
`GET /reef/harness` returns. Zero GPU by default: the six enabled tools are classical
(DCP dehaze, gamma/CLAHE, bilateral/median, unsharp) and the model is a remote
text LLM. JarvisIR's twelve expert models sit in the same registry behind
`enabled: false`; turning them on needs a container image and a card, and
changes nothing on the loop side - the docker backend has the same one-in
one-out contract as builtin. `restore doctor` says which of the two a tool is
in right now.

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

`run.sh` needs an upstream endpoint and a key, from the environment
(`REEF_UPSTREAM_URL` / `REEF_UPSTREAM_API_KEY`) or from `~/.dsh/.env`
(`RESTORE_UPSTREAM_URL=` / `RESTORE_UPSTREAM_KEY=`). Any OpenAI-compatible
chat-completions endpoint works - give the base URL without `/v1`, reef appends
it. The model name lives in `serve.yaml` (`litellm/DeepSeek-V4-Flash` here);
neither the endpoint nor the key is ever written into the repo.

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
| `REEF_UPSTREAM_URL` | env, or run.sh from `~/.dsh/.env` `RESTORE_UPSTREAM_URL` | interpolated into serve.yaml `reef.upstream_url` |
| `REEF_UPSTREAM_API_KEY` | env, or run.sh from `~/.dsh/.env` `RESTORE_UPSTREAM_KEY` | interpolated into serve.yaml `reef.upstream_api_key` |
| `HTTP(S)_PROXY / ALL_PROXY / *_proxy` | run.sh **unsets** | a local proxy hijacks the upstream calls; reef's urllib/aiohttp honour them |
| `NO_PROXY` | run.sh | the upstream host (derived from `REEF_UPSTREAM_URL`), `127.0.0.1`, `localhost` |
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

坑与教训统一记在下面的「已知坑」一节(中文,11 条)。另外两条纯环境的:
- **DeepSeek-V4-Flash 的推理与正文共用 `max_tokens`**;proposer 不设上限,单次允许 240 s。
- **`pi` 版本**:descriptor 只为安装脚本钉 0.84.2;本地 0.84.3 配 `PI_SKIP_VERSION_CHECK=1`(reef 会设)可正常跑。

## 实测结论(2026-09-04 定稿)

修好观测、奖励刻度、题干泄题、episode id 之后重测。**闸门仍不具判别力,但原因查清了。**

### 噪声地板:同一棵种子树重复评估 6 次

| 任务 | 极差 | 天花板(≤4 步) | 噪声占可达区间 | 闸门实际判胜负的差 |
|---|---|---|---|---|
| haze_low_light_00 | 0.5722 | 0.5994 | **95%** | 0.016 – 0.156 |
| haze_low_light_01 | 0.1455 | 0.3863 | 38% | 0.000 – 0.217 |
| low_light_noise_00 | 0.6462 | 0.6268 | **103%** | 0.044 – 0.620 |

效应比噪声小 2–50 倍,三题都无判别力。证据 `evidence/2026-09-04-noise-floor-final/`。

### 机制:不是抖动,是双峰

每个 0.0 都从内容寻址缓存里把图捞回来重新打分验证过:

```
dehaze_dcp > lowlight_gamma                    ->  0.2165 / 0.4158
dehaze_dcp > lowlight_gamma > lowlight_clahe   ->  0.0000
```

多走一步就归零,两个不同任务上同一条链。所以分布不是围绕均值的抖动,是"干净的两步链"与"多走一步毁掉"的二选一。

### 根因:盲评拿掉了 agent 的自纠错能力

盲评堵住了泄题(实测:18 个 episode 零 score 事件),但 agent 因此**无法知道自己上一步把图弄坏了**。它只有诊断读数,而诊断测的是"还剩多少退化",不是"离原图多远"——过度处理时读数继续下降、图却在变差。

**开放问题:盲策略靠什么信号决定何时停手。** 这是无参考指标可靠性问题,不是工程细节。

### 修复本身是有效的(行为侧验证)

任务名已改成不可读哈希,agent 唯一能读的就是诊断,所以下面的变化只能归因于修好的观测:

| 工具 | 修复前 | 修复后 | 真实情况 |
|---|---|---|---|
| `sharpen_unsharp` | 23.7% | **5.8%** | 三题的任何最优链都不含它 |
| `dehaze_dcp` | 9.2% | **17.3%** | 含雾题最优链第一步 |
| `lowlight_gamma` | 20.8% | **40.4%** | 最优链主力 |

循环也首次发布了经验,内容与穷举最优链一致("暗+雾先提亮再去雾")。中继最好一次跑出该顺序,得 0.5021(天花板 0.5994)。

### 诊断 v1 vs v0

| | v0 | v1 |
|---|---|---|
| 三族能否分开 | 雾**重叠** | 全分开,间隔 0.232 / 0.290 / 0.528 |
| blur 在 12 题(无一含模糊) | 6 题读 0.94–0.96 | 全读 0.000 |
| blur 在干净参考图上 | **0.912** | 0.000 |
| blur 能否测出真模糊 | 测不出 | r=1 → 0.257,r=3 → 0.960 |
| 掩盖(同图同雾强度加暗) | 掉 91% | 掉 22% |

### 奖励刻度 v1 vs v0

下限 15 dB 高于 12 题里 8 题的起点(三题:13.19 / 12.76 / 8.59 dB),39% 分数被钉死 0.0。改成相对退化输入的改善量后:

| | 旧 | 新 |
|---|---|---|
| 三题天花板 | 0.36 / 0.18 / 0.15 | 0.60 / 0.39 / 0.63 |
| 单工具非零个数 | 0–1 / 6 | 3–4 / 6 |

逐题只是减一个常数,不改胜负,只把被地板压平的那段还原。

## 停手判据:核心问题,已量化

盲评堵住了泄题,同时也让 agent 无法知道上一步把图弄坏了。这个问题可以判定:动作
空间可枚举、任务有真值,所以能穷举所有链的每一步,把「agent 看得见的观测」和
「真实分数变没变好」配对。`tasks/stopping_signal.py` 采数据,`stopping_analysis.py` 出结论。

### 每个盲信号能预测「下一步会不会变好」吗

3096 个决策点(12 题 × ≤3 步),其中 36.0% 是「下一步真的变好」。

| 信号(全部不含参考图) | AUC |
|---|---|
| **试走一步后诊断总分的下降量** | **0.789** |
| 试走后的诊断总分 | 0.650 |
| 暗光读数下降量 | 0.630 |
| 已偏离输入多远(PSNR vs 输入) | 0.608 |
| 噪声读数下降量 | 0.604 |
| 模糊读数下降量 / 链长 / 当前诊断总分 / 雾读数下降量 | 0.43–0.49(≈抛硬币) |

被发布的那条 skill 教的规则是「读数降了就说明这步对」。实测它的精确率:

| 阈值 τ(降幅 > τ 才走) | 说该走的次数 | 其中真变好 | 精确率 | 召回 |
|---|---|---|---|---|
| 0.00(当前规则) | 1703 | 974 | **57.2%** | 87.4% |
| 0.10 | 893 | 583 | 65.3% | 52.3% |
| 0.30 | 385 | 276 | **71.7%** | 24.8% |

**换阈值救不了**:精确率最高 72%,而且召回掉到 25%。这个观测量本身不够。

### 但 argmax 比逐个判断容易得多

逐步判断「这一步好不好」只有 57% 精确率,可**从 6 个工具里挑降幅最大的那个**是个
容易得多的问题。确定性贪心策略(每步试走全部 6 个工具,选诊断总分降幅最大的,降不动
就停)在 12 题上:

| 任务 | 贪心停在哪 | 同一轨迹上的最高分 | 白丢 |
|---|---|---|---|
| low_light_00 | 0.4096 | **1.0000** | 0.59 |
| low_light_01 | 0.6730 | **0.9650** | 0.29 |
| noise_01 | **0.0568** | **0.8276** | **0.77** |
| 其余 9 题 | — | 与贪心一致 | 0 |
| **合计** | **5.6092** | **7.2625** | **1.65** |

**「不知道该在哪停手」值 23% 的可达分数。** 而且过头是真的会发生:`noise_01` 走过
0.83,最后停在 0.057。

### 这条基线必须先跨过

同样只用 agent 看得见的信息,贪心在三个评测任务上拿到 **0.607 / 0.267 / 0.649**,
**比 LLM agent 的典型表现好,而且从不归零**(agent 的 episode 分数在 0.0–0.57 之间跳)。

**也就是说:自进化循环现在想学的东西,一条十行的确定性规则已经做得更好。** 任何
「循环学会了什么」的说法都要先跨过这条线。`tests/test_stopping_baseline.py` 把它钉住,
挪动了就说明对照失效。

证据 `evidence/2026-09-04-stopping-signal/`。

## 上界工具的正确用法

`tasks/oracle_chains.py` 穷举所有 ≤N 步工具链,用**闸门自己的打分公式**(原样抄自 `cli.py`)。它给的是**某个步数预算下**的上界,不是工具箱的全局上界——上界随链长持续上涨:

| 链长 | haze_low_light_00 | low_light_noise_00 |
|---|---|---|
| ≤3(穷举) | 0.3384 | 0.0881 |
| ≤5(穷举 9331 条) | 0.3696 | 0.1672 |
| ≤6(穷举 55987 条) | **0.3799** | — |
| ≤15(beam 宽 24) | **0.3957** | **0.2313** |

beam 的准度可校准:深度 5 时 beam 给 0.3680、穷举 0.3696,只低估 0.0016,是紧下界。

**所以只能拿它卡同样受该步数预算限制的策略。** 中继版 agent 受 `RECORD_MAX_TURNS=6` 限制(一轮一条命令),可以用 ≤5 步上界;pi episode 有真 shell、调用次数不受限(实测跑到 23 步),不能用。

## 已知坑

1. **artifact 路径必须绝对**。reef 用 `git ls-remote <artifact_repository>` 探测已注册 scenario;git 对相对路径会先 chdir 到*外层* git 仓顶层再解析——而本目录就在 restore-rsi 仓里,于是 `work/artifacts.git` 永远"不是 git 仓",服务起不来。`run.sh` 现在把 serve.yaml 里五个 `work/` 路径解析成绝对路径写进 `work/serve.resolved.yaml` 再启动。

2. **evolve 期间服务不响应 HTTP**。reef 同步跑 evaluate 的 pi episode,期间 `/reef/harness`、`/reef/status` 都会 socket 超时。`run.py` 的轮询原本只接 `ReefClientError`(404),裸 `TimeoutError` 直接把 run.py 崩掉;`run.sh` 的 `trap ... EXIT` 随即杀掉 reef,正在跑的 episode 一起死——表现为"记录成功、evolve 永远等不到"。现已把超时当"还在进化"处理。

3. **重跑前必须清 `work/agent-record`**,否则同名 record 409 Conflict。

4. **本机代理会劫持上游**。不 unset 的话每个调用都失败或挂起,第一症状是所有 episode 得 0、所有闸门打平。

5. **report 引用超过一个 record → 整批被静默丢弃**。reef `train/cordis_backend/processor.py:48`:

   ```python
   if len(context.references) != 1 or not self._min_score <= score <= self._max_score:
       return NEVER
   ```

   多轮中继每轮各有一个 receipt,全传进去就 `!= 1` → 判 NEVER、记录释放、**日志零报错**。三个任务全部低于阈值却一个 evolve 都没触发,`batch_ready: false`、`scenario_step: 0`,表现得完全像"模型没学到东西"。
   改法:`references=receipts[:1]`,而且**必须是第 1 轮的**——`propose` 从被引用 payload 的 `messages[-1]` 取任务 prompt,只有第 1 轮那条是干净的 `[task_id] ...`。

6. **模型不按纯文本回命令**。DeepSeek-V4-Flash 把命令包在自家 tool-call 标记里(`<||DSML||parameter name="command" ...>cd ... && restore diagnose ...`),行锚定的正则一条都匹配不到 → 判成 "no plan",三题全 0,看起来像策略问题。
   但**无锚点匹配又会执行模型的散文**:"I should **NOT** restore run sharpen_unsharp" 会真去跑它,"Plan: 1) ... 2) ..." 只吞第一条。现在要求命令处于**命令位**(行首 / 反引号 / `$` / `>` / `&&` 之后),11 个真实回复形态钉在 `tests/test_relay_commands.py`。

7. **奖励作弊出现过两次**。①首个手工 pi episode(2026-09-02)找到 `tasks/data/manifest.json`、读出精确退化参数、用 numpy 反演,通过官方打分器得 1.0。②参考图路径和 `restore score` 还在 prompt 里的时候,agent 合法地用它们对着 GT 爬山而不去复原。
   `evaluate` 会把碰 `clean_*.png` / `--ref` / `manifest.json` / 退化代码 / python·PIL·numpy·imagemagick 的 episode 记 0;但堵住**被授权**的那条路要靠盲评协议,shortcut 模式只管未授权的那条。
   接口本身可平凡刷满:`restore score IN <参考图> --ref <参考图>` → `REEF_SCORE=1.0000`。

8. **`0.07`→`0.35` 这类等长改动会命中 `.pyc` 缓存**。Python 按 (mtime, size) 判失效,等长替换 + 亚秒写入会复用旧字节码。变异测试第一版因此得出"改 severity 表全绿"的假结论。变异测试必须 `PYTHONDONTWRITEBYTECODE=1` + 清 `__pycache__` + `-p no:randomly`(`pytest-randomly` 会重播随机种子,同一变异体能跑出 7 或 10 个失败)。

9. **package shadowing**:仓库根曾有个空的 `harness/`,editable 安装下会遮蔽 `reef_example/harness`。已删,不要重建根级 `harness` 包。

10. **`pi` 版本**:descriptor 只为安装脚本钉 0.84.2;本地 0.84.3 配 `PI_SKIP_VERSION_CHECK=1`(reef 会设)可正常跑。

11. **trace 是唯一的运行记录,重置会丢**。泄题那轮 1653 条 trace 在重置时被删,机制无法复盘。`run.sh` 现在每轮开跑前把上一轮轮转成 `trace.<时间戳>.jsonl`,关键证据固化进受版本管理的 `evidence/`。

以上都没有改动 reef 源码。


12. **实验不能依赖"任务会失败"**。`noise_floor.sh --limit 1` 靠"记录的任务分数 ≤ max_score"来触发 evolve step;有一轮记录段跑出 0.5021(高于阈值 0.5),于是不进批、不触发、无样本,**且没有任何报错**,重试逻辑还一遍遍撞同一面墙。现在测量时用 `RESTORE_MAX_SCORE=1.0` 覆盖阈值。

13. **episode id 必须一份、且与 cwd 无关**。wrapper 原来从 cwd 的父目录取 id,agent 一 `cd` 就取不到,兜底又是 `adhoc-$(date +%s)-$$` —— 同一 episode 每次调用一个新 id,trace 打散,判分方按它看到的 id 找不到成功的 run,把能跑的 episode 记成 0.0(实测 3 次)。现在:cwd 里任意位置的 `reef-episode-<id>` → pi 自己的 `PI_CODING_AGENT_DIR` → 由 cwd 派生的稳定兜底;并且 wrapper 每次调用都 echo 标记,`_episode_id` 收集全部标记、不一致就报错。

## 核对通过的项

| 项 | 做法 | 结果 |
|---|---|---|
| 判据一致性 | oracle 最优链**用真 `restore` CLI 重跑再打分** | 与 oracle 差 0.000000 |
| 盲评协议成立 | 数 trace 里 episode 侧的 score 事件 | 18 个 episode 只有 run(207)+ diagnose(197),**score 事件 0** |
| 噪声地板前提 | 比对三步 commit 的 `algorithm_state.entries` 哈希 | 三步逐字相同 |
| 判分对象来源 | 用 trace 的 `input_sha`/`output_sha` 逐 episode 连链回任务输入 | 18/18 可追溯,无断链 |
| 工具确定性 | 6 个算子各跑 3 次比输出哈希 | 全一致 |

## 未修清单

| # | 问题 | 修法 |
|---|---|---|
| 1 | `diagnose` 在复合退化上失效、`blur` 系统性误报(根因 A) | 读数解耦(先估光照,再在归一化后的图上估雾);无模糊族时 `blur` 恒为 0 |
| 2 | 打分刻度下限高于任务起点,39% 分数被截断(根因 B) | 下限按每题退化图 PSNR 设,即改成"相对起点的改善量" |
| 3 | 任务 id 与输入路径把退化组合泄漏进 prompt | 任务 id 改成不可读的哈希 |
| 4 | `data.max_score: 0.5` 是空条件(任何步数预算下都到不了) | 随 #2 一起解决 |
| 5 | `propose` 只看得到第 1 轮 payload,中继轨迹进不去 | 轨迹摘要写进 report 的 feedback 字段 |
| 6 | 中继强制线性链、判分取最后一次成功 `run`(agent 不能退回上一张),系统性压低分数 | 允许声明最终图,用 trace 哈希校验其来源(数据已足够,judge 尚未做) |

**#1 和 #2 是口径变更**:一改,今天所有数字都不可比,所以先不动。

## 证据

| 内容 | 路径 |
|---|---|
| 三步闸门全 metrics、3 条 skill 提议原文、443 条事件 trace、噪声地板 | `evidence/2026-09-03-blind-3task/` |
| 自查数字(诊断 vs 真值 / 刻度截断 / 工具用量 / 闸门截断计数) | `evidence/2026-09-03-audit.json` |
| 各步数预算下的上界 | `evidence/oracle_d3.json` / `d4` / `d5` / `d6` |
| 泄题那轮的对照数据 | `evidence/2026-09-03-leak-vs-blind/` |
