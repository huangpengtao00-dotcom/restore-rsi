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

坑与教训统一记在下面的「已知坑」一节(中文,11 条)。另外两条纯环境的:
- **DeepSeek-V4-Flash 的推理与正文共用 `max_tokens`**;proposer 不设上限,单次允许 240 s。
- **`pi` 版本**:descriptor 只为安装脚本钉 0.84.2;本地 0.84.3 配 `PI_SKIP_VERSION_CHECK=1`(reef 会设)可正常跑。

## 实测结论(2026-09-03,三题盲评)

三个 evolve step,**全部 reject、零发布**。证据 `evidence/2026-09-03-blind-3task/`。

| step | current | candidate | wins/losses/ties | outcome |
|---|---|---|---|---|
| 1 | 0.1985 / 0.0133 / 0.0 = 0.2118 | 0.1435 / 0.0709 / 0.0 = **0.2144** | 1 / 1 / 1 | reject |
| 2 | 0.0 / 0.0 / 0.0558 = 0.0558 | 0.0 / 0.0459 / 0.0 = 0.0459 | 1 / 1 / 1 | reject |
| 3 | 0.1034 / 0.0123 / 0.1106 = 0.2263 | 0.1366 / 0.0 / 0.0146 = 0.1512 | 1 / 2 / 0 | reject |

step 1 拒掉的候选**总分更高**——`score_comparison` 按逐题胜负算,不按总分。

### 闸门不具备判别力

因为每步都 reject、零发布,每步的 current 树始终是同一棵种子树(entries 哈希逐字相同 `8fef307384e2620e`)——那些 current 分数就是同一配置的重掷:

| 任务 | 掷1 | 掷2 | 掷3 | 掷4 | 极差 |
|---|---|---|---|---|---|
| haze_low_light_00 | 0.2374 | 0.1985 | 0.0000 | 0.1034 | **0.2374** |
| haze_low_light_01 | 0.0501 | 0.0133 | 0.0000 | 0.0123 | **0.0501** |
| low_light_noise_00 | 0.1069 | 0.0000 | 0.0558 | 0.1106 | **0.1106** |

候选-现版之差非零项均值 **0.051**、最大 0.096 → **噪声是效应的三倍**。三步 reject 结果大概对,理由和抛硬币没区别。

排除了"episode 崩了"这个解释:18 个 episode 各跑 5–23 次工具,三步 `episode_failures = 0`。
每题 4 个样本,量级估计而非方差估计;且是在**空种子 skill**(零策略指导)上量的。

**方法学**:闸门在量噪声地板之前不具判别力。定阈值——同配置重掷 N 次取抖动上界,候选只有超过它才算赢;或用多次重掷的均值。量它几乎免费:闸门连续 reject 时的 current 分数天然就是重掷样本。

### 根因 A:`diagnose` 在复合退化上失效

**口径**:manifest 的 `gt.strengths` 是各退化器自己的混合系数(`t~U(0.4,0.9)`),**与读数不同尺度**,不能直接比大小。以下全部用尺度无关的检验。

| 检验 | 结果 |
|---|---|
| 能否判断该退化在不在场 | 雾:在场 [0.073, 0.755] vs 不在场 [0.000, 0.202] **重叠 → 判断不了**;暗、噪声区分开(噪声饱和在 1.000) |
| 能否给强度排序(同族 n=6) | Spearman ρ = **+0.086**(雾)/ **−0.200**(暗)/ −0.200(噪声饱和) |
| 复合退化是否掩盖读数 | 雾单独 0.541/0.755(真实 0.44/0.84)→ 与低光复合 **0.073/0.130**(真实 **0.70/0.73**,更高)。低光方向同理 0.529/0.543 → 0.194/0.034 |
| `blur` | 12 个任务**没有一个**加过模糊,**6 个被报成 >0.5**(最高 0.96) |

机制在公式里:`haze` 依赖暗通道均值高,图一暗就被按下去;`low_light` 依赖亮度均值低,雾的大气光提亮图又把它按下去;`blur` 算的是梯度能量,暗图雾图梯度本来就小。

**因果有内部对照**(207 次工具调用按任务归类):

| 退化 | 在任务名里? | agent 行为 |
|---|---|---|
| 模糊 | **不在** → 只能靠读数 | 误报的两题锐化 **31.9%**,读数 0.000 那题 **13.2%** |
| 雾 | **在**(`[haze_low_light_00]`) | 含雾题去雾 18%/14.5%,无雾题 0% —— 跟着名字走,不跟着读数 |

即:只能靠读数的维度 agent 被带偏;任务名泄漏了的维度它绕过读数做对了。而锐化在三题的任何最优链里都不出现、单用一次得 0.0000。

### 根因 B:奖励在起点处被截断

打分是 `clip((PSNR-15)/20, 0, 1)`,而任务起点:

| 任务 | 退化图 PSNR | 归一分 |
|---|---|---|
| haze_low_light_00 | 13.19 dB | 0.0000 |
| haze_low_light_01 | 12.76 dB | 0.0000 |
| low_light_noise_00 | 8.59 dB | 0.0000 |

**12 个任务里 8 个的起点在刻度下限之下**。于是 8.6 dB 和 14.9 dB 都是 0.0:18 个 episode 分数里 **7 个(39%)钉死在 0**,9 次逐题比较里 **5 次至少一边是 0**——记的"平"不是平。单工具得分表里三个评测任务只有 `lowlight_gamma` 在两题非零,其余全 0.0000 → **第一步几乎必然没有反馈**。

### 归因

不是"LLM 规划天生就抖",是**观测是错的、奖励在它工作的区间里是平的**——这种条件下策略选择近似随机。可推广的说法:**观测函数与奖励刻度的缺陷会伪装成策略噪声**,而且两者都能用便宜探针查出来。

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
