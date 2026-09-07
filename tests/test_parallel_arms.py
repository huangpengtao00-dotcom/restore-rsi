"""同一台机器并行跑多个消融臂:每个臂的状态必须互不干扰。

reef 的 evaluate 串行跑 episode(`tuple(... for task in self._tasks)`),改不动
也不该改。所以并行的粒度放在**配置之间**:消融那几个臂互不相干,同时跑,总时间
等于最慢的那个而不是它们的和。实测一次完整跑 42 分钟(其中 evolve 段 37 分钟),
五个臂串行是三个多小时,并行就还是四十分钟。

前提是状态真的隔离。共用 work 目录会让 trace、缓存、task_refs 互相覆盖 ——
**而那种污染事后完全看不出来**:数字都在,只是来自别的臂。所以这里钉住隔离。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reef_example"))


def _work_dir_with(value: str | None) -> Path:
    """在给定 RESTORE_WORK 下,harness 认为状态目录在哪。"""
    import importlib

    import harness.evolution as ev

    old = os.environ.get("RESTORE_WORK")
    try:
        if value is None:
            os.environ.pop("RESTORE_WORK", None)
        else:
            os.environ["RESTORE_WORK"] = value
        importlib.reload(ev)
        return ev._work_dir()
    finally:
        if old is None:
            os.environ.pop("RESTORE_WORK", None)
        else:
            os.environ["RESTORE_WORK"] = old
        importlib.reload(ev)


def test_default_work_dir_is_unchanged():
    """没设变量时行为必须和以前完全一样,否则这个改动会悄悄搬走所有既有状态。"""
    assert _work_dir_with(None).name == "work"


def test_two_arms_get_disjoint_state():
    a = _work_dir_with("work/abl-mf-zero")
    b = _work_dir_with("work/abl-gate")
    assert a != b
    # 三样都必须分开:trace 记录发生了什么、cache 是内容寻址的产物、
    # task_refs 是判分用的参考图。任何一样共用都会串味。
    for sub in ("results/trace.jsonl", "cache", "task_refs.json"):
        assert (a / sub) != (b / sub)
    assert not str(a).startswith(str(b)) and not str(b).startswith(str(a)), "两个臂的目录不能互相嵌套"


def test_service_url_follows_the_port():
    """端口不分开,第二个臂根本起不来(或者更糟:连到第一个臂的服务上)。"""
    src = (ROOT / "reef_example" / "run.py").read_text()
    assert "REEF_PORT" in src, "run.py 必须从 REEF_PORT 取端口"
    assert '"http://127.0.0.1:8900"' not in src, "端口不能再是硬编码字面量"


@pytest.fixture
def scratch_arm(request):
    """一个临时的臂目录,测完删掉。

    materialize 把产物写在 `reef_example/<RESTORE_WORK>` 下(那是它的设计:臂目录
    要和 serve.yaml 同级才能被 reef 解析),所以用不了 tmp_path,得自己收拾。
    """
    import shutil

    arm = request.param
    target = ROOT / "reef_example" / arm
    shutil.rmtree(target, ignore_errors=True)
    try:
        yield arm
    finally:
        shutil.rmtree(target, ignore_errors=True)


@pytest.mark.parametrize("scratch_arm", ["work/t-arm-a", "work/t-arm-b"], indirect=True)
def test_materialize_writes_into_its_own_arm(scratch_arm):
    arm = scratch_arm
    """真跑一次 materialize,确认产物落在该臂目录里,而不是共用的 work/。"""
    if not (ROOT / "tasks" / "data" / "manifest.json").exists():
        pytest.skip("tasks/data 未生成:先跑 tasks/make_tasks.py")
    env = {**os.environ, "RESTORE_WORK": arm}
    env.pop("RESTORE_ABLATION", None)
    proc = subprocess.run(
        [sys.executable, "materialize.py"],
        cwd=ROOT / "reef_example", env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    produced = ROOT / "reef_example" / arm
    for name in ("tasks.json", "task_refs.json", "recipes/harness_evolve.yaml"):
        assert (produced / name).exists(), f"{arm} 缺 {name};stdout={proc.stdout[-300:]}"


# ------------------------------------------------- episode 侧的路径隔离

@pytest.fixture
def arm_wrapper(request, tmp_path):
    """用**生产脚本**给一个臂生成 wrapper —— 判据从被测代码取,不在测试里重写一份。"""
    import shutil
    import subprocess

    arm = request.param
    here = ROOT / "reef_example"
    target = here / arm
    shutil.rmtree(target, ignore_errors=True)
    proc = subprocess.run(
        [str(here / "bin" / "make-arm-wrapper.sh"), str(here), arm],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    try:
        yield Path(proc.stdout.strip())
    finally:
        shutil.rmtree(target, ignore_errors=True)


@pytest.mark.parametrize("arm_wrapper", ["work/t-wrap-a", "work/t-wrap-b"], indirect=True)
def test_the_arm_wrapper_pins_paths_into_its_own_arm(arm_wrapper):
    """episode 侧的路径必须烧进 wrapper,不能靠环境变量传。

    reef 把 episode 的环境剥到只剩 PATH/TMPDIR,所以 RESTORE_RESULTS_DIR 传不进
    pi。bin/restore 的 `${VAR:-default}` 于是总取默认的 `work/` —— 并行跑臂时那是
    错的。实测代价:baseline 跑在 work/baseline,pi 的 376 条 trace 全写进共用的
    work/,judge 在本臂找不到成功的 run,把每个 episode 判 0 分,48 分钟跑完两边
    都是 0.0,日志一切正常。
    """
    arm = arm_wrapper.parent.parent.name  # .../work/t-wrap-a/bin/restore -> t-wrap-a
    assert arm_wrapper.exists() and arm_wrapper.stat().st_mode & 0o111, "wrapper 要可执行"
    body = arm_wrapper.read_text(encoding="utf-8")
    for var in ("RESTORE_RESULTS_DIR", "RESTORE_CACHE_DIR", "RESTORE_TRACE"):
        assert var in body, f"{var} 必须被烧进 wrapper"
        line = next(l for l in body.splitlines() if l.startswith(f"export {var}="))
        assert f"/{arm}/" in line, f"{var} 指向的不是本臂:{line}"
    assert body.rstrip().endswith('"$@"'), "应当把参数透传给原 wrapper"


@pytest.mark.parametrize("arm_wrapper", ["work/t-wrap-c"], indirect=True)
def test_the_wrapper_actually_exports_those_paths(arm_wrapper):
    """真跑一次,看它导出的值。只读文本会漏掉引号/展开出错的情形。"""
    import subprocess

    proc = subprocess.run(
        ["bash", "-c", f'source "{arm_wrapper}" 2>/dev/null; echo "$RESTORE_TRACE"'],
        capture_output=True, text=True, timeout=30,
    )
    # source 会执行到 exec 那行,所以用一个只取变量的方式:直接解析并 eval 前几行
    body = "\n".join(l for l in arm_wrapper.read_text(encoding="utf-8").splitlines() if l.startswith("export "))
    proc = subprocess.run(
        ["bash", "-c", f'{body}\necho "$RESTORE_TRACE|$RESTORE_CACHE_DIR"'],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    trace, cache = proc.stdout.strip().split("|")
    assert trace.endswith("/work/t-wrap-c/results/trace.jsonl"), trace
    assert cache.endswith("/work/t-wrap-c/cache"), cache


def test_run_sh_puts_the_arm_wrapper_first_on_path():
    """臂 wrapper 必须排在共用的 bin/ 前面,否则烧进去的路径根本不生效。"""
    src = (ROOT / "reef_example" / "run.sh").read_text(encoding="utf-8")
    line = next(l for l in src.splitlines() if l.startswith("export PATH="))
    arm_pos, shared_pos = line.index('$WORK/bin'), line.index('$PWD/bin')
    assert arm_pos < shared_pos, f"臂 wrapper 应当在共用 bin 之前:{line}"
    assert "make-arm-wrapper.sh" in src, "run.sh 必须调用生成脚本"


def test_run_sh_refuses_to_start_on_a_busy_port():
    """端口被占必须退出,不能接着跑。

    实测过后果:上一轮的 reef 占着 8900,新一轮起不来(Errno 48),而 run.py 连上了
    **旧实例** —— 它的 work 目录是上一个臂的,pi 的 trace 全写进别处,判分全 0,
    唯一线索是一句 409 "agent_record_id already has different content" 埋在堆栈里。

    并行跑臂时这最危险:端口一撞,新臂那一行数据整批来自旧配置,而没有任何地方
    会说出来。
    """
    src = (ROOT / "reef_example" / "run.sh").read_text(encoding="utf-8")
    assert 'lsof -ti:"$PORT"' in src, "启动前必须检查端口"
    guard = src[src.index('lsof -ti:"$PORT"'):]
    assert "exit 1" in guard[:600], "占用时必须退出,而不是警告后继续"


# ------------------------------------------------- 发车必须脱离会话组

def test_launcher_detaches_the_session():
    """`nohup &` 不够,发车必须 start_new_session。

    2026-09-07 实测:四个臂 record 段全跑完、evolve 正在跑,一次会话中断把它们全杀了,
    每臂只留下 1–2 个 step 的落盘数据。而日志尾部是正常的 "evolve step still running",
    没有任何报错 —— **从日志完全看不出它们是被杀的**,只能从「进程没了但没有收尾行」推断。

    记忆里已有同款教训(那次是 Popen 派工人),这次换成 nohup 又栽一遍,所以钉成判据。
    """
    src = (ROOT / "reef_example" / "bin" / "launch_arm.py").read_text(encoding="utf-8")
    assert "start_new_session=True" in src, "发车必须让子进程 setsid,否则主控中断会连坐"
    assert "run.pid" in src, "pid 要落盘 —— 停车按 pid,不按 pkill -f 模式(会误杀同类)"


def test_launcher_clears_ablation_for_baseline():
    """baseline 臂必须**删掉**环境变量,而不是传一个 'baseline' 字符串进去 ——
    那会被 Ablation.parse 当成拼错的开关而报错。"""
    src = (ROOT / "reef_example" / "bin" / "launch_arm.py").read_text(encoding="utf-8")
    assert 'env.pop("RESTORE_ABLATION"' in src
