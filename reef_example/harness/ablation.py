"""消融开关:缺陷不是被删掉,是被做成配置项。

这个模块存在的理由是一次文献核查(Obsidian 31 号)。"自进化循环的闸门不可信"
在 2026 年已经有人做了 —— PACE(arXiv 2606.08106)把"分数涨了就留"定性为未受控
的自适应多重检验,量出 30–42% false / 10–33% harmful edits,方法是 anytime-valid
的 e-process,比这里的符号检验强一档;HarnessFix(2606.06324)做了 harness 缺陷
的七层分类,用的还正好是 one-sided sign test。

但它们全在文本/代码/游戏域(GSM8K、SWE-Bench、GAIA、AppWorld、Atari)。那些域
**没有天花板**:能比 A/B 谁高,不能问"离最优还差多少"。所以 PACE 只能报**错误率**,
报不了**代价**。

复原任务能:工具确定性、动作空间 6 个、退化由构造给出、原图在手,于是每道题的上界
可以穷举。有了分母,问题就从"这个缺陷错多少次"变成"**这个缺陷值多少可达分数**" ——
那是文本域结构上给不出的量,也是这条线唯一别人抄不走的位置。

要回答那个问题,缺陷就必须**能被打开**,否则跑不出对照。所以今天修掉的那几处都不删,
退化成这里的开关:

    gate                 sign_test | wins_over_losses
    batch_criterion      attained  | absolute
    diagnosis            decoupled | coupled_v0
    measurement_failure  raise     | zero | exclude | guess

默认全是修好的那一侧 —— 打开缺陷必须是显式动作,不能是忘了设。

    RESTORE_ABLATION="measurement_failure=zero,gate=wins_over_losses" ./run.sh
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

#: 每个开关的合法取值。第一个是默认值(= 修好的那一侧)。
CHOICES: dict[str, tuple[str, ...]] = {
    # 闸门判据。wins_over_losses 是 reef 自带的规则,在这套系统上假阳性率 50%
    # (候选与现版无差别时,"正面多于反面"本来就有 1/2 的概率),与重复次数无关。
    # e_process = PACE(arXiv 2606.08106)的 testing-by-betting 闸门。零假设与符号检验
    # 相同(配对、丢平局、H0 下不一致对各半),差别是序贯判定:E<-E*(1+λ(2w-1)),
    # E>=1/α 即提交,由 Ville 不等式在任意停时控制假提交率。它不是"缺陷臂"——
    # 是一个**更强的对照**:我们这个域有天花板,能回答"错误率控住了,质量真的更好吗"。
    "gate": ("sign_test", "wins_over_losses", "e_process"),
    # 什么算"失败"、于是进 batch。absolute 是一把绝对尺子(data.max_score=0.5),
    # 而 12 道题的上界差 46.5 倍,实测 4/12 判反 —— 有巨大改进空间的题永不进 batch,
    # 已经做到头的题反而进。
    "batch_criterion": ("attained", "absolute"),
    # 诊断读数是否互相掩盖。coupled_v0 是修复前的版本:同一张图同样的雾,只是加了
    # 暗光,雾读数就掉 91%;而"模糊"在干净参考图上读 0.912。
    "diagnosis": ("decoupled", "coupled_v0"),
    # 测量失败怎么被编码成分数。三种非默认值各自对应一篇顶会工作的公开代码:
    #   zero     JarvisArt 的 reward 路径上 28 处(grpo_lr.py 9 + lrc_tools_reward.py 19)
    #            -> 悲观偏差,而且污染 GRPO 的组内优势归一化
    #   exclude  JarvisEvo 的 `if result: group_list.append(result)` -> 失败样本从分母
    #            消失,**乐观**偏差(与 zero 方向相反),且失败数不报告
    #   guess    VIEScore 的 `give_up_parsing` -> 解析不出来就猜一个分,纯噪声
    "measurement_failure": ("raise", "zero", "exclude", "guess"),
}


@dataclass(frozen=True)
class Ablation:
    gate: str = "sign_test"
    batch_criterion: str = "attained"
    diagnosis: str = "decoupled"
    measurement_failure: str = "raise"

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value not in CHOICES[field.name]:
                raise ValueError(
                    f"ablation {field.name}={value!r} 不是合法取值;可选 {CHOICES[field.name]}。"
                    f"拼错一个开关而静默用了默认值,会让整批实验的标签是错的 —— 那比跑失败更糟。"
                )

    @property
    def is_baseline(self) -> bool:
        """全是修好的那一侧。"""
        return all(getattr(self, f.name) == CHOICES[f.name][0] for f in fields(self))

    def label(self) -> str:
        """跑出来的每份结果都要带上它的配置标签,否则事后分不清哪个数出自哪个配置。"""
        if self.is_baseline:
            return "baseline"
        return ",".join(
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) != CHOICES[f.name][0]
        )

    def as_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def parse(cls, spec: str | None) -> "Ablation":
        """`"gate=wins_over_losses,measurement_failure=zero"` -> Ablation。

        空串/None 给基线。**不认识的键要报错**:静默忽略一个拼错的开关,意味着
        实验标签说打开了缺陷而实际没打开,那种数据看起来完全正常。
        """
        if not spec or not spec.strip():
            return cls()
        values: dict[str, str] = {}
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"ablation 片段 {chunk!r} 不是 key=value 形式")
            key, value = (part.strip() for part in chunk.split("=", 1))
            if key not in CHOICES:
                raise ValueError(f"未知的 ablation 开关 {key!r};可选 {sorted(CHOICES)}")
            values[key] = value
        return cls(**values)

    @classmethod
    def from_env(cls) -> "Ablation":
        return cls.parse(os.environ.get("RESTORE_ABLATION"))


def current() -> Ablation:
    """进程当前的消融配置。每次读环境变量,便于测试里 monkeypatch。"""
    return Ablation.from_env()
