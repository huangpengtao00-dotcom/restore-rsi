# 一次 DNS 抽风,落盘成"模型这轮没提出改动"

2026-09-07 19:36。n36-mf 臂(measurement_failure=zero)开跑 50 分钟就"跑完"了
三个 evolve step,而同批其他四个臂 100 分钟连第一步都没结束。

## 落盘长什么样

    {"metrics": {"skipped": "no proposal", "steps": 1, "traces": 1}}
    {"metrics": {"skipped": "no proposal", "steps": 2, "traces": 1}}
    {"metrics": {"skipped": "no proposal", "steps": 3, "traces": 1}}

步数照涨,run.log 无异常,读起来像一个合法的算法结果:**模型这轮没想到能改的地方。**

## 真实原因

reef.log(见 reef-propose-errors.log):

    propose: served model call failed: ModelBindingError(
      'model endpoint unreachable: <urlopen error [Errno 8]
       nodename nor servname provided, or not known>')

DNS 解析失败,三次。事后复测:解析 0.01–0.04s,网关 HTTP 200。同批另外四个臂 0 次。

**订正(当晚稍后查明):最初写的"瞬时网络抽风"是错的,真因是这台 Mac 睡了。**
`pmset -g log` 显示 19:33:03 `Clamshell Sleep`(合盖)、19:36:57 `Maintenance Sleep`
睡了 938 秒 —— 三次 propose 失败正落在这两段里。DNS 解析失败只是症状:
系统进 Deep Idle 时网络栈断开,而进程在被冻结前后各发出了一次调用。

同一轮 148 分钟里机器共睡 16 次、累计 73 分钟,**占墙钟时间的 50%**。
睡眠期间进程冻结、不发请求,因此**在任何日志里都不留痕迹** —— run.log 只是看起来慢。

## 为什么只有 propose 中招

episode 的推理调用有重试 —— 同一晚 n36-baseline 吃到一个 502,日志是
`inference attempt 1/4 failed (ReefClientError 502); retrying in 5s`,重试后恢复。

**propose 没有重试:一次失败就记 skipped,继续下一步。** 同一个网络故障,在
episode 路径上是"重试一下就好",在 propose 路径上是"这一步没有候选"。

## 这正是这条线要讲的东西

论文的说法是:自进化循环里的基础设施故障会伪装成算法效应。这是一个不用构造的实例 ——

- 故障是外部的(DNS),与算法无关
- 编码后的结果(`no proposal`)在语义上完全合法
- 计数照常推进,没有任何一处报错
- 若不与其他臂对比耗时,不会有人去看

**"这一步没有候选"和"这一步的候选没赢"在下游是两回事,但落盘后长得一样。**
它与 measurement_failure 那组开关是同一个病:测量侧的失败被编码成了被测对象的属性。
区别只在于,那三种编码(zero/exclude/guess)是从别人的公开代码里抄来的,
这一个是今晚自己撞上的。

## 已做

- 该臂数据作废,原样归档在 `work/n36-mf-void-dns-20260907/`,重跑
- `tasks/check_arm_validity.py` 新增两条判据:
  1. 落盘里出现 `skipped` 就报红(拿这份废数据验过会红)
  2. **报告这一臂跑的那段时间里机器睡了多久**,超过 60 秒即红。判据取自 pmset,
     不靠猜:`Entering Sleep state` 行末尾的 `N secs` 就是时长(用两组数据与随后的
     DarkWake 时间戳交叉核对过)
- `caffeinate -dimsu` 阻止空闲睡眠。**合盖睡眠(Clamshell)阻止不了** ——
  那需要人把盖子打开,或接外接显示器

## 已核:出结论的那批数据没有受影响

同日报告的 12 步结论出自两个窗口,两个都全程清醒:

    基线 3 步      01:05-02:15   睡 0 次   0s
    四缺陷臂 9 步  17:14-18:26   睡 0 次   0s

并逐步核过完整性:12 步的候选/现版分数数组都是 9+9,排除 0、episode 失败 0,
胜+负+平与配对数一致。

## 没做

没给 propose 加重试 —— 那在 reef 内部,不是这个仓的代码。真要做,应当是让
`skipped` 带上原因分类(模型无输出 / 调用失败),而不是共用一个字符串。
