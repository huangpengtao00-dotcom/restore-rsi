# 深度 4 的天花板:分母算小了,循环会显得已经做到头

2026-09-07。`tasks/oracle_chains.py --depth 4`,12 道题,纯 CPU 不碰模型,
判据直接调 `restore score` 那两个函数(不在 oracle 里重推)。

## 为什么要跑

同日的消融实验里有一条 episode 报出可达率 **106%** —— 分母被越过了。查清楚:
那正是**唯一一条走了 4 步的链**,而当时的天花板是深度 3 穷举的。深度 3 的上界
不是走 4 步时的上界,所以那不是算错,是口径没对齐。

## 订正结果

同一批 12 条 episode,换分母重算:

    d3 口径:12 条里 1 条超 100%
    d4 口径:12 条里 0 条

更值得记的是另外两条。d3 口径下它们是**正好 1.000** —— 读起来像"完美发挥,
没得学了"。d4 口径下是 0.955 和 0.937:

    haze_low_light_00   得分 0.5722   d3 上界 0.5722(1.000)   d4 上界 0.5994(0.955)
    haze_low_light_01   得分 0.3620   d3 上界 0.3620(1.000)   d4 上界 0.3863(0.937)

**分母算小了,不只是让个别数字超过 100%,还会让"还有多少可学"整体被低估。**
而"还有多少可学"正是 batch 该按什么进的判据(见 harness/scoring.py: attained)。

## 各题上界

题                         d3      d4     涨幅   d4 最优链
haze_00               0.0215  0.0244    13%   sharpen_unsharp>lowlight_clahe>denoise_median>denoise_bilateral
haze_01               0.5180  0.5183     0%   denoise_bilateral>dehaze_dcp>denoise_bilateral>denoise_bilateral
low_light_00          1.0000  1.0000     0%   lowlight_gamma>lowlight_gamma
low_light_01          0.9650  0.9874     2%   lowlight_gamma>lowlight_gamma>lowlight_gamma>lowlight_clahe
noise_00              0.9580  1.0000     4%   denoise_bilateral>denoise_bilateral>denoise_bilateral>denoise_bilateral
noise_01              1.0000  1.0000     0%   lowlight_gamma>denoise_bilateral>denoise_bilateral>denoise_bilateral
haze_low_light_00     0.5722  0.5994     5%   lowlight_gamma>dehaze_dcp>lowlight_gamma>lowlight_gamma
haze_low_light_01     0.3620  0.3863     7%   dehaze_dcp>lowlight_gamma>lowlight_gamma>lowlight_gamma
low_light_noise_00    0.5444  0.6268    15%   denoise_median>denoise_bilateral>lowlight_gamma>lowlight_gamma
low_light_noise_01    0.6220  0.7610    22%   denoise_median>denoise_bilateral>lowlight_gamma>lowlight_gamma
haze_noise_00         0.2019  0.2468    22%   denoise_median>denoise_bilateral>dehaze_dcp>lowlight_gamma
haze_noise_01         0.4071  0.4652    14%   dehaze_dcp>denoise_median>denoise_median>denoise_bilateral

## 还没闭合

链长 5 及以上仍可能越过深度 4 的上界。深度 5 是 6 倍量、约 2.6 小时(本机被
其他任务占着时更久),没跑。中继允许 6 轮,所以理论上最长 6 步;实测最长 4 步。
