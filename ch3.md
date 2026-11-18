# 第3章 量化的数学基础：scale 与 zero-point

**一句话**：`quantization` 用 `scale` 与 `zero-point` 把浮点映成整数，精度损失就藏在这两个数的选择里。

**为什么需要它**：模型里的 `weight` 与 `activation` 默认用浮点存，一个数是 32 位；量化把它们映成 `int8` 甚至更低比特，体积直接降到四分之一或更少，访存与计算也跟着变快。但映射必然有误差——理解 `scale` 与 `zero-point` 怎么定，才能判断精度为什么会掉、怎么补。这是后面所有量化方案的公共基础。

**核心结构**：量化的核心是把一个浮点区间映到整数区间。`scale` 决定区间怎么缩放，`zero-point` 决定零点落在哪个整数上。`symmetric quantization` 让零点对齐在 0，映射简单；`asymmetric quantization` 允许零点偏移，能更好地贴合不居中的分布。`dequantization` 是反方向——把整数还原回浮点，推理时在需要的地方做。`per-tensor` 对整个张量用一组参数，`per-channel` 对每个通道单独算，精度更高但参数更多。

**关键设计**：`scale` 由数据的动态范围决定：范围越大、`scale` 越粗，量化误差越大。这就是为什么 `outlier`（离群值）这么麻烦——一个极大的值会把整个范围的 `scale` 撑大，让大多数正常值挤在很窄的整数区间里。`LLM.int8()` 论文正是发现大模型激活里存在系统性离群值，才专门设计了混合精度方案来处理。

**拆解**：
### 1. 一次映射怎么算
给定浮点值、给定的 `scale` 与 `zero-point`，量化就是除以 `scale`、加上 `zero-point`、再取整到最近的整数；反量化是减 `zero-point`、乘 `scale`。论文把这套映射写成整数算术的形式，目标是让推理全程用 `integer arithmetic`，避免反复在浮点与整数间转换。理解这一步，就知道量化误差来自「取整」与「范围压缩」两处。

### 2. 对称与非对称怎么选
`symmetric quantization` 的零点固定在 0，好处是计算简单、`dequantization` 只需一次乘法；缺点是要求数据分布大致以 0 居中。`asymmetric quantization` 的 `zero-point` 可以偏移，能把非居中的区间（比如 ReLU 之后全正的激活）贴得更紧，精度更好。权重大体对称，常用对称量化；激活常常非对称，需要非对称量化。

### 3. per-tensor 与 per-channel 的差别
`per-tensor` 一个张量一组 `scale` 与 `zero-point`，实现简单、开销小；`per-channel` 每个输出通道各算一组，能适应不同通道差异很大的权重分布，精度明显更高。代价是多了若干组参数，计算时也要按通道取对应的 `scale`。`integer-arithmetic-only inference` 论文对这套 per-channel 的整数算术实现有详细展开。

### 4. 范围怎么定：calibration 的位置
`scale` 与 `zero-point` 定得准不准，取决于用来估计动态范围的样本。`calibration` 就是用一小批代表性数据跑一遍，记录激活的实际范围，再据此定量化参数。样本选得偏，范围就估不准——这是训练后量化精度掉的主因之一，第四章专门讲。

**今天怎么做**：量化前先看权重的分布是不是以 0 居中、有没有明显的 `outlier`。权重可试 `symmetric quantization` 加 `per-channel`，激活多半要 `asymmetric quantization`。精度掉了先怀疑动态范围估得准不准，再怀疑位宽够不够。

<div class="chapend" markdown="1">

**容易误解的地方**：`zero-point` 不是浮点里的 0，是整数区间里代表 0 的那个整数值。`per-channel` 更准但更贵，不是所有后端都支持。量化误差不全来自位宽，范围估计不当同样致命。

**记住这一句**：`scale` 定粗细、`zero-point` 定零点，量化的精度损失就藏在这两个数的选择里。

</div>
