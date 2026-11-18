# 第9章 轻量架构与深度可分离卷积

**一句话**：`MobileNets` 用 `depthwise separable` 把标准卷积拆成两步，从源头上省参数，是压缩链路的第一道省。

**为什么需要它**：前面几章讲的剪枝、量化都是在给定结构上做减法；如果结构本身就选错了，后面再怎么压也是事倍功半。`MobileNets` 论文专门针对移动端与嵌入式视觉，提出 `depthwise separable` 卷积，把标准卷积的乘法量降一个量级，再用 `width multiplier` 与 `resolution multiplier` 提供两个连续的取舍旋钮。这是「结构设计本身就是压缩」的代表。

**核心结构**：标准卷积同时做空间滤波与通道混合，`depthwise separable` 把它拆成两步：`depthwise convolution` 对每个输入通道单独做空间卷积，不做通道混合；`pointwise convolution` 用 1×1 卷积做通道混合。两步合起来逼近标准卷积的效果，但计算量大幅下降。`width multiplier` 按比例缩窄每一层的通道数，`resolution multiplier` 按比例缩小输入分辨率，两者都能在精度与体积/速度之间连续权衡。

**关键设计**：`depthwise separable` 的收益来自「把空间与通道解耦」——大部分计算量在标准卷积的通道混合里，拆开后 `depthwise` 只做轻量空间滤波，`pointwise` 的 1×1 卷积远比标准卷积便宜。`MobileNets` 论文给出计算量的对比与两个 multiplier 的用法，让同一个结构能适配不同资源预算的设备。

**拆解**：
### 1. 标准卷积贵在哪
一个标准卷积要对每个输出通道、每个空间位置，把所有输入通道都乘一遍，计算量正比于输入通道数乘输出通道数乘卷积核空间大小。通道多、核大时乘法量爆炸。这是移动端跑不动大卷积网的根本原因。

### 2. depthwise + pointwise 怎么省
`depthwise convolution` 让每个输入通道配一个滤波器，只在该通道内做空间卷积，计算量正比于通道数乘核大小，不做跨通道；`pointwise convolution` 用 1×1 卷积把通道混起来，计算量正比于输入通道乘输出通道。两步相加比一步标准卷积少一个量级。`MobileNets` 论文给出了这个对比。

### 3. width multiplier 与 resolution multiplier
`width multiplier` 统一缩放每层通道数，模型整体变瘦；`resolution multiplier` 缩放输入分辨率，特征图变小。两者都提供连续的取舍——想要更快更小就调低，想要更准就调高。综合报告口径常把这两个 multiplier 当成端侧调参的第一组旋钮，先调结构再调量化。

### 4. 轻量结构与后几道门的关系
选了 `MobileNets` 这类轻量结构，参数量已经降下来了，后续 `pruning` 与 `quantization` 的负担也小。但轻量结构对量化可能更敏感——参数少了，每个参数的误差影响更大。所以轻量结构常常配合 QAT 或蒸馏，把精度补回来。结构是压缩链路的第一道省，不是唯一一道。

**今天怎么做**：端侧视觉任务先选轻量骨架，`MobileNets` 的 `depthwise separable` 是起点。用 `width multiplier` 与 `resolution multiplier` 按设备预算定容量，再叠量化。轻量结构对量化敏感，准备好用 QAT 或蒸馏补精度。

<div class="chapend" markdown="1">

**容易误解的地方**：`depthwise separable` 不是无损替代品，精度有损失但换来计算量下降。两个 multiplier 是连续旋钮，不是固定档位。轻量结构不等于不用再量化，两者是叠加关系。

**记住这一句**：`depthwise separable` 把空间与通道解耦，从结构上省下第一个量级的计算，是端侧压缩的第一道门。

</div>
