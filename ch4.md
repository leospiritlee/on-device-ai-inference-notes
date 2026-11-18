# 第4章 训练后量化与校准

**一句话**：`post-training quantization` 不需要重训，用 `representative_dataset` 做一次 `calibration` 就能把模型压下来，精度全看校准样本选得准不准。

**为什么需要它**：大多数时候你手上只有一个训练好的浮点模型，没有训练数据、没有训练代码，也不想再训一遍。`post-training quantization`（PTQ）就是为这种场景准备的：直接拿现成模型，用少量样本跑一遍定出量化参数，转成 `int8` 模型。它是端侧落地里最常用、成本最低的一道门。

**核心结构**：TensorFlow Lite 官方文档把 `post-training quantization` 分成几档：`dynamic range quantization` 只量化权重、激活在运行时动态定范围，体积降得多、实现简单；`full integer quantization` 把权重和激活都量化成整数，需要一份 `representative_dataset` 来做 `calibration`；`float16 quantization` 把权重压到半精度，体积减半但精度几乎不损。用 `tf.lite.TFLiteConverter` 转换时，通过配置决定走哪一档，量化参数由 `calibration` 从样本里统计出来。

**关键设计**：PTQ 的关键不在算法，在样本。`representative_dataset` 必须代表真实输入的分布——亮度、尺度、类别比例都要接近线上。样本太少或太偏，`scale` 就估不准，某些通道会被截顶，精度断崖式下跌。官方文档反复强调要提供能代表典型输入的样本集，就是这个原因。

**拆解**：
### 1. PTQ 的三档各适合什么
`dynamic range quantization` 无需校准数据，转换最快，适合先跑通看效果；`full integer quantization` 需要校准数据，能全程整数推理，适合有整数加速后端的设备；`float16 quantization` 适合有半精度支持、对精度敏感的场景。综合报告口径里常在「能不能拿到代表性数据」与「有没有整数后端」之间做选择，这两条决定档位。

### 2. representative_dataset 怎么准备
样本要从真实输入里抽，覆盖常见与边界情况，数量不必多但要典型。官方示例里用一小批数据喂给转换器，转换器跑前向记录每个激活张量的范围。用随机噪声或完全不相关的图片做校准，得到的 `scale` 跟真实分布对不上，量化后精度必然掉。

### 3. calibration 在转换里的位置
`calibration` 是转换期的一步，不是运行期。转换器用 `representative_dataset` 跑若干次前向，统计出每个张量的动态范围，据此算出 `scale` 与 `zero-point`，固化进模型。运行期不再重算。这意味着校准做完，模型的量化参数就定死了——换分布就得重新校准。

### 4. TFLiteConverter 的配置要点
`tf.lite.TFLiteConverter` 是转换入口，通过选项打开量化、指定是否全整数、挂上校准数据生成器。综合报告口径提醒：转化后用同一批验证集对比浮点与量化输出的误差，别只看转换成功与否；有些算子不支持全整数，会回退到浮点，要检查回退列表。

**今天怎么做**：先做一版 `dynamic range quantization` 看体积与精度基线；能拿到代表性数据就上 `full integer quantization`，认真准备 `representative_dataset`；转换后逐层对比浮点输出，定位误差最大的层。对精度敏感的层可以单独保留浮点。

<div class="chapend" markdown="1">

**容易误解的地方**：PTQ 不需要训练数据不等于不需要数据，校准样本Still必需。转换成功不等于精度可接受，必须实测。全整数量化遇不支持的算子会回退浮点，不是所有计算都变整数。

**记住这一句**：PTQ 的精度上限由 `representative_dataset` 决定，校准样本偏一寸，量化精度掉一截。

</div>
