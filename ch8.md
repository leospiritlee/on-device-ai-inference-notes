# 第8章 知识蒸馏

**一句话**：`knowledge distillation` 让大 `teacher` 的 `soft target` 教小 `student`，把「暗知识」传过去，小模型因此能用更小容量逼近大模型的行为。

**为什么需要它**：直接训一个小模型，往往精度不够；把大模型压小， quantization 和剪枝又掉精度。`knowledge distillation` 走第三条路：先有一个精度高的大 `teacher`，再训一个小 `student` 去模仿它。`Distilling the Knowledge` 论文的核心洞察是，`teacher` 输出的概率分布里藏着「哪些类别相似」的暗知识，比硬标签提供的信息丰富得多，小模型学了这些，精度能超过直接从硬标签学。

**核心结构**：`teacher` 是大而准的模型，`student` 是要部署的小模型。训练时 `student` 同时拟合硬标签和 `teacher` 的 `soft target`。`soft target` 来自 `teacher` 的 `softmax` 输出，经过 `temperature` 软化——提高 `temperature` 让概率分布更平滑，暴露类别间的相对关系。综合损失通常是硬标签损失与软标签损失的加权和。

**关键设计**：`temperature` 是蒸馏的旋钮。它只在训练时抬高，让 `teacher` 的 `softmax` 输出变软；`student` 训练好后用正常温度推理。`temperature` 太低，`soft target` 跟硬标签差不多，蒸馏没收益；太高，信号全被抹平。`Distilling the Knowledge` 论文对温度的作用有清楚论证，这是蒸馏最容易被调错的地方。

**拆解**：
### 1. soft target 里的暗知识在哪
硬标签只告诉模型「这是猫」，`teacher` 的 `soft target` 还会告诉它「这张图有点像狗、不太像车」。这种类间相似度是模型学到的隐含结构，硬标签里没有。`student` 学到这些相对关系，泛化更好。论文用 `softmax` 加 `temperature` 把这种关系显出来。

### 2. temperature 怎么调
训练时把 `teacher` 和 `student` 的 `softmax` 都除以 `temperature` 再算软损失，`student` 的硬标签损失仍用正常温度。`temperature` 常取大于 1 的值，具体看任务。综合报告口径提醒：`temperature` 与软硬损失的权重一起调，单独动一个往往不收敛到好结果。

### 3. 蒸馏与量化的配合
第八章前面讲 QAT 时提过，蒸馏常和量化叠加：让量化后的 `student` 拟合浮点 `teacher` 的输出。这样即使位宽很低，`student` 也能从 `teacher` 那里补回精度。这是端侧最常见的组合——结构变小、位宽变低，再用蒸馏把精度拉回来。

### 4. teacher 从哪来
`teacher` 可以是现成的大模型，也可以是同一结构的浮点高精度版本。端侧常见做法是：先在云端训一个大模型当 `teacher`，再在端侧约束下训 `student`，让 `student` 学 `teacher` 的行为。这与端云协同的判断框架呼应——重训练留云端，蒸馏产物上端侧。

**今天怎么做**：先有一个可信的 `teacher`，确定 `student` 的目标容量。训练时同时用硬标签损失与 `teacher` 的 `soft target` 损失，`temperature` 从常用区间起步再调。若同时要量化，把量化后的 `student` 接进蒸馏，用浮点 `teacher` 补精度。

<div class="chapend" markdown="1">

**容易误解的地方**：`soft target` 不是随便 smoothed 的标签，是 `teacher` 的 `softmax` 输出经 `temperature` 处理的结果。`temperature` 只训练时抬高，推理时恢复正常。蒸馏不等于压缩，它换的是行为逼近，通常还要配合结构与量化才真正省资源。

**记住这一句**：蒸馏把 `teacher` 的 `soft target` 当更丰富的标签，让 `student` 用更小容量学到类间关系。

</div>
