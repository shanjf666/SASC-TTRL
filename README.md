<div align="center">

# SASC-TTRL: Stability-Aware Self-Consistency Test-Time Reinforcement Learning

# 基于熵稳定性加权过滤的测试时强化学习系统

</div>

<div align="center">

**通过不确定度感知（Uncertainty-Aware）生成的偏好数据，解决大模型在复杂推理中的“盲目自信”与“错误共识”问题。**

[项目介绍](https://www.google.com/search?q=%23-%E9%A1%B9%E7%9B%AE%E4%BB%8B%E7%BB%8D) • [核心方法](https://www.google.com/search?q=%23-%E6%A0%B8%E5%BF%83%E6%96%B9%E6%B3%95) • [实验结果](https://www.google.com/search?q=%23-%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C)

</div>

---

## 📖 项目介绍

**SASC-TTRL** 提出了一种全新的自动化偏好数据生成与筛选框架，旨在提升大语言模型（LLM）在复杂数学推理任务中的稳定性与可靠性 。

传统的测试时强化学习（TTRL）方法通常依赖“多数投票（Self-Consistency）”来生成伪标签。然而，在面对高难度或诱导性问题时，模型容易产生“错误共识”，即生成大量逻辑错误但答案一致的路径 。此外，全词表熵计算在长文本推理中极易导致显存溢出（OOM） 。

本项目引入 **熵标准差（Entropy Standard Deviation）** 作为推理稳定性的度量指标，结合 **Top-K 局部熵** 计算与 **Z-Score 动态相对加权** 机制，实现了低成本、高质量的偏好数据生成，有效指导模型在测试时进行自我进化 。

## ⚙️ 核心方法

我们的方法包含“推理验证（TTS）”与“训练固化（TTRL）”两个阶段，核心流程如下 ：

1. **Top-K 局部熵计算 (Top-K Local Entropy)**:
为避免 OOM，仅针对每步生成概率最高的  个 Token（）计算局部熵，在保持关键信息的同时大幅降低计算开销 。


2. **推理稳定性度量 (Stability Metric)**:
计算推理路径中熵序列的标准差。标准差越小，代表推理过程越稳定，逻辑断裂的可能性越低 。


3. **Z-Score 难度自适应加权 (Task Adaptation)**:
针对不同难度的测试样本，对其 Rollout 集合的稳定性得分进行 Z-Score 归一化。这消除了简单与困难任务间的置信度基准差异，使模型能依据“相对优势”进行学习 。


4. **动态过滤与加权投票 (Filter & Weighted Vote)**:
* 
**过滤 (Gate)**: 剔除稳定性最差（标准差最大）的 20% 路径 。


* 
**加权**: 基于相对稳定性权重  进行投票，生成高质量伪标签 。




5. **GRPO 策略更新**:
利用筛选出的伪标签作为监督信号，通过 GRPO 算法计算组内优势，更新模型策略 。



## 📊 实验结果

实验基于 **Qwen2.5-Math-1.5B** 模型，在 **MATH500** 和 **AMC** 数据集上进行验证。

### 1. 推理阶段 (Test-Time Scaling)

在仅进行推理筛选的情况下，SASC 策略显著优于传统的多数投票（Majority Vote）基线 。

| 方法 | 准确率 (MATH500) | 提升幅度 |
| --- | --- | --- |
| Baseline: Majority Vote | 68.8% | - |
| **SASC (Filter 20% + Weighted Vote)** | **72.0%** | **+3.2%** |

### 2. 训练阶段 (TTRL Performance)

将生成的偏好数据用于 GRPO 训练，SASC-TTRL 在不同算力预算下（Pass@1 ~ Pass@64）均取得了最优效果，并有效防止了模型崩塌 。

**MATH500 数据集 (Pass@k)** 

| 模型 | Pass@1 | Pass@4 | Pass@64 |
| --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | 45.79% | 75.09% | 93.40% |
| TTRL (Baseline) | 69.31% | 81.90% | 93.80% |
| **SASC-TTRL (Ours)** | **70.3%** | **82.2%** | **94.2%** |

**AMC 数据集 (Pass@k)** 

| 模型 | Pass@1 | Pass@4 | Pass@64 |
| --- | --- | --- | --- |
| Qwen2.5-Math-1.5B | 45.7% | 75.0% | 93.4% |
| TTRL (Baseline) | 47.35% | 61.53% | 81.93% |
| **SASC-TTRL (Ours)** | **48.66%** | **63.68%** | **85.54%** |


