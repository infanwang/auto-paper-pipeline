# 🏆 评分最高的5篇论文 - 详细技术总结

## #1 [7.8分] Patch Policy: Efficient Embodied Control via Dense Visual Representations

**主题**: LLM推理优化 | **标签**: Vision Transformer, Robot Learning

**核心问题**:
现有机器人策略要么将每个观测压缩为单个全局token，要么从头训练视觉骨干网络，导致空间细节丢失和预训练优势浪费。

**创新方法**:
- 提出Patch Policy架构，直接使用预训练ViT的密集patch特征
- 引入block-causal注意力掩码保持时序因果性
- 轻量级设计，无需完整VLM骨干网络

**实验结论**:
- 在4个仿真和3个真实环境上，相比全局池化方法提升40%
- 超越微调OpenVLA-OFT 18%，参数量仅为其0.7%
- 实现高频率反应式控制

**意义**: 为机器人社区提供了利用视觉表示学习进展的高效pipeline

---

## #2 [7.8分] HOMIE: Human-object Centric Video Personalization via Multimodal Intelligent Enhancement

**主题**: 多模态大模型 | **标签**: Video Generation, Personalization

**核心问题**:
现有主体驱动视频生成方法存在两个关键限制：难以平衡高主体保真度和背景一致性。

**创新方法**:
- 以人为-物体为中心的视频个性化框架
- 多模态智能增强
- 联合建模人-物交互

**实验结论**:
- 在多个基准上达到SOTA
- 同时保持高主体保真度和背景一致性

**意义**: 推动个性化视频生成技术发展

---

## #3 [7.3分] MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning

**主题**: LLM推理优化 | **标签**: RL, Multi-Agent, Compact Models

**核心问题**:
大语言模型推理性能强但训练成本高，特别是对于紧凑模型（≤4B参数）。

**创新方法**:
- MADA-RL后训练框架
- 专门化生成器和评论家角色
- 辩论感知学习信号
- LoRA适配器高效微调

**实验结论**:
- DeepSeek-R1-Distill-Qwen-1.5B准确率从39.9%提升至41.9%
- 使用16倍更少的可训练参数
- 在准确率-参数量Pareto前沿

**意义**: 为紧凑模型提供高效的后训练方案

---

## #4 [7.3分] Sparse Evidence Can Suffice: Agentic Evidence Seeking

**主题**: 芯片验证 | **标签**: Video, Misinformation, Evidence

**核心问题**:
多模态视频虚假信息检测通常将整个视频作为整体处理，但真实虚假信息通常具有稀疏和组合的证据结构。

**创新方法**:
- 智能证据寻求代理
- 选择性检索稀疏但充分的证据
- 减少60%的证据检索量

**实验结论**:
- 达到92%准确率
- 同时减少60%证据检索
- 证明稀疏证据足以检测虚假信息

**意义**: 提高视频虚假信息检测效率

---

## #5 [6.3分] SWE-Pruner Pro: The Coder LLM Already Knows What to Prune

**主题**: AI_Agent | **标签**: Pruning, Context Management

**核心问题**:
代码代理的长上下文剪枝是高效上下文管理的关键技术，但现有方法需要单独的代码分类器。

**创新方法**:
- 发现代理自身编码了表示代码相关性的内部表示
- 无需额外分类器即可进行剪枝
- 利用代理内部表示指导剪枝决策

**实验结论**:
- 证明代理内部表示可用于剪枝
- 简化剪枝流程
- 保持或提升性能

**意义**: 为代码代理提供更高效的上下文管理方案

---

## 📊 总结

| 论文 | 评分 | 核心创新 | 潜在影响 |
|------|------|---------|---------|
| Patch Policy | 7.8 | ViT密集特征用于机器人 | ⭐⭐⭐⭐⭐ |
| HOMIE | 7.8 | 人-物中心视频个性化 | ⭐⭐⭐⭐ |
| MADA-RL | 7.3 | 紧凑模型多Agent RL | ⭐⭐⭐⭐ |
| Sparse Evidence | 7.3 | 稀疏证据检测虚假信息 | ⭐⭐⭐ |
| SWE-Pruner Pro | 6.3 | 代理内部表示剪枝 | ⭐⭐⭐ |
