# 📚 Auto Paper Pipeline

> **让论文研究自动化** — 从爬取到复现，全流程AI驱动

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00d2ff.svg)](LICENSE)
[![ArXiv](https://img.shields.io/badge/ArXiv-2026-b31b1b.svg)](https://arxiv.org)

---

## 🎯 这是什么？

Auto Paper Pipeline 是一个**全自动化的论文研究系统**，覆盖6个热门研究领域，30篇论文从爬取到复现全流程自动化。

**解决的痛点**：
- ❌ 手动搜索ArXiv，错过重要论文
- ❌ 下载PDF后混乱分类
- ❌ 阅读论文后难以复现实验
- ❌ 整理技术报告耗时耗力

**Auto Paper Pipeline**：
- ✅ 每日自动爬取最新论文
- ✅ 智能排序+PDF分类存储
- ✅ 自动生成复现指南+实验代码
- ✅ 每日发送HTML美化简报

---

## ✨ 核心特性

| 特性 | 描述 | 技术实现 |
|------|------|---------|
| 🔍 智能爬取 | 多关键词、多类别搜索 | ArXiv API + Semantic Scholar |
| 📊 引用分析 | 获取引用数、AI摘要 | Semantic Scholar API |
| 📥 PDF管理 | 自动下载+按领域分类 | Python + os |
| 🧪 实验复现 | 30篇论文完整复现 | NumPy/PyTorch |
| 📧 邮件简报 | HTML美化+附件 | SMTP + reportlab |
| ⏰ 定时任务 | 每日/每周自动执行 | cron |

---

## 🗂️ 覆盖领域

```
┌─────────────────────────────────────────────────────────┐
│                    6大研究领域                            │
├──────────────┬──────────────┬──────────────┬────────────┤
│  AI Agent    │ LLM推理优化  │  多模态大模型 │  代码生成  │
│  5篇论文     │  5篇论文     │  5篇论文     │  5篇论文   │
│  智能体/协作  │  加速/量化   │  视觉语言    │  AI编程    │
├──────────────┼──────────────┼──────────────┼────────────┤
│  芯片验证    │  5G移动通信  │              │            │
│  5篇论文     │  5篇论文     │              │            │
│  形式验证    │  信道建模    │              │            │
└──────────────┴──────────────┴──────────────┴────────────┘
```

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/infanwang/auto-paper-pipeline.git
cd auto-paper-pipeline
pip install pyyaml numpy
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
# 研究领域
topics:
  - name: "AI Agent"
    queries: ["LLM agent tool use", "multi-agent collaboration"]
    categories: ["cs.AI", "cs.CL"]

# 邮件配置
email:
  smtp_host: "smtp.163.com"
  sender: "your@email.com"
  recipient: "your@email.com"
```

### 3. 运行

```bash
# 完整管道：爬取 + 分析 + 简报
python scripts/enhanced_scheduler.py daily

# 仅爬取论文
python scripts/enhanced_crawler.py

# 仅生成报告
python scripts/enhanced_report.py
```

---

## 📁 项目结构

```
auto-paper-pipeline/
├── 📄 config.yaml                 # 配置文件
├── 📄 README.md                   # 项目说明
│
├── 📂 scripts/                    # 核心脚本
│   ├── arxiv.py                   # ArXiv API封装
│   ├── enhanced_crawler.py        # 增强爬取（含引用数据）
│   ├── enhanced_report.py         # 报告生成器
│   ├── enhanced_scheduler.py      # 主调度器
│   └── reproducer.py              # 复现指南生成
│
├── 📂 data/                       # 论文数据
│   └── YYYY-MM-DD/
│       ├── papers.json            # 基础数据
│       └── papers_enhanced.json   # 增强数据（含引用）
│
├── 📂 pdfs/                       # PDF存储（按领域分类）
│   ├── AI_Agent/
│   ├── LLM推理优化/
│   ├── 多模态大模型/
│   ├── 代码生成/
│   ├── 芯片验证/
│   └── 5G移动通信/
│
├── 📂 reproduction/               # 论文复现
│   ├── ai_agent/                  # AI Agent复现
│   ├── llm_inference/             # LLM推理复现
│   ├── multimodal/                # 多模态复现
│   ├── code_gen/                  # 代码生成复现
│   ├── chip_verify/               # 芯片验证复现
│   ├── 5g_comm/                   # 5G通信复现
│   └── improvements/              # 改进实现
│
└── 📄 COMPLETE_REPORT.pdf         # 完整实验报告
```

---

## 📊 实验成果

### 30篇论文全部复现完成

| 领域 | 完成 | 最佳结果 | 亮点 |
|------|------|---------|------|
| AI Agent | 5/5 ✅ | mIoU 0.309 | 时序树搜索3倍提升 |
| LLM推理优化 | 5/5 ✅ | 73%开销降低 | FVAttn稀疏注意力 |
| 多模态大模型 | 5/5 ✅ | 44.9%提升 | VTLoc视觉触觉融合 |
| 代码生成 | 5/5 ✅ | 100%覆盖率 | ADA-ST自适应故障注入 |
| 芯片验证 | 5/5 ✅ | 全部匹配 | 3DGS图处理器渲染 |
| 5G移动通信 | 5/5 ✅ | 77.4%缩减 | DPNexT轻量化架构 |

### 8个改进实现

| 改进 | 原论文 | 效果 |
|------|--------|------|
| PagedWeight自适应量化 | 2607.16184 | 内存节省20-23% |
| FVAttn动态负载均衡 | 2607.16190 | 不平衡降低33% |
| DPNeXt知识蒸馏 | 2607.16102 | 181x参数压缩 |
| CLIFE Transformer融合 | 2607.16154 | 1.4x推理加速 |
| ADA-ST智能故障注入 | 2607.16161 | 故障减少1.38x |
| ECC ML预测 | 2607.16042 | 自适应ECC选择 |
| DoSQ DL检测器 | 2607.16102 | 96%检测率 |
| DPNeXt NAS搜索 | 2607.16102 | 频段自适应+ONNX |

---

## ⏰ 自动化任务

| 任务 | 时间 | 内容 |
|------|------|------|
| 每日论文爬取 | 每天 08:00 | 搜索 → 下载 → 分析 → 邮件简报 |
| 每周数据备份 | 每周一 02:00 | 备份数据 → 生成周报 → 邮件通知 |

---

## 📧 邮件简报示例

每日自动发送HTML美化简报，包含：

```html
📚 AI论文日报 2026-07-21
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 今日摘要
• 总计: 154 篇论文 | AI Agent: 31 | LLM推理: 36 | ...

🔥 Top 5 候选复现论文
1. [98分] IoT for Smart Manufacturing
2. [38分] UAV-DualCog: Dual-Cognition Benchmark
...

📥 PDF下载: 10篇论文已下载
📊 引用数据: Semantic Scholar实时更新
```

---

## 🛠️ 扩展指南

### 添加新领域

编辑 `config.yaml`：

```yaml
topics:
  - name: "新领域"
    queries: ["关键词1", "关键词2"]
    categories: ["cs.XX"]
```

### 自定义邮件模板

编辑 `scripts/enhanced_report.py` 中的HTML模板。

### 集成其他API

在 `scripts/enhanced_crawler.py` 中添加新的数据源。

---

## 📈 路线图

- [ ] 支持更多研究领域
- [ ] 集成LLM自动论文解读
- [ ] 添加论文影响力预测
- [ ] 开发Web仪表盘
- [ ] 支持多语言论文
- [ ] 添加论文对比分析

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

```bash
# Fork项目
# 创建特性分支
git checkout -b feature/amazing-feature

# 提交更改
git commit -m 'Add amazing feature'

# 推送分支
git push origin feature/amazing-feature

# 创建Pull Request
```

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [ArXiv](https://arxiv.org/) - 论文数据源
- [Semantic Scholar](https://www.semanticscholar.org/) - 引用数据
- [MiMo Code](https://mimo.xiaomi.com/) - AI编程助手

---

<div align="center">

**如果这个项目对你有帮助，请给个⭐支持一下！**

Made with ❤️ by [infanwang](https://github.com/infanwang)

</div>
