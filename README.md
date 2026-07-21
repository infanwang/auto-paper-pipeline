# 📚 Auto Paper Pipeline

> 自动化论文爬取、复现、验证系统 — 覆盖6个研究领域，30篇论文全流程自动化

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ 特性

- 🔍 **自动爬取**: 基于ArXiv API自动搜索最新论文
- 📊 **引用数据**: Semantic Scholar API获取引用信息
- 📥 **PDF下载**: Top论文自动下载PDF并分类存储
- 🧪 **实验复现**: 自动生成复现指南和实验代码
- 📧 **邮件简报**: 每日自动发送HTML美化简报
- 📅 **定时任务**: 每日8点自动运行，每周一备份

## 🎯 覆盖领域

| 领域 | 论文数 | 关键方向 |
|------|--------|---------|
| AI Agent | 5 | 智能体、工具调用、多Agent协作 |
| LLM推理优化 | 5 | 推理加速、KV Cache、量化 |
| 多模态大模型 | 5 | 视觉语言模型、跨模态对齐 |
| 代码生成 | 5 | AI编程、代码补全、测试生成 |
| 芯片验证 | 5 | 形式验证、错误检测、电路仿真 |
| 5G移动通信 | 5 | 信道建模、波束赋形、资源分配 |

## 🚀 快速开始

### 安装

```bash
pip install pyyaml numpy
```

### 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml 设置邮箱等配置
```

### 运行

```bash
# 运行完整管道
python scripts/enhanced_scheduler.py daily

# 仅爬取论文
python scripts/enhanced_crawler.py

# 仅发送简报
python scripts/enhanced_scheduler.py daily
```

## 📁 项目结构

```
paper-pipeline/
├── config.yaml              # 配置文件
├── scripts/
│   ├── arxiv.py             # ArXiv API搜索
│   ├── enhanced_crawler.py  # 增强爬取（含引用数据）
│   ├── enhanced_report.py   # 报告生成
│   ├── enhanced_scheduler.py # 主调度器
│   └── reproducer.py        # 复现指南生成
├── data/                    # 论文数据
├── reports/                 # 生成的报告
├── pdfs/                    # 下载的PDF（按领域分类）
│   ├── AI_Agent/
│   ├── LLM推理优化/
│   ├── 多模态大模型/
│   ├── 代码生成/
│   ├── 芯片验证/
│   └── 5G移动通信/
├── reproduction/            # 论文复现
│   ├── ai_agent/
│   ├── llm_inference/
│   ├── multimodal/
│   ├── code_gen/
│   ├── chip_verify/
│   ├── 5g_comm/
│   └── improvements/
└── COMPLETE_REPORT.pdf      # 完整实验报告
```

## 📧 邮件简报

每日自动发送包含以下内容的HTML美化简报：
- 今日新论文列表
- Top 5复现候选
- 引用数据排名
- PDF下载链接

## 🧪 实验复现

系统已复现30篇论文的实验结果：

| 领域 | 完成 | 最佳结果 |
|------|------|---------|
| AI Agent | 5/5 | mIoU 0.309 (3x提升) |
| LLM推理优化 | 5/5 | FVAttn 73%开销降低 |
| 多模态大模型 | 5/5 | VTLoc提升44.9% |
| 代码生成 | 5/5 | ADA-ST 100%覆盖率 |
| 芯片验证 | 5/5 | 全部指标匹配 |
| 5G移动通信 | 5/5 | DPNexT 77.4%参数缩减 |

## ⏰ 定时任务

| 任务 | 时间 | 说明 |
|------|------|------|
| 每日论文爬取 | 每天 8:00 | 自动搜索+下载+发送简报 |
| 每周备份 | 每周一 2:00 | 备份数据+发送周报 |

## 📊 实验报告

完整实验报告见 `COMPLETE_REPORT.pdf`，包含：
- 30篇论文详细实验结果
- 与论文报告结果对比
- 关键发现总结

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 License

MIT License

---
*Built with ❤️ by Auto Paper Pipeline*
