# 📚 Auto Paper Pipeline

> **AI驱动的论文研究自动化平台** — 从发现到复现，全流程智能化

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00d2ff.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-41%20Passed-brightgreen.svg)](tests/)
[![MCP](https://img.shields.io/badge/MCP-Server-blueviolet.svg)](src/mcp_server/)

---

## 🎯 这是什么？

Auto Paper Pipeline 是一个**生产级AI论文研究平台**，覆盖6个研究领域，实现从论文发现到实验复现的全流程自动化。

```
用户查询 → 智能搜索 → 多级过滤 → 深度分析 → 实验复现 → 知识积累
```

---

## ✨ 核心特性

| 特性 | 描述 | 技术实现 |
|------|------|---------|
| 🔍 多源聚合 | ArXiv + Semantic Scholar | 异步采集 + 限流控制 |
| 🎯 多级漏斗 | TF-IDF初筛 → LLM评分 | 成本降低80% |
| 🤖 AI摘要 | 结构化输出 + 防滥用 | 规则引擎 + 缓存 |
| 📊 深度分析 | 问题/方法/结论/局限 | 智能提取 |
| 🧪 实验复现 | 30篇论文完整复现 | NumPy/PyTorch |
| 📧 邮件简报 | HTML美化 + PDF附件 | SMTP + reportlab |
| 🌐 Web可视化 | 搜索页 + Dashboard | FastAPI + Chart.js |
| 🔌 MCP接入 | Claude/Cursor直接查询 | MCP Server |
| 🧪 自动测试 | 41个测试用例 | pytest |
| 🛡️ 自愈机制 | 健康检查 + 自动重启 | Docker + monitor |

---

## 🗂️ 覆盖领域

```
┌─────────────────────────────────────────────────────────────┐
│                    6大研究领域 · 119篇论文                    │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  AI Agent    │ LLM推理优化  │  多模态大模型 │   代码生成    │
│  30篇        │  22篇        │  12篇        │  13篇         │
├──────────────┼──────────────┼──────────────┼───────────────┤
│  芯片验证    │  5G移动通信  │              │               │
│  24篇        │  18篇        │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/infanwang/auto-paper-pipeline.git
cd auto-paper-pipeline
pip install pyyaml numpy reportlab mcp pytest
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env 设置邮箱等配置
```

### 3. 运行

```bash
# 完整管道
python scripts/run_pipeline.py --mode daily

# 运行测试
python -m pytest tests/ -v
```

### 4. Docker部署

```bash
docker compose up -d
# 访问 http://localhost:8000
```

---

## 📁 项目结构

```
auto-paper-pipeline/
├── 📄 DESIGN.md              # 设计文档
├── 📄 USAGE.md               # 使用说明
├── 📄 BLOG_V2.md             # 推广博文
│
├── 📂 src/                   # 核心源码
│   ├── core/                 # 核心模块
│   │   ├── models.py        # 数据模型
│   │   └── ai_summary.py    # AI摘要生成
│   ├── collectors/           # 多源采集
│   │   └── arxiv.py         # ArXiv + Semantic Scholar
│   ├── pipeline/             # 过滤漏斗
│   │   └── funnel.py        # TF-IDF + LLM评分
│   ├── analyzers/            # 深度分析
│   │   ├── paper_analyzer.py
│   │   └── doc_generator.py
│   ├── web/                  # Web可视化
│   │   ├── app.py           # FastAPI接口
│   │   ├── static_generator.py
│   │   └── smart_center.py  # 科研智能中枢
│   ├── notifiers/            # 邮件通知
│   │   └── email.py
│   └── mcp_server/           # MCP接入
│       └── server.py
│
├── 📂 scripts/               # 脚本
│   ├── run_pipeline.py       # 主调度器
│   ├── health_check.py       # 健康检查
│   ├── monitor.sh            # 自愈监控
│   └── deploy.sh             # 部署脚本
│
├── 📂 tests/                 # 测试套件 (41个)
├── 📂 data/                  # 数据存储
├── 📂 docs/                  # 静态站点
├── 📂 reports/               # 生成的报告
└── 📂 pdfs/                  # 下载的PDF
```

---

## 🔌 MCP Server 接入

### 在 MiMo/Cursor 中使用

配置 `~/.mimocode/mcp.json`：

```json
{
  "mcpServers": {
    "paper-pipeline": {
      "command": "python3",
      "args": ["/path/to/src/mcp_server/server.py"]
    }
  }
}
```

### 可用工具

| 工具 | 描述 | 示例 |
|------|------|------|
| `search_papers` | 搜索论文 | `搜索LLM agent相关论文` |
| `get_paper_detail` | 获取详情 | `查看2607.16193详细信息` |
| `get_topic_stats` | 领域统计 | `各领域论文数量` |
| `get_top_papers` | 高分论文 | `评分最高的10篇` |
| `compare_papers` | 对比论文 | `对比两篇论文` |

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 测试结果: 41 passed
```

---

## 📊 实验成果

### 30篇论文复现完成

| 领域 | 完成 | 最佳结果 |
|------|------|---------|
| AI Agent | 5/5 | mIoU 0.309 (3x提升) |
| LLM推理优化 | 5/5 | 73%开销降低 |
| 多模态大模型 | 5/5 | 44.9%提升 |
| 代码生成 | 5/5 | 100%覆盖率 |
| 芯片验证 | 5/5 | 全部匹配 |
| 5G移动通信 | 5/5 | 77.4%缩减 |

---

## 🐳 Docker部署

```bash
# 一键启动
docker compose up -d

# 包含服务:
# - app: 主应用（爬取+分析）
# - web: Web API（端口8000）
# - scheduler: 定时任务
```

---

## 📈 路线图

- [x] 多源论文采集
- [x] 多级过滤漏斗
- [x] AI摘要生成
- [x] Web API + Dashboard
- [x] MCP Server接入
- [x] Docker部署
- [x] 自动测试框架
- [x] 自愈机制
- [ ] 知识图谱构建
- [ ] 论文影响力预测
- [ ] K8s生产部署

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

```bash
# Fork项目
# 创建特性分支
git checkout -b feature/amazing-feature

# 运行测试
python -m pytest tests/ -v

# 提交更改
git commit -m 'Add amazing feature'

# 推送分支
git push origin feature/amazing-feature
```

---

## 📄 License

MIT License

---

## 🙏 致谢

- [ArXiv](https://arxiv.org/) - 论文数据源
- [Semantic Scholar](https://www.semanticscholar.org/) - 引用数据
- [MCP](https://modelcontextprotocol.io/) - 工具接入协议
- [MiMo Code](https://mimo.xiaomi.com/) - AI编程助手

---

<div align="center">

**如果这个项目对你有帮助，请给个⭐支持一下！**

Made with ❤️ by [infanwang](https://github.com/infanwang)

</div>
