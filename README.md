# 📚 Auto Paper Pipeline

> **AI驱动的论文研究自动化平台** — 从发现到复现，全流程智能化

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00d2ff.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-39%20Passed-brightgreen.svg)](tests/)
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
| 📊 深度分析 | 结构化摘要 + 可复现性评估 | Pydantic输出 |
| 🧪 实验复现 | 30篇论文完整复现 | NumPy/PyTorch |
| 📧 邮件简报 | HTML美化 + PDF附件 | SMTP + reportlab |
| 🌐 Web可视化 | 静态站点 + API | FastAPI + Pagefind |
| 🔌 MCP接入 | Claude/Cursor直接查询 | MCP Server |
| 🧪 自动测试 | 39个测试用例 | pytest |

---

## 🗂️ 覆盖领域

```
┌─────────────────────────────────────────────────────────────┐
│                    6大研究领域 · 30篇论文                      │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  AI Agent    │ LLM推理优化  │  多模态大模型 │   代码生成    │
│  智能体/协作  │  加速/量化   │  视觉语言    │   AI编程     │
├──────────────┼──────────────┼──────────────┼───────────────┤
│  芯片验证    │  5G移动通信  │              │               │
│  形式验证    │  信道建模    │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/infanwang/auto-paper-pipeline.git
cd auto-paper-pipeline
pip install pyyaml numpy reportlab
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

# 仅爬取论文
python src/collectors/arxiv.py

# 运行测试
python -m pytest tests/ -v
```

---

## 📁 项目结构

```
auto-paper-pipeline/
├── 📄 .env.example              # 环境变量模板
├── 📄 pyproject.toml            # 依赖管理
├── 📄 docker-compose.yml        # Docker部署
│
├── 📂 src/                      # 核心源码
│   ├── core/                    # 核心基类
│   │   └── models.py           # SQLAlchemy数据模型
│   ├── collectors/              # 多源采集
│   │   └── arxiv.py            # ArXiv + Semantic Scholar
│   ├── pipeline/                # 过滤漏斗
│   │   └── funnel.py           # TF-IDF + LLM评分
│   ├── analyzers/               # 深度分析
│   │   ├── paper_analyzer.py   # 结构化分析
│   │   └── doc_generator.py    # PDF/Word生成
│   ├── web/                     # Web可视化
│   │   ├── app.py              # FastAPI接口
│   │   └── static_generator.py # 静态站点
│   ├── notifiers/               # 多渠道推送
│   │   └── email.py            # 邮件通知
│   └── mcp_server/              # MCP接入
│       └── server.py           # Claude/Cursor工具
│
├── 📂 scripts/                  # 脚本
│   └── run_pipeline.py         # 主调度器
│
├── 📂 tests/                    # 测试套件
│   ├── test_collectors.py      # 采集器测试
│   ├── test_pipeline.py        # 管道测试
│   ├── test_analyzers.py       # 分析器测试
│   ├── test_web.py             # Web测试
│   └── test_notifiers.py       # 通知器测试
│
├── 📂 data/                     # 数据存储
├── 📂 docs/                     # 静态站点
├── 📂 reports/                  # 生成的报告
└── 📂 pdfs/                     # 下载的PDF
```

---

## 🔌 MCP Server 接入

### 在 Claude Desktop 中使用

1. 安装MCP SDK:
```bash
pip install mcp
```

2. 在 Claude Desktop 配置中添加:
```json
{
  "mcpServers": {
    "paper-pipeline": {
      "command": "python",
      "args": ["/path/to/auto-paper-pipeline/src/mcp_server/server.py"]
    }
  }
}
```

3. 在 Claude 中使用:
```
帮我搜索关于MoE推理优化的最新论文
```

### 可用工具

| 工具 | 描述 | 示例 |
|------|------|------|
| `search_papers` | 搜索论文 | `搜索LLM agent相关论文` |
| `get_paper_detail` | 获取论文详情 | `查看2607.16193的详细信息` |
| `get_topic_stats` | 获取领域统计 | `各领域有多少论文？` |
| `get_top_papers` | 获取高分论文 | `评分最高的10篇论文` |
| `compare_papers` | 对比论文 | `对比2607.16193和2607.16189` |

### 命令行使用

```bash
# 列出所有工具
python src/mcp_server/server.py tools

# 搜索论文
python src/mcp_server/server.py search --query "agent" --limit 5

# 获取论文详情
python src/mcp_server/server.py detail --paper-id 2607.16193

# 获取领域统计
python src/mcp_server/server.py stats

# 获取高分论文
python src/mcp_server/server.py top --limit 10
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

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_pipeline.py -v

# 查看测试覆盖率
python -m pytest tests/ --cov=src
```

**测试覆盖**: 39个测试用例，覆盖采集器/管道/分析器/Web/通知器

---

## 🐳 Docker部署

```bash
# 一键启动
docker-compose up -d

# 包含服务:
# - app: 主应用（爬取+分析）
# - web: Web API（端口8000）
```

---

## 📧 邮件简报

每日自动发送HTML美化简报，包含:
- 今日新论文列表
- Top 5复现候选
- 引用数据排名
- PDF下载链接

```bash
# 手动发送
export SMTP_HOST="smtp.163.com"
export SMTP_SENDER="your@email.com"
export SMTP_AUTH_CODE="your_auth_code"
python scripts/run_pipeline.py --mode daily
```

---

## 📈 路线图

- [x] 多源论文采集
- [x] 多级过滤漏斗
- [x] 深度分析引擎
- [x] 静态站点生成
- [x] PDF报告生成
- [x] MCP Server接入
- [x] 自动测试框架
- [ ] Web UI管理后台
- [ ] 知识图谱构建
- [ ] 论文影响力预测
- [ ] 多语言支持
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

MIT License - 详见 [LICENSE](LICENSE)

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
