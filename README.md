# 📚 Auto Paper Pipeline

> AI驱动的论文研究自动化平台 — 从发现到复现，全流程智能化（个人自用）

---

## 🎯 这是什么？

Auto Paper Pipeline 是一个**个人使用的AI论文研究平台**，覆盖6个研究领域，112篇论文从爬取到复现全流程自动化。

```
用户查询 → 智能搜索 → 多级过滤 → 深度分析 → 实验复现 → 知识积累
```

---

## ✨ 核心功能

| 功能 | 描述 | 技术 |
|------|------|------|
| 🔍 智能搜索 | 关键词+领域筛选 | ArXiv API |
| 🤖 AI摘要 | 结构化输出 | 规则引擎+缓存 |
| 📊 影响力预测 | 多特征评分 | 加权算法 |
| 🌐 多语言 | 中英文支持 | 关键词映射 |
| 🧪 实验复现 | 30篇论文 | NumPy/PyTorch |
| 📧 邮件简报 | 每日自动 | SMTP |

---

## 🚀 快速开始

### Windows
```cmd
cd C:\Users\infanwang\Desktop\auto-paper-pipeline
pip install --break-system-packages pyyaml numpy reportlab mcp pytest fastapi uvicorn openpyxl
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

### Linux
```bash
cd /root/git/mimo/paper-pipeline
docker compose up -d
```

### 访问
- 主页: http://localhost:8000
- 搜索: http://localhost:8000/search
- 仪表盘: http://localhost:8000/dashboard

---

## 📁 项目结构

```
auto-paper-pipeline/
├── src/           # 核心模块 (16个)
├── tests/         # 测试 (41个)
├── scripts/       # 脚本
├── docs/          # 静态站点
├── reproduction/  # 论文复现 (30篇)
├── data/          # 论文数据 (112篇)
├── pdfs/          # PDF文件 (118篇)
└── reports/       # 报告
```

---

## 🧪 测试

```bash
python -m pytest tests/ -v
# 结果: 41 passed
```

---

*Auto Paper Pipeline V2.0 — 个人自用*
