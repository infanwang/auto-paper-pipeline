# Auto Paper Pipeline: 个人AI论文研究助手

## 一句话

个人自用的AI论文研究平台，6个领域112篇论文全流程自动化。

## 解决的问题

```
每天100+ AI论文 → 哪些重要？→ 为什么？→ 有代码？→ 怎么跑？
```

## 核心功能

1. **智能搜索** - 关键词+领域筛选
2. **AI摘要** - 结构化输出(核心问题/方法/结论)
3. **影响力预测** - 多特征评分
4. **多语言** - 中英文支持
5. **邮件简报** - 每日自动发送
6. **实验复现** - 30篇论文代码

## 使用方式

```bash
# Windows
cd C:\Users\infanwang\Desktop\auto-paper-pipeline
python -m uvicorn src.web.app:app --port 8000

# Linux
docker compose up -d
```

## 数据

- 112篇论文，6个领域
- 118篇PDF下载
- 30篇论文复现
- 41个测试用例

## 技术栈

- Python 3.12 + FastAPI
- ArXiv API + Semantic Scholar
- Docker + K8s
- MCP Server

---
*个人自用，不对外发布*
