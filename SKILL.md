# Skill: Auto Paper Pipeline

> AI驱动的论文研究自动化平台 — 从发现到复现，全流程智能化

## 触发条件

当用户提到以下关键词时激活此技能：
- "论文爬取" / "paper crawl"
- "论文搜索" / "paper search"
- "AI摘要" / "paper summary"
- "论文复现" / "paper reproduction"
- "ArXiv搜索" / "arxiv search"
- "技术简报" / "tech briefing"
- "论文邮件" / "paper email"
- "RAG" / "检索增强"

## 功能概述

### 1. 智能论文搜索

**命令**：
```bash
# API搜索
curl "http://localhost:8000/api/papers/search?q=LLM&limit=10"

# MCP工具
python src/mcp_server/server.py search --query "agent" --limit 5
```

**功能**：
- 关键词搜索
- 领域筛选（6个领域）
- 评分过滤
- AI语义搜索

### 2. AI摘要生成

**命令**：
```bash
# API获取详情（自动生成AI摘要）
curl "http://localhost:8000/api/paper/2607.16193"
```

**功能**：
- 结构化输出（核心问题/创新方法/实验结论/局限展望）
- 防滥用截断（500字符）
- 缓存机制

### 3. 论文复现

**命令**：
```bash
# 复现指定论文
python scripts/reproducer.py --paper <arxiv_id>

# 运行完整管道
python scripts/run_pipeline.py --mode daily
```

### 4. 邮件简报

**命令**：
```bash
# 手动发送
python scripts/email_briefing.py
```

## 核心算法

### 多级过滤漏斗
```
Stage 1: TF-IDF快速过滤 → 过滤60%噪音
Stage 2: LLM深度评分 → 精准筛选Top 20
```

### AI摘要生成
```
输入: 标题 + 摘要(截断500字)
输出: JSON格式结构化摘要
缓存: 避免重复调用
```

## MCP Server

### 可用工具
- `search_papers` — 搜索论文
- `get_paper_detail` — 获取详情
- `get_topic_stats` — 领域统计
- `get_top_papers` — 高分论文
- `compare_papers` — 对比论文

### 配置
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

## 示例对话

**用户**：搜索关于MoE推理优化的最新论文

**响应**：
```
正在搜索MoE推理优化论文...
找到12篇论文，按评分排序：
1. PagedWeight (7.8分) - 动态per-expert量化
2. C²KV (6.3分) - KV缓存重用
...
邮件已发送至 cl0udp1k@163.com
```

**用户**：查看2607.16193的详细信息

**响应**：
```
论文: UAV-DualCog: Dual-Cognition Benchmark
评分: 7.5
标签: Agent, 多模态
AI摘要:
  核心问题: 提升无人机双认知能力
  创新方法: 双认知基准测试
```

**用户**：帮我复现FlashAttention-2

**响应**：
```
正在复现FlashAttention-2...
论文标题: FlashAttention-2: Faster Attention
实验结果: 开销降低73%
复现代码: reproduction/llm_inference/
```

## 注意事项

1. **邮箱配置**：需要在.env中配置SMTP授权码
2. **API限制**：ArXiv API有频率限制，建议间隔3秒
3. **缓存**：AI摘要结果会缓存，重复请求直接返回
4. **Docker**：生产环境建议使用Docker部署

## 相关资源

- [GitHub仓库](https://github.com/infanwang/auto-paper-pipeline)
- [设计文档](DESIGN.md)
- [使用说明](USAGE.md)
- [推广博文](BLOG_V2.md)
