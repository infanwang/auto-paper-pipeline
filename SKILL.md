# Skill: Auto Paper Pipeline

> AI驱动的论文研究自动化平台（个人自用）

## 触发条件

- "论文搜索" / "paper search"
- "AI摘要" / "paper summary"
- "论文复现" / "paper reproduction"
- "技术简报" / "tech briefing"
- "论文邮件" / "paper email"

## 核心功能

### 1. 论文搜索
```bash
curl "http://localhost:8000/api/papers/search?q=agent&limit=10"
```

### 2. AI摘要
```bash
curl "http://localhost:8000/api/paper/2607.16193"
```

### 3. 领域统计
```bash
curl "http://localhost:8000/api/stats"
```

### 4. 运行管道
```bash
python scripts/run_pipeline.py --mode daily
```

## MCP Server

配置 `~/.mimocode/mcp.json`:
```json
{
  "mcpServers": {
    "paper-pipeline": {
      "command": "python3",
      "args": ["/root/git/mimo/paper-pipeline/src/mcp_server/server.py"]
    }
  }
}
```

## 示例

用户: "搜索关于MoE推理优化的最新论文"
响应: 自动搜索并返回相关论文列表

用户: "查看2607.16193的详细信息"
响应: 返回论文详情+AI摘要
