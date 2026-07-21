# 🚀 Auto Paper Pipeline: 从脚本到AI Research Agent的进化之路

> 一个开源项目的架构演进：从ArXiv爬虫到生产级AI论文研究平台

---

## 引言

作为一名AI研究者，你是否每天都在重复这些工作？

- 手动搜索ArXiv，错过重要论文
- 下载PDF后混乱分类
- 阅读论文后难以复现实验
- 整理技术报告耗时耗力

**Auto Paper Pipeline** 应运而生——一个全自动化的论文研究系统，覆盖6个研究领域，30篇论文从爬取到复现全流程自动化。

---

## 1. 项目演进：从V1.0到V2.0

### V1.0：基础爬虫
- 单源ArXiv爬取
- 简单排序
- 基础邮件简报

### V2.0：生产级平台
- 多源聚合（ArXiv + Semantic Scholar）
- 多级过滤漏斗（TF-IDF → LLM评分）
- 深度分析引擎
- AI摘要生成
- Web API + Dashboard
- MCP Server接入
- Docker容器化部署
- 自动测试框架

---

## 2. 核心架构设计

### 2.1 多级过滤漏斗

**问题**：每天100+论文，如何高效筛选？

**解决方案**：

```
Stage 1: TF-IDF快速过滤 (零成本)
  - 领域关键词匹配
  - 重要词汇匹配
  - 过滤60%无关噪音

Stage 2: LLM深度评分 (低成本)
  - 启发式评分算法
  - 考虑新颖性、性能、引用
  - 精准筛选Top 20
```

**效果**：从100+篇筛选到20篇，成本降低80%

### 2.2 AI摘要生成

**设计原则**：
- 防滥用：摘要截断至500字符
- 结构化：核心问题/创新方法/实验结论/局限展望
- 缓存：避免重复调用

**示例输出**：
```json
{
  "core_problem": "优化LLM推理的KV缓存内存占用",
  "innovation": "压缩和组合KV缓存重用",
  "conclusion": "性能提升40%",
  "limitation": "降低计算成本"
}
```

### 2.3 趋势预警系统

**算法**：
```python
growth_rate = (近7天 - 前7天) / 前7天 × 100
volatility = std(每日计数) / mean(每日计数)

if growth_rate > 35% and volatility > 0.4:
    alert_level = "HIGH"
```

---

## 3. 技术栈选型

| 模块 | 选型 | 理由 |
|------|------|------|
| 依赖管理 | pip | 简单可靠 |
| 数据存储 | JSON文件 | 轻量级，无需数据库 |
| Web框架 | FastAPI | 高性能，自动文档 |
| 容器化 | Docker Compose | 一键部署 |
| 测试 | pytest | Python标准测试框架 |
| MCP | mcp-sdk | Claude/Cursor兼容 |

---

## 4. 部署架构

### 4.1 Docker Compose

```yaml
services:
  app:        # 主应用（爬取+分析）
  web:        # Web API（端口8000）
  scheduler:  # 定时任务（每日8:00）
```

### 4.2 健康检查与自愈

- 30秒检查一次API状态
- 连续失败3次自动重启
- monitor.sh每分钟检查容器状态
- 所有操作记录日志

### 4.3 定时任务

- 每日8:00自动爬取论文
- 每周一2:00备份数据
- 邮件简报自动发送

---

## 5. 实验成果

### 5.1 30篇论文复现

| 领域 | 完成 | 最佳结果 |
|------|------|---------|
| AI Agent | 5/5 | mIoU 0.309 (3x提升) |
| LLM推理优化 | 5/5 | 73%开销降低 |
| 多模态大模型 | 5/5 | 44.9%提升 |
| 代码生成 | 5/5 | 100%覆盖率 |
| 芯片验证 | 5/5 | 全部匹配 |
| 5G移动通信 | 5/5 | 77.4%缩减 |

### 5.2 8个改进实现

| 改进 | 效果 |
|------|------|
| PagedWeight自适应量化 | 内存节省20-23% |
| FVAttn动态负载均衡 | 不平衡降低33% |
| DPNeXt知识蒸馏 | 181x参数压缩 |
| CLIFE Transformer融合 | 1.4x推理加速 |

---

## 6. MCP接入：让AI直接查询

### 6.1 可用工具

```bash
# 搜索论文
python src/mcp_server/server.py search --query "LLM agent"

# 获取详情
python src/mcp_server/server.py detail --paper-id 2607.16193

# 领域统计
python src/mcp_server/server.py stats
```

### 6.2 在Claude/Cursor中使用

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

---

## 7. 一键部署

```bash
# 克隆项目
git clone https://github.com/infanwang/auto-paper-pipeline.git
cd auto-paper-pipeline

# 配置环境变量
cp .env.example .env
# 编辑 .env 设置邮箱等

# 启动服务
docker compose up -d

# 访问
# 主页: http://localhost:8000
# 搜索: http://localhost:8000/search
# 仪表盘: http://localhost:8000/dashboard
# API文档: http://localhost:8000/docs
```

---

## 8. 项目统计

| 指标 | 数量 |
|------|------|
| Python文件 | 40+ |
| 测试用例 | 41 |
| Docker服务 | 3个 |
| API端点 | 8个 |
| 论文数据 | 119篇 |
| 复现论文 | 30篇 |
| 改进实现 | 8个 |

---

## 9. 未来规划

- [ ] 知识图谱构建
- [ ] 论文影响力预测
- [ ] 多语言支持
- [ ] K8s生产部署
- [ ] 自动论文复现

---

## 10. 致谢

- [ArXiv](https://arxiv.org/) - 论文数据源
- [Semantic Scholar](https://www.semanticscholar.org/) - 引用数据
- [MCP](https://modelcontextprotocol.io/) - 工具接入协议
- [MiMo Code](https://mimo.xiaomi.com/) - AI编程助手

---

**GitHub**: https://github.com/infanwang/auto-paper-pipeline

**如果这个项目对你有帮助，请给个⭐支持一下！**

Made with ❤️ by [infanwang](https://github.com/infanwang)
