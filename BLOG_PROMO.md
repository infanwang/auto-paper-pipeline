# 🚀 Auto Paper Pipeline: 开源AI论文研究平台全景解读

> 从ArXiv爬虫到生产级Agent，一个开源项目的完整进化之路

---

## 一句话介绍

**Auto Paper Pipeline** 是一个AI驱动的论文研究自动化平台，覆盖6个研究领域，119篇论文从发现到复现全流程智能化。

---

## 为什么需要这个项目？

### 痛点

```
每天100+ AI论文
    ↓
哪些重要？
    ↓
为什么重要？
    ↓
有没有代码？
    ↓
怎么跑？
    ↓
是否超过baseline？
```

### 解决方案

```
Auto Paper Pipeline

用户: "搜索关于MoE推理优化的最新论文"
    ↓
自动:
- 找论文 (ArXiv + Semantic Scholar)
- 评分 (TF-IDF + LLM)
- 摘要 (AI结构化)
- 分类 (6个领域)
- 报告 (PDF + HTML)
```

---

## 核心架构

### 多级过滤漏斗

```
100+ 篇论文
    ↓ Stage 1: TF-IDF (零成本)
20 篇论文
    ↓ Stage 2: LLM评分 (低成本)
Top 10 精选论文
    ↓ Stage 3: 深度分析
结构化摘要 + 可复现性评估
```

### AI摘要引擎

```json
{
  "core_problem": "优化LLM推理的KV缓存内存占用",
  "innovation": "压缩和组合KV缓存重用",
  "conclusion": "性能提升40%",
  "limitation": "降低计算成本"
}
```

---

## 技术亮点

### 1. 多源聚合
- ArXiv API: 论文元数据
- Semantic Scholar: 引用数据
- 自动去重 + 限流控制

### 2. 智能评分
- TF-IDF快速过滤 (零成本)
- LLM深度评分 (低成本)
- 综合考虑新颖性、性能、引用

### 3. MCP Server
```bash
# 在Claude/Cursor中直接查询
python src/mcp_server/server.py search --query "LLM agent"
```

### 4. 生产级部署
```bash
# 一键启动
docker compose up -d

# 自动健康检查
# 自愈重启机制
# 定时任务调度
```

---

## 实验成果

### 30篇论文复现

| 领域 | 最佳结果 |
|------|---------|
| AI Agent | mIoU 0.309 (3x提升) |
| LLM推理 | 73%开销降低 |
| 多模态 | 44.9%提升 |
| 代码生成 | 100%覆盖率 |
| 芯片验证 | 全部匹配 |
| 5G通信 | 77.4%缩减 |

### 8个改进实现

| 改进 | 效果 |
|------|------|
| PagedWeight自适应量化 | 内存节省20-23% |
| FVAttn动态负载均衡 | 不平衡降低33% |
| DPNeXt知识蒸馏 | 181x参数压缩 |
| CLIFE Transformer融合 | 1.4x推理加速 |

---

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/infanwang/auto-paper-pipeline.git
cd auto-paper-pipeline

# 2. 配置
cp .env.example .env
# 编辑 .env

# 3. 启动
docker compose up -d

# 4. 访问
# 搜索: http://localhost:8000/search
# 仪表盘: http://localhost:8000/dashboard
# API: http://localhost:8000/docs
```

---

## 项目统计

| 指标 | 数量 |
|------|------|
| Python文件 | 40+ |
| 测试用例 | 41 |
| 论文数据 | 119篇 |
| 复现论文 | 30篇 |
| 改进实现 | 8个 |
| Docker服务 | 3个 |

---

## 未来规划

- [ ] 知识图谱构建
- [ ] 论文影响力预测
- [ ] 多语言支持
- [ ] K8s生产部署
- [ ] 自动论文复现

---

## 致谢

- [ArXiv](https://arxiv.org/) - 论文数据源
- [Semantic Scholar](https://www.semanticscholar.org/) - 引用数据
- [MCP](https://modelcontextprotocol.io/) - 工具接入协议
- [MiMo Code](https://mimo.xiaomi.com/) - AI编程助手

---

**GitHub**: https://github.com/infanwang/auto-paper-pipeline

**如果这个项目对你有帮助，请给个⭐支持一下！**

Made with ❤️ by [infanwang](https://github.com/infanwang)
