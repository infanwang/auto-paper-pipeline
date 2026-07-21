# Skill: Auto Paper Pipeline

> 自动化论文爬取、分析、复现系统 — 让论文研究全流程自动化

## 触发条件

当用户提到以下关键词时激活此技能：
- "论文爬取" / "paper crawl"
- "论文分析" / "paper analysis"
- "论文复现" / "paper reproduction"
- "ArXiv搜索" / "arxiv search"
- "技术简报" / "tech briefing"
- "论文邮件" / "paper email"

## 功能概述

### 1. 智能论文爬取

**命令**：
```bash
python scripts/enhanced_crawler.py
```

**功能**：
- 基于ArXiv API自动搜索最新论文
- Semantic Scholar API获取引用数据
- 按重要性排序
- 下载Top论文PDF并分类存储

**配置**：编辑 `config.yaml`
```yaml
topics:
  - name: "AI Agent"
    queries: ["LLM agent tool use", "multi-agent collaboration"]
    categories: ["cs.AI", "cs.CL"]
```

### 2. 实验复现

**命令**：
```bash
python scripts/reproducer.py --paper <arxiv_id>
```

**功能**：
- 自动生成复现指南
- 实现核心算法
- 运行实验并记录结果
- 与论文报告结果对比

**复现结果位置**：`reproduction/<topic>/experiments/`

### 3. 技术简报

**命令**：
```bash
python scripts/enhanced_scheduler.py daily
```

**功能**：
- 生成每日技术简报
- HTML美化格式
- 附带PDF附件
- 自动发送邮件

### 4. 实验报告

**命令**：
```bash
python reproduction/md_to_pdf_v2.py
```

**功能**：
- 将实验结果转换为PDF
- 支持中文显示
- 表格和格式美化

## 工作流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. 爬取     │───▶│  2. 分析     │───▶│  3. 复现     │───▶│  4. 报告     │
│  ArXiv API  │    │  引用数据    │    │  实验代码    │    │  PDF/邮件    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 论文数据 | `data/YYYY-MM-DD/` | JSON格式论文信息 |
| PDF文件 | `pdfs/<topic>/` | 按领域分类的PDF |
| 复现代码 | `reproduction/<topic>/` | Python实验脚本 |
| 实验结果 | `reproduction/<topic>/experiments/` | JSON结果文件 |
| 技术简报 | 邮件附件 | HTML格式简报 |
| 实验报告 | `reproduction/COMPLETE_REPORT.pdf` | 完整PDF报告 |

## 配置说明

### config.yaml 主要配置项

```yaml
# 研究领域
topics:
  - name: "领域名称"
    queries: ["搜索关键词"]
    categories: ["arxiv分类"]

# 时间范围
time_range: "2026-01-01"

# Top N复现
top_n: 10

# 邮件配置
email:
  smtp_host: "smtp.163.com"
  sender: "发件人邮箱"
  auth_code: "SMTP授权码"
  recipient: "收件人邮箱"

# 定时任务
daily_update_hour: 8
weekly_backup_day: 1
```

## 依赖

```
pyyaml
numpy
reportlab  # PDF生成
```

## 示例对话

**用户**：帮我搜索最新的AI Agent论文

**响应**：
```
正在搜索AI Agent领域最新论文...
找到31篇论文，按引用数排序：
1. IoT for Smart Manufacturing (385 citations)
2. UAV-DualCog (0 citations, 2026-07-17)
...
PDF已下载到 pdfs/AI_Agent/
```

**用户**：复现FVAttn这篇论文

**响应**：
```
正在复现FVAttn (2607.16190)...
论文标题：Adaptive Sparse Attention with Runtime Load Balancing
实验结果：开销降低73% (2.6ms→0.7ms)
与论文对比：✅ 完美匹配
复现代码：reproduction/llm_inference/experiments/fvattn/
```

**用户**：发送今日技术简报

**响应**：
```
正在生成技术简报...
今日新论文：15篇
Top 5复现候选：FVAttn, VTLoc, ADA-ST...
邮件已发送至 cl0udp1k@163.com
```

## 注意事项

1. **邮箱配置**：需要在config.yaml中配置SMTP授权码
2. **API限制**：ArXiv API有频率限制，建议间隔3秒
3. **PDF存储**：大量PDF会占用磁盘空间，建议定期清理
4. **复现环境**：部分论文需要GPU或特定数据集

## 相关资源

- [ArXiv API文档](https://info.arxiv.org/help/api/index.html)
- [Semantic Scholar API](https://api.semanticscholar.org/)
- [项目GitHub](https://github.com/infanwang/auto-paper-pipeline)
