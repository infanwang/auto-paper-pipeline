# 🚀 Auto Paper Pipeline: 让论文研究自动化

## 引言

作为一名AI研究者，你是否每天都在重复这些工作？

- 手动搜索ArXiv上的最新论文
- 下载PDF并分类整理
- 阅读论文并理解实验设置
- 复现实验结果
- 整理技术报告

**Auto Paper Pipeline** 应运而生——一个全自动化的论文研究系统，覆盖6个研究领域，30篇论文从爬取到复现全流程自动化。

---

## 🎯 核心功能

### 1. 智能论文爬取

基于ArXiv API + Semantic Scholar，自动：
- 搜索最新论文（支持多领域、多关键词）
- 获取引用数据和AI摘要
- 按重要性排序
- 下载Top论文PDF并分类存储

```bash
# 一键爬取6个领域的最新论文
python scripts/enhanced_crawler.py
```

### 2. 自动实验复现

内置30篇论文的复现框架：
- 自动读取论文获取实验设置
- 实现核心算法
- 运行实验并记录结果
- 与论文报告结果对比

### 3. 技术简报邮件

每日自动发送HTML美化简报：
- 今日新论文列表
- Top 5复现候选
- 引用数据排名
- 关键发现总结

### 4. 定时任务

- **每日8:00**：自动爬取+下载+发送简报
- **每周一2:00**：备份数据+发送周报

---

## 📊 覆盖领域

| 领域 | 论文数 | 关键方向 |
|------|--------|---------|
| AI Agent | 5 | 智能体、工具调用、多Agent协作 |
| LLM推理优化 | 5 | 推理加速、KV Cache、量化 |
| 多模态大模型 | 5 | 视觉语言模型、跨模态对齐 |
| 代码生成 | 5 | AI编程、代码补全、测试生成 |
| 芯片验证 | 5 | 形式验证、错误检测、电路仿真 |
| 5G移动通信 | 5 | 信道建模、波束赋形、资源分配 |

---

## 🧪 实验结果

30篇论文全部复现完成，关键结果：

| 论文 | 核心发现 |
|------|---------|
| FVAttn | 稀疏注意力开销降低73% |
| VTLoc | 视觉-触觉融合提升44.9% |
| ADA-ST | 自适应故障注入覆盖率达100% |
| DPNexT | 深度可分离架构参数缩减77.4% |
| CLIFE | Camera-LiDAR融合MOTA 78.7% |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/infanwang/auto-paper-pipeline.git
cd auto-paper-pipeline
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

---

## 📁 项目结构

```
auto-paper-pipeline/
├── config.yaml              # 配置文件
├── scripts/                 # 核心脚本
│   ├── enhanced_crawler.py  # 增强爬取
│   ├── enhanced_report.py   # 报告生成
│   └── enhanced_scheduler.py # 主调度器
├── data/                    # 论文数据
├── pdfs/                    # 下载的PDF
├── reproduction/            # 论文复现
└── COMPLETE_REPORT.pdf      # 完整实验报告
```

---

## 💡 为什么选择 Auto Paper Pipeline？

1. **全自动化**：从爬取到复现到报告，一键完成
2. **多领域覆盖**：6个研究领域，30篇论文
3. **引用数据**：Semantic Scholar API获取引用信息
4. **邮件简报**：每日自动发送美化简报
5. **定时任务**：无需手动干预
6. **开源免费**：MIT协议，欢迎贡献

---

## 🔮 未来计划

- [ ] 支持更多研究领域
- [ ] 集成LLM自动论文解读
- [ ] 添加论文影响力预测
- [ ] 支持多语言论文
- [ ] 开发Web界面

---

## 📧 联系方式

- GitHub: [infanwang](https://github.com/infanwang)
- Email: your@email.com

---

## 📄 License

MIT License

---

*如果这个项目对你有帮助，请给个⭐支持一下！*
