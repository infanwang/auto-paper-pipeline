# Auto Paper Pipeline 软件设计文档

**版本**: 2.0  
**日期**: 2026-07-24  
**作者**: MiMoCode  
**仓库**: https://github.com/infanwang/auto-paper-pipeline

---

## 目录

1. [软件设计方案](#一软件设计方案)
2. [需求说明](#二需求说明)
3. [架构设计](#三架构设计)
4. [算法和数据结构设计](#四算法和数据结构设计)
5. [未来演进](#五未来演进)
6. [运维测试](#六运维测试)

---

## 一、软件设计方案

### 1.1 项目概述

Auto Paper Pipeline 是一个自动化论文研究平台，覆盖从论文发现、爬取、分析到复现的全流程。

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| 自动化 | 全流程自动化，无需人工干预 |
| 可扩展 | 模块化设计，易于添加新功能 |
| 可靠性 | 反爬措施、去重、错误恢复 |
| 多语言 | 支持 7 种联合国官方语言 |
| 可观测 | 完整的日志和监控 |

### 1.3 核心功能

```
┌─────────────────────────────────────────────────────────┐
│                    Paper Pipeline                       │
├─────────────────────────────────────────────────────────┤
│  1. 论文采集    →  arXiv, ACL, NeurIPS, ICML           │
│  2. 数据增强    →  Semantic Scholar 引用数据            │
│  3. 智能过滤    →  TF-IDF + LLM 评分                   │
│  4. 深度分析    →  多维度评分、知识图谱                  │
│  5. 报告生成    →  Markdown, PDF, 多语言                │
│  6. 论文复现    →  Python 实现、测试验证                │
│  7. 邮件推送    →  定时发送论文日报                     │
└─────────────────────────────────────────────────────────┘
```

---

## 二、需求说明

### 2.1 功能需求

#### FR-1: 论文采集
- 支持多源爬取 (arXiv, ACL, NeurIPS, ICML)
- 反爬措施 (User-Agent 轮换、请求间隔)
- 自动重试和错误恢复

#### FR-2: 数据增强
- Semantic Scholar 引用数据
- 论文元数据补全
- 相关论文推荐

#### FR-3: 智能过滤
- TF-IDF 关键词过滤
- LLM 多维度评分
- 按主题分类

#### FR-4: 多语言支持
- 7 种联合国官方语言
- 语言检测 (100% 准确率)
- 跨语言翻译

#### FR-5: 报告生成
- Markdown 格式报告
- PDF 格式报告
- 多语言报告

#### FR-6: 论文复现
- Python 代码实现
- 单元测试验证
- 模型保存

#### FR-7: 定时任务
- 隔天执行
- 邮件推送
- 日志记录

### 2.2 非功能需求

| 需求 | 指标 |
|------|------|
| 可用性 | 99.9% |
| 性能 | 单次运行 < 10 分钟 |
| 可扩展性 | 支持 100+ 论文源 |
| 安全性 | API 密钥加密存储 |
| 可维护性 | 模块化、文档化 |

---

## 三、架构设计

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Web UI  │  │ CLI     │  │ API     │  │ Email   │      │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
├───────┴────────────┴────────────┴────────────┴─────────────┤
│                      Application Layer                     │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                 Pipeline Orchestrator                │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Crawler │  │ Analyzer│  │ Reporter│  │Reproducer│      │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
├───────┴────────────┴────────────┴────────────┴─────────────┤
│                       Core Layer                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │AntiCrawl│  │  Dedup  │  │Multilin │  │ Scorer  │      │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
├───────┴────────────┴────────────┴────────────┴─────────────┤
│                      Data Layer                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Files   │  │  JSON   │  │ SQLite  │  │  Cache  │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模块设计

#### 3.2.1 爬虫模块

```python
# 模块结构
scripts/
├── enhanced_crawler.py      # 主爬虫
├── multi_source_crawler.py  # 多源爬虫
├── arxiv.py                 # arXiv API
├── anti_crawl.py            # 反爬措施
└── dedup.py                 # 去重功能
```

**职责**:
- 从多个论文源采集论文
- 实现反爬措施
- 去重和数据清洗

#### 3.2.2 分析模块

```python
# 模块结构
scripts/
├── enhanced_scorer.py       # 增强评分
├── knowledge_graph.py       # 知识图谱
├── recommender.py           # 推荐系统
└── multilingual.py          # 多语言支持
```

**职责**:
- 多维度论文评分
- 知识图谱构建
- 论文推荐

#### 3.2.3 报告模块

```python
# 模块结构
scripts/
├── enhanced_report.py       # 报告生成
├── multilingual_report.py   # 多语言报告
└── translator.py            # 翻译模块
```

**职责**:
- 生成 Markdown/PDF 报告
- 多语言报告支持
- 论文翻译

#### 3.2.4 复现模块

```python
# 模块结构
reproduction/
├── ai_agent/                # AI Agent 论文
├── llm_inference/           # LLM 推理优化
└── chip_verification/       # 芯片验证
```

**职责**:
- 论文方法实现
- 单元测试验证
- 模型保存

---

## 四、算法和数据结构设计

### 4.1 反爬算法

```python
class AntiCrawl:
    """反爬措施实现"""
    
    def __init__(self, min_delay=3, max_delay=8, max_retries=3):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.last_request_time = 0
    
    def get_headers(self) -> dict:
        """获取随机请求头"""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,...",
            "Accept-Language": "en-US,en;q=0.9",
        }
    
    def wait(self):
        """随机等待"""
        elapsed = time.time() - self.last_request_time
        delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request_time = time.time()
    
    def retry_request(self, func, *args, **kwargs):
        """指数退避重试"""
        for attempt in range(self.max_retries):
            try:
                self.wait()
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 5
                time.sleep(wait_time)
```

### 4.2 去重算法

```python
class Deduplicator:
    """三级去重实现"""
    
    def __init__(self):
        self.index = {"papers": {}, "hashes": {}}
    
    def is_duplicate(self, paper_id: str, title: str = None) -> bool:
        """检查是否重复"""
        # 1. ID 去重
        if paper_id in self.index["papers"]:
            return True
        
        # 2. 标题相似度去重
        if title:
            for existing in self.index["papers"].values():
                similarity = SequenceMatcher(
                    None, title.lower(), existing["title"].lower()
                ).ratio()
                if similarity > 0.85:
                    return True
        
        return False
    
    def file_hash(self, file_path: str) -> str:
        """文件哈希"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
```

### 4.3 语言检测算法

```python
class LanguageDetector:
    """语言检测实现"""
    
    def detect(self, text: str) -> Tuple[Language, float]:
        """检测语言"""
        scores = {}
        
        # 基于关键词匹配
        for lang, pattern in self.patterns.items():
            matches = pattern.findall(text)
            scores[lang] = len(matches)
        
        # 字符集检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
        
        if chinese_chars > 0:
            scores[Language.CHINESE] = max(
                scores.get(Language.CHINESE, 0), 
                chinese_chars * 10
            )
        
        # 计算置信度
        total_score = sum(scores.values())
        best_lang = max(scores, key=scores.get)
        confidence = scores[best_lang] / total_score if total_score > 0 else 0.0
        
        return best_lang, confidence
```

### 4.4 LLM 评分算法

```python
class EnhancedLLMScorer:
    """多维度评分实现"""
    
    def score_paper(self, title: str, abstract: str, citations: int = 0):
        """评分"""
        scores = PaperScore(
            novelty=self.score_novelty(title, abstract),
            impact=self.score_impact(title, abstract, citations),
            rigor=self.score_rigor(title, abstract),
            clarity=self.score_clarity(title, abstract),
            reproducibility=self.score_reproducibility(title, abstract),
        )
        scores.overall = self.calculate_overall(scores)
        return scores
    
    def calculate_overall(self, scores: PaperScore) -> float:
        """计算综合评分"""
        weights = {
            "novelty": 0.25,
            "impact": 0.25,
            "rigor": 0.20,
            "clarity": 0.15,
            "reproducibility": 0.15,
        }
        return sum(getattr(scores, k) * v for k, v in weights.items())
```

### 4.5 知识图谱数据结构

```python
class PaperKnowledgeGraph:
    """知识图谱实现"""
    
    def __init__(self):
        self.graph = {
            "papers": {},           # 论文节点
            "authors": {},          # 作者节点
            "topics": {},           # 主题节点
            "citations": {},        # 引用关系
            "co_authors": {},       # 合作关系
            "topic_papers": {},     # 主题-论文关系
            "temporal": {},         # 时间关系
        }
    
    def add_paper(self, paper: dict):
        """添加论文到图谱"""
        paper_id = paper.get("id", "")
        self.graph["papers"][paper_id] = {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "topics": paper.get("topics", []),
        }
        
        # 添加作者关系
        for author in paper.get("authors", []):
            if author not in self.graph["authors"]:
                self.graph["authors"][author] = {"papers": [], "co_authors": set()}
            self.graph["authors"][author]["papers"].append(paper_id)
```

### 4.6 推荐系统算法

```python
class PaperRecommender:
    """推荐系统实现"""
    
    def __init__(self):
        self.papers = {}
        self.idf = {}
    
    def compute_tfidf(self, text: str) -> Dict[str, float]:
        """计算 TF-IDF"""
        tf = self.compute_tf(text)
        tfidf = {}
        for token, tf_val in tf.items():
            idf_val = self.idf.get(token, 1.0)
            tfidf[token] = tf_val * idf_val
        return tfidf
    
    def cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """计算余弦相似度"""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        
        dot = sum(vec1[k] * vec2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0
    
    def recommend(self, user_papers: List[str], n: int = 10):
        """推荐论文"""
        # 构建用户画像
        user_text = ""
        for pid in user_papers:
            if pid in self.papers:
                user_text += " " + self.get_paper_text(self.papers[pid])
        
        user_tfidf = self.compute_tfidf(user_text)
        
        # 计算相似度
        scores = []
        for pid, paper in self.papers.items():
            if pid in user_papers:
                continue
            text = self.get_paper_text(paper)
            tfidf = self.compute_tfidf(text)
            sim = self.cosine_similarity(user_tfidf, tfidf)
            if sim > 0:
                scores.append((pid, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
```

---

## 五、未来演进

### 5.1 短期演进 (3-6 个月)

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 更多论文源 | ACL, NeurIPS, ICML, EMNLP | 高 |
| 改进评分 | 集成 GPT-4 评分 | 高 |
| 摘要翻译 | 自动翻译论文摘要 | 中 |
| Web UI 改进 | 更好的可视化 | 中 |

### 5.2 中期演进 (6-12 个月)

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 知识图谱可视化 | 交互式图谱展示 | 高 |
| 论文推荐系统 | 个性化推荐 | 高 |
| 多语言优化 | 改进翻译质量 | 中 |
| 移动端支持 | 移动端阅读 | 低 |

### 5.3 长期演进 (1-2 年)

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 论文复现自动化 | 自动实现论文方法 | 高 |
| 研究趋势预测 | 预测研究方向 | 中 |
| 学术社交网络 | 研究者协作 | 中 |
| 商业化 | SaaS 服务 | 低 |

### 5.4 技术演进路线

```
2026 Q3: 基础功能完善
├── 多源爬虫
├── 多语言支持
└── CI/CD 流水线

2026 Q4: 智能化升级
├── LLM 集成评分
├── 知识图谱可视化
└── 推荐系统

2027 Q1: 平台化
├── Web UI 完善
├── API 开放
└── 移动端支持

2027 Q2+: 商业化
├── SaaS 服务
├── 企业版
└── 国际化
```

---

## 六、运维测试

### 6.1 测试策略

#### 单元测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_pipeline.py::TestAntiCrawl -v

# 生成覆盖率报告
pytest tests/ --cov=scripts --cov-report=html
```

#### 集成测试

```bash
# 运行完整管道
python scripts/run_pipeline.py --mode daily --dry-run

# 测试多语言功能
python scripts/multilingual_report.py

# 测试论文复现
python reproduction/ai_agent/papers_analysis.py
```

#### 性能测试

```bash
# 测量爬虫性能
time python scripts/enhanced_crawler.py

# 测量翻译性能
python -c "
from multilingual import MultilingualTranslator
import time
t = MultilingualTranslator()
start = time.time()
for i in range(10):
    t.translate_text('Test sentence', target_lang='zh')
print(f'Average: {(time.time() - start) / 10:.3f}s per translation')
"
```

### 6.2 监控指标

| 指标 | 描述 | 告警阈值 |
|------|------|----------|
| 爬虫成功率 | 成功爬取的论文比例 | < 90% |
| 翻译成功率 | 成功翻译的比例 | < 95% |
| 报告生成时间 | 生成报告耗时 | > 5 分钟 |
| 邮件发送成功率 | 邮件发送成功比例 | < 99% |
| 错误率 | 系统错误比例 | > 1% |

### 6.3 日志规范

```python
# 日志格式
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# 日志级别
- DEBUG: 调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误
```

### 6.4 部署检查清单

- [ ] 环境变量配置正确
- [ ] 依赖安装完整
- [ ] 数据目录权限正确
- [ ] SMTP 配置测试通过
- [ ] 定时任务配置正确
- [ ] 日志目录可写
- [ ] 备份策略配置

### 6.5 故障恢复

| 故障 | 恢复方案 |
|------|----------|
| 爬虫限流 | 增加请求间隔，使用代理 |
| 翻译失败 | 使用缓存，降级到原文 |
| 邮件发送失败 | 重试，记录失败日志 |
| 磁盘空间不足 | 清理旧数据，压缩日志 |
| 内存溢出 | 分批处理，优化数据结构 |

---

## 七、附录

### 7.1 文件结构

```
paper-pipeline/
├── .github/workflows/      # CI/CD 配置
├── scripts/                # 核心脚本
├── tests/                  # 测试套件
├── reproduction/           # 论文复现
├── skills/                 # 技能文档
├── reports/                # 生成的报告
├── data/                   # 数据文件
├── docs/                   # 静态站点
├── pdfs/                   # 下载的论文
├── Dockerfile              # Docker 配置
├── docker-compose.yml      # Docker Compose
├── requirements.txt        # 依赖列表
├── EXPERIENCE.md           # 经验总结
└── README.md               # 项目说明
```

### 7.2 依赖列表

```
# 核心依赖
pyyaml>=6.0
numpy>=1.24.0
torch>=2.0.0

# 测试
pytest>=7.0
pytest-cov>=4.0

# Web API
fastapi>=0.100.0
uvicorn>=0.23.0

# 代码质量
flake8>=6.0
black>=23.0
```

### 7.3 环境变量

```bash
# SMTP 配置
SMTP_HOST=smtp.163.com
SMTP_PORT=465
SMTP_SENDER=your@email.com
SMTP_AUTH_CODE=your_auth_code
SMTP_RECIPIENT=recipient@email.com

# 服务器配置
SERVER_HOST=your_server
SERVER_USER=root
SSH_KEY=your_ssh_key
```

---

*文档生成日期: 2026-07-24*  
*生成工具: MiMoCode*
