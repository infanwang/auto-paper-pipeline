---
name: paper-pipeline
description: Automated paper crawling, analysis, and reporting pipeline. Use when user mentions "paper pipeline", "论文管道", "crawl papers", "fetch arxiv", "run pipeline", "daily paper report", or "论文日报". Triggers on requests to search, download, analyze, or report on academic papers from arXiv and Semantic Scholar.
---

# Paper Pipeline

## Overview

Automated system for crawling, analyzing, and reporting on academic papers from arXiv with Semantic Scholar enrichment.

## Quick Start

```bash
# Run full pipeline
cd /root/git/mimo/paper-pipeline && export $(cat .env | xargs) && python3 scripts/run_pipeline.py --mode daily

# Run crawler only
python3 scripts/enhanced_crawler.py

# Test anti-crawl and dedup
python3 scripts/test_modules.py
```

## Pipeline Steps

### 1. Paper Crawling
- **Source**: arXiv API via `arxiv.py`
- **Topics**: AI Agent, LLM推理优化, 多模态大模型, 代码生成, 芯片验证, 5G移动通信
- **Anti-crawl**: User-Agent rotation, random delays (3-8s), exponential backoff retry

### 2. Semantic Scholar Enrichment
- Citation counts
- TLDR summaries
- Fields of study

### 3. Multi-level Filtering
- **Stage 1**: TF-IDF filtering (139 → 84 papers)
- **Stage 2**: LLM scoring (Top 20 papers)

### 4. Report Generation
- Markdown report: `reports/report_YYYY-MM-DD.md`
- PDF report: `reports/report_YYYY-MM-DD.pdf`
- Static site: `docs/`

### 5. Email Delivery
- Sends to configured recipient via 163 SMTP
- Includes PDF attachment

## Key Files

| File | Purpose |
|------|---------|
| `scripts/run_pipeline.py` | Main pipeline orchestrator |
| `scripts/enhanced_crawler.py` | Paper crawler with anti-crawl |
| `scripts/anti_crawl.py` | Anti-crawling measures |
| `scripts/dedup.py` | Deduplication module |
| `scripts/arxiv.py` | arXiv API client |
| `data/dedup_index.json` | Dedup index |
| `data/pipeline_YYYY-MM-DD.json` | Daily pipeline data |

## Configuration

### Environment Variables (.env)
```
SMTP_HOST=smtp.163.com
SMTP_PORT=465
SMTP_SENDER=your@email.com
SMTP_AUTH_CODE=your_auth_code
SMTP_RECIPIENT=recipient@email.com
```

### Cron Schedule
```bash
# Every other day at 23:00 Beijing time (15:00 UTC)
0 15 1-31/2 * * cd /root/git/mimo/paper-pipeline && export $(cat .env | xargs) && python3 scripts/run_pipeline.py --mode daily
```

## Anti-Crawling Features

- **User-Agent Rotation**: 7 browser UAs
- **Request Delays**: 3-8 seconds random
- **Exponential Backoff**: 5s → 10s → 20s on 429/5xx
- **PDF Validation**: Check %PDF- header
- **File Hash**: MD5 for dedup

## Deduplication

Three-level dedup:
1. **ID-based**: arXiv ID
2. **Title Similarity**: 85% threshold
3. **File Hash**: MD5 of PDF content

Index stored at: `data/dedup_index.json`

## Troubleshooting

### ArXiv Rate Limiting (429)
- Pipeline auto-retries with exponential backoff
- Reduce request frequency if persistent

### Download Failures
- Check network connectivity
- Increase timeout in `anti_crawl.py`
- Verify PDF URL is valid

### Email Not Sending
- Verify SMTP credentials in `.env`
- Check 163 authorization code
- Ensure port 465 is open

## Customization

### Add New Topic
Edit `TOPICS` dict in `enhanced_crawler.py`:
```python
TOPICS["New Topic"] = {
    "queries": ["query1", "query2"],
    "categories": ["cs.XX"],
}
```

### Adjust Filtering
Modify thresholds in `run_pipeline.py`:
- `tfidf_threshold`: TF-IDF cutoff (default 0.2)
- `llm_top_n`: LLM scoring top N (default 20)
