"""Web API - V2.0"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
from pathlib import Path

app = FastAPI(title="Auto Paper Pipeline API", version="2.0.0")

# 数据目录
DATA_DIR = Path("/root/git/mimo/paper-pipeline/data")


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    return """
    <html>
    <head>
        <title>Auto Paper Pipeline V2.0</title>
        <style>
            body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #1a1a2e; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px 0; }
            .card h3 { margin-top: 0; }
            a { color: #0f3460; }
        </style>
    </head>
    <body>
        <h1>📚 Auto Paper Pipeline V2.0</h1>
        <p>AI驱动的论文研究自动化平台</p>
        
        <div class="card">
            <h3>📊 API 端点</h3>
            <ul>
                <li><a href="/api/papers">/api/papers</a> - 论文列表</li>
                <li><a href="/api/papers/search?q=agent">/api/papers/search</a> - 搜索论文</li>
                <li><a href="/api/topics">/api/topics</a> - 研究领域</li>
                <li><a href="/api/stats">/api/stats</a> - 统计信息</li>
            </ul>
        </div>
        
        <div class="card">
            <h3>📁 文件</h3>
            <ul>
                <li><a href="/reports/">/reports/</a> - 报告目录</li>
                <li><a href="/data/">/data/</a> - 数据目录</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/api/papers")
async def list_papers(
    topic: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0
):
    """获取论文列表"""
    papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        return {"papers": [], "total": 0}
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
    # 过滤
    if topic:
        papers = [p for p in papers if p.get('topic') == topic]
    
    total = len(papers)
    papers = papers[offset:offset + limit]
    
    return {
        "papers": papers,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/papers/search")
async def search_papers(
    q: str = Query(..., description="搜索关键词"),
    topic: Optional[str] = None,
    limit: int = Query(20, le=100)
):
    """搜索论文"""
    papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        return {"papers": [], "total": 0}
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
    # 关键词搜索
    q_lower = q.lower()
    results = []
    
    for p in papers:
        text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
        if q_lower in text:
            if topic and p.get('topic') != topic:
                continue
            results.append(p)
    
    return {
        "query": q,
        "papers": results[:limit],
        "total": len(results)
    }


@app.get("/api/topics")
async def list_topics():
    """获取研究领域列表"""
    topics = {
        "AI_Agent": "智能体、工具调用、多Agent协作",
        "LLM推理优化": "推理加速、KV Cache、量化",
        "多模态大模型": "视觉语言模型、跨模态对齐",
        "代码生成": "AI编程、代码补全、测试生成",
        "芯片验证": "形式验证、错误检测、电路仿真",
        "5G移动通信": "信道建模、波束赋形、资源分配"
    }
    return {"topics": topics}


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        return {"total": 0, "by_topic": {}}
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
    by_topic = {}
    for p in papers:
        topic = p.get('topic', 'Unknown')
        by_topic[topic] = by_topic.get(topic, 0) + 1
    
    return {
        "total": len(papers),
        "by_topic": by_topic,
        "date": data.get('date', 'N/A')
    }


@app.get("/api/paper/{paper_id}")
async def get_paper(paper_id: str):
    """获取单篇论文详情"""
    papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        raise HTTPException(status_code=404, detail="No data found")
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
    for p in papers:
        if p.get('id') == paper_id:
            return p
    
    raise HTTPException(status_code=404, detail="Paper not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
