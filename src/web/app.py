"""Web API - V2.0 带日志中间件"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# 日志配置
LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("paper-pipeline")

app = FastAPI(title="Auto Paper Pipeline API", version="2.0.0")

# 数据目录
DATA_DIR = Path("/app/data")

# 智能中枢导入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from web.smart_center import AISummaryGenerator, CitationNetwork, TrendAlertSystem

# AI摘要生成器
ai_summary_gen = AISummaryGenerator()


def _extract_keywords(query: str) -> List[str]:
    """从自然语言提取关键词"""
    # 简单的关键词提取（实际应用中可调用LLM）
    keyword_map = {
        '大模型': ['LLM', 'large language model'],
        '推理优化': ['inference', 'optimization', 'efficient'],
        'agent': ['agent', 'agentic'],
        '多模态': ['multimodal', 'vision', 'language'],
        '代码生成': ['code generation', 'programming'],
        '5G': ['5G', 'wireless', 'MIMO'],
        '芯片': ['chip', 'verification', 'hardware'],
    }
    
    keywords = []
    for cn, en_keywords in keyword_map.items():
        if cn in query:
            keywords.extend(en_keywords)
    
    if not keywords:
        # 默认：提取原始查询中的单词
        keywords = query.split()
    
    return keywords


# ============ 中间件：请求日志 ============
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # 获取请求信息
    method = request.method
    path = request.url.path
    query_params = dict(request.query_params)
    
    # 脱敏处理（移除敏感参数）
    safe_params = {k: v[:50] + "..." if len(str(v)) > 50 else v 
                   for k, v in query_params.items()}
    
    # 处理请求
    response = await call_next(request)
    
    # 计算耗时
    duration_ms = (time.time() - start_time) * 1000
    status_code = response.status_code
    
    # 慢请求标记
    slow_flag = "[SLOW] " if duration_ms > 2000 else ""
    
    # 日志格式
    log_entry = f"{slow_flag}[{datetime.now().strftime('%H:%M:%S')}] {method} {path} {safe_params} -> {status_code} ({duration_ms:.1f}ms)"
    
    # 输出到控制台
    logger.info(log_entry)
    
    # 写入日志文件
    log_file = LOG_DIR / "access.log"
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")
    
    # 慢请求记录堆栈
    if duration_ms > 2000:
        logger.warning(f"[SLOW] 慢请求: {path} 耗时 {duration_ms:.1f}ms")
        import traceback
        logger.warning(traceback.format_exc())
    
    return response


# ============ API 端点 ============
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
                <li><a href="/api/health">/api/health</a> - 健康检查</li>
                <li><a href="/dashboard">/dashboard</a> - 监控仪表盘</li>
            </ul>
        </div>
        
        <div class="card">
            <h3>📁 文件</h3>
            <ul>
                <li><a href="/reports/">/reports/</a> - 报告目录</li>
                <li><a href="/data/">/data/</a> - 数据目录</li>
                <li><a href="/docs/">/docs/</a> - 静态站点</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/search", response_class=HTMLResponse)
async def search_page():
    """搜索页面"""
    search_html = (Path(__file__).parent / "templates" / "search.html").read_text()
    return search_html


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    try:
        papers_file = DATA_DIR / "pipeline_2026-07-21.json"
        if not papers_file.exists():
            papers_file = DATA_DIR / "discovery_2026-07-21.json"
        
        if papers_file.exists():
            data = json.loads(papers_file.read_text())
            total = len(data.get('papers', []))
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "papers_count": total,
                "uptime": "running"
            }
        else:
            return {"status": "unhealthy", "error": "No data files found"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/api/papers")
async def list_papers(
    topic: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0
):
    """获取论文列表"""
    papers_file = DATA_DIR / "pipeline_2026-07-21.json"
    if not papers_file.exists():
        papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        return {"papers": [], "total": 0}
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
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
    min_score: float = 0,
    limit: int = Query(20, le=100),
    semantic: bool = Query(False, description="AI语义搜索")
):
    """搜索论文（支持AI语义搜索）"""
    papers_file = DATA_DIR / "pipeline_2026-07-21.json"
    if not papers_file.exists():
        papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        return {"papers": [], "total": 0}
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
    # AI语义搜索：提取关键词
    if semantic:
        keywords = _extract_keywords(q)
        q_lower = ' '.join(keywords).lower()
    else:
        q_lower = q.lower()
    
    q_lower = q.lower()
    results = []
    
    for p in papers:
        text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
        if q_lower in text:
            if topic and p.get('topic') != topic:
                continue
            if p.get('llm_score', 0) < min_score:
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
    papers_file = DATA_DIR / "pipeline_2026-07-21.json"
    if not papers_file.exists():
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
    papers_file = DATA_DIR / "pipeline_2026-07-21.json"
    if not papers_file.exists():
        papers_file = DATA_DIR / "discovery_2026-07-21.json"
    
    if not papers_file.exists():
        raise HTTPException(status_code=404, detail="No data found")
    
    data = json.loads(papers_file.read_text())
    papers = data.get('papers', [])
    
    for p in papers:
        if p.get('id') == paper_id:
            return p
    
    raise HTTPException(status_code=404, detail="Paper not found")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """监控仪表盘"""
    return DASHBOARD_HTML


# 仪表盘HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>监控仪表盘 - Auto Paper Pipeline</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f0f2f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
        .header h1 { font-size: 1.8em; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { color: #666; font-size: 0.9em; margin-bottom: 10px; }
        .card .value { font-size: 2em; font-weight: bold; color: #1a1a2e; }
        .card .change { font-size: 0.8em; color: #4CAF50; }
        .chart-container { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }
        .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
        .status-healthy { background: #e8f5e9; color: #2e7d32; }
        .status-warning { background: #fff3e0; color: #ef6c00; }
        .status-error { background: #ffebee; color: #c62828; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 监控仪表盘</h1>
        <p>Auto Paper Pipeline 实时状态</p>
    </div>
    
    <div class="container">
        <div class="cards">
            <div class="card">
                <h3>📄 总论文数</h3>
                <div class="value" id="total-papers">--</div>
                <div class="change" id="today-papers">今日新增: --</div>
            </div>
            <div class="card">
                <h3>📁 研究领域</h3>
                <div class="value" id="total-topics">--</div>
            </div>
            <div class="card">
                <h3>⭐ 高分论文</h3>
                <div class="value" id="high-score">--</div>
            </div>
            <div class="card">
                <h3>🔧 系统状态</h3>
                <div class="value" id="system-status">--</div>
            </div>
        </div>
        
        <div class="chart-row">
            <div class="chart-container">
                <h3>📊 论文分布（饼图）</h3>
                <canvas id="topicChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>📈 论文收录趋势</h3>
                <canvas id="trendChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📋 最新论文</h3>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background:#f8f9fa;">
                    <th style="padding:10px; text-align:left;">标题</th>
                    <th style="padding:10px; text-align:left;">领域</th>
                    <th style="padding:10px; text-align:left;">评分</th>
                </tr>
                <tbody id="papers-table"></tbody>
            </table>
        </div>
    </div>
    
    <script>
    // 加载统计数据
    async function loadStats() {
        try {
            const resp = await fetch('/api/stats');
            const data = await resp.json();
            
            document.getElementById('total-papers').textContent = data.total || 0;
            document.getElementById('total-topics').textContent = Object.keys(data.by_topic || {}).length;
            
            // 绘制饼图
            const topics = data.by_topic || {};
            const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b'];
            new Chart(document.getElementById('topicChart'), {
                type: 'doughnut',
                data: {
                    labels: Object.keys(topics),
                    datasets: [{
                        data: Object.values(topics),
                        backgroundColor: colors.slice(0, Object.keys(topics).length)
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'right' } }
                }
            });
            
            // 模拟趋势数据
            const trendLabels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
            const trendData = [12, 19, 15, 25, 22, 30, data.total || 20];
            new Chart(document.getElementById('trendChart'), {
                type: 'line',
                data: {
                    labels: trendLabels,
                    datasets: [{
                        label: '论文数',
                        data: trendData,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } }
                }
            });
        } catch (e) {
            console.error('加载统计失败:', e);
        }
    }
    
    // 加载最新论文
    async function loadPapers() {
        try {
            const resp = await fetch('/api/papers?limit=5');
            const data = await resp.json();
            const tbody = document.getElementById('papers-table');
            tbody.innerHTML = '';
            (data.papers || []).forEach(p => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td style="padding:10px; border-bottom:1px solid #eee;">${(p.title||'').substring(0,50)}...</td>
                               <td style="padding:10px; border-bottom:1px solid #eee;">${p.topic||'N/A'}</td>
                               <td style="padding:10px; border-bottom:1px solid #eee;"><span style="background:#667eea;color:white;padding:2px 8px;border-radius:10px;">${(p.llm_score||0).toFixed(1)}</span></td>`;
                tbody.appendChild(tr);
            });
        } catch (e) {
            console.error('加载论文失败:', e);
        }
    }
    
    // 健康检查
    async function checkHealth() {
        try {
            const resp = await fetch('/api/health');
            const data = await resp.json();
            const el = document.getElementById('system-status');
            if (data.status === 'healthy') {
                el.innerHTML = '<span class="status-badge status-healthy">✅ 健康</span>';
            } else {
                el.innerHTML = '<span class="status-badge status-error">❌ 异常</span>';
            }
        } catch (e) {
            document.getElementById('system-status').innerHTML = '<span class="status-badge status-error">❌ 离线</span>';
        }
    }
    
    // 初始化
    loadStats();
    loadPapers();
    checkHealth();
    
    // 每30秒刷新
    setInterval(() => { loadStats(); checkHealth(); }, 30000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
