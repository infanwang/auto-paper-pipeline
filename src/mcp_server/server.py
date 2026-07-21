"""
Auto Paper Pipeline - MCP Server
让Claude/Cursor直接查询论文数据库
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent

# 创建MCP服务器
app = Server("auto-paper-pipeline")


def load_papers() -> List[Dict]:
    """加载论文数据"""
    data_dir = Path("/root/git/mimo/paper-pipeline/data")
    json_files = sorted(data_dir.glob("pipeline_*.json"), reverse=True)
    if not json_files:
        json_files = sorted(data_dir.glob("discovery_*.json"), reverse=True)
    if json_files:
        data = json.loads(json_files[0].read_text())
        return data.get('papers', [])
    return []


def search_papers(query: str, topic: str = None, min_score: float = 0) -> List[Dict]:
    """搜索论文"""
    papers = load_papers()
    results = []
    query_lower = query.lower()
    for p in papers:
        text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
        if query_lower in text:
            if topic and p.get('topic') != topic:
                continue
            if p.get('llm_score', 0) < min_score:
                continue
            results.append(p)
    return results


def get_paper_by_id(paper_id: str) -> Dict:
    """根据ID获取论文"""
    papers = load_papers()
    for p in papers:
        if p.get('id') == paper_id:
            return p
    return {}


def get_topic_stats() -> Dict:
    """获取领域统计"""
    papers = load_papers()
    stats = {}
    for p in papers:
        topic = p.get('topic', 'Unknown')
        stats[topic] = stats.get(topic, 0) + 1
    return stats


# MCP工具定义
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description="搜索AI论文。支持关键词搜索、领域筛选、评分过滤。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词（如 'LLM agent', 'MoE inference'）"},
                    "topic": {"type": "string", "description": "筛选领域", "enum": ["AI_Agent", "LLM推理优化", "多模态大模型", "代码生成", "芯片验证", "5G移动通信"]},
                    "min_score": {"type": "number", "description": "最低评分（0-10）", "default": 0},
                    "limit": {"type": "integer", "description": "返回数量", "default": 10}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_paper_detail",
            description="获取单篇论文的详细信息，包括摘要、作者、评分、分析结果。",
            inputSchema={
                "type": "object",
                "properties": {"paper_id": {"type": "string", "description": "论文ArXiv ID（如 2607.16193）"}},
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="get_topic_stats",
            description="获取各研究领域的论文统计信息。",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_top_papers",
            description="获取评分最高的论文列表。",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回数量", "default": 10},
                    "topic": {"type": "string", "description": "筛选领域"}
                }
            }
        ),
        Tool(
            name="compare_papers",
            description="对比两篇论文的异同。",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id_1": {"type": "string", "description": "第一篇论文ID"},
                    "paper_id_2": {"type": "string", "description": "第二篇论文ID"}
                },
                "required": ["paper_id_1", "paper_id_2"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    """处理工具调用"""
    result = ""
    
    if name == "search_papers":
        query = arguments.get("query", "")
        topic = arguments.get("topic")
        min_score = arguments.get("min_score", 0)
        limit = arguments.get("limit", 10)
        results = search_papers(query, topic, min_score)[:limit]
        output = {
            "query": query,
            "total_found": len(results),
            "papers": [{"id": p.get("id"), "title": p.get("title"), "topic": p.get("topic"),
                       "score": p.get("llm_score", 0), "published": p.get("published_date"),
                       "abstract_preview": p.get("abstract", "")[:100] + "..."} for p in results]
        }
        result = json.dumps(output, ensure_ascii=False, indent=2)
    
    elif name == "get_paper_detail":
        paper_id = arguments.get("paper_id", "")
        paper = get_paper_by_id(paper_id)
        if not paper:
            result = json.dumps({"error": f"论文 {paper_id} 未找到"})
        else:
            output = {
                "id": paper.get("id"), "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "topic": paper.get("topic"), "score": paper.get("llm_score", 0),
                "tags": paper.get("llm_tags", []), "published": paper.get("published_date"),
                "analysis": paper.get("analysis", {})
            }
            result = json.dumps(output, ensure_ascii=False, indent=2)
    
    elif name == "get_topic_stats":
        stats = get_topic_stats()
        result = json.dumps({"topics": stats, "total": sum(stats.values())}, ensure_ascii=False, indent=2)
    
    elif name == "get_top_papers":
        limit = arguments.get("limit", 10)
        topic = arguments.get("topic")
        papers = load_papers()
        if topic:
            papers = [p for p in papers if p.get("topic") == topic]
        papers.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
        output = [{"id": p.get("id"), "title": p.get("title"), "score": p.get("llm_score", 0), "topic": p.get("topic")} for p in papers[:limit]]
        result = json.dumps({"top_papers": output}, ensure_ascii=False, indent=2)
    
    elif name == "compare_papers":
        id1, id2 = arguments.get("paper_id_1", ""), arguments.get("paper_id_2", "")
        p1, p2 = get_paper_by_id(id1), get_paper_by_id(id2)
        if not p1 or not p2:
            result = json.dumps({"error": "一篇或多篇论文未找到"})
        else:
            comparison = {
                "paper_1": {"id": p1.get("id"), "title": p1.get("title"), "score": p1.get("llm_score", 0), "topic": p1.get("topic")},
                "paper_2": {"id": p2.get("id"), "title": p2.get("title"), "score": p2.get("llm_score", 0), "topic": p2.get("topic")},
                "score_diff": p1.get("llm_score", 0) - p2.get("llm_score", 0),
                "same_topic": p1.get("topic") == p2.get("topic")
            }
            result = json.dumps(comparison, ensure_ascii=False, indent=2)
    
    else:
        result = json.dumps({"error": f"未知工具: {name}"})
    
    return [TextContent(type="text", text=result)]


# 命令行接口
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto Paper Pipeline MCP Server")
    parser.add_argument("command", choices=["tools", "search", "detail", "stats", "top"])
    parser.add_argument("--query", type=str)
    parser.add_argument("--topic", type=str)
    parser.add_argument("--paper-id", type=str)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    if args.command == "tools":
        tools = [
            {"name": "search_papers", "description": "搜索AI论文"},
            {"name": "get_paper_detail", "description": "获取论文详情"},
            {"name": "get_topic_stats", "description": "获取领域统计"},
            {"name": "get_top_papers", "description": "获取高分论文"},
            {"name": "compare_papers", "description": "对比两篇论文"},
        ]
        print(json.dumps(tools, ensure_ascii=False, indent=2))
    elif args.command == "search":
        results = search_papers(args.query, args.topic)[:args.limit]
        print(json.dumps([{"id": p.get("id"), "title": p.get("title"), "score": p.get("llm_score", 0)} for p in results], ensure_ascii=False, indent=2))
    elif args.command == "detail":
        paper = get_paper_by_id(args.paper_id)
        print(json.dumps(paper, ensure_ascii=False, indent=2) if paper else "未找到")
    elif args.command == "stats":
        print(json.dumps(get_topic_stats(), ensure_ascii=False, indent=2))
    elif args.command == "top":
        papers = sorted(load_papers(), key=lambda x: x.get("llm_score", 0), reverse=True)[:args.limit]
        print(json.dumps([{"id": p.get("id"), "title": p.get("title"), "score": p.get("llm_score", 0)} for p in papers], ensure_ascii=False, indent=2))
