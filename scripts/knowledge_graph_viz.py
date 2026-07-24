#!/usr/bin/env python3
"""Generate knowledge graph visualization."""

import json
from pathlib import Path
from collections import defaultdict


def generate_knowledge_graph_html(papers: list, output_path: str):
    """Generate interactive knowledge graph visualization."""
    
    # Build graph data
    nodes = []
    edges = []
    node_ids = set()
    
    # Add paper nodes
    for paper in papers:
        paper_id = paper.get("id", "")
        if paper_id and paper_id not in node_ids:
            node_ids.add(paper_id)
            nodes.append({
                "id": paper_id,
                "label": paper.get("title", "")[:30] + "...",
                "group": "paper",
                "title": paper.get("title", ""),
                "score": paper.get("llm_score", 0),
            })
    
    # Add author nodes and edges
    author_papers = defaultdict(list)
    for paper in papers:
        paper_id = paper.get("id", "")
        authors = paper.get("authors", [])
        for author in authors:
            if isinstance(author, dict):
                author_name = author.get("name", "")
            else:
                author_name = str(author)
            
            if author_name:
                author_papers[author_name].append(paper_id)
                
                if author_name not in node_ids:
                    node_ids.add(author_name)
                    nodes.append({
                        "id": author_name,
                        "label": author_name,
                        "group": "author",
                    })
                
                edges.append({
                    "from": paper_id,
                    "to": author_name,
                    "label": "authored",
                })
    
    # Add topic nodes and edges
    topic_papers = defaultdict(list)
    for paper in papers:
        paper_id = paper.get("id", "")
        topic = paper.get("topic", "")
        if topic:
            topic_papers[topic].append(paper_id)
            
            if topic not in node_ids:
                node_ids.add(topic)
                nodes.append({
                    "id": topic,
                    "label": topic,
                    "group": "topic",
                })
            
            edges.append({
                "from": paper_id,
                "to": topic,
                "label": "belongs_to",
            })
    
    # Add co-author edges
    for author, paper_list in author_papers.items():
        if len(paper_list) > 1:
            # Connect author to first paper (already done)
            pass
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Knowledge Graph</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        #graph-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 8px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-dot {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        .stats {{
            text-align: center;
            margin: 20px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>Paper Knowledge Graph</h1>
    
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background: #e74c3c;"></div>
            <span>Papers</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #3498db;"></div>
            <span>Authors</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #2ecc71;"></div>
            <span>Topics</span>
        </div>
    </div>
    
    <div class="stats">
        <strong>{len(nodes)}</strong> nodes | <strong>{len(edges)}</strong> edges | 
        <strong>{sum(1 for n in nodes if n['group'] == 'paper')}</strong> papers |
        <strong>{sum(1 for n in nodes if n['group'] == 'author')}</strong> authors |
        <strong>{sum(1 for n in nodes if n['group'] == 'topic')}</strong> topics
    </div>
    
    <div id="graph-container"></div>
    
    <script>
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        
        var container = document.getElementById('graph-container');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{
                    size: 12,
                    face: 'Arial'
                }}
            }},
            edges: {{
                arrows: 'to',
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04
                }}
            }},
            groups: {{
                paper: {{
                    color: {{
                        background: '#e74c3c',
                        border: '#c0392b'
                    }},
                    size: 25
                }},
                author: {{
                    color: {{
                        background: '#3498db',
                        border: '#2980b9'
                    }},
                    size: 20
                }},
                topic: {{
                    color: {{
                        background: '#2ecc71',
                        border: '#27ae60'
                    }},
                    size: 30,
                    font: {{
                        size: 14,
                        bold: true
                    }}
                }}
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Click event
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                if (node && node.title) {{
                    alert('Paper: ' + node.title);
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    # Write HTML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding='utf-8')
    
    return output_file


def main():
    """Main function."""
    # Load papers
    data_dir = Path("/root/git/mimo/paper-pipeline/data")
    papers = []
    
    for json_file in data_dir.glob("pipeline_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            papers.extend(data.get('papers', []))
        except Exception:
            continue
    
    print(f"Loaded {len(papers)} papers")
    
    # Generate visualization
    output_path = "/root/git/mimo/paper-pipeline/docs/knowledge_graph.html"
    generate_knowledge_graph_html(papers, output_path)
    
    print(f"Generated visualization: {output_path}")


if __name__ == "__main__":
    main()
