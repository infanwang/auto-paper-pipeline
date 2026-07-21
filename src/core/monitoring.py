"""
监控与日志收集模块
支持Prometheus指标 + 结构化日志
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import logging


class MetricsCollector:
    """Prometheus风格指标收集器"""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
    
    def inc(self, name: str, value: int = 1):
        """增加计数器"""
        self.counters[name] += value
    
    def set(self, name: str, value: float):
        """设置仪表盘"""
        self.gauges[name] = value
    
    def observe(self, name: str, value: float):
        """记录直方图"""
        self.histograms[name].append(value)
    
    def get_metrics(self) -> Dict:
        """获取所有指标"""
        metrics = {}
        
        for name, value in self.counters.items():
            metrics[f"paper_{name}_total"] = value
        
        for name, value in self.gauges.items():
            metrics[f"paper_{name}_current"] = value
        
        for name, values in self.histograms.items():
            if values:
                metrics[f"paper_{name}_avg"] = sum(values) / len(values)
                metrics[f"paper_{name}_p99"] = sorted(values)[int(len(values) * 0.99)]
        
        return metrics
    
    def export_prometheus(self) -> str:
        """导出Prometheus格式"""
        lines = []
        for name, value in self.counters.items():
            lines.append(f"paper_{name}_total {value}")
        for name, value in self.gauges.items():
            lines.append(f"paper_{name}_current {value}")
        for name, values in self.histograms.items():
            if values:
                lines.append(f"paper_{name}_avg {sum(values) / len(values)}")
        return "\n".join(lines)


class StructuredLogger:
    """结构化日志"""
    
    def __init__(self, log_dir: str = "/app/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        self.logger = logging.getLogger("paper-pipeline")
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str, message: str, data: Dict = None):
        """记录事件"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "message": message,
            "data": data or {}
        }
        
        # 写入JSON日志
        log_file = self.log_dir / f"events_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # 写入访问日志
        access_file = self.log_dir / "access.log"
        with open(access_file, "a") as f:
            f.write(f"[{entry['timestamp']}] [{event_type}] {message}\n")
        
        return entry
    
    def log_api_request(self, method: str, path: str, status: int, duration_ms: float):
        """记录API请求"""
        self.log_event("api_request", f"{method} {path} -> {status}", {
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": duration_ms
        })
    
    def log_paper_crawl(self, count: int, duration: float):
        """记录论文爬取"""
        self.log_event("paper_crawl", f"Crawled {count} papers in {duration:.1f}s", {
            "count": count,
            "duration": duration
        })
    
    def log_error(self, error: str, context: Dict = None):
        """记录错误"""
        self.log_event("error", error, context)
        self.logger.error(error)


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
    
    def check_api(self) -> Dict:
        """检查API健康"""
        import urllib.request
        
        try:
            req = urllib.request.Request(f"{self.api_url}/api/health")
            response = urllib.request.urlopen(req, timeout=5)
            data = json.loads(response.read().decode())
            return {"status": "healthy", "data": data}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def check_disk(self) -> Dict:
        """检查磁盘空间"""
        import os
        
        data_dir = Path("/app/data")
        if data_dir.exists():
            usage = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
            return {"status": "ok", "usage_mb": usage / 1024 / 1024}
        return {"status": "warning", "message": "Data directory not found"}
    
    def full_check(self) -> Dict:
        """完整健康检查"""
        return {
            "timestamp": datetime.now().isoformat(),
            "api": self.check_api(),
            "disk": self.check_disk()
        }
