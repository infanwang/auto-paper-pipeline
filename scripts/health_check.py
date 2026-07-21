#!/usr/bin/env python3
"""
Auto Paper Pipeline - 健康检查脚本
定期检查API状态，如果挂了自动重启
"""

import subprocess
import urllib.request
import time
import json
from datetime import datetime
from pathlib import Path

# 配置
API_URL = "http://localhost:8000"
CHECK_INTERVAL = 60  # 检查间隔（秒）
MAX_RETRIES = 3
LOG_FILE = Path("/root/git/mimo/paper-pipeline/data/health_check.log")


def check_api_health():
    """检查API健康状态"""
    try:
        req = urllib.request.Request(f"{API_URL}/api/stats")
        response = urllib.request.urlopen(req, timeout=5)
        data = json.loads(response.read().decode())
        
        if "total" in data:
            return True, f"API正常，{data['total']}篇论文"
        else:
            return False, "API返回数据格式错误"
    except Exception as e:
        return False, f"API连接失败: {str(e)}"


def check_docker_status():
    """检查Docker容器状态"""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        container = json.loads(line)
                        containers.append(container)
                    except:
                        pass
            return True, containers
        return False, "Docker命令失败"
    except Exception as e:
        return False, f"Docker检查失败: {str(e)}"


def restart_service(service_name="web"):
    """重启服务"""
    try:
        result = subprocess.run(
            ["docker", "compose", "restart", service_name],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        return False


def log_message(message):
    """记录日志"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)
    
    print(log_entry.strip())


def run_health_check():
    """执行健康检查"""
    log_message("=" * 50)
    log_message("开始健康检查")
    
    # 1. 检查API状态
    api_ok, api_msg = check_api_health()
    log_message(f"API状态: {'✅ 正常' if api_ok else '❌ 异常'} - {api_msg}")
    
    # 2. 检查Docker状态
    docker_ok, docker_msg = check_docker_status()
    if docker_ok:
        running = sum(1 for c in docker_msg if c.get("State") == "running")
        log_message(f"Docker状态: {running}个容器运行中")
    else:
        log_message(f"Docker状态: ❌ {docker_msg}")
    
    # 3. 如果API挂了，尝试重启
    if not api_ok:
        log_message("⚠️ API异常，尝试重启web服务...")
        for attempt in range(MAX_RETRIES):
            if restart_service("web"):
                time.sleep(5)
                api_ok, api_msg = check_api_health()
                if api_ok:
                    log_message(f"✅ 重启成功，API恢复正常")
                    break
                else:
                    log_message(f"⚠️ 第{attempt+1}次重启失败")
            else:
                log_message(f"❌ 重启命令执行失败")
    
    log_message("健康检查完成")
    log_message("=" * 50)
    
    return api_ok


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="健康检查脚本")
    parser.add_argument("--daemon", action="store_true", help="守护进程模式（持续运行）")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL, help="检查间隔（秒）")
    args = parser.parse_args()
    
    if args.daemon:
        log_message("启动守护进程模式")
        while True:
            try:
                run_health_check()
            except Exception as e:
                log_message(f"检查异常: {e}")
            time.sleep(args.interval)
    else:
        run_health_check()
