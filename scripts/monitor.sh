#!/bin/bash
# Auto Paper Pipeline - 自愈监控脚本
# 每分钟检查容器状态，自动重启异常服务

LOG_DIR="/root/git/mimo/paper-pipeline/logs"
LOG_FILE="$LOG_DIR/heal.log"
CHECK_INTERVAL=60

mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_container() {
    local name=$1
    local status=$(docker inspect --format='{{.State.Status}}' "$name" 2>/dev/null)
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null)
    
    if [ -z "$status" ] || [ "$status" = "exited" ] || [ "$health" = "unhealthy" ]; then
        return 1
    fi
    return 0
}

restart_container() {
    local name=$1
    local reason=$2
    log_message "[RESTART] 重启 $name - 原因: $reason"
    docker restart "$name" 2>/dev/null
    log_message "[RESTART] $name 重启完成"
}

# 主循环
while true; do
    # 检查 Web 服务
    if ! check_container "paper-pipeline-web"; then
        restart_container "paper-pipeline-web" "容器异常或停止"
    fi
    
    # 检查 API 健康
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health | grep -q "500\|000"; then
        log_message "[HEALTH] API 响应异常，重启 web"
        restart_container "paper-pipeline-web" "API 响应异常"
    fi
    
    sleep $CHECK_INTERVAL
done
