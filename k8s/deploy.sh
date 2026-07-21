#!/bin/bash
# K8s部署脚本

set -e

echo "=========================================="
echo "  Auto Paper Pipeline - K8s部署"
echo "=========================================="

# 检查kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl未安装"
    exit 1
fi

# 检查集群连接
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ 无法连接K8s集群"
    exit 1
fi

echo "✅ K8s集群连接正常"

# 创建命名空间
echo "📦 创建命名空间..."
kubectl create namespace paper-pipeline --dry-run=client -o yaml | kubectl apply -f -

# 应用配置
echo "🔧 应用K8s配置..."
kubectl apply -f k8s/deployment.yaml -n paper-pipeline

# 等待部署完成
echo "⏳ 等待部署..."
kubectl rollout status deployment/paper-pipeline-web -n paper-pipeline --timeout=120s

# 显示状态
echo ""
echo "✅ 部署完成！"
echo ""
echo "服务状态:"
kubectl get pods -n paper-pipeline
echo ""
echo "服务访问:"
kubectl get service paper-pipeline-service -n paper-pipeline
