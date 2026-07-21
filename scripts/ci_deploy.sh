#!/bin/bash
# CI/CD部署脚本

set -e

echo "=========================================="
echo "  Auto Paper Pipeline - CI/CD部署"
echo "=========================================="

# 配置
REGISTRY="ghcr.io"
IMAGE_NAME="infanwang/auto-paper-pipeline"
VERSION=$(git rev-parse --short HEAD)

# 构建镜像
echo "🔨 构建Docker镜像..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} .
docker tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:latest

# 推送到Registry
echo "📤 推送到GitHub Container Registry..."
echo "${{ secrets.GITHUB_TOKEN }}" | docker login ${REGISTRY} -u ${{ github.actor }} --password-stdin
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
docker push ${REGISTRY}/${IMAGE_NAME}:latest

# 部署到K8s
echo "☸️ 部署到K8s..."
kubectl set image deployment/paper-pipeline-web \
  web=${REGISTRY}/${IMAGE_NAME}:${VERSION} \
  -n paper-pipeline

# 等待部署完成
echo "⏳ 等待部署..."
kubectl rollout status deployment/paper-pipeline-web -n paper-pipeline --timeout=120s

echo ""
echo "✅ 部署完成！"
echo "版本: ${VERSION}"
echo "镜像: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
