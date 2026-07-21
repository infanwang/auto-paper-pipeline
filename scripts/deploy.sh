#!/bin/bash
# Auto Paper Pipeline - 部署脚本

set -e

echo "=========================================="
echo "  Auto Paper Pipeline - 生产部署"
echo "=========================================="

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  .env文件不存在，从.env.example复制..."
    cp .env.example .env
    echo "请编辑 .env 文件配置邮箱等信息"
    exit 1
fi

# 构建镜像
echo "🔨 构建Docker镜像..."
docker-compose build

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 显示状态
echo ""
echo "✅ 部署完成！"
echo ""
echo "服务状态:"
docker-compose ps
echo ""
echo "访问地址:"
echo "  Web API: http://localhost:8000"
echo "  API文档: http://localhost:8000/docs"
echo ""
echo "常用命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  运行管道: docker-compose exec app python scripts/run_pipeline.py"
