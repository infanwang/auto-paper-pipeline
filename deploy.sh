#!/bin/bash
# Paper Pipeline Deployment Script

set -e

echo "=========================================="
echo "Paper Pipeline Deployment"
echo "=========================================="
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please create .env file with required environment variables"
    exit 1
fi

# Build and start containers
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo
echo "Services:"
echo "  - Pipeline: docker-compose logs -f paper-pipeline"
echo "  - Web UI: http://localhost:8000"
echo
echo "Commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop: docker-compose down"
echo "  - Restart: docker-compose restart"
echo "  - Update: docker-compose pull && docker-compose up -d"
