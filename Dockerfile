FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY config/ ./config/

# 创建数据目录
RUN mkdir -p data docs reports pdfs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "scripts/run_pipeline.py", "--mode", "daily"]
