#!/bin/bash
# 快速部署脚本

set -e

echo "🚀 开始部署 ssb-backend..."

# 1. Clone 代码（如果目录不存在）
if [ ! -d "ssb-backend" ]; then
    echo "📦 Cloning 代码..."
    git clone https://github.com/FredLiuuuu/ssb-backend.git
    cd ssb-backend
else
    echo "📦 更新代码..."
    cd ssb-backend
    git pull
fi

# 2. Build Docker 镜像
echo "🔨 构建 Docker 镜像..."
docker build -t ssb-backend .

# 3. 停止并删除旧容器（如果存在）
if [ "$(docker ps -aq -f name=ssb-backend)" ]; then
    echo "🛑 停止旧容器..."
    docker stop ssb-backend || true
    docker rm ssb-backend || true
fi

# 4. Run 新容器
echo "▶️  启动新容器..."
docker run -d \
  --name ssb-backend \
  -p 8000:8000 \
  --restart unless-stopped \
  ssb-backend

# 5. 等待服务启动
echo "⏳ 等待服务启动..."
sleep 3

# 6. 验证服务
echo "✅ 验证服务..."
if curl -f http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "✅ 服务运行正常！"
    echo "📊 健康检查: http://localhost:8000/healthz"
    echo "🏠 根端点: http://localhost:8000/"
else
    echo "❌ 服务启动失败，查看日志: docker logs ssb-backend"
    exit 1
fi

echo "🎉 部署完成！"
