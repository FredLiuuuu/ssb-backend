# 服务器部署指南

## 1. Clone 代码
```bash
git clone https://github.com/FredLiuuuu/ssb-backend.git
cd ssb-backend
```

## 2. Build Docker 镜像
```bash
docker build -t ssb-backend .
```

## 3. Run 容器
```bash
docker run -d \
  --name ssb-backend \
  -p 8000:8000 \
  --restart unless-stopped \
  ssb-backend
```

## 4. 验证服务
```bash
# 检查容器状态
docker ps

# 测试健康检查端点
curl http://localhost:8000/healthz

# 或者测试根端点
curl http://localhost:8000/
```

## 5. 查看日志
```bash
docker logs ssb-backend
```

## 6. 停止服务
```bash
docker stop ssb-backend
docker rm ssb-backend
```

## 7. 使用环境变量（可选）
如果需要使用 .env 文件：
```bash
docker run -d \
  --name ssb-backend \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  ssb-backend
```
