# SSB Backend (FastAPI + Postgres/pgvector) 运维与使用说明

本文档用于在 **阿里云轻量服务器（Docker 镜像）** 上部署与维护 `ssb-backend` 后端服务，支持通过 **Gitee push/pull** 更新代码，通过 **.env** 更新密钥/配置，并说明服务器上的目录、容器、网络、数据卷等关键信息。

---

## 0. 当前部署信息（按本次实际环境）

- 公网 IP：`121.196.235.145`
- FastAPI 服务端口：`8000`
- FastAPI 容器名：`ssb-api`
- Postgres 容器名：`ssb-pg`
- Docker network：`ssb-net`
- Postgres 数据卷（宿主机路径）：`/opt/ssb-pg/data`
- 后端代码目录（建议固定）：`/opt/ssb-api`

> 备注：Postgres 容器未映射 `5432` 到公网（安全做法），仅供同一 Docker network 内访问。

---

## 1. 仓库与目录结构

### 1.1 代码仓库（Gitee）
- 仓库：`ssb-backend`（示例名）
- 典型包含文件：
  - `app.py`：FastAPI 入口
  - `requirements.txt`：依赖
  - `Dockerfile`：构建镜像
  - `deploy.sh`：服务器部署脚本（仓库内）
  - `DEPLOY.md`：部署说明（仓库内）
  - `.env`：**不提交 Git**（服务器本地维护）

### 1.2 服务器路径约定（强烈建议固定）
- 后端代码目录：`/opt/ssb-api`
- Postgres 数据目录（持久化）：`/opt/ssb-pg/data`
- 可选备份目录：`/opt/ssb-backup`

---

## 2. Docker 资源说明（容器 / 网络 / 数据）

### 2.1 容器
- `ssb-api`：FastAPI 服务容器
- `ssb-pg`：Postgres + pgvector 容器（镜像 `pgvector/pgvector:pg16`）

查看容器：
```bash
docker ps
docker ps -a
```

查看日志：

```bash
docker logs -n 200 ssb-api
docker logs -n 200 ssb-pg
```

### 2.2 Docker network

本项目使用自定义网络：`ssb-net`
创建（仅一次）：

```bash
docker network create ssb-net
```

意义：

* `ssb-api` 可以用 `ssb-pg:5432` 访问数据库（容器名当 host）
* 不需要把 Postgres 暴露到公网

查看网络：

```bash
docker network ls
docker network inspect ssb-net
```

### 2.3 Postgres 持久化数据

宿主机路径：

* `/opt/ssb-pg/data` 绑定到容器内 `/var/lib/postgresql/data`

确认数据卷目录：

```bash
ls -la /opt/ssb-pg/data
```

---

## 3. 环境变量与 `.env` 管理（密钥/配置都在这里）

### 3.1 `.env` 放置位置

`.env` 放在后端代码目录（例如 `/opt/ssb-api/.env`），不提交 git。

示例 `.env`（按当前同机数据库方案）：

```env
ENV=prod
PORT=8000

# Postgres (同一 Docker network 内通过容器名访问)
DATABASE_URL=postgresql+psycopg://ssb:ChangeMe_Strong@ssb-pg:5432/ssb

# 预留：模型端（组员提供的模型服务）
MODEL_API_URL=
MODEL_API_KEY=
```

查看 `.env`：

```bash
cd /opt/ssb-api
cat .env
```

> 修改 `.env` 后需要 **重启 ssb-api 容器** 才会生效（见第 6 章）。

### 3.2 密钥更新方式

只改 `.env`，不要改代码：

* 模型端地址/密钥：`MODEL_API_URL` / `MODEL_API_KEY`
* 未来 JWT/CORS 等也建议走环境变量

---

## 4. 初始化与启动（同机 Postgres + API）

> 如果你已经跑起来了，可跳到第 5/6 章。

### 4.1 启动 Postgres（pgvector）

（确保数据目录存在）

```bash
mkdir -p /opt/ssb-pg/data
```

启动 `ssb-pg`：

```bash
docker run -d --name ssb-pg --restart unless-stopped \
  --network ssb-net \
  -e POSTGRES_DB=ssb \
  -e POSTGRES_USER=ssb \
  -e POSTGRES_PASSWORD='ChangeMe_Strong' \
  -v /opt/ssb-pg/data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

创建 pgvector 扩展（一次性/可重复执行）：

```bash
docker exec -it ssb-pg psql -U ssb -d ssb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4.2 拉取代码并构建/启动 API

```bash
mkdir -p /opt/ssb-api
cd /opt/ssb-api

git clone <你的Gitee仓库URL> .
```

创建 `.env`（见第 3 章）。

构建镜像：

```bash
docker build -t ssb-api .
```

启动 `ssb-api`：

```bash
docker run -d --name ssb-api --restart unless-stopped \
  --network ssb-net \
  --env-file .env \
  -p 8000:8000 \
  ssb-api
```

---

## 5. API 验证方式（本机 / 公网）

### 5.1 服务器本机验证

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/
```

### 5.2 公网验证

在你的电脑（macOS/zsh）：

```bash
curl -s http://121.196.235.145:8000/healthz
curl -s "http://121.196.235.145:8000/v1/records?limit=5"
```

> zsh 下 URL 里包含 `?` 必须用引号包裹，否则会触发 glob 报错。

---

## 6. 日常更新流程（Gitee push/pull + 重建容器）

### 6.1 推荐流程（标准）

1. 本地开发机修改代码 → `git commit` → `git push`
2. 服务器更新并重启容器

服务器执行：

```bash
cd /opt/ssb-api
git pull

docker build -t ssb-api .
docker rm -f ssb-api || true
docker run -d --name ssb-api --restart unless-stopped \
  --network ssb-net \
  --env-file .env \
  -p 8000:8000 \
  ssb-api
```

### 6.2 使用 `deploy.sh` 一键部署（如果仓库已有）

```bash
cd /opt/ssb-api
chmod +x deploy.sh
./deploy.sh
```

### 6.3 修改 `.env` 后如何生效

只需要重启 `ssb-api`（不必重建镜像）：

```bash
cd /opt/ssb-api
docker rm -f ssb-api || true
docker run -d --name ssb-api --restart unless-stopped \
  --network ssb-net \
  --env-file .env \
  -p 8000:8000 \
  ssb-api
```

---

## 7. 数据写入/读取（records 接口）

### 7.1 写入一条 record

```bash
curl -s http://127.0.0.1:8000/v1/records/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "id":"test-1",
    "ts": 1768596000,
    "user_text":"hello",
    "assistant_text":"world",
    "summary":"demo",
    "tags":[{"name":"demo","confidence":0.9}]
  }'
```

### 7.2 读取 records

```bash
curl -s "http://127.0.0.1:8000/v1/records?limit=5"
```

---

## 8. Postgres 检查与备份

### 8.1 进入 psql

```bash
docker exec -it ssb-pg psql -U ssb -d ssb
```

### 8.2 快速检查

```bash
docker exec -it ssb-pg psql -U ssb -d ssb -c "select now();"
docker exec -it ssb-pg psql -U ssb -d ssb -c "\dt"
docker exec -it ssb-pg psql -U ssb -d ssb -c "select id, summary from records order by ts desc limit 5;"
```

### 8.3 备份（pg_dump）

建议备份到宿主机目录 `/opt/ssb-backup`：

```bash
mkdir -p /opt/ssb-backup
docker exec -t ssb-pg pg_dump -U ssb -d ssb > /opt/ssb-backup/ssb_$(date +%F_%H%M%S).sql
ls -lh /opt/ssb-backup | tail
```

### 8.4 恢复（危险操作，谨慎）

```bash
cat /opt/ssb-backup/<file>.sql | docker exec -i ssb-pg psql -U ssb -d ssb
```

---

## 9. 常见问题排查

### 9.1 公网访问不通（本机通）

* 确认轻量服务器防火墙放行 `TCP 8000`
* 服务器上确认端口监听：

  ```bash
  ss -lntp | grep 8000 || true
  ```
* 确认容器端口映射：

  ```bash
  docker ps | grep ssb-api
  ```

### 9.2 `git pull` 要求输入用户名密码

Gitee HTTPS pull 默认要输入账号/密码（或 token）。
推荐改为 SSH 方式（长期方案）：

* 在服务器生成 SSH key，添加到 Gitee
* 将远端改为 SSH URL

### 9.3 `DATABASE_URL is required` 或 DB 连接失败

* 检查 `.env` 是否存在且包含 `DATABASE_URL`

  ```bash
  cd /opt/ssb-api
  cat .env
  ```
* 检查 `ssb-pg` 是否在同一网络并运行：

  ```bash
  docker ps | grep ssb-pg
  docker network inspect ssb-net | grep ssb-pg -n || true
  ```
* 在 `ssb-api` 容器内测试 DNS（可选）：

  ```bash
  docker exec -it ssb-api sh -lc "python -c 'import socket; print(socket.gethostbyname(\"ssb-pg\"))'"
  ```

### 9.4 Postgres 数据"看起来没了"

只要 `/opt/ssb-pg/data` 没被删，数据不会丢。确认目录仍在：

```bash
ls -la /opt/ssb-pg/data | head
```

---

## 10. 安全建议（最低限度）

* 不要在轻量防火墙放行 `5432`（数据库不暴露公网）
* `.env` 不提交 git，避免泄露密钥
* `POSTGRES_PASSWORD` 改为强密码
* 定期 `pg_dump` 备份（尤其是 hackathon 期间）

---

## 11. 下一阶段扩展建议（与模型端/前端联动）

### 11.1 建议的服务职责

* 模型端：提供 HTTP API，返回 `assistant_text + summary + tags[]`（结构化 JSON）
* 后端（ssb-api）：调用模型端、落库、对前端提供统一 API（未来加 SSE stream）
* 前端：只对接后端 `http://121.196.235.145:8000`

### 11.2 建议新增接口（后续）

* `POST /v1/chat`：非流式（最快联调）
* `POST /v1/chat/stream`：SSE/WS 流式
* `projects/graph/revisions/recommendations`：逐步补齐产品功能

---

## 12. 快速命令速查表

### 查看状态

```bash
docker ps
docker logs -n 80 ssb-api
docker logs -n 80 ssb-pg
```

### 重启 API（.env 改动后）

```bash
cd /opt/ssb-api
docker rm -f ssb-api || true
docker run -d --name ssb-api --restart unless-stopped \
  --network ssb-net \
  --env-file .env \
  -p 8000:8000 \
  ssb-api
```

### 更新代码并部署

```bash
cd /opt/ssb-api
git pull
docker build -t ssb-api .
docker rm -f ssb-api || true
docker run -d --name ssb-api --restart unless-stopped \
  --network ssb-net \
  --env-file .env \
  -p 8000:8000 \
  ssb-api
```

### 数据库检查

```bash
docker exec -it ssb-pg psql -U ssb -d ssb -c "select 1;"
docker exec -it ssb-pg psql -U ssb -d ssb -c "select id, summary from records order by ts desc limit 5;"
```

---
