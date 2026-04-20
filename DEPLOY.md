# 南一高中 Demo — 部署文档

## 一、架构概览

```
┌────────────────────────────────────────┐
│  浏览器                                │
│  - 登录页（/login）                    │
│  - 主页面（/）双 tab                   │
│    - 🗺️ LBS 地图（HTML/CSS）            │
│    - 🎮 像素地图（Phaser 3）            │
│  - 右侧面板：角色 / 对话 / 上帝指令      │
│  - WebSocket 实时推送                  │
└────────────────┬───────────────────────┘
                 │ HTTP/WS
┌────────────────▼───────────────────────┐
│  FastAPI 服务（端口 8090）              │
│  - /api/login, /api/state, /api/chat   │
│  - /api/god, /api/visitor/*            │
│  - /ws 实时广播                         │
│  - 后台 asyncio 任务：sim loop          │
└────────────────┬───────────────────────┘
                 │
        ┌────────┼─────────┐
        │                  │
  ┌─────▼──────┐    ┌─────▼──────┐
  │ Sim Engine │    │ Postgres   │
  │ 3 NPCs +   │    │ (or SQLite)│
  │ N visitors │    │            │
  │ LiteLLM ↗  │    │ users      │
  └────────────┘    │ sessions   │
                    │ chat       │
                    │ snapshot   │
                    └────────────┘
```

## 二、前置要求

- Linux 服务器（Ubuntu 20.04+ 或 Debian 11+）
- Docker + Docker Compose
- 公网/内网可访问的端口 8090
- LiteLLM 网关可访问（已有）
- Postgres 14+（你们公司已有）

## 三、部署步骤

### 1. 拉代码

```bash
git clone <你的仓库> nanyi-demo
cd nanyi-demo
```

### 2. 配置环境变量

```bash
cp .env.example .env
vim .env
```

填写：
- `LITELLM_API_KEY`：LiteLLM 网关 key
- `DATABASE_URL`：Postgres 连接串，如 `postgresql://user:pwd@10.0.0.5:5432/nanyi_demo`
  - 如果暂时不用 Postgres，留默认 SQLite 也能跑（单容器部署推荐）

### 3. 预创建 Postgres 数据库（如果用 Postgres）

```bash
psql -h <db_host> -U <admin> -c "CREATE DATABASE nanyi_demo;"
psql -h <db_host> -U <admin> -c "CREATE USER nanyi_app WITH PASSWORD 'xxx';"
psql -h <db_host> -U <admin> -c "GRANT ALL ON DATABASE nanyi_demo TO nanyi_app;"
```

表会在服务首次启动时自动创建（参见 `server/db.py` 的 `init_schema`）。

### 4. 启动

```bash
docker compose up -d --build
docker compose logs -f app
```

看到 `Uvicorn running on http://0.0.0.0:8090` 即启动成功。

### 5. 访问

- 浏览器打开 `http://<服务器IP>:8090/`
- 登录账号：
  - `admin` / `nanyi2026`
  - `guest` / `nanyi2026`
  - `user1`, `user2` / `nanyi2026`

## 四、用户操作说明

### 登录后的主界面

- **顶部**：当前模拟时间 + 「🚪 进入世界」按钮
- **左侧**：可切换「🗺️ LBS 地图」和「🎮 像素地图」两个 tab
- **右侧**：三个面板
  - **角色**：看 3 个 NPC 的当前状态
  - **对话**：跟任意 NPC 对话（LLM 实时生成回复）
  - **指令**：上帝视角直接移动 NPC 到任意地点

### 作为访客进入世界

点击顶部「🚪 进入世界」按钮后：
- 地图上会出现你的绿色头像
- **键盘方向键 / WASD** 可移动
- 下拉菜单可快速传送到任一地点
- 离开时点「🚪 离开世界」

## 五、监控与维护

### 查看日志

```bash
docker compose logs -f app
```

### 重启（会保留 NPC 位置，不保留当前活动描述）

```bash
docker compose restart app
```

### 数据备份（Postgres）

```bash
pg_dump -h <host> -U <user> nanyi_demo > backup_$(date +%F).sql
```

### 数据备份（SQLite）

```bash
cp server/app.db server/app.db.$(date +%F).bak
```

## 六、扩展与修改

### 增加新用户

```sql
-- 密码 nanyi2026 的 sha256
INSERT INTO users (username, pwd_hash) VALUES
  ('newuser', '<sha256 hash>');
```

或者改 `server/db.py` 里 `init_schema` 中默认用户，重启服务。

### 修改 3 个 NPC 的人设

编辑：
- `environment/frontend_server/storage/base_nan_yi_high/personas/钟辰时/bootstrap_memory/scratch.json`
- 同上，`周往`、`陈昔`

修改 `innate`、`learned`、`daily_plan_req` 等字段，**重启服务生效**。

### 修改地图

- LBS 地图：`server/static/map_lbs.html`
- 像素地图：`server/static/map_phaser.html`

修改后浏览器刷新即可。

### 接入新 LLM

改 `.env` 的 `LLM_MODEL` 字段即可，前提是 LiteLLM 网关支持该模型。

## 七、已知限制

1. **3 个 NPC 共享一个世界**：所有用户看同一套数据，用户 A 的对话记录用户 B 也能看到。
2. **访客记忆不持久**：重启服务后，所有访客（用户）需要重新「进入世界」。
3. **LLM 失败兜底**：如果 LLM 调用失败，角色会进入 fallback 状态（显示 idle），不会崩溃。
4. **当前只用 SQLite/Postgres 存对话 + 角色快照**；角色的深度记忆（a_mem）仍在 bootstrap_memory 目录下，重启服务时重置为空。生产版本需要把 a_mem 也迁到 Postgres（TODO）。
5. **没有并发对话排队**：多个用户同时对同一个 NPC 说话会按到达顺序 FIFO 处理。

## 八、故障排查

**问：访问页面白屏**
- 检查 `docker compose logs -f app`
- 检查是否 LLM 调用超时阻塞了启动

**问：NPC 一直停在 "校门口"**
- 说明 LLM 调用全部失败（fallback 到 living_area）
- 检查 LITELLM_API_KEY 是否正确
- 手动试 `curl https://litellm.quickcan.com/v1/models`

**问：时间不推进**
- 检查 sim loop 是否在运行：`docker exec -it nanyi-demo sh -c "ps aux"`
- 默认每 3 秒推进一个 tick（模拟时间 10 分钟）
- 如需调整，改 `server/sim_engine.py` 的 `tick_interval_sec`

**问：对话没反应**
- 后端会异步生成 NPC 回复，正常需要 3-10 秒
- 查看 `docker compose logs -f app` 是否有 error

## 九、从本地开发转到线上

本地开发：

```bash
cd generative_agents
.venv/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port 8090 --reload
```

线上部署：使用上面的 Docker Compose。
