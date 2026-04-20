# 南一高中 Demo — 部署交接清单

> 给同事的一页文档。详细部署步骤见 `DEPLOY.md`。

---

## 一、这个 Demo 是什么

LBS 角色世界模拟器：3 个高中生 NPC（钟辰时、周往、陈昔）在南一高中地图上自主活动，能感知、规划、反思、彼此对话；用户登录后进入世界，可以跟 NPC 聊天，也能被 NPC 感知到并主动打招呼。

**技术栈**：FastAPI + WebSocket + LangChain + LiteLLM 网关 + Langfuse trace + Postgres/SQLite

---

## 二、需要部署的东西

| 项目 | 内容 |
|------|------|
| **代码仓库** | `https://github.com/kkmh-org/stanford-town.git` （main 分支） |
| **入口文件** | `Dockerfile` + `docker-compose.yml` 已写好，开箱即用 |
| **服务端口** | `8090` |
| **依赖** | Docker + Docker Compose（无需 Python 环境） |

---

## 三、部署 5 步走

### 1. 拉代码
```bash
git clone https://github.com/kkmh-org/stanford-town.git nanyi-demo
cd nanyi-demo
```

### 2. 配置环境变量
```bash
cp .env.example .env
vim .env
```

**必填**：
- `LITELLM_API_KEY` — LiteLLM 网关 key（找我要）

**可选**：
- `DATABASE_URL` — 不填默认用 SQLite，用 Postgres 改成 `postgresql://user:pwd@host:5432/nanyi_demo`
- `LANGFUSE_*` — 已经预填了我们 generative-agents 项目的 key，不用改

### 3. （可选）准备 Postgres
```bash
psql -h <db_host> -U <admin> -c "CREATE DATABASE nanyi_demo;"
```
表会在首次启动时自动创建。

### 4. 启动
```bash
docker compose up -d --build
docker compose logs -f app
```
看到 `Uvicorn running on http://0.0.0.0:8090` 即成功。

### 5. 访问
- 浏览器打开 `http://<服务器IP>:8090/`
- 登录账号：`admin / nanyi2026`（或 guest/user1/user2，密码同）

---

## 四、需要注意的事

1. **LLM 费用**：每个 NPC 每个 tick（默认 3 秒一次）会调 4-6 次 LLM。3 个 NPC 跑一天约 5000-8000 次调用。建议设置 LiteLLM 的预算告警。

2. **Langfuse trace**：所有 LLM 调用会自动上报到 `https://langfuse.quickcan.com` 的 `generative-agents` 项目，用同事建的项目 key（已经在 .env.example 里）。

3. **NPC 记忆不持久**：重启容器后角色的对话/感知记忆会清空（bootstrap_memory 还在，但 a_mem 重置）。生产环境如果需要持久化，需要把 `a_mem` 也存 Postgres（TODO）。

4. **首次访问慢**：服务启动后 NPC 需要先生成日程，约 30-60 秒后地图上才有动静。可以先观察右侧"角色"面板的 emoji 变化。

---

## 五、常见故障速查

| 现象 | 原因 / 解决 |
|------|------------|
| 页面白屏 | `docker compose logs -f app` 看启动日志 |
| NPC 一直停在校门口 | LLM 全失败了，检查 `LITELLM_API_KEY` 和 `curl https://litellm.quickcan.com/v1/models` |
| 对话没反应 | 后端异步生成回复，正常 3-10 秒；如果一直等不到看后端 log |
| 时间不推进 | sim loop 挂了，`docker compose restart app` |
| Langfuse 看不到 trace | 检查 .env 三个 LANGFUSE_* 是否生效，重启容器 |

---

## 六、Demo 的核心功能（让同事知道在演示什么）

1. **角色自主活动**：每个 NPC 有自己的 24 小时日程（LLM 生成的，不是脚本）
2. **角色感知世界**：周围发生的事会被 NPC 感知到，重要的事记住，不重要的忽略
3. **角色反思进化**：累积一定重要性后触发反思，产生新的"想法"，影响后续行为
4. **NPC↔NPC 自动互动**：两个 NPC 走到同一地点会自发对话（按角色性格）
5. **用户被感知**：用户进入 NPC 所在地点，NPC 会主动打招呼
6. **用户主动对话**：右侧面板可以跟任意 NPC 聊天，回复符合人设
7. **上帝视角**：可以强制传送 NPC 到任意地点

---

## 七、后续迭代（同事不用关心，留个底）

- 持久化角色 a_mem 到 Postgres
- X Agent + 编剧 Agent（harness 文档里设计的氛围/事件机制）
- 偶发事件机制（下雨、考试等）
- 多用户隔离的私人对话历史
