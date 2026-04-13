# Stanford Town — Generative Agents (LangChain 改造版)

<p align="center" width="100%">
<img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

基于斯坦福 [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) 论文的代码实现，经过 LangChain 改造，对接团队 LiteLLM 网关，支持多模型切换和向量数据库。

---

## 项目架构

### 整体结构

```
stanford-town/
├── environment/           # Django 前端（地图渲染、动画）
│   └── frontend_server/
│       ├── manage.py
│       └── storage/       # 模拟存档数据
├── reverie/               # Agent 模拟后端
│   └── backend_server/
│       ├── reverie.py     # 主循环入口
│       ├── maze.py        # 地图网格 + 碰撞检测
│       ├── path_finder.py # A* 寻路
│       ├── llm_config.py  # LiteLLM 网关配置
│       └── persona/       # Agent 实现
│           ├── persona.py                    # Agent 主类，认知管线
│           ├── memory_structures/
│           │   ├── spatial_memory.py         # 空间记忆树
│           │   ├── associative_memory.py     # 关联记忆 + VectorStore
│           │   └── scratch.py               # 工作记忆（身份/计划/状态）
│           ├── cognitive_modules/
│           │   ├── perceive.py              # 感知
│           │   ├── retrieve.py              # 检索（三因素加权）
│           │   ├── plan.py                  # 规划（长期/短期/社交）
│           │   ├── reflect.py               # 反思
│           │   ├── execute.py               # 执行（寻路）
│           │   └── converse.py              # 对话
│           └── prompt_template/
│               ├── gpt_structure.py          # LLM 调用层（LangChain）
│               ├── chain_utils.py            # Prompt 函数 v2 实现
│               ├── run_gpt_prompt.py         # Prompt 函数注册入口
│               └── v2/, v3_ChatGPT/          # .txt 模板文件
└── tests/                 # 自动化测试（34 个）
```

### Agent 认知架构

每个 Agent 由 **三层记忆 + 五个认知模块** 组成：

```
┌─────────────────────────────────────────────────────────┐
│                      Persona                            │
│                                                         │
│  ┌─────────────────── 记忆层 ───────────────────────┐   │
│  │  Spatial Memory    Associative Memory    Scratch  │   │
│  │  (空间地图树)       (记忆流+VectorStore)  (工作记忆)│   │
│  └───────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────── 认知管线 (每 tick) ───────────────┐   │
│  │  Perceive → Retrieve → Plan → Reflect → Execute  │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

每个 tick 的数据流：

```
前端写 environment/{step}.json
  → Reverie 读取，同步位置
  → 每个 Agent 执行 move()：感知→检索→规划→反思→执行
  → 写 movement/{step}.json
  → 前端读取并渲染动画
```

详细的代码阅读笔记见 [code_reading.md](code_reading.md)。

---

## 技术栈（改造后）

| 层面 | 实现 |
|------|------|
| **LLM 调用** | LangChain `ChatOpenAI` → LiteLLM 网关（支持 deepseek / claude / gpt / gemini） |
| **Embedding** | LangChain `OpenAIEmbeddings` → LiteLLM 网关（`text-embedding-3-small`） |
| **Prompt 管理** | `ChatPromptTemplate` + `run_chain` / `run_prompt_str` |
| **向量检索** | FAISS VectorStore（可切换腾讯云向量数据库） |
| **记忆存储** | JSON/CSV 文件 + FAISS 索引持久化 |
| **Web 框架** | Django（前端地图渲染） |
| **测试** | pytest，34 个自动化测试 |

---

## 快速开始

### 1. 环境准备

```bash
cd reverie/backend_server

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install langchain-core langchain-openai langchain-community faiss-cpu
pip install -r ../../requirements.txt
```

### 2. 配置 LLM

编辑 `reverie/backend_server/llm_config.py`，或通过环境变量设置：

```bash
export LITELLM_BASE_URL="https://litellm.quickcan.com"
export LITELLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-5-mini"                    # 可选：claude-4.6-sonnet, deepseek-v3.2-ali 等
export EMBEDDING_MODEL="text-embedding-3-small"
```

> 不再需要手动创建 `utils.py`，路径配置已自动生成。

### 3. 运行测试

```bash
# 冒烟测试（验证 LLM 网关连通性）
.venv/bin/pytest tests/test_smoke.py -v

# 全量测试
.venv/bin/pytest tests/ -v
```

### 4. 启动模拟

```bash
# 终端 1：启动前端
cd environment/frontend_server
python manage.py runserver

# 终端 2：启动 Agent 后端
cd reverie/backend_server
python reverie.py
```

按提示输入 fork 模拟名（如 `base_the_ville_isabella_maria_klaus`）和新模拟名，然后 `run 100` 开始模拟。

---

## 改造详情

本仓库在原版 Stanford Generative Agents 基础上完成了以下改造：

### 阶段一：LLM 调用层

- `openai==0.27.0` → LangChain `ChatOpenAI` + `OpenAIEmbeddings`
- 所有 LLM 调用通过 LiteLLM 网关，可切换任意模型
- 函数签名保持不变，上层认知模块零改动

### 阶段二三：Prompt 模板 + 输出解析

- 27 个 prompt 函数全部迁移到 `chain_utils.py` 的 v2 实现
- 17 个简单/中等函数：`.txt` 模板内联为 `ChatPromptTemplate`
- 10 个复杂函数：保留 `.txt` 模板渲染，替换 LLM 调用为 `run_prompt_str`
- 统一的重试 + fail_safe 机制（`run_chain` / `run_prompt_str`）

### 阶段四：VectorStore 向量检索

- `AssociativeMemory` 新增 FAISS VectorStore 双写
- `extract_relevance` 优先走 FAISS 索引（O(log n)），fallback 到 cos_sim（O(n)）
- 三因素加权逻辑（recency × relevance × importance）完全保留
- 上线时切腾讯云向量数据库只需改一行 import

完整改造方案和执行记录见 [2026-04-13-langchain-migration-plan.md](2026-04-13-langchain-migration-plan.md)。

---

## 测试

```
tests/test_smoke.py               8 passed   LLM 调用层 + LiteLLM 网关连通性
tests/test_phase2_prompts.py      19 passed   27 个 prompt 函数输出格式验证
tests/test_phase4_vectorstore.py   5 passed   VectorStore 写入/检索/持久化
tests/test_integration.py          2 passed   Retrieve 模块端到端
──────────────────────────────────────────────
Total                             34 passed
```

---

## 切换腾讯云向量数据库

上线时修改 `associative_memory.py` 中的 `_init_vectorstore` 方法：

```python
# 开发阶段（当前）
from langchain_community.vectorstores import FAISS

# 上线阶段
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import ConnectionParams

vectorstore = TencentVectorDB(
    embedding=_embeddings,
    connection_params=ConnectionParams(url="your_url", key="your_key"),
    collection_name="agent_memory",
)
```

安装依赖：`pip install tcvectordb`

---

## 原始论文

> Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein. **Generative Agents: Interactive Simulacra of Human Behavior.** UIST 2023.

```bibtex
@inproceedings{Park2023GenerativeAgents,
  author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title = {Generative Agents: Interactive Simulacra of Human Behavior},
  year = {2023},
  publisher = {Association for Computing Machinery},
  booktitle = {UIST '23},
}
```

## 致谢

原始项目游戏素材设计：
- 背景：[PixyMoon (@_PixyMoon\_)](https://twitter.com/_PixyMoon_)
- 家具/室内：[LimeZu (@lime_px)](https://twitter.com/lime_px)
- 角色：[ぴぽ (@pipohi)](https://twitter.com/pipohi)
