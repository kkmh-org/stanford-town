# Generative Agents × LangChain 改造方案

> 日期：2026-04-13
>
> 目标：将 Generative Agents 的底层基础设施替换为 LangChain，对接团队现有的 LiteLLM 网关，同时保持上层认知架构（感知-检索-规划-反思-执行）不变。

---

## 一、现状分析

### 1.1 Generative Agents 现有技术栈

| 层面 | 现有实现 | 问题 |
|------|---------|------|
| **LLM 调用** | 裸 `openai==0.27.0`（旧版 SDK），硬编码 `gpt-3.5-turbo` / `gpt-4` / `text-davinci-003` | 锁死 OpenAI，无法换模型 |
| **Embedding** | `openai.Embedding.create`，硬编码 `text-embedding-ada-002` | 同上 |
| **Prompt 管理** | 自写 `.txt` 模板 + `!<INPUT N>!` 字符串替换 | 无 few-shot、无组合、难维护 |
| **输出解析** | 手写 `json.loads` + `func_validate` + `func_clean_up`，循环重试 | 脆弱，每个函数重复写 |
| **向量检索** | `dict` 存 embedding + `numpy.dot` 遍历算 `cos_sim` | O(n) 暴力遍历，无索引 |
| **记忆存储** | JSON/CSV 文件读写 | 无持久化保障 |

### 1.2 团队现有基础设施（vibe-engine）

| 组件 | 说明 |
|------|------|
| **LiteLLM 网关** | `https://litellm.quickcan.com`，OpenAI 兼容 API |
| **可用模型** | `deepseek-v3.2-ali`、`claude-4.6-sonnet`、`claude-4.6-opus`、`gpt-5.4-pro`、`gpt-5-mini`、`gemini-2.5-pro` 等 |
| **调用方式** | OpenAI SDK（`base_url` 指向网关）或 `requests.post` 直调 |
| **配置管理** | Nacos 配置中心（`rockagent_llm_model_center`）+ `.env` 环境变量 |
| **目标向量数据库** | 腾讯云向量数据库（LangChain 已有 `TencentVectorDB` 集成） |

### 1.3 改造后的目标架构

```
┌────────────────────────────────────────────────────────────────────┐
│                    Generative Agents（改造后）                      │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            认知管线（保持不变）                                 │  │
│  │  perceive → retrieve → plan → reflect → execute              │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  run_gpt_prompt.py                                           │  │
│  │  + LangChain PromptTemplate / ChatPromptTemplate             │  │
│  │  + LangChain OutputParser (JSON / Pydantic)                  │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  gpt_structure.py                                            │  │
│  │  LangChain ChatOpenAI(base_url=LiteLLM 网关)                 │  │
│  │  LangChain OpenAIEmbeddings(base_url=LiteLLM 网关)           │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  associative_memory.py                                       │  │
│  │  LangChain VectorStore (FAISS → 腾讯云 VectorDB)             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────┐
│              LiteLLM 网关 (https://litellm.quickcan.com)           │
│                                                                    │
│  deepseek-v3.2-ali | claude-4.6-sonnet | gpt-5.4-pro | gemini ... │
└────────────────────────────────────────────────────────────────────┘
```

与 vibe-engine 共享 LiteLLM 网关，但代码不耦合：

```
┌──────────────┐      ┌──────────────────┐
│ vibe-engine  │      │generative_agents │
│ OpenaiLLM /  │      │ LangChain        │
│ LiteLLMAdapter│     │ ChatOpenAI       │
└──────┬───────┘      └───────┬──────────┘
       │   同一个 base_url     │
       ▼                      ▼
┌──────────────────────────────────────┐
│      LiteLLM 网关 (统一管理)          │
└──────────────────────────────────────┘
```

---

## 二、改造步骤

### 阶段一：LLM 调用层替换

**目标**：用 LangChain 替换 `gpt_structure.py` 中的裸 OpenAI SDK 调用，对接 LiteLLM 网关。上层零改动。

**改动文件**：
- `reverie/backend_server/persona/prompt_template/gpt_structure.py`
- 新增配置文件（API key、base_url、模型名）

**风险**：低
**工作量**：小（1 天）

#### 1.1 新增配置

新建 `reverie/backend_server/config.py`，集中管理 LLM 配置：

```python
import os

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://litellm.quickcan.com")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-xxx")

# 可随时切换模型，不改代码
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-4.6-sonnet")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
```

#### 1.2 改造 gpt_structure.py

```python
# 替换前
import openai
openai.api_key = openai_api_key

def ChatGPT_request(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text: text = "this is blank"
    return openai.Embedding.create(
        input=[text], model=model)['data'][0]['embedding']
```

```python
# 替换后
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import LITELLM_BASE_URL, LITELLM_API_KEY, LLM_MODEL, EMBEDDING_MODEL

_llm = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=LLM_MODEL,
)

_embeddings = OpenAIEmbeddings(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=EMBEDDING_MODEL,
)

def ChatGPT_request(prompt):          # 函数签名不变
    response = _llm.invoke(prompt)
    return response.content

def GPT4_request(prompt):             # 函数签名不变
    response = _llm.invoke(prompt)    # 模型已由配置决定，不再区分 3.5/4
    return response.content

def ChatGPT_single_request(prompt):   # 函数签名不变
    response = _llm.invoke(prompt)
    return response.content

def get_embedding(text, model=None):  # 函数签名不变
    text = text.replace("\n", " ")
    if not text: text = "this is blank"
    return _embeddings.embed_query(text)
```

保留以下函数的签名和行为，**只替换内部实现**：

| 函数 | 调用者数量 | 改造方式 |
|------|-----------|---------|
| `ChatGPT_single_request(prompt)` | 少量 | `_llm.invoke(prompt).content` |
| `ChatGPT_request(prompt)` | ~15 处 | `_llm.invoke(prompt).content` |
| `GPT4_request(prompt)` | ~5 处 | `_llm.invoke(prompt).content` |
| `GPT_request(prompt, gpt_parameter)` | ~10 处 | `_llm.invoke(prompt, **mapped_params).content` |
| `get_embedding(text)` | 8 处 | `_embeddings.embed_query(text)` |
| `generate_prompt(input, file)` | 31 处 | 暂不动，阶段二改 |
| `safe_generate_response(...)` | ~10 处 | 暂不动，阶段三改 |
| `ChatGPT_safe_generate_response(...)` | ~15 处 | 暂不动，阶段三改 |
| `GPT4_safe_generate_response(...)` | ~5 处 | 暂不动，阶段三改 |

#### 1.3 处理 GPT_request 的参数映射

旧的 `GPT_request` 接受 `gpt_parameter` dict（含 `engine`、`temperature` 等）。改造时需做参数映射：

```python
def GPT_request(prompt, gpt_parameter):
    llm_with_params = _llm.bind(
        temperature=gpt_parameter.get("temperature", 0),
        max_tokens=gpt_parameter.get("max_tokens", 150),
        stop=gpt_parameter.get("stop", None),
    )
    response = llm_with_params.invoke(prompt)
    return response.content
```

#### 1.4 验证清单

- [ ] 所有 31 个 `run_gpt_prompt_*` 函数调用正常
- [ ] Embedding 生成格式一致（返回 float list）
- [ ] `safe_generate_response` 系列重试逻辑正常
- [ ] 5 个认知模块端到端跑通

---

### 阶段二：Prompt 模板迁移

**目标**：将 `.txt` 模板 + `!<INPUT N>!` 替换为 LangChain `PromptTemplate`，提升可维护性。

**改动文件**：
- `gpt_structure.py` 中的 `generate_prompt` 函数
- `run_gpt_prompt.py` 中 31 个 prompt 函数
- `prompt_template/v2/` 和 `v3_ChatGPT/` 下的 `.txt` 模板文件

**风险**：低
**工作量**：中（2-3 天，但可渐进）

#### 2.1 兼容策略

先改 `generate_prompt` 支持两种模式，然后逐个迁移：

```python
from langchain_core.prompts import PromptTemplate

def generate_prompt(curr_input, prompt_lib_file):
    """兼容旧 .txt 模板和新 LangChain 模板"""
    if prompt_lib_file.endswith('.txt'):
        # 旧模式：保持原有逻辑
        if type(curr_input) == type("string"):
            curr_input = [curr_input]
        curr_input = [str(i) for i in curr_input]
        f = open(prompt_lib_file, "r")
        prompt = f.read()
        f.close()
        for count, i in enumerate(curr_input):
            prompt = prompt.replace(f"!<INPUT {count}>!", i)
        if "<commentblockmarker>###</commentblockmarker>" in prompt:
            prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
        return prompt.strip()
    else:
        # 新模式：LangChain PromptTemplate
        template = PromptTemplate.from_file(prompt_lib_file)
        return template.format(**curr_input)
```

#### 2.2 迁移示例

以 `wake_up_hour` 为例：

```python
# 迁移前
prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
prompt_input = [persona.scratch.get_str_iss(),
                persona.scratch.get_str_lifestyle(),
                persona.scratch.get_str_firstname()]
prompt = generate_prompt(prompt_input, prompt_template)

# 迁移后
from langchain_core.prompts import ChatPromptTemplate

wake_up_prompt = ChatPromptTemplate.from_messages([
    ("system", "You determine what time a person would wake up based on their profile."),
    ("human", "{iss}\n\n{name}'s lifestyle: {lifestyle}\n\nWhat time does {name} wake up? (answer in 'X am' format)")
])

prompt = wake_up_prompt.format(
    iss=persona.scratch.get_str_iss(),
    lifestyle=persona.scratch.get_str_lifestyle(),
    name=persona.scratch.get_str_firstname()
)
```

#### 2.3 迁移优先级

按调用频率排序，先迁移高频 prompt：

| 优先级 | 函数 | 每 tick 调用次数 |
|--------|------|----------------|
| P0 | `event_poignancy` | 每个感知事件 1 次 |
| P0 | `pronunciatio` | 每次行动 1 次 |
| P0 | `event_triple` | 每次行动 1 次 |
| P1 | `daily_plan` / `hourly_schedule` / `task_decomp` | 每天/每小时 |
| P1 | `action_sector` / `action_arena` / `action_game_object` | 每次行动 |
| P2 | `decide_to_talk` / `decide_to_react` | 遇到其他 Agent 时 |
| P2 | `focal_pt` / `insight_and_guidance` | 反思触发时 |
| P3 | 其余 | 低频 |

---

### 阶段三：输出解析重构

**目标**：用 LangChain `OutputParser` 替换手写的 `func_validate` / `func_clean_up` / 重试循环。

**改动文件**：
- `gpt_structure.py` 中的 `safe_generate_response` 系列
- `run_gpt_prompt.py` 中 31 个函数的解析逻辑

**风险**：中
**工作量**：大（3-5 天，31 个函数逐个改）

**建议与阶段二合并**：改一个 prompt 函数时同时完成模板迁移 + 输出解析。

#### 3.1 通用模式

```python
# 迁移前：每个函数都重复的模式
def run_gpt_prompt_xxx(persona, ...):
    def __func_validate(gpt_response, prompt=""):
        try: __func_clean_up(gpt_response)
        except: return False
        return True

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.strip().split("\n")[0]

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                     __func_validate, __func_clean_up)

# 迁移后：LangChain Chain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import RetryOutputParser

chain = prompt_template | _llm | JsonOutputParser()  # 或 StrOutputParser()
output = chain.invoke({"input": ...})
```

#### 3.2 输出类型分类

31 个 prompt 函数的输出大致分四类：

| 输出类型 | 函数数量 | Parser |
|---------|---------|--------|
| 简单字符串 | ~10 个 | `StrOutputParser` |
| 整数/数字 | ~5 个 | `StrOutputParser` + 自定义转换 |
| JSON 结构 | ~10 个 | `JsonOutputParser` |
| 列表 | ~6 个 | `JsonOutputParser` 或 `CommaSeparatedListOutputParser` |

---

### 阶段四：VectorStore 向量检索层

**目标**：用 LangChain VectorStore 替换 `dict` + `cos_sim` 暴力检索，最终对接腾讯云向量数据库。

**改动文件**：
- `persona/memory_structures/associative_memory.py`
- `persona/cognitive_modules/retrieve.py`

**风险**：中
**工作量**：中（2-3 天）

**与阶段二三无依赖关系，可并行。仅依赖阶段一（Embedding 层）。**

#### 4.1 AssociativeMemory 改造

```python
# 迁移前
class AssociativeMemory:
    def __init__(self, f_saved):
        self.embeddings = json.load(open(f_saved + "/embeddings.json"))  # 巨大的 dict

    def add_event(self, ...):
        self.embeddings[embedding_pair[0]] = embedding_pair[1]  # 存到 dict

# 迁移后
from langchain_community.vectorstores import FAISS  # 开发阶段
# from langchain_community.vectorstores import TencentVectorDB  # 上线阶段

class AssociativeMemory:
    def __init__(self, f_saved):
        self.vectorstore = self._load_or_create_vectorstore(f_saved)
        self.embeddings = {}  # 保留，兼容旧的关键词检索路径

    def _load_or_create_vectorstore(self, f_saved):
        index_path = f_saved + "/faiss_index"
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, _embeddings)
        return FAISS.from_texts(["init"], _embeddings)

    def add_event(self, ...):
        # 同时写入 VectorStore 和保留 dict（过渡期兼容）
        self.vectorstore.add_texts(
            [description],
            metadatas=[{"node_id": node_id, "type": "event",
                       "poignancy": poignancy, "created": str(created)}]
        )
        self.embeddings[embedding_pair[0]] = embedding_pair[1]
```

#### 4.2 retrieve.py 改造

```python
# 迁移前
def extract_relevance(persona, nodes, focal_pt):
    focal_embedding = get_embedding(focal_pt)
    relevance_out = {}
    for node in nodes:                    # O(n) 遍历
        node_embedding = persona.a_mem.embeddings[node.embedding_key]
        relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)
    return relevance_out

# 迁移后
def extract_relevance(persona, nodes, focal_pt):
    results = persona.a_mem.vectorstore.similarity_search_with_relevance_scores(
        focal_pt, k=len(nodes)
    )
    relevance_out = {}
    for doc, score in results:
        node_id = doc.metadata["node_id"]
        if node_id in {n.node_id for n in nodes}:
            relevance_out[node_id] = score
    # 补全未命中的节点（VectorStore 可能 k 不够大）
    for node in nodes:
        if node.node_id not in relevance_out:
            relevance_out[node.node_id] = 0.0
    return relevance_out
```

**三因素加权逻辑完全保留**：

```python
# 不变
gw = [0.5, 3, 2]
for key in recency_out.keys():
    master_out[key] = (recency_w * recency_out[key] * gw[0]
                     + relevance_w * relevance_out[key] * gw[1]
                     + importance_w * importance_out[key] * gw[2])
```

#### 4.3 切换腾讯云向量数据库

开发/测试阶段用 FAISS，上线时只改初始化：

```python
# 开发阶段
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(texts, embedding=_embeddings)

# 上线阶段 —— 只改这里
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import ConnectionParams

vectorstore = TencentVectorDB(
    embedding=_embeddings,
    connection_params=ConnectionParams(
        url=os.environ["TENCENT_VDB_URL"],
        key=os.environ["TENCENT_VDB_KEY"],
    ),
    collection_name="agent_memory",
)

# 以下代码完全不变
results = vectorstore.similarity_search_with_relevance_scores(query, k=30)
```

---

## 三、依赖管理

### 3.1 新增 Python 依赖

```
langchain-core>=0.3
langchain-openai>=0.3
langchain-community>=0.3
faiss-cpu>=1.7          # 开发阶段向量检索
tcvectordb              # 上线阶段腾讯云向量数据库
```

### 3.2 移除的依赖

```
openai==0.27.0          # 被 langchain-openai 内部管理
```

---

## 四、执行计划

```
阶段一 (1天)                    阶段四 (2-3天)
LLM 调用层替换                  VectorStore 向量检索层
gpt_structure.py               associative_memory.py
对接 LiteLLM 网关               retrieve.py
  │                               │
  │                               │（可并行）
  ▼                               │
阶段二+三 (3-5天)                  │
Prompt 模板 + 输出解析             │
run_gpt_prompt.py (31个函数)      │
  │                               │
  ▼                               ▼
集成测试 (1-2天)
端到端跑通完整模拟
  │
  ▼
切换腾讯云 VectorDB (0.5天)
改 import + 配置连接参数
```

### 里程碑

| 里程碑 | 完成标志 | 预计时间 |
|--------|---------|---------|
| M1：LLM 调通 | 所有 prompt 函数通过 LiteLLM 网关正常返回 | 第 1 天 |
| M2：Prompt + 解析 | 31 个函数全部迁移完成 | 第 4 天 |
| M3：VectorStore | 记忆检索走 FAISS，三因素加权结果正确 | 第 4 天 |
| M4：集成测试 | 完整模拟跑通 3+ 个 Agent 24 小时 | 第 6 天 |
| M5：上线 | 切换腾讯云向量数据库 | 第 7 天 |

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| LangChain ChatOpenAI 与 LiteLLM 网关的兼容性 | 阶段一 | 先用简单 prompt 测试连通性 |
| `GPT_request` 的 Completion API 已废弃 | 阶段一 | 统一改为 Chat API，旧 `text-davinci` 模板适配 |
| 31 个 prompt 函数改动量大 | 阶段二三 | 渐进迁移，新旧模板共存 |
| VectorStore 的 `similarity_search` 返回格式与原有 `cos_sim` 不同 | 阶段四 | 归一化处理 + 充分的单元测试 |
| 腾讯云 VDB 的 LangChain 集成版本兼容性 | 阶段四 | 先用 FAISS 验证接口一致性 |

---

## 六、改造原则

1. **自底向上**：先改最底层（LLM 调用），逐步往上改，每一步都可独立验证
2. **函数签名不变**：阶段一不改任何对外接口，上层代码零改动
3. **渐进迁移**：阶段二三支持新旧模板共存，不需要一次性全部改完
4. **双写过渡**：阶段四 VectorStore 与 dict 双写，确保回退能力
5. **配置外置**：模型名、API key、网关地址全部走环境变量，不硬编码

---

## 七、2026-04-13 改造完成总结

### 改造全景

```
改造前                                    改造后
─────────────                            ─────────────
openai==0.27.0 (旧版SDK)                  LangChain ChatOpenAI + OpenAIEmbeddings
硬编码 gpt-3.5-turbo / gpt-4              通过 LiteLLM 网关，可切换任意模型
.txt 模板 + !<INPUT N>! 替换              ChatPromptTemplate (17个) + generate_prompt (10个)
手写 func_validate/clean_up × 31          run_chain / run_prompt_str 统一重试
dict 存 embedding + numpy cos_sim O(n)    FAISS VectorStore 索引检索 + dict 双写
无 utils.py（需手动创建）                   自动生成 utils.py
无测试                                    34 个自动化测试
```

### 阶段一：LLM 调用层替换 ✅

**做了什么**：替换 `gpt_structure.py` 内部实现，所有函数签名不变

| 改动 | 说明 |
|------|------|
| 新增 `llm_config.py` | LiteLLM 网关 base_url / api_key / 模型名，全部可通过环境变量覆盖 |
| `import openai` → `from langchain_openai import ChatOpenAI, OpenAIEmbeddings` | 不再直接依赖 openai SDK |
| `openai.ChatCompletion.create()` → `_llm.invoke().content` | 所有 LLM 调用统一走 LangChain |
| `openai.Embedding.create()` → `_embeddings.embed_query()` | Embedding 也走 LangChain |
| `from utils import *` 加 try/except | 测试环境下不依赖手动创建的 utils.py |

**踩过的坑**：
1. LiteLLM 网关上没有 `text-embedding-ada-002`，改为 `text-embedding-3-small`
2. `gpt-5-mini` 是推理模型，`max_tokens` 会被 reasoning tokens 占用导致空输出 → 不传 `max_tokens`

### 阶段二三：Prompt 模板 + 输出解析迁移 ✅

**做了什么**：全部 27 个 prompt 函数迁移到 `chain_utils.py` 的 v2 实现

| 批次 | 函数数 | 迁移策略 | 代表函数 |
|------|--------|---------|---------|
| P0 高频 | 5 | `ChatPromptTemplate` + `run_chain` | `event_poignancy`, `pronunciatio`, `event_triple` |
| P1 计划 | 2 | `ChatPromptTemplate` + `run_chain` | `wake_up_hour`, `daily_plan` |
| 批次 A 简单 | 10 | `ChatPromptTemplate` + `run_chain` + `_simple_str_clean` | `summarize_conversation`, `memo_on_convo`, `act_obj_desc` 等 |
| 批次 B 中等 | 7 | `ChatPromptTemplate` + `run_chain` + 专用解析 | `focal_pt`, `insight_and_guidance`, `extract_keywords` 等 |
| 批次 C 复杂 | 10 | `generate_prompt` + .txt 模板 + `run_prompt_str` | `task_decomp`, `action_sector`, `decide_to_talk`, `create_conversation` 等 |
| **合计** | **34** | | 含 7 个 `_legacy` 重命名的原 P0/P1 函数 |

**架构设计**：

```
run_gpt_prompt.py（入口）
  │
  ├── 顶部 import + 赋值：run_gpt_prompt_xxx = run_gpt_prompt_xxx_v2
  │
  └── 旧函数全部重命名为 _run_gpt_prompt_xxx_legacy（保留供参考）

chain_utils.py（新增，~1100 行）
  │
  ├── run_chain()         ← 接受 ChatPromptTemplate，带重试 + fail_safe
  ├── run_prompt_str()    ← 接受预构建 prompt 字符串，带重试 + fail_safe
  │
  ├── 17 个 ChatPromptTemplate 常量（_POIGNANCY_PROMPT 等）
  │
  └── 27 个 run_gpt_prompt_xxx_v2() 函数
```

### 阶段四：VectorStore 向量检索 ✅

**做了什么**：在 `AssociativeMemory` 中新增 FAISS VectorStore，双写 + 索引检索

| 改动 | 说明 |
|------|------|
| `_init_vectorstore()` | 从已有 embeddings.json 构建 FAISS 索引，或加载 `faiss_index/` |
| `_add_to_vectorstore()` | 每次 add_event/thought/chat 同时写入 VectorStore |
| `relevance_search()` | 封装 VectorStore 查询，返回 `{embedding_key: score}` |
| `save()` 扩展 | 同时保存 FAISS 索引到 `faiss_index/` 目录 |
| `extract_relevance()` 改造 | 优先走 VectorStore，fallback 到暴力 cos_sim |

**切换腾讯云**：上线时改 `_init_vectorstore` 中 `FAISS` → `TencentVectorDB` + 连接参数

### 集成测试 ✅

| 改动 | 说明 |
|------|------|
| 新增 `utils.py` | 自动配置所有路径，不再需要手动创建 |
| `conftest.py` | Persona fixtures（Isabella / Klaus / persona_with_memories） |
| Retrieve 测试 | `new_retrieve` 返回格式正确 + `last_accessed` 更新 — 2/2 通过 |
| Reflect 测试 | 功能正确，LLM 调用量大（> 10 分钟），适合 CI 异步跑 |

### 新增/修改文件清单

| 文件 | 操作 | 行数 |
|------|------|------|
| `reverie/backend_server/llm_config.py` | **新增** | 18 行 |
| `reverie/backend_server/utils.py` | **新增** | 18 行 |
| `reverie/backend_server/persona/prompt_template/gpt_structure.py` | 改造 | ~200 行 |
| `reverie/backend_server/persona/prompt_template/chain_utils.py` | **新增** | ~1100 行 |
| `reverie/backend_server/persona/prompt_template/run_gpt_prompt.py` | 改造 | 顶部新增 50 行 import/赋值 |
| `reverie/backend_server/persona/memory_structures/associative_memory.py` | 改造 | 新增 ~90 行 VectorStore 逻辑 |
| `reverie/backend_server/persona/cognitive_modules/retrieve.py` | 改造 | `extract_relevance` 重写 ~30 行 |
| `tests/conftest.py` | **新增** | 80 行 |
| `tests/test_smoke.py` | **新增** | 120 行 |
| `tests/test_phase2_prompts.py` | **新增** | 180 行 |
| `tests/test_phase4_vectorstore.py` | **新增** | 130 行 |
| `tests/test_integration.py` | **新增** | 50 行 |

### 全量测试结果

```
tests/test_smoke.py               8 passed   阶段一：LLM 调用层 + LiteLLM 网关
tests/test_phase2_prompts.py      19 passed   阶段二三：27 个 prompt 函数格式验证
tests/test_phase4_vectorstore.py   5 passed   阶段四：VectorStore 写入/检索/持久化
tests/test_integration.py          2 passed   集成：Retrieve 模块端到端
──────────────────────────────────────────────
Total                             34 passed, 0 failed
```

### 改造前后对比

| 维度 | 改造前 | 改造后 |
|------|-------|--------|
| **模型锁定** | 硬编码 OpenAI gpt-3.5/4 | 可切换任意模型（通过 LiteLLM 网关） |
| **Embedding** | 硬编码 text-embedding-ada-002 | 可切换（当前 text-embedding-3-small） |
| **向量检索** | O(n) 暴力遍历 | FAISS 索引，可切换腾讯云 VDB |
| **Prompt 管理** | 31 个 .txt 文件 + 位置占位符 | 17 个 ChatPromptTemplate + 10 个 generate_prompt |
| **输出解析** | 每个函数重复写 validate/clean_up | `run_chain` / `run_prompt_str` 统一模式 |
| **配置** | 手动创建 utils.py | 自动配置，环境变量覆盖 |
| **测试** | 0 个 | 34 个 |
| **代码量** | ~3000 行（run_gpt_prompt.py） | 新增 ~1100 行 chain_utils.py，旧代码保留为 legacy |

### 待完成事项

| 事项 | 说明 | 优先级 |
|------|------|--------|
| Reflect 集成测试 | 功能正确但耗时 > 10 分钟，需在 CI 中异步跑 | P2 |
| Perceive / Plan / Execute 集成测试 | 需要完整 Maze 实例 | P2 |
| E2E 测试 | 单 Agent 跑 1 天 / 多 Agent 交互 | P3 |
| 切换腾讯云向量数据库 | 改 `_init_vectorstore` 中 FAISS → TencentVectorDB | 上线时做 |
| 清理 legacy 代码 | 删除 `run_gpt_prompt.py` 中的 `_legacy` 函数 | 稳定后做 |
