# LangGraph 集成 Langfuse

## 1. 概述

[Langfuse](https://langfuse.com) 是一个开源的 LLM 可观测性平台，支持 Trace、评估、Prompt 管理等功能。[LangGraph](https://github.com/langchain-ai/langgraph) 是 LangChain 生态中用于构建多 Agent 有状态工作流的框架。

由于 LangGraph 底层基于 LangChain，因此可以直接使用 **Langfuse 的 LangChain CallbackHandler** 来实现 trace 上报，无需额外适配。

**集成后你可以获得：**

*   🔍 每次 Graph 执行的完整调用链路（Trace）
    
*   📊 每个 LLM 调用的 token 用量、耗时和费用
    
*   🔗 Agent → Tool → LLM 的嵌套层级关系
    
*   🏷️ 自定义 trace 名称、session、用户等元数据
    

---

## 2. 环境准备

### 2.1 依赖安装

```bash
# 核心依赖
pip install langgraph langchain langchain-openai langfuse

# 如果使用 classic agent（AgentExecutor）
pip install langchain-classic

```

**版本要求：**

| 包名 | 最低版本 | 说明 |
| --- | --- | --- |
| `langfuse` | \>= 2.0.0 | Langfuse Python SDK |
| `langgraph` | \>= 0.0.20 | LangGraph 核心库 |
| `langchain-core` | \>= 0.1.0 | LangChain 核心 |
| `langchain-openai` | \>= 0.0.5 | OpenAI 集成 |

### 2.2 环境变量配置

在项目根目录创建 `.env` 文件：

```env
# ===== LLM 配置 =====
OPENAI_API_KEY="your-openai-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"       # 可选，使用代理时配置

# ===== Langfuse 配置 =====
LANGFUSE_SECRET_KEY="sk-lf-xxxxxxxx"               # Langfuse Secret Key
LANGFUSE_PUBLIC_KEY="pk-lf-xxxxxxxx"               # Langfuse Public Key
LANGFUSE_HOST="https://cloud.langfuse.com"         # Langfuse 服务地址（自部署时修改）

```
> **获取 Langfuse Key：** 登录 Langfuse → 进入项目 → Settings → API Keys

---

## 3. 集成方式

### 3.1 Langfuse LangChain CallbackHandler

LangGraph 底层基于 LangChain，因此可以直接利用 LangChain 的 Callback 机制，通过 Langfuse 提供的 `CallbackHandler` 自动捕获 LangGraph 中所有 LLM 调用、Tool 调用和 Chain 执行。

```python
from langfuse.langchain import CallbackHandler

# 初始化（自动读取环境变量）
langfuse_handler = CallbackHandler()

# 在 graph 执行时传入
result = graph.stream(
    {"messages": [HumanMessage(content="用户输入")]},
    config={"callbacks": [langfuse_handler]}
)

```

**优点：**

*   ✅ 零侵入，无需修改业务代码
    
*   ✅ 自动捕获完整的嵌套调用链路
    
*   ✅ 支持 token 统计和费用计算
    
*   ✅ 支持自定义 trace name、session、user 等
    

---

## 4. 核心用法详解

### 4.1 初始化 CallbackHandler

```python
from langfuse.langchain import CallbackHandler

# 方式 A：自动从环境变量读取配置（推荐）
langfuse_handler = CallbackHandler()

# 方式 B：手动指定配置
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-xxxxxxxx",
    public_key="pk-lf-xxxxxxxx",
    host="https://cloud.langfuse.com"
)

```

### 4.2 在 LangGraph stream 中传入回调

`stream()` 方法会逐步输出每个节点的执行结果，适合实时展示：

```python
for step in graph.stream(
    {"messages": [HumanMessage(content="帮我检查 nginx-pod-123 为什么一直重启")]},
    config={"callbacks": [langfuse_handler]}
):
    if "__end__" not in step:
        print(step)

```

### 4.3 在 LangGraph invoke 中传入回调

`invoke()` 方法会等待整个 graph 执行完毕后返回最终结果：

```python
result = graph.invoke(
    {"messages": [HumanMessage(content="帮我检查 Pod 状态")]},
    config={"callbacks": [langfuse_handler]}
)
print(result)

```

### 4.4 异步方法 astream / ainvoke

如果你的应用使用了 `async/await`（如 FastAPI），可以使用异步版本的方法，callback 传入方式完全一致：

```python
import asyncio

async def run_async():
    # 异步流式
    async for step in graph.astream(
        {"messages": [HumanMessage(content="你的问题")]},
        config={"callbacks": [langfuse_handler]}
    ):
        print(step)

    # 异步调用
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="你的问题")]},
        config={"callbacks": [langfuse_handler]}
    )
    print(result)

asyncio.run(run_async())

```

### 4.5 批量执行 batch / abatch

当需要一次性处理多个输入时，可以使用 `batch()`（同步）或 `abatch()`（异步）：

```python
# 同步批量
results = graph.batch(
    [
        {"messages": [HumanMessage(content="问题1")]},
        {"messages": [HumanMessage(content="问题2")]},
    ],
    config={"callbacks": [langfuse_handler]}
)

# 异步批量
results = await graph.abatch(
    [
        {"messages": [HumanMessage(content="问题1")]},
        {"messages": [HumanMessage(content="问题2")]},
    ],
    config={"callbacks": [langfuse_handler]}
)

```
> 每个输入会生成独立的 Trace，方便在 Langfuse 中逐条查看。

### 4.6 编译时预绑定 with\_config（推荐）

如果你希望 **编译时一次性绑定 callback**，后续每次调用无需重复传入 `config`，可以使用 `with_config()`：

```python
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

# 编译时绑定 callback
graph = workflow.compile().with_config({"callbacks": [langfuse_handler]})

# 后续调用自动携带 callback，无需再传 config
for step in graph.stream({"messages": [HumanMessage(content="你的问题")]}):
    print(step)

result = graph.invoke({"messages": [HumanMessage(content="你的问题")]})

```
> 💡 此方式特别适合 **LangGraph Server** 场景，因为 Server 自动处理 graph 调用，无法在每次请求时手动传入 `config`。

### 4.7 自定义 Trace 名称

通过 `config` 中的 `run_name` 参数可以设置在 Langfuse 中显示的 trace 名称，便于区分不同的调用场景：

```python
for step in graph.stream(
    {"messages": [HumanMessage(content="检查 Pod 状态")]},
    config={
        "callbacks": [langfuse_handler],
        "run_name": "Pod重启诊断"            # 👈 自定义 trace 名称
    }
):
    print(step)

```

在 Langfuse 界面中，该 trace 将以 "Pod重启诊断" 作为名称显示，方便检索和辨识。

**所有支持 callback 的入口方法汇总：**

| 方法 | 类型 | 说明 |
| --- | --- | --- |
| `graph.invoke()` | 同步 | 执行完毕后返回最终结果 |
| `graph.stream()` | 同步流式 | 逐步输出每个节点的执行结果 |
| `graph.ainvoke()` | 异步 | `invoke()` 的 async 版本 |
| `graph.astream()` | 异步流式 | `stream()` 的 async 版本 |
| `graph.batch()` | 同步批量 | 批量处理多个输入 |
| `graph.abatch()` | 异步批量 | `batch()` 的 async 版本 |
| `compile().with_config()` | 预绑定 | 编译时绑定，后续调用自动生效 |

---

## 5. 完整示例

### 5.1 基础示例：多 Agent Supervisor 模式

以下是一个最小化的 LangGraph + Langfuse 集成示例：

```python
import dotenv
dotenv.load_dotenv()

import functools
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langgraph.graph import END, StateGraph, START
from langfuse.langchain import CallbackHandler  # ← 导入 Langfuse Handler


# 1. 定义 State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# 2. 创建 Agent
def create_agent(llm, system_prompt, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# 3. 创建 Supervisor
llm = ChatOpenAI(model="gpt-4o")
members = ["Worker1", "Worker2"]

# ... (supervisor 逻辑省略，参考 demo-1.py)


# 4. 构建 Graph
workflow = StateGraph(AgentState)
workflow.add_node("Worker1", worker1_node)
workflow.add_node("Worker2", worker2_node)
workflow.add_node("supervisor", supervisor_node)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()


# 5. 执行并上报 Trace ← 关键步骤
langfuse_handler = CallbackHandler()

for step in graph.stream(
    {"messages": [HumanMessage(content="你的问题")]},
    config={
        "callbacks": [langfuse_handler],  # ← 传入 handler
        "run_name": "我的LangGraph调用"     # ← 可选：自定义名称
    }
):
    print(step)

```
---

## 6. Langfuse 中的 Trace 结构

一次 LangGraph 执行在 Langfuse 中会生成如下层级的 Trace（以 Supervisor 多 Agent 模式为例）：

```plaintext
📦 Trace: "自定义Trace名称"
│
├── 🔗 Chain: "supervisor"  (Supervisor 节点 - 第一次路由)
│   └── 🤖 LLM: ChatOpenAI (gpt-4o)
│       ├── Input: system prompt + messages
│       ├── Output: route → Worker1
│       ├── Tokens: prompt=320, completion=12
│       └── Duration: 1.2s
│
├── 🔗 Chain: "Worker1"  (Agent 节点)
│   ├── 🤖 LLM: ChatOpenAI (gpt-4o)   ← Agent 推理
│   │   └── Output: 决定调用某个工具
│   ├── 🔧 Tool: tool_name             ← 工具执行
│   │   └── Output: 工具返回结果
│   └── 🤖 LLM: ChatOpenAI (gpt-4o)   ← Agent 总结
│       └── Output: 任务执行结果
│
├── 🔗 Chain: "supervisor"  (Supervisor 节点 - 第二次路由)
│   └── 🤖 LLM: ChatOpenAI → route → Worker2
│
├── 🔗 Chain: "Worker2"  (Agent 节点)
│   ├── 🤖 LLM → 🔧 Tool → 🤖 LLM
│   └── Output: 任务执行结果
│
└── 🔗 Chain: "supervisor"  (Supervisor 节点 - 最终路由)
    └── 🤖 LLM → route → FINISH

```

**在 Langfuse 界面中你可以看到：**

| 信息 | 说明 |
| --- | --- |
| Trace Name | 通过 `run_name` 设置的名称 |
| 总耗时 | 整个 Graph 执行的端到端时间 |
| Token 用量 | 每次 LLM 调用的 prompt/completion tokens |
| 费用估算 | 基于模型定价自动计算 |
| 输入/输出 | 每个节点的输入消息和输出结果 |
| 调用层级 | Chain → LLM / Tool 的嵌套关系 |

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65ApGE2N9Oke/img/eb3c4b84-d3b7-4455-bbc0-dde4847b7331.png)

---

## 7. 高级配置

### 7.1 自定义 Session / User / Metadata

```python
langfuse_handler = CallbackHandler(
    session_id="diagnosis-session-001",       # 关联同一会话的多次调用
    user_id="sre-engineer-zhangsan",          # 标记用户
    metadata={"env": "production", "pod": "nginx-pod-123"},  # 自定义元数据
    tags=["sre", "k8s", "diagnosis"]          # 标签，方便筛选
)

```

### 7.2 多次调用使用同一 Trace

如果你希望多个 Graph 执行归属到同一条 Trace 下（例如多轮对话）：

```python
from langfuse.langchain import CallbackHandler

# 第一次调用
handler = CallbackHandler(
    trace_name="多轮诊断会话",
    session_id="session-123"
)

graph.invoke(
    {"messages": [HumanMessage(content="检查 Pod 状态")]},
    config={"callbacks": [handler]}
)

# 第二次调用，使用相同的 session_id
handler2 = CallbackHandler(
    trace_name="多轮诊断会话-后续",
    session_id="session-123"        # 同一 session
)

graph.invoke(
    {"messages": [HumanMessage(content="再查看一下日志")]},
    config={"callbacks": [handler2]}
)

```

### 7.3 通过 metadata 动态设置 Trace 属性

除了在 `CallbackHandler` 构造函数中设置 `user_id`、`session_id` 等属性外，还可以通过 `config.metadata` 在每次调用时动态指定：

```python
langfuse_handler = CallbackHandler()

result = graph.invoke(
    {"messages": [HumanMessage(content="你的问题")]},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": "user-zhangsan",       # 动态设置用户
            "langfuse_session_id": "session-456",       # 动态设置会话
            "langfuse_tags": ["production", "v2"]       # 动态设置标签
        }
    }
)

```
> 💡 此方式在每次调用时灵活指定不同的 trace 属性，无需为每次调用创建新的 `CallbackHandler` 实例，非常适合 Web 服务按请求区分用户的场景。

---

## 8. 常见问题与排查

### Q1: Trace 没有出现在 Langfuse 中？

**排查步骤：**

1.  检查环境变量是否正确加载：
    
    ```python
    import os
    print(os.getenv("LANGFUSE_PUBLIC_KEY"))  # 应有值
    print(os.getenv("LANGFUSE_SECRET_KEY"))  # 应有值
    print(os.getenv("LANGFUSE_HOST"))        # 应有值
    
    ```
    
2.  确认 `dotenv.load_dotenv()` 在最前面被调用。
    
3.  确认 `config={"callbacks": [langfuse_handler]}` 确实传入了 `stream()` 或 `invoke()`。
    
4.  Langfuse 上报是异步的，等待几秒后刷新页面。
    

### Q2: 只看到部分节点的 Trace？

确保 `callbacks` 是通过顶层 `config` 传入的，而不是在单个节点内部传入。LangGraph 会自动将 `config` 中的 `callbacks` 传播到所有子节点。

### Q3: 如何区分不同场景的 Trace？

使用 `run_name` 和 `tags`：

```python
config={
    "callbacks": [langfuse_handler],
    "run_name": "场景名称"
}

```

或在 handler 上设置 tags：

```python
handler = CallbackHandler(tags=["production", "sre-diagnosis"])

```

### Q4: Token 用量或费用不准确？

*   确认使用的是 `langchain-openai` 包（而非旧版 `langchain` 内置的 OpenAI 集成）
    
*   如果使用自定义 `OPENAI_BASE_URL`（如 LiteLLM 代理），token 统计依赖上游是否正确返回 `usage` 字段
    
*   可在 Langfuse 的 Settings → Models 中配置自定义模型的定价
    

### Q5: 如何手动 flush 确保 trace 上报完成？

```python
langfuse_handler.flush()

```

在脚本结束前调用，确保所有 trace 数据都已发送到 Langfuse 服务端。

---

## 9. 参考链接

| 资源 | 链接 |
| --- | --- |
| Langfuse 官方文档 | https://langfuse.com/docs |
| Langfuse LangChain 集成文档 | https://langfuse.com/docs/integrations/langchain |
| LangGraph 官方文档 | https://langchain-ai.github.io/langgraph/ |
| Langfuse Python SDK | https://github.com/langfuse/langfuse-python |
| LangGraph GitHub | https://github.com/langchain-ai/langgraph |