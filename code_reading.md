# Generative Agents 代码阅读笔记

> 论文：*Generative Agents: Interactive Simulacra of Human Behavior* (Stanford, 2023)

---

## 一、整体架构：两个进程 + 文件通信

项目分为两部分，通过 **JSON 文件** 互相通信：

| 组件 | 路径 | 作用 |
|------|------|------|
| **Django 前端** | `environment/frontend_server/` | 渲染地图 UI，写 `environment/{step}.json`，读 `movement/{step}.json` |
| **Reverie 后端** | `reverie/backend_server/` | 运行 Agent 的认知循环，读环境数据，写行动数据 |

通信流程：

```
前端输出环境状态 → 后端读取 → Agent 决策 → 后端输出行动 → 前端读取并渲染
```

---

## 二、主循环：`reverie.py`

入口在 `reverie/backend_server/reverie.py`，核心类是 `ReverieServer`。

### 启动流程

1. 用户输入要 fork 的模拟名 + 新模拟名
2. `__init__` 复制基础模拟文件夹，加载 `Maze`（地图），实例化每个 `Persona`（Agent）
3. 用户在 REPL 中输入 `run N`，触发 `start_server(N)`

### 每个 tick 的循环（`start_server`）

1. 等待前端写出 `environment/{step}.json`（当前世界状态）
2. 同步每个 Agent 在地图上的位置
3. **对每个 Persona 调用 `persona.move()`** — 这是核心
4. 收集所有 Agent 的行动，写入 `movement/{step}.json`
5. 时间推进 `sec_per_step`（默认 10 秒），步数 +1

---

## 三、Agent 架构总览

一个 Agent（`Persona` 类）由 **三层记忆 + 五个认知模块** 组成：

```
┌─────────────────────────────────────────────────────────┐
│                      Persona                            │
│                                                         │
│  ┌─────────────────── 记忆层 ───────────────────────┐   │
│  │                                                   │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌───────────┐  │   │
│  │  │ Spatial Mem │ │ Associative  │ │  Scratch   │  │   │
│  │  │ (空间记忆)   │ │ Memory       │ │ (工作记忆) │  │   │
│  │  │             │ │ (记忆流)      │ │            │  │   │
│  │  │ 世界地图树   │ │ event/thought│ │ 身份/计划/ │  │   │
│  │  │ world→sector│ │ /chat 节点   │ │ 当前行动   │  │   │
│  │  │ →arena→obj  │ │ +embedding   │ │            │  │   │
│  │  └─────────────┘ └──────────────┘ └───────────┘  │   │
│  └───────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────── 认知管线 (每 tick 执行) ──────────┐   │
│  │                                                   │   │
│  │  Perceive → Retrieve → Plan → Reflect → Execute  │   │
│  │                                                   │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

核心代码在 `persona.py` 的 `move()` 方法：

```python
def move(self, maze, personas, curr_tile, curr_time):
    new_day = False
    if not self.scratch.curr_time:
        new_day = "First day"
    elif self.scratch.curr_time.strftime('%A %B %d') != curr_time.strftime('%A %B %d'):
        new_day = "New day"
    self.scratch.curr_time = curr_time

    perceived = self.perceive(maze)
    retrieved = self.retrieve(perceived)
    plan = self.plan(maze, personas, new_day, retrieved)
    self.reflect()
    return self.execute(maze, personas, plan)
```

---

## 四、三层记忆

### 4.1 Spatial Memory（空间记忆）— 「我知道世界长什么样」

文件：`persona/memory_structures/spatial_memory.py`

一棵 JSON 树，四层结构：**world → sector → arena → [objects]**

```json
{
  "the Ville": {
    "Hobbs Cafe": {
      "cafe": ["counter", "espresso machine", "menu board"],
      "kitchen": ["stove", "fridge"]
    },
    "Dolores's apartment": {
      "bedroom": ["bed", "closet", "painting"]
    }
  }
}
```

用途：当 Agent 在 Plan 阶段需要决定"去哪里"时，通过 `get_str_accessible_sectors` / `get_str_accessible_sector_arenas` / `get_str_accessible_arena_game_objects` 逐层询问 LLM 选择目的地。

---

### 4.2 Associative Memory（关联记忆 / 记忆流）— 「我经历过什么」

文件：`persona/memory_structures/associative_memory.py`

这是论文中最核心的 **Memory Stream**，由 `ConceptNode` 节点组成。

#### ConceptNode 结构

| 字段 | 含义 | 示例 |
|------|------|------|
| `type` | 三种类型 | `event`（感知事件）、`thought`（反思洞察）、`chat`（对话） |
| `s, p, o` | 主谓宾三元组 | `("Isabella", "is", "cooking dinner")` |
| `poignancy` | 重要性（1-10，LLM 打分） | 日常=1, 重大事件=8 |
| `embedding_key` | 对应的 embedding 向量键 | 用于语义相似度检索 |
| `depth` | 抽象层级 | event=0, thought=1+（基于证据链深度递增） |
| `filling` | 证据链 | thought 节点指向它依据的 node_id 列表 |
| `last_accessed` | 上次访问时间 | 用于 recency 衰减计算 |
| `created` / `expiration` | 创建和过期时间 | thought 默认 30 天过期 |

#### 存储索引

```python
self.id_to_node = dict()        # node_id → ConceptNode

self.seq_event = []             # 按时间排序的事件序列
self.seq_thought = []           # 按时间排序的想法序列
self.seq_chat = []              # 按时间排序的对话序列

self.kw_to_event = dict()       # 关键词 → 事件节点列表
self.kw_to_thought = dict()     # 关键词 → 想法节点列表
self.kw_to_chat = dict()        # 关键词 → 对话节点列表

self.kw_strength_event = dict() # 关键词出现频次（事件）
self.kw_strength_thought = dict() # 关键词出现频次（想法）
```

---

### 4.3 Scratch（工作记忆）— 「我现在在干嘛 / 我是谁」

文件：`persona/memory_structures/scratch.py`

Agent 的短期状态和身份信息，分四大类：

#### 身份层（三级）

```python
self.innate = None      # L0：天生特质，不变（如 "hard-edged, independent, loyal"）
self.learned = None     # L1：后天特质（如 "Dolores is a painter who wants to live quietly"）
self.currently = None   # L2：当前状态（如 "preparing for her first solo show"）
self.lifestyle = None   # 生活习惯（如 "goes to bed around 11pm, sleeps for 7 hours"）
self.living_area = None # 居住区域
```

这三层拼在一起形成 **ISS（Identity Stable Set）**，几乎每个 prompt 都会塞进去作为角色设定：

```
Name: Dolores Heitmiller
Age: 28
Innate traits: hard-edged, independent, loyal
Learned traits: Dolores is a painter who wants live quietly and paint while enjoying her everyday life.
Currently: Dolores is preparing for her first solo show. She mostly works from home.
Lifestyle: Dolores goes to bed around 11pm, sleeps for 7 hours, eats dinner around 6pm.
Daily plan requirement: Dolores is planning to stay at home all day and never go out.
```

#### 计划层

```python
self.daily_req = []                   # 今天的目标列表
self.f_daily_schedule = []            # 分解后的日程 [['sleeping', 360], ['waking up', 5], ...]
self.f_daily_schedule_hourly_org = [] # 原始小时级日程（未分解版）
```

日程是一个 `[任务描述, 持续分钟数]` 的列表，粗粒度小时计划会逐步被分解成 5-10 分钟粒度。

#### 当前行动

```python
self.act_address = None        # "world:sector:arena:object" 四级地址
self.act_start_time = None     # 行动开始时间
self.act_duration = None       # 持续分钟数
self.act_description = None    # 行动描述文本
self.act_pronunciatio = None   # emoji 表情
self.act_event = (name, None, None)  # SPO 三元组
self.chatting_with = None      # 正在聊天的对象
self.chat = None               # 对话内容 [["Alice", "Hi"], ["Bob", "Hello"]]
self.planned_path = []         # A* 寻路的路径坐标列表
```

#### 反思计数器

```python
self.recency_w = 1                  # 时效性权重
self.relevance_w = 1                # 相关性权重
self.importance_w = 1               # 重要性权重
self.recency_decay = 0.99           # 时效衰减率
self.importance_trigger_max = 150   # 反思阈值
self.importance_trigger_curr = 150  # 当前剩余阈值（每次感知扣减，归零触发反思）
self.importance_ele_n = 0           # 累计感知事件数
```

---

## 五、五个认知模块

### 5.1 Perceive（感知） — `cognitive_modules/perceive.py`

- 扫描 `vision_r`（默认 4 格）范围内的地图事件
- 用 `att_bandwidth`（默认 3）限制同时注意的事件数
- 用 `retention`（默认 5）跳过已记住的旧事件
- 对新事件调用 LLM 打 **poignancy 分**（1-10），然后存入 Associative Memory
- 同时扣减 `importance_trigger_curr`（为反思触发做准备）

**超参数：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `vision_r` | 4 | 视觉半径（格） |
| `att_bandwidth` | 3 | 注意力带宽（同时感知事件数上限） |
| `retention` | 5 | 记忆保持（跳过最近 N 个已记忆事件） |

---

### 5.2 Retrieve（检索记忆） — `cognitive_modules/retrieve.py`

#### 基础检索（用于日常感知）

对每个新感知到的事件，用 SPO 关键词去记忆流里做**关键词匹配**，取出相关 events 和 thoughts：

```python
relevant_events = persona.a_mem.retrieve_relevant_events(event.subject, event.predicate, event.object)
relevant_thoughts = persona.a_mem.retrieve_relevant_thoughts(event.subject, event.predicate, event.object)
```

#### 高级检索 `new_retrieve`（用于反思和对话）

三因素加权排序：

```python
recency_out = extract_recency(persona, nodes)       # 时间衰减：0.99^i
importance_out = extract_importance(persona, nodes)  # poignancy 分数
relevance_out = extract_relevance(persona, nodes, focal_pt)  # 余弦相似度
```

三个维度分别归一化到 [0, 1]，然后加权求和：

```python
gw = [0.5, 3, 2]  # recency, relevance, importance 的全局权重
score = 0.5 × recency + 3 × relevance + 2 × importance
```

取 top-N 节点返回，同时更新这些节点的 `last_accessed` 时间。

---

### 5.3 Plan（规划） — `cognitive_modules/plan.py`

最复杂的模块，分三个层次：

#### 1) 长期规划（`_long_term_planning`）

新的一天开始时触发：

1. LLM 生成当天的大致日程（`daily_planning` prompt）
2. 分解为小时级计划（`generate_hourly_schedule`）
3. 当行动即将执行时，再分解为 5-10 分钟的子任务（`task_decomp`）

#### 2) 行动决策（`_determine_action`）

当前任务完成后：

1. 从日程栈取下一个任务描述
2. LLM 逐层选择目的地：sector → arena → object（通过 Spatial Memory 提供候选项）
3. 设置 `act_address`, `act_duration`, `act_description` 等

#### 3) 社交反应

感知到其他 Agent 时：

- `decide_to_talk`：判断是否发起对话
- `decide_to_react`：判断是否做出反应
- 结果可能是：**聊天（chat）**、**等待（wait）** 或 **忽略**

---

### 5.4 Reflect（反思） — `cognitive_modules/reflect.py`

#### 触发条件

```python
def reflection_trigger(persona):
    if persona.scratch.importance_trigger_curr <= 0:
        return True
    return False
```

当 `importance_trigger_curr` 被感知阶段不断扣减到 ≤ 0 时触发（阈值默认 150）。

#### 反思流程

1. **生成焦点问题**：从最近记忆中让 LLM 生成 3 个焦点问题（`generate_focal_points`）
2. **检索相关记忆**：对每个焦点用 `new_retrieve` 取回相关记忆节点
3. **生成洞察**：LLM 基于这些节点生成 **洞察（insights）+ 证据链（evidence）**
4. **存入记忆**：洞察作为 `thought` 节点（depth ≥ 1）写回记忆流
5. **重置计数器**：`importance_trigger_curr` 恢复为 `importance_trigger_max`

#### 对话后反思

对话结束后还会额外生成两个 thought 节点：
- **planning thought**：基于对话的规划想法（如 "For Alice's planning: ..."）
- **memo**：对话备忘录（如 "Alice learned that ..."）

---

### 5.5 Execute（执行） — `cognitive_modules/execute.py`

将 Plan 输出的目标地址（`act_address`）翻译为具体地图操作：

1. 将 `act_address`（如 `"the Ville:Hobbs Cafe:cafe:counter"`）转化为地图坐标
2. 用 `path_finder.py` 做 **A\* 寻路**
3. 返回三元组 `(next_tile, pronunciatio, description)`
   - `next_tile`：下一步坐标，如 `(58, 9)`
   - `pronunciatio`：emoji 表情，如 `💤`
   - `description`：行为描述，如 `"writing her next novel @ double studio:common room:sofa"`

---

## 六、Prompt 模板系统

文件：`persona/prompt_template/` 目录

### 模板文件格式

`.txt` 文件位于 `v3_ChatGPT/` 目录下，用 `!<INPUT N>!` 占位符：

```
daily_planning_v6.txt

Variables:
!<INPUT 0>! -- Commonset (ISS)
!<INPUT 1>! -- General description
!<INPUT 2>! -- Today's date
!<INPUT 3>! -- Persona name

<commentblockmarker>###</commentblockmarker>
!<INPUT 0>!

In general, !<INPUT 1>!
Today is !<INPUT 2>!. Here is !<INPUT 3>!'s plan today in broad-strokes ...
```

`###` 之前的注释区在发送前会被裁掉，只保留之后的内容作为实际 prompt。

### 核心文件

| 文件 | 作用 |
|------|------|
| `run_gpt_prompt.py` | 每个认知功能对应一个函数，负责组装 prompt 输入、调用 LLM、解析输出 |
| `gpt_structure.py` | 封装 OpenAI API 调用（Completion / Chat / GPT-4），包含 `safe_generate_response` 做重试和校验 |
| `v3_ChatGPT/*.txt` | 各功能的 prompt 模板文件 |

### 主要 Prompt 模板一览

| 模板 | 用于 |
|------|------|
| `daily_planning_v*.txt` | 生成每日计划 |
| `generate_hourly_schedule_v*.txt` | 生成小时级日程 |
| `task_decomp_v*.txt` | 任务分解为子任务 |
| `decide_to_talk_v*.txt` | 判断是否发起对话 |
| `decide_to_react_v*.txt` | 判断是否做出反应 |
| `create_conversation_v*.txt` | 生成对话内容 |
| `poignancy_event_v1.txt` | 对事件打重要性分 |
| `generate_focal_pt_v1.txt` | 生成反思焦点问题 |
| `insight_and_evidence_v1.txt` | 生成洞察与证据 |
| `wake_up_hour_v1.txt` | 决定起床时间 |
| `summarize_conversation_v1.txt` | 总结对话 |
| `memo_on_convo_v1.txt` | 对话备忘录 |

---

## 七、数据流总结

```
                              ┌──────────────────┐
                              │   Django 前端     │
                              │  (地图渲染/动画)  │
                              └──────┬───────────┘
                                     │
                          写 environment/{step}.json
                          读 movement/{step}.json
                                     │
                              ┌──────▼───────────┐
                              │  ReverieServer    │
                              │  (reverie.py)     │
                              └──────┬───────────┘
                                     │
                          对每个 Persona 调用 move()
                                     │
              ┌──────────────────────▼──────────────────────┐
              │                   move()                     │
              │                                              │
              │  ① perceive(maze)                            │
              │     └→ 扫描周围事件，打 poignancy 分，存记忆   │
              │                                              │
              │  ② retrieve(perceived)                       │
              │     └→ 关键词匹配取出相关 events/thoughts      │
              │                                              │
              │  ③ plan(maze, personas, new_day, retrieved)   │
              │     └→ 长期规划 / 行动决策 / 社交反应          │
              │                                              │
              │  ④ reflect()                                  │
              │     └→ 重要性阈值触发 → 焦点→检索→洞察→存记忆  │
              │                                              │
              │  ⑤ execute(maze, personas, plan)              │
              │     └→ A* 寻路 → (next_tile, emoji, desc)     │
              └──────────────────────────────────────────────┘
```

---

## 八、设计哲学

这个 Agent 的架构本质上是 **"认知心理学的计算化"**：

- **感知**模拟人类的注意力（有限带宽 + 遗忘）
- **记忆流**模拟人类的情景记忆（SPO 三元组 + 时间衰减 + 情感权重）
- **规划**模拟人类的分层规划（今天做什么 → 这小时做什么 → 下 5 分钟做什么）
- **反思**模拟人类的元认知（"最近发生的事让我想到了什么"）
- **三层身份**模拟人格的稳定性（天性不变 → 习得缓慢变化 → 状态随时变化）

---

## 九、关键文件索引

| 文件 | 作用 |
|------|------|
| `reverie/backend_server/reverie.py` | 主循环入口，`ReverieServer` |
| `reverie/backend_server/maze.py` | 地图网格 + 碰撞检测 |
| `reverie/backend_server/path_finder.py` | A* 寻路 |
| `reverie/backend_server/persona/persona.py` | Agent 主类，`move()` 认知管线 |
| `persona/memory_structures/spatial_memory.py` | 空间记忆树 |
| `persona/memory_structures/associative_memory.py` | 关联记忆 / 记忆流 |
| `persona/memory_structures/scratch.py` | 工作记忆（身份 + 计划 + 状态） |
| `persona/cognitive_modules/perceive.py` | 感知模块 |
| `persona/cognitive_modules/retrieve.py` | 检索模块 |
| `persona/cognitive_modules/plan.py` | 规划模块 |
| `persona/cognitive_modules/reflect.py` | 反思模块 |
| `persona/cognitive_modules/execute.py` | 执行模块 |
| `persona/cognitive_modules/converse.py` | 对话模块 |
| `persona/prompt_template/run_gpt_prompt.py` | Prompt 组装与 LLM 调用 |
| `persona/prompt_template/gpt_structure.py` | OpenAI API 封装 |
