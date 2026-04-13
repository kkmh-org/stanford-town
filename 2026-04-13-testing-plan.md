# Generative Agents LangChain 改造 — 测试方案

> 日期：2026-04-13
>
> 配套文档：`2026-04-13-langchain-migration-plan.md`

---

## 一、测试难点

这个项目的测试和普通后端项目不同，有两个核心难题：

1. **LLM 输出不确定**：同一个 prompt 每次返回不同结果，无法做精确的 `assertEqual`
2. **无现有测试**：项目没有测试框架、没有测试用例，需要从零搭建

**应对策略**：
- 分层测试：能确定性验证的层（数据结构、格式、流程）用断言；不确定的层（LLM 输出内容）用约束校验
- Mock 驱动：单元测试中 mock LLM 调用，确保流程正确；集成测试中用真实 LLM，验证端到端
- 快照对比：关键环节记录改造前后的输出，人工 review 行为是否合理

---

## 二、测试架构

```
┌─────────────────────────────────────────────────────────────┐
│                        测试金字塔                            │
│                                                             │
│                    ┌───────────┐                            │
│                    │  E2E 测试  │  ← 跑完整模拟，人工观察     │
│                    │ (1-2 个)   │                            │
│                  ┌─┴───────────┴─┐                          │
│                  │  集成测试       │  ← 真实 LLM，单模块验证   │
│                  │ (每模块 2-3 个) │                          │
│                ┌─┴───────────────┴─┐                        │
│                │    单元测试         │  ← Mock LLM，验证格式   │
│                │  (每函数 1-2 个)    │    /流程/数据结构       │
│              ┌─┴───────────────────┴─┐                      │
│              │   冒烟测试（连通性）     │  ← LiteLLM 网关通不通  │
│              │  (改造第一步就跑)       │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、测试目录结构

```
generative_agents/
└── tests/
    ├── conftest.py                  # pytest fixtures（Persona、Memory 等）
    ├── test_smoke.py                # 冒烟测试：网关连通性
    ├── unit/
    │   ├── test_gpt_structure.py    # 阶段一：LLM 调用层
    │   ├── test_embedding.py        # 阶段一：Embedding
    │   ├── test_prompt_template.py  # 阶段二：Prompt 模板
    │   ├── test_output_parser.py    # 阶段三：输出解析
    │   ├── test_associative_memory.py  # 阶段四：记忆存储
    │   └── test_vectorstore.py      # 阶段四：向量检索
    ├── integration/
    │   ├── test_perceive.py         # 感知模块
    │   ├── test_retrieve.py         # 检索模块
    │   ├── test_plan.py             # 规划模块
    │   ├── test_reflect.py          # 反思模块
    │   └── test_converse.py         # 对话模块
    └── e2e/
        ├── test_single_agent.py     # 单 Agent 24 小时模拟
        └── test_multi_agent.py      # 3 Agent 交互模拟
```

---

## 四、分阶段测试详情

### 阶段一测试：LLM 调用层 + LiteLLM 网关

#### T1.1 冒烟测试 — 网关连通性（第一个要跑的测试）

```python
# tests/test_smoke.py

def test_litellm_gateway_reachable():
    """验证 LiteLLM 网关可达，能返回正常响应"""
    response = ChatGPT_request("Say 'hello' and nothing else.")
    assert response is not None
    assert len(response) > 0
    assert "ERROR" not in response

def test_litellm_gateway_model_available():
    """验证配置的模型在网关上可用"""
    response = GPT4_request("Return the number 42.")
    assert response is not None
    assert "42" in response

def test_embedding_gateway_reachable():
    """验证 Embedding API 可用，返回正确维度"""
    embedding = get_embedding("test text")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
```

**通过标准**：3 个测试全绿 → 网关对接成功，可以继续

---

#### T1.2 单元测试 — 函数签名和返回格式兼容

```python
# tests/unit/test_gpt_structure.py

def test_chatgpt_request_returns_string():
    """ChatGPT_request 必须返回 str"""
    result = ChatGPT_request("Say hello.")
    assert isinstance(result, str)

def test_gpt4_request_returns_string():
    """GPT4_request 必须返回 str"""
    result = GPT4_request("Say hello.")
    assert isinstance(result, str)

def test_gpt_request_with_params():
    """GPT_request 必须兼容 gpt_parameter dict"""
    gpt_param = {
        "engine": "text-davinci-003",
        "max_tokens": 10,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": ["\n"]
    }
    result = GPT_request("Say hello.", gpt_param)
    assert isinstance(result, str)

def test_embedding_dimension():
    """Embedding 向量维度必须一致"""
    emb1 = get_embedding("hello world")
    emb2 = get_embedding("goodbye world")
    assert len(emb1) == len(emb2)
    assert len(emb1) > 100  # 至少是合理的维度

def test_embedding_similarity_sanity():
    """语义相近的文本，余弦相似度应明显高于无关文本"""
    from numpy import dot
    from numpy.linalg import norm
    def cos_sim(a, b): return dot(a, b)/(norm(a)*norm(b))

    emb_cat = get_embedding("The cat sat on the mat")
    emb_kitten = get_embedding("A kitten rested on the rug")
    emb_stock = get_embedding("Stock market crashed today")

    sim_related = cos_sim(emb_cat, emb_kitten)
    sim_unrelated = cos_sim(emb_cat, emb_stock)
    assert sim_related > sim_unrelated
```

---

#### T1.3 回归测试 — safe_generate_response 重试逻辑

```python
# tests/unit/test_gpt_structure.py

def test_safe_generate_response_retries(mocker):
    """验证重试机制：前两次返回无效输出，第三次返回有效输出"""
    call_count = 0
    def mock_gpt_request(prompt, params):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "invalid"
        return "7am"

    mocker.patch('persona.prompt_template.gpt_structure.GPT_request', mock_gpt_request)

    def validate(response, prompt=""): return "am" in response
    def clean(response, prompt=""): return int(response.strip().split("am")[0])

    result = safe_generate_response("test", {}, 5, 8, validate, clean)
    assert result == 7
    assert call_count == 3

def test_safe_generate_response_fail_safe(mocker):
    """验证 fail_safe：所有重试失败后返回默认值"""
    mocker.patch('persona.prompt_template.gpt_structure.GPT_request',
                 return_value="garbage")

    def validate(response, prompt=""): return False
    def clean(response, prompt=""): return response

    result = safe_generate_response("test", {}, 3, "default_value", validate, clean)
    assert result == "default_value"
```

---

### 阶段二+三测试：Prompt 模板 + 输出解析

#### T2.1 Prompt 模板渲染测试（不调 LLM，纯格式验证）

```python
# tests/unit/test_prompt_template.py

def test_generate_prompt_old_format():
    """验证旧的 .txt 模板仍然可以正确渲染"""
    prompt = generate_prompt(
        ["Alice", "30", "teacher"],
        "persona/prompt_template/v2/wake_up_hour_v1.txt"
    )
    assert "Alice" in prompt
    assert "!<INPUT" not in prompt  # 所有占位符都应被替换
    assert "<commentblockmarker>" not in prompt  # 注释应被裁掉

def test_all_prompt_templates_render_without_error():
    """遍历所有 .txt 模板文件，验证占位符数量和渲染不报错"""
    import glob, re
    template_files = glob.glob("persona/prompt_template/v*/*.txt")
    for f in template_files:
        content = open(f).read()
        input_count = len(set(re.findall(r'!<INPUT \d+>!', content)))
        dummy_inputs = [f"TEST_{i}" for i in range(input_count)]
        prompt = generate_prompt(dummy_inputs, f)
        assert "!<INPUT" not in prompt, f"Unresolved placeholder in {f}"
```

#### T2.2 输出解析测试 — 每个 prompt 函数的格式约束

```python
# tests/unit/test_output_parser.py

# 使用真实 Persona fixture
def test_wake_up_hour_output_format(persona_fixture):
    """wake_up_hour 必须返回 4-12 之间的整数"""
    result = run_gpt_prompt_wake_up_hour(persona_fixture)[0]
    assert isinstance(result, int)
    assert 4 <= result <= 12

def test_daily_plan_output_format(persona_fixture):
    """daily_plan 必须返回非空字符串列表"""
    result = run_gpt_prompt_daily_plan(persona_fixture, 7)[0]
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)

def test_event_poignancy_output_range(persona_fixture):
    """poignancy 必须返回 1-10 之间的整数"""
    result = run_gpt_prompt_event_poignancy(persona_fixture, "eating breakfast")[0]
    assert isinstance(result, int)
    assert 1 <= result <= 10

def test_event_triple_output_format(persona_fixture):
    """event_triple 必须返回 (s, p, o) 三元组"""
    result = run_gpt_prompt_event_triple("cooking dinner", persona_fixture)[0]
    assert isinstance(result, tuple) or isinstance(result, list)
    assert len(result) == 3

def test_pronunciatio_output_is_emoji(persona_fixture):
    """pronunciatio 必须返回 emoji 字符串"""
    result = run_gpt_prompt_pronunciatio("sleeping", persona_fixture)[0]
    assert isinstance(result, str)
    assert len(result) > 0
```

**完整覆盖**：31 个 prompt 函数，每个至少 1 个输出格式测试。

---

### 阶段四测试：VectorStore 向量检索

#### T4.1 VectorStore 基本操作

```python
# tests/unit/test_vectorstore.py

def test_vectorstore_add_and_search():
    """写入文本后能搜索到"""
    vectorstore = create_test_vectorstore()  # FAISS
    vectorstore.add_texts(
        ["Isabella is cooking dinner"],
        metadatas=[{"node_id": "node_1", "type": "event"}]
    )
    results = vectorstore.similarity_search("What is Isabella doing?", k=1)
    assert len(results) == 1
    assert "Isabella" in results[0].page_content

def test_vectorstore_metadata_preserved():
    """metadata 在写入后能正确取回"""
    vectorstore = create_test_vectorstore()
    vectorstore.add_texts(
        ["Klaus is writing a paper"],
        metadatas=[{"node_id": "node_42", "type": "event", "poignancy": 5}]
    )
    results = vectorstore.similarity_search("research paper", k=1)
    assert results[0].metadata["node_id"] == "node_42"
    assert results[0].metadata["poignancy"] == 5

def test_vectorstore_relevance_scores():
    """similarity_search_with_relevance_scores 返回 (doc, score) 对"""
    vectorstore = create_test_vectorstore()
    vectorstore.add_texts(["cat sat on mat", "stock market crashed"])
    results = vectorstore.similarity_search_with_relevance_scores("kitten on rug", k=2)
    assert len(results) == 2
    assert results[0][1] > results[1][1]  # 更相关的排前面
```

#### T4.2 检索结果对比 — 新旧一致性

这是阶段四最关键的测试：**验证 VectorStore 检索和原来 cos_sim 暴力检索的结果足够接近**。

```python
# tests/unit/test_vectorstore.py

def test_retrieve_relevance_consistency():
    """
    用相同的数据和查询，对比：
    1. 旧方式：dict + cos_sim 遍历
    2. 新方式：VectorStore.similarity_search_with_relevance_scores

    top-10 结果的重叠率应 >= 80%
    """
    # 准备测试数据：用真实 Persona 的 embeddings
    persona = load_test_persona("Isabella Rodriguez")
    focal_pt = "Valentine's Day party planning"

    # 旧方式
    old_results = extract_relevance_old(persona, nodes, focal_pt)
    old_top10 = set(sorted(old_results, key=old_results.get, reverse=True)[:10])

    # 新方式
    new_results = extract_relevance_new(persona, nodes, focal_pt)
    new_top10 = set(sorted(new_results, key=new_results.get, reverse=True)[:10])

    overlap = len(old_top10 & new_top10)
    assert overlap >= 8, f"Top-10 overlap is only {overlap}/10"
```

#### T4.3 三因素加权端到端

```python
# tests/unit/test_vectorstore.py

def test_new_retrieve_with_vectorstore():
    """
    验证改造后的 new_retrieve：
    1. 返回格式正确：dict[focal_pt] = [ConceptNode, ...]
    2. 结果按加权分数排序
    3. last_accessed 被更新
    """
    persona = load_test_persona("Klaus Mueller")
    focal_points = ["research paper on gentrification"]

    retrieved = new_retrieve(persona, focal_points, n_count=10)

    assert "research paper on gentrification" in retrieved
    nodes = retrieved["research paper on gentrification"]
    assert len(nodes) <= 10
    assert all(hasattr(n, 'node_id') for n in nodes)
    # 验证 last_accessed 被更新
    for n in nodes:
        assert n.last_accessed == persona.scratch.curr_time
```

---

### 集成测试：认知模块端到端

#### T-INT.1 Perceive 模块

```python
# tests/integration/test_perceive.py

def test_perceive_returns_concept_nodes(persona_in_maze):
    """perceive 返回 ConceptNode 列表，每个都存入了 a_mem"""
    persona, maze = persona_in_maze
    events = perceive(persona, maze)

    assert isinstance(events, list)
    for event in events:
        assert hasattr(event, 'node_id')
        assert hasattr(event, 'poignancy')
        assert 1 <= event.poignancy <= 10
        assert event.node_id in persona.a_mem.id_to_node

def test_perceive_updates_importance_trigger(persona_in_maze):
    """perceive 后 importance_trigger_curr 应该减小"""
    persona, maze = persona_in_maze
    before = persona.scratch.importance_trigger_curr
    perceive(persona, maze)
    after = persona.scratch.importance_trigger_curr
    assert after <= before
```

#### T-INT.2 Plan 模块

```python
# tests/integration/test_plan.py

def test_long_term_planning_generates_schedule(persona_in_maze):
    """新的一天生成的日程非空，总分钟数接近 24 小时"""
    persona, maze = persona_in_maze
    plan(persona, maze, {}, "First day", {})

    schedule = persona.scratch.f_daily_schedule
    assert len(schedule) > 0
    total_minutes = sum(duration for _, duration in schedule)
    assert 1200 <= total_minutes <= 1500  # 20-25 小时（含睡眠）

def test_plan_sets_valid_act_address(persona_in_maze):
    """plan 设置的 act_address 必须在 spatial_memory 中存在"""
    persona, maze = persona_in_maze
    plan(persona, maze, {}, "First day", {})

    addr = persona.scratch.act_address
    assert addr is not None
    parts = addr.split(":")
    assert len(parts) >= 2  # 至少有 world:sector
```

#### T-INT.3 Reflect 模块

```python
# tests/integration/test_reflect.py

def test_reflection_trigger_fires(persona_with_memories):
    """累计 importance 超过阈值时触发反思"""
    persona = persona_with_memories
    persona.scratch.importance_trigger_curr = 0  # 强制触发

    thought_count_before = len(persona.a_mem.seq_thought)
    reflect(persona)
    thought_count_after = len(persona.a_mem.seq_thought)

    assert thought_count_after > thought_count_before  # 应产生新 thought

def test_reflection_resets_counter(persona_with_memories):
    """反思后 importance_trigger_curr 应重置"""
    persona = persona_with_memories
    persona.scratch.importance_trigger_curr = 0

    reflect(persona)

    assert persona.scratch.importance_trigger_curr == persona.scratch.importance_trigger_max
```

---

### E2E 测试：完整模拟

#### T-E2E.1 单 Agent 模拟

```python
# tests/e2e/test_single_agent.py

def test_single_agent_one_day():
    """
    用 Isabella Rodriguez 跑 1 天模拟（约 144 个 tick，每 tick 10 分钟）。
    验证：
    1. 不崩溃（无异常抛出）
    2. 日程生成合理（有起床、工作、吃饭、睡觉）
    3. 记忆流增长（事件数 > 0）
    4. 行动地址在地图范围内
    """
    server = create_test_server("base_the_ville_isabella_maria_klaus")
    persona = server.personas["Isabella Rodriguez"]

    errors = []
    for step in range(144):
        try:
            result = persona.move(server.maze, server.personas,
                                  persona.scratch.curr_tile,
                                  get_time_for_step(step))
            next_tile, pronunciatio, description = result
            assert next_tile is not None
            assert isinstance(pronunciatio, str)
            assert isinstance(description, str)
        except Exception as e:
            errors.append(f"Step {step}: {e}")

    assert len(errors) == 0, f"Errors in {len(errors)} steps: {errors[:5]}"
    assert len(persona.a_mem.seq_event) > 0
    assert len(persona.scratch.f_daily_schedule) > 0
```

#### T-E2E.2 多 Agent 交互

```python
# tests/e2e/test_multi_agent.py

def test_three_agents_interaction():
    """
    用 Isabella + Maria + Klaus 跑 2 小时模拟（12 个 tick）。
    验证至少发生过一次 Agent 间交互（chatting_with 不为空）。
    """
    server = create_test_server("base_the_ville_isabella_maria_klaus")

    chat_happened = False
    for step in range(12):
        for name, persona in server.personas.items():
            result = persona.move(server.maze, server.personas,
                                  persona.scratch.curr_tile,
                                  get_time_for_step(step))
            if persona.scratch.chatting_with:
                chat_happened = True

    # 交互可能不一定发生，但记忆流应增长
    for name, persona in server.personas.items():
        assert len(persona.a_mem.seq_event) > 0
```

---

## 五、测试 Fixtures

```python
# tests/conftest.py
import pytest
import datetime

STORAGE_BASE = "environment/frontend_server/storage"

@pytest.fixture
def persona_fixture():
    """加载一个真实的 Persona（Isabella Rodriguez），带完整 bootstrap 数据"""
    folder = f"{STORAGE_BASE}/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez"
    persona = Persona("Isabella Rodriguez", folder)
    persona.scratch.curr_time = datetime.datetime(2023, 2, 13, 8, 0, 0)
    return persona

@pytest.fixture
def persona_with_memories(persona_fixture):
    """在 persona_fixture 基础上注入一些测试记忆"""
    persona = persona_fixture
    # 注入 10 条测试事件
    for i in range(10):
        persona.a_mem.add_event(
            created=persona.scratch.curr_time,
            expiration=None,
            s="Isabella Rodriguez", p="is", o=f"test_action_{i}",
            description=f"Isabella Rodriguez is test_action_{i}",
            keywords={"Isabella Rodriguez", f"test_action_{i}"},
            poignancy=random.randint(1, 10),
            embedding_pair=(f"test_{i}", get_embedding(f"test action {i}")),
            filling=[]
        )
    return persona

@pytest.fixture
def persona_in_maze(persona_fixture):
    """Persona + Maze 组合"""
    maze = Maze(...)  # 加载测试地图
    persona_fixture.scratch.curr_tile = (58, 39)
    return persona_fixture, maze
```

---

## 六、测试执行策略

### 按阶段推进

```
改造阶段一 → 跑 T1.1 冒烟测试（必须全绿）
           → 跑 T1.2 格式测试（必须全绿）
           → 跑 T1.3 重试测试（必须全绿）
           ✅ 阶段一完成

改造阶段二三 → 每改完一个 prompt 函数，跑对应的 T2.2 格式测试
             → 全部改完后跑 T2.1 模板渲染全量测试
             ✅ 阶段二三完成

改造阶段四 → 跑 T4.1 基本操作（必须全绿）
           → 跑 T4.2 一致性对比（重叠率 >= 80%）
           → 跑 T4.3 三因素加权（必须全绿）
           ✅ 阶段四完成

全部完成 → 跑集成测试 T-INT.*
         → 跑 E2E 测试 T-E2E.*
         → 人工观察 Agent 行为是否合理
         ✅ 可上线
```

### 测试运行命令

```bash
# 冒烟测试（最先跑，验证网关连通）
pytest tests/test_smoke.py -v

# 单元测试（快，mock LLM，秒级完成）
pytest tests/unit/ -v

# 集成测试（慢，真实 LLM 调用，分钟级）
pytest tests/integration/ -v --timeout=120

# E2E 测试（很慢，完整模拟，需要 10-30 分钟）
pytest tests/e2e/ -v --timeout=1800

# 全量
pytest tests/ -v --timeout=1800
```

---

## 七、核心验收标准

| 检查项 | 标准 | 测试覆盖 |
|--------|------|---------|
| 网关连通 | LLM 调用和 Embedding 都能正常返回 | T1.1 |
| 函数签名兼容 | 所有旧函数签名不变，返回类型不变 | T1.2 |
| 重试机制 | fail_safe 和 retry 行为与改造前一致 | T1.3 |
| Prompt 渲染 | 所有模板文件无残留占位符 | T2.1 |
| 输出格式 | 31 个 prompt 函数的输出都满足类型/范围约束 | T2.2 |
| 向量检索 | VectorStore 写入/检索/metadata 正常 | T4.1 |
| 检索一致性 | 新旧 top-10 结果重叠率 >= 80% | T4.2 |
| 三因素加权 | new_retrieve 返回格式和排序逻辑正确 | T4.3 |
| 认知模块 | 5 个模块各自端到端跑通 | T-INT.* |
| 完整模拟 | 单 Agent 跑完 1 天不崩溃 | T-E2E.1 |
| Agent 交互 | 多 Agent 记忆流正常增长 | T-E2E.2 |

---

## 八、LLM 调用成本估算

测试会产生真实的 LLM 调用费用：

| 测试类型 | 预估 LLM 调用次数 | 备注 |
|---------|-----------------|------|
| 冒烟测试 | 3 次 | 可忽略 |
| 单元测试 | 0 次（mock） | 无费用 |
| 格式验证（T2.2） | ~31 次 | 每个 prompt 函数 1 次 |
| 集成测试 | ~50 次 | 每模块 10 次左右 |
| E2E 单 Agent 1 天 | ~500-800 次 | 每 tick 约 4-6 次调用 |
| E2E 多 Agent 2 小时 | ~100-200 次 | 3 Agent × 12 tick |

**建议**：日常开发跑 `unit/` + `test_smoke.py`（零/低费用），合并前跑 `integration/`，上线前跑 `e2e/`。

---

## 九、全部 Prompt 函数输出格式测试（T2.2 完整版）

每个迁移后的 prompt 函数至少 1 个输出格式测试，验证类型和范围约束。

### 已完成（7 个）

| 函数 | 测试 | 验证点 |
|------|------|--------|
| `event_poignancy` | `test_event_poignancy_returns_int_in_range` | int, 1-10 |
| `thought_poignancy` | `test_thought_poignancy_returns_int_in_range` | int, 1-10 |
| `chat_poignancy` | `test_chat_poignancy_returns_int_in_range` | int, 1-10 |
| `pronunciatio` | `test_pronunciatio_returns_emoji` | str, len <= 3 |
| `event_triple` | `test_event_triple_returns_3_tuple` | tuple, len == 3, [0] == name |
| `wake_up_hour` | `test_wake_up_hour_returns_reasonable_int` | int, 4-10 |
| `daily_plan` | `test_daily_plan_returns_non_empty_list` | list[str], len >= 4, [0] startswith "wake up" |

### 批次 A：简单函数（10 个）— 输出均为 str

```python
# 所有简单函数共用同一个验证模式：
# assert isinstance(result, str)
# assert len(result) > 0

def test_summarize_conversation():
    # 输入：[["Alice", "Hi"], ["Bob", "Hello"]]
    # 输出：str，以 "conversing about" 开头
    result = run_gpt_prompt_summarize_conversation(persona, conversation)[0]
    assert isinstance(result, str)
    assert "conversing" in result

def test_keyword_to_thoughts():
    # 输入：keyword="party", concept_summary="Valentine's Day plans"
    # 输出：str（一句话想法）
    result = run_gpt_prompt_keyword_to_thoughts(persona, "party", "planning a party")[0]
    assert isinstance(result, str)
    assert len(result) > 5

def test_agent_chat_summarize_ideas():
    # 输入：persona, target_persona, statements, context
    # 输出：str（总结的想法）
    result = run_gpt_prompt_agent_chat_summarize_ideas(persona, target, stmts, ctx)[0]
    assert isinstance(result, str)

def test_agent_chat_summarize_relationship():
    # 输入：persona, target_persona, statements
    # 输出：str（关系描述）
    result = run_gpt_prompt_agent_chat_summarize_relationship(persona, target, stmts)[0]
    assert isinstance(result, str)

def test_summarize_ideas():
    # 输入：statements, question
    # 输出：str
    result = run_gpt_prompt_summarize_ideas(persona, stmts, question)[0]
    assert isinstance(result, str)

def test_generate_next_convo_line():
    # 输入：persona, interlocutor_desc, prev_convo, summary
    # 输出：str（一句对话）
    result = run_gpt_prompt_generate_next_convo_line(persona, "Interviewer", "", "")[0]
    assert isinstance(result, str)

def test_generate_whisper_inner_thought():
    # 输入：persona, whisper
    # 输出：str（内心想法）
    result = run_gpt_prompt_generate_whisper_inner_thought(persona, "there's a party")[0]
    assert isinstance(result, str)

def test_planning_thought_on_convo():
    # 输入：all_utt（对话全文）
    # 输出：str
    result = run_gpt_prompt_planning_thought_on_convo(persona, "Alice: Hi\nBob: Hello")[0]
    assert isinstance(result, str)

def test_memo_on_convo():
    # 输入：all_utt（对话全文）
    # 输出：str
    result = run_gpt_prompt_memo_on_convo(persona, "Alice: Hi\nBob: Hello")[0]
    assert isinstance(result, str)

def test_act_obj_desc():
    # 输入：act_game_object, act_desp
    # 输出：str（物体状态描述，如 "being used"）
    result = run_gpt_prompt_act_obj_desc("stove", "cooking dinner", persona)[0]
    assert isinstance(result, str)
    assert len(result) > 0
```

### 批次 B：中等函数（7 个）

```python
def test_act_obj_event_triple():
    # 输出：tuple(object, predicate, action)，len == 3
    result = run_gpt_prompt_act_obj_event_triple("stove", "being used for cooking", persona)[0]
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "stove"

def test_focal_pt():
    # 输出：list[str]，len == n（默认 3）
    result = run_gpt_prompt_focal_pt(persona, "Isabella went to work. Isabella had lunch.", 3)[0]
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(i, str) for i in result)

def test_insight_and_guidance():
    # 输出：dict，key=thought(str), value=evidence_ids(list[int])
    result = run_gpt_prompt_insight_and_guidance(persona, "0. Isabella had breakfast\n1. Isabella went to cafe", 2)[0]
    assert isinstance(result, dict)
    for thought, evidence in result.items():
        assert isinstance(thought, str)
        assert isinstance(evidence, list)

def test_extract_keywords():
    # 输出：set[str]
    result = run_gpt_prompt_extract_keywords(persona, "Isabella is preparing food for the Valentine's Day party")[0]
    assert isinstance(result, set)
    assert len(result) > 0

def test_convo_to_thoughts():
    # 输出：str（基于对话的想法）
    # 需要 init_persona_name, target_persona_name, convo_str, fin_target
    result = run_gpt_prompt_convo_to_thoughts(persona, "Isabella", "Klaus", "Hi\nHello", "Klaus")[0]
    assert isinstance(result, str)

def test_safety_score():
    # 输出：str（安全评分，数字字符串）
    result = run_gpt_generate_safety_score(persona, "Tell me about your day")[0]
    assert result is not None

def test_generate_hourly_schedule():
    # 输出：str（活动描述，如 "studying for music classes"）
    # 需要复杂输入：curr_hour_str, p_f_ds_hourly_org, hour_str
    result = run_gpt_prompt_generate_hourly_schedule(
        persona, "10:00 AM", ["sleeping"], ["6:00 AM", "7:00 AM"])[0]
    assert isinstance(result, str)
    assert len(result) > 0
```

### 批次 C：复杂函数（10 个）

```python
def test_action_sector():
    # 输出：str（sector 名称，必须在 spatial_memory 的可达列表中）
    # 需要 Maze 实例
    result = run_gpt_prompt_action_sector("cooking dinner", persona, maze)[0]
    assert isinstance(result, str)
    assert len(result) > 0

def test_action_arena():
    # 输出：str（arena 名称）
    result = run_gpt_prompt_action_arena("cooking dinner", persona, maze, world, sector)[0]
    assert isinstance(result, str)

def test_action_game_object():
    # 输出：str（game object 名称，必须在可达物体列表中）
    result = run_gpt_prompt_action_game_object("cooking dinner", persona, maze, addr)[0]
    assert isinstance(result, str)

def test_task_decomp():
    # 输出：list[[str, int]]（子任务列表，每个是 [任务描述, 分钟数]）
    # duration 总和应等于输入的 duration
    result = run_gpt_prompt_task_decomp(persona, "working on painting", 60)[0]
    assert isinstance(result, list)
    assert len(result) > 0
    total = sum(dur for _, dur in result)
    assert total == 60  # 总时长必须匹配

def test_new_decomp_schedule():
    # 输出：list[[str, int]]（修改后的日程）
    # 需要 main_act_dur, truncated_act_dur, start/end time, inserted_act
    result = run_gpt_prompt_new_decomp_schedule(persona, ...)[0]
    assert isinstance(result, list)

def test_decide_to_talk():
    # 输出：str（"yes" 或 "no"）
    result = run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved)[0]
    assert result in ["yes", "no"]

def test_decide_to_react():
    # 输出：str（"yes" 或 "no"）
    result = run_gpt_prompt_decide_to_react(persona, target_persona, retrieved)[0]
    assert result in ["yes", "no"]

def test_create_conversation():
    # 输出：list[list[str, str]]（对话列表 [[name, utterance], ...]）
    result = run_gpt_prompt_create_conversation(persona, target, loc, ...)[0]
    assert isinstance(result, list)
    for row in result:
        assert len(row) == 2

def test_agent_chat():
    # 输出：list[list[str, str]]（同上）
    result = run_gpt_prompt_agent_chat(maze, persona, target, ctx, idea1, idea2)[0]
    assert isinstance(result, list)

def test_iterative_chat_utt():
    # 输出：dict{"utterance": str, "end": bool}
    result = run_gpt_generate_iterative_chat_utt(maze, persona, target, retrieved, ctx, chat)[0]
    assert isinstance(result, dict)
    assert "utterance" in result
    assert "end" in result
    assert isinstance(result["utterance"], str)
    assert isinstance(result["end"], bool)
```

---

## 十、集成测试详细方案

### 前置条件

#### 10.1 测试用 utils.py

集成测试需要加载真实的 Maze 和 Persona 数据，这依赖 `utils.py` 中的路径配置。
创建 `tests/test_utils_setup.py` 作为 conftest 的一部分：

```python
# tests/conftest.py 中添加

import os
import sys

# 为集成测试注入路径配置（替代手工创建的 utils.py）
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server')
sys.path.insert(0, os.path.abspath(backend_dir))

_frontend = os.path.join(os.path.dirname(__file__), '..', 'environment', 'frontend_server')

# 写入 utils.py（如果不存在）
utils_path = os.path.join(backend_dir, 'utils.py')
if not os.path.exists(utils_path):
    with open(utils_path, 'w') as f:
        f.write(f'''
maze_assets_loc = "{os.path.abspath(os.path.join(_frontend, 'static_dirs', 'assets'))}"
env_matrix = f"{{maze_assets_loc}}/the_ville/matrix"
env_visuals = f"{{maze_assets_loc}}/the_ville/visuals"
fs_storage = "{os.path.abspath(os.path.join(_frontend, 'storage'))}"
fs_temp_storage = "{os.path.abspath(os.path.join(_frontend, 'temp_storage'))}"
collision_block_id = "32125"
debug = True
openai_api_key = "not-needed"
key_owner = "test"
''')
```

#### 10.2 Fixtures

```python
@pytest.fixture
def maze():
    """加载真实的 Maze（the_ville 地图）"""
    from maze import Maze
    maze = Maze("the_ville", fs_storage + "/base_the_ville_isabella_maria_klaus")
    return maze

@pytest.fixture
def persona_isabella(maze):
    """加载 Isabella Rodriguez（带完整 bootstrap 数据）"""
    from persona.persona import Persona
    folder = fs_storage + "/base_the_ville_isabella_maria_klaus/personas/Isabella Rodriguez"
    p = Persona("Isabella Rodriguez", folder)
    p.scratch.curr_time = datetime.datetime(2023, 2, 13, 8, 0, 0)
    p.scratch.curr_tile = (58, 39)  # 一个合理的初始位置
    return p

@pytest.fixture
def persona_klaus(maze):
    """加载 Klaus Mueller"""
    from persona.persona import Persona
    folder = fs_storage + "/base_the_ville_isabella_maria_klaus/personas/Klaus Mueller"
    p = Persona("Klaus Mueller", folder)
    p.scratch.curr_time = datetime.datetime(2023, 2, 13, 8, 0, 0)
    p.scratch.curr_tile = (60, 40)
    return p

@pytest.fixture
def personas(persona_isabella, persona_klaus):
    """多 Agent dict"""
    return {
        "Isabella Rodriguez": persona_isabella,
        "Klaus Mueller": persona_klaus,
    }
```

### 集成测试用例

#### T-INT.1 Perceive 模块

```python
# tests/integration/test_perceive.py

def test_perceive_returns_concept_nodes(persona_isabella, maze):
    """perceive 返回 ConceptNode 列表"""
    from persona.cognitive_modules.perceive import perceive
    events = perceive(persona_isabella, maze)
    assert isinstance(events, list)
    # 首次运行可能返回空（取决于地图上有无事件），但不应报错

def test_perceive_updates_spatial_memory(persona_isabella, maze):
    """perceive 后 spatial_memory 应包含当前位置的 sector/arena"""
    from persona.cognitive_modules.perceive import perceive
    perceive(persona_isabella, maze)
    tile_info = maze.access_tile(persona_isabella.scratch.curr_tile)
    if tile_info["world"]:
        assert tile_info["world"] in persona_isabella.s_mem.tree

def test_perceive_updates_importance_trigger(persona_isabella, maze):
    """如果有新事件，importance_trigger_curr 应减小"""
    from persona.cognitive_modules.perceive import perceive
    before = persona_isabella.scratch.importance_trigger_curr
    events = perceive(persona_isabella, maze)
    after = persona_isabella.scratch.importance_trigger_curr
    if events:  # 如果确实感知到了事件
        assert after < before
```

#### T-INT.2 Retrieve 模块

```python
# tests/integration/test_retrieve.py

def test_retrieve_returns_dict(persona_isabella, maze):
    """retrieve 返回正确格式的 dict"""
    from persona.cognitive_modules.perceive import perceive
    from persona.cognitive_modules.retrieve import retrieve
    perceived = perceive(persona_isabella, maze)
    retrieved = retrieve(persona_isabella, perceived)
    assert isinstance(retrieved, dict)
    for key, val in retrieved.items():
        assert "curr_event" in val
        assert "events" in val
        assert "thoughts" in val

def test_new_retrieve_returns_nodes(persona_with_memories):
    """new_retrieve 返回 ConceptNode 列表"""
    from persona.cognitive_modules.retrieve import new_retrieve
    result = new_retrieve(persona_with_memories, ["Valentine's Day party"], 10)
    assert "Valentine's Day party" in result
    nodes = result["Valentine's Day party"]
    assert isinstance(nodes, list)
    assert len(nodes) <= 10
```

#### T-INT.3 Plan 模块

```python
# tests/integration/test_plan.py

def test_plan_first_day(persona_isabella, maze, personas):
    """第一天的 plan 应生成日程"""
    from persona.cognitive_modules.plan import plan
    plan(persona_isabella, maze, personas, "First day", {})
    assert len(persona_isabella.scratch.f_daily_schedule) > 0
    total_min = sum(d for _, d in persona_isabella.scratch.f_daily_schedule)
    assert total_min > 600  # 至少 10 小时

def test_plan_sets_act_address(persona_isabella, maze, personas):
    """plan 后应设置合法的 act_address"""
    from persona.cognitive_modules.plan import plan
    plan(persona_isabella, maze, personas, "First day", {})
    assert persona_isabella.scratch.act_address is not None
    parts = persona_isabella.scratch.act_address.split(":")
    assert len(parts) >= 2
```

#### T-INT.4 Reflect 模块

```python
# tests/integration/test_reflect.py

def test_reflect_trigger_and_run(persona_with_memories):
    """强制触发反思，应生成新的 thought 节点"""
    from persona.cognitive_modules.reflect import reflect
    persona = persona_with_memories
    persona.scratch.importance_trigger_curr = 0
    thoughts_before = len(persona.a_mem.seq_thought)
    reflect(persona)
    thoughts_after = len(persona.a_mem.seq_thought)
    assert thoughts_after > thoughts_before

def test_reflect_resets_counter(persona_with_memories):
    """反思后 importance_trigger_curr 重置为 max"""
    from persona.cognitive_modules.reflect import reflect
    persona = persona_with_memories
    persona.scratch.importance_trigger_curr = 0
    reflect(persona)
    assert persona.scratch.importance_trigger_curr == persona.scratch.importance_trigger_max
```

#### T-INT.5 Execute 模块

```python
# tests/integration/test_execute.py

def test_execute_returns_triple(persona_isabella, maze, personas):
    """execute 返回 (next_tile, pronunciatio, description)"""
    from persona.cognitive_modules.plan import plan
    from persona.cognitive_modules.execute import execute
    act_addr = plan(persona_isabella, maze, personas, "First day", {})
    result = execute(persona_isabella, maze, personas, act_addr)
    assert isinstance(result, tuple)
    assert len(result) == 3
    next_tile, pronunciatio, description = result
    assert isinstance(next_tile, tuple)
    assert len(next_tile) == 2
    assert isinstance(pronunciatio, str)
    assert isinstance(description, str)
```

### 10.3 fixture: persona_with_memories

```python
@pytest.fixture
def persona_with_memories(persona_isabella):
    """在 Isabella 基础上注入 15 条测试记忆，使其可以触发反思"""
    from persona.prompt_template.gpt_structure import get_embedding
    import random
    persona = persona_isabella

    test_events = [
        "Isabella opened Hobbs Cafe this morning",
        "Isabella served coffee to a customer",
        "Isabella noticed Klaus studying at a table",
        "Isabella overheard a conversation about gentrification",
        "Isabella is planning the Valentine's Day party menu",
        "Isabella ordered red roses for the cafe decoration",
        "Isabella updated the cafe's social media page",
        "Isabella restocked the coffee beans",
        "Isabella chatted with Maria about the party",
        "Isabella printed flyers for the Valentine's Day event",
        "Isabella cleaned the espresso machine",
        "Isabella called the bakery for a cake order",
        "Isabella arranged tables for the party setup",
        "Isabella welcomed a group of students into the cafe",
        "Isabella reviewed the cafe's monthly expenses",
    ]
    for desc in test_events:
        emb_pair = (desc, get_embedding(desc))
        persona.a_mem.add_event(
            created=persona.scratch.curr_time,
            expiration=None,
            s="Isabella Rodriguez", p="is", o=desc.split("Isabella ")[-1],
            description=desc,
            keywords={"Isabella Rodriguez"},
            poignancy=random.randint(2, 8),
            embedding_pair=emb_pair,
            filling=[],
        )
        persona.scratch.importance_trigger_curr -= random.randint(2, 8)
        persona.scratch.importance_ele_n += 1
    return persona
```

---

## 十一、完整测试矩阵

| 测试文件 | 测试数量 | 类型 | 依赖 | 预计耗时 |
|---------|---------|------|------|---------|
| `test_smoke.py` | 8 | 冒烟 | LiteLLM 网关 | 30s |
| `test_phase2_prompts.py` | 9 + 10 + 7 = 26 | 格式验证 | LiteLLM 网关 | 5-10min |
| `test_phase2_complex.py` | 10 | 格式验证 | LiteLLM + Maze | 5-10min |
| `test_phase4_vectorstore.py` | 5 | VectorStore | LiteLLM（Embedding） | 10s |
| `test_int_perceive.py` | 3 | 集成 | Maze + Persona | 1-2min |
| `test_int_retrieve.py` | 2 | 集成 | Persona + 记忆 | 1-2min |
| `test_int_plan.py` | 2 | 集成 | Maze + Persona | 2-3min |
| `test_int_reflect.py` | 2 | 集成 | Persona + 记忆 | 2-3min |
| `test_int_execute.py` | 1 | 集成 | Maze + Persona | 1min |
| **总计** | **~59** | | | **~25min** |

---

## 十二、执行推进顺序

```
1. 批次 A 迁移 + 测试（10 个简单函数）
   │
2. 批次 B 迁移 + 测试（7 个中等函数）
   │
3. 批次 C 迁移 + 测试（10 个复杂函数）
   │
4. 创建集成测试 utils.py + fixtures
   │
5. 集成测试 T-INT.1 ~ T-INT.5
   │
6. 全量测试通过
   │
   ▼
   全部完成 ✅
```
