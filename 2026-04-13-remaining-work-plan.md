# 剩余工作执行计划

> 日期：2026-04-13
>
> 范围：阶段二三剩余 24 个函数迁移 + 集成测试

---

## 一、剩余 24 个 Prompt 函数迁移（阶段二三合并）

### 分类和分批

按输出解析复杂度分为三批，每批内按功能模块分组：

#### 批次 A：简单函数（10 个）— 输出是字符串，clean_up 只做 strip

| # | 函数名 | 输入参数 | 输出类型 | 对应 .txt 模板 |
|---|--------|---------|---------|---------------|
| 1 | `summarize_conversation` | conversation | str | `v3_ChatGPT/summarize_conversation_v1.txt` |
| 2 | `keyword_to_thoughts` | keyword, concept_summary | str | `v2/keyword_to_thoughts_v1.txt` |
| 3 | `agent_chat_summarize_ideas` | persona, target_persona, statements, context | str | `v3_ChatGPT/summarize_chat_ideas_v1.txt` |
| 4 | `agent_chat_summarize_relationship` | persona, target_persona, statements | str | `v3_ChatGPT/summarize_chat_relationship_v2.txt` |
| 5 | `summarize_ideas` | statements, question | str | `v3_ChatGPT/summarize_ideas_v1.txt` |
| 6 | `generate_next_convo_line` | persona, interlocutor_desc, prev_convo, summary | str | `v2/generate_next_convo_line_v1.txt` |
| 7 | `generate_whisper_inner_thought` | persona, whisper | str | `v2/whisper_inner_thought_v1.txt` |
| 8 | `planning_thought_on_convo` | all_utt | str | `v2/planning_thought_on_convo_v1.txt` |
| 9 | `memo_on_convo` | all_utt | str | `v3_ChatGPT/memo_on_convo_v1.txt` |
| 10 | `act_obj_desc` | act_game_object, act_desp | str | `v3_ChatGPT/generate_obj_event_v1.txt` |

**迁移模式**：全部相同 — 读 .txt → 内联为 ChatPromptTemplate → run_chain → strip 返回

**预计工作量**：每个 5 分钟，共 50 分钟

---

#### 批次 B：中等函数（7 个）— 输出需要解析（列表、字典、整数等）

| # | 函数名 | 输出类型 | 解析逻辑 |
|---|--------|---------|---------|
| 11 | `act_obj_event_triple` | tuple(s,p,o) | 与 event_triple 相同模式 |
| 12 | `focal_pt` | list[str] | split("\n") + split(") ") |
| 13 | `insight_and_guidance` | dict{thought: [evidence_ids]} | split("\n") + "because of" 解析 |
| 14 | `extract_keywords` | set[str] | split("Emotive keywords:") |
| 15 | `convo_to_thoughts` | str | strip，但 prompt 输入较多 |
| 16 | `safety_score` | str (JSON output) | json.loads |
| 17 | `generate_hourly_schedule` | str | strip + 去尾部句号 |

**预计工作量**：每个 10 分钟，共 70 分钟

---

#### 批次 C：复杂函数（7 个）— prompt 输入依赖运行时数据（maze、retrieved、多 persona）

| # | 函数名 | 复杂点 | 策略 |
|---|--------|-------|------|
| 18 | `action_sector` | 输入依赖 maze + spatial_memory + 过滤逻辑 | 保留 create_prompt_input 逻辑，模板内联，run_chain |
| 19 | `action_arena` | 同上 | 同上 |
| 20 | `action_game_object` | 同上 + random fallback | 同上 |
| 21 | `task_decomp` | 60 行解析逻辑（duration 归一化） | 保留解析逻辑不变，只替换 LLM 调用 |
| 22 | `new_decomp_schedule` | 时间段计算 + 解析 | 同上 |
| 23 | `decide_to_talk` | 输入依赖 retrieved + last_chat | 保留 create_prompt_input，模板内联 |
| 24 | `decide_to_react` | 同上 | 同上 |

**注意**：create_conversation、agent_chat、iterative_chat_utt 这三个不在上面 24 个里吗？

让我重新数：已迁移 7 个，总共函数列表有 33 个（含 legacy），去掉 legacy 是 26 个活跃函数，减去 7 个已迁移 = **19 个未迁移**。但 create_conversation / agent_chat / iterative_chat_utt 也需要迁移。

**修正：实际剩余 26 个未迁移函数**

增加：
| 25 | `create_conversation` | 多 persona + 复杂 prompt 构建 | 保留 create_prompt_input |
| 26 | `agent_chat` | 最复杂，多 persona + maze + 多轮 | 保留 create_prompt_input |
| 27 | `iterative_chat_utt` | maze + retrieved + JSON 输出 | 保留 create_prompt_input |

**预计工作量**：每个 15-20 分钟，共 2-3 小时

---

### 执行方式

每批的操作步骤固定：

1. **读 .txt 模板**：获取原始 prompt 内容
2. **在 chain_utils.py 中**：
   - 定义 `_XXX_PROMPT = ChatPromptTemplate.from_messages([...])`（把 `!<INPUT N>!` 转为 `{named_var}`）
   - 定义 `run_gpt_prompt_xxx_v2()` 函数（用 `run_chain`）
   - 对于复杂函数，把原有 `create_prompt_input` 的逻辑搬到 v2 函数里
3. **在 run_gpt_prompt.py 顶部**：
   - 添加 import
   - 添加 `run_gpt_prompt_xxx = run_gpt_prompt_xxx_v2`
   - 把旧函数 `def` 改名为 `_run_gpt_prompt_xxx_legacy`
4. **在 test_phase2_prompts.py 中**：
   - 添加对应的输出格式测试

### 验收标准

- [ ] 全部 33 个函数（含 7 个已完成）指向 v2 实现
- [ ] run_gpt_prompt.py 中所有活跃 `def` 都是 `_legacy` 后缀
- [ ] 所有测试通过

---

## 二、集成测试

### 前置条件：创建测试用 utils.py

```python
# tests/test_utils.py — 测试环境的 utils.py 替代品
import os

_base = os.path.join(os.path.dirname(__file__), '..',
                     'environment', 'frontend_server')

openai_api_key = "not-needed-anymore"  # Phase 1 已改为 LangChain
key_owner = "test"

maze_assets_loc = os.path.join(_base, "static_dirs", "assets")
env_matrix = os.path.join(maze_assets_loc, "the_ville", "matrix")
env_visuals = os.path.join(maze_assets_loc, "the_ville", "visuals")

fs_storage = os.path.join(_base, "storage")
fs_temp_storage = os.path.join(_base, "temp_storage")

collision_block_id = "32125"

debug = True
```

### 测试用例

#### T-INT.1 Perceive 模块

| 测试 | 输入 | 验证点 |
|------|------|--------|
| `test_perceive_returns_concept_nodes` | Persona("Isabella Rodriguez") + Maze | 返回 list[ConceptNode]，每个都在 a_mem 中 |
| `test_perceive_updates_importance_trigger` | 同上 | importance_trigger_curr 减小 |
| `test_perceive_respects_att_bandwidth` | 同上 | 返回数量 <= att_bandwidth |

#### T-INT.2 Retrieve 模块

| 测试 | 输入 | 验证点 |
|------|------|--------|
| `test_retrieve_returns_correct_format` | Persona + perceived events | 返回 dict[desc] = {curr_event, events, thoughts} |
| `test_new_retrieve_returns_sorted_nodes` | Persona + focal_points | 返回按加权分数排序的 ConceptNode 列表 |
| `test_new_retrieve_updates_last_accessed` | 同上 | 检索到的节点 last_accessed 被更新 |

#### T-INT.3 Plan 模块

| 测试 | 输入 | 验证点 |
|------|------|--------|
| `test_long_term_planning_generates_schedule` | Persona + new_day="First day" | f_daily_schedule 非空，总分钟数合理 |
| `test_plan_sets_valid_act_address` | 同上 | act_address 格式正确（world:sector:...） |

#### T-INT.4 Reflect 模块

| 测试 | 输入 | 验证点 |
|------|------|--------|
| `test_reflection_trigger_fires` | importance_trigger_curr=0 | 反思后 seq_thought 增长 |
| `test_reflection_resets_counter` | 同上 | importance_trigger_curr 重置为 max |

#### T-INT.5 Execute 模块

| 测试 | 输入 | 验证点 |
|------|------|--------|
| `test_execute_returns_valid_tile` | Persona + Maze + plan | 返回 (tile, emoji, desc) 三元组 |

### 执行步骤

1. 创建 `tests/test_utils_env.py`（测试用 utils.py）
2. 创建 `tests/conftest.py` 中的 Persona + Maze fixtures（加载真实 bootstrap 数据）
3. 逐模块写测试
4. 运行并修复

**预计工作量**：2-3 小时

---

## 三、执行顺序

```
批次 A (10个简单函数)     ← 先做，快速推进
  │
  ▼
批次 B (7个中等函数)      ← 接着做
  │
  ▼
批次 C (10个复杂函数)     ← 最后做，最耗时
  │
  ▼
创建测试用 utils.py       ← 前置条件
  │
  ▼
集成测试 (5个模块)        ← 最终验证
```

### 里程碑

| 里程碑 | 完成标志 | 预计耗时 |
|--------|---------|---------|
| M1：批次 A 完成 | 10 个简单函数全部迁移 + 测试通过 | 1 小时 |
| M2：批次 B 完成 | 7 个中等函数全部迁移 + 测试通过 | 1.5 小时 |
| M3：批次 C 完成 | 10 个复杂函数全部迁移 + 测试通过 | 3 小时 |
| M4：集成测试完成 | 5 个认知模块端到端测试通过 | 2 小时 |
| **总计** | **27 个函数迁移 + 集成测试** | **~7.5 小时** |
