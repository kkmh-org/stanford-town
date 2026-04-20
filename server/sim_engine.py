"""
Sim engine: runs the shared world simulation in the background.
Manages 3 NPC personas (钟辰时/周往/陈昔) + N visitor users.
"""
import asyncio
import datetime
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Callable

# Path setup
BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reverie", "backend_server")
sys.path.insert(0, BACKEND)

from lbs_maze import LBSMaze
from persona.persona import Persona

# Monkey patch path_finder for LBS
import persona.cognitive_modules.execute as _exec_mod
def _lbs_path_finder(maze, start, end, collision_block_char, verbose=False):
    return [start, end]
_exec_mod.path_finder = _lbs_path_finder


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPATIAL_MEM = os.path.join(ROOT, "lbs_demo_nan_yi", "spatial_memory.json")
PERSONA_ROOT = os.path.join(ROOT, "environment", "frontend_server",
                            "storage", "base_nan_yi_high", "personas")
NPC_NAMES = ["钟辰时", "周往", "陈昔"]
TICK_MINUTES = 10
START_DATETIME = datetime.datetime(2026, 4, 14, 7, 0)


class Visitor:
    """Lightweight visitor persona for users entering the world."""
    def __init__(self, username: str, tile):
        self.name = username
        self.tile = tile
        self.act_description = "idle"
        self.emoji = "👤"


class SimEngine:
    def __init__(self, tick_interval_sec: float = 3.0):
        """tick_interval_sec: wall-clock seconds between simulation ticks."""
        self.tick_interval = tick_interval_sec
        self.maze = LBSMaze(SPATIAL_MEM)
        self.personas: Dict[str, Persona] = {}
        self.visitors: Dict[str, Visitor] = {}
        self.sim_time = START_DATETIME
        self.tick_count = 0
        self.listeners: List[Callable] = []   # broadcast callbacks
        self.chat_log: List[dict] = []        # {time, from, to, msg}
        self.god_commands: List[dict] = []    # pending commands
        self.pending_user_messages: List[dict] = []  # {from_user, to_npc, msg}
        self._running = False
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Cooldowns to avoid spammy interactions
        self._npc_chat_cooldown: Dict[tuple, int] = {}     # (a, b) -> tick_count when last chatted
        self._npc_visitor_greeting: Dict[tuple, int] = {}  # (npc, visitor) -> tick when last greeted
        self.NPC_CHAT_COOLDOWN_TICKS = 18  # 3 hours @ 10min/tick
        self.NPC_VISITOR_GREETING_COOLDOWN = 6  # 1 hour

    def load_personas(self):
        for name in NPC_NAMES:
            folder = os.path.join(PERSONA_ROOT, name)
            p = Persona(name, folder_mem_saved=folder)
            p.scratch.curr_tile = self.maze.get_spawning_tile(p.scratch.living_area)
            self.personas[name] = p

    def snapshot(self) -> dict:
        """Current world state for broadcasting to clients."""
        npcs = []
        for name, p in self.personas.items():
            tile = p.scratch.curr_tile or (0, 0)
            td = self.maze.access_tile(tile)
            # Latest thought (reflection) — seq_thought[0] is newest
            recent_thoughts = []
            try:
                for t in p.a_mem.seq_thought[:3]:
                    recent_thoughts.append({
                        "desc": t.description,
                        "time": t.created.strftime("%H:%M") if t.created else "",
                    })
            except Exception:
                pass
            # Recent perceived events (so user sees what NPC noticed)
            recent_events = []
            try:
                for e in p.a_mem.seq_event[:5]:
                    recent_events.append({
                        "desc": e.description,
                        "poignancy": e.poignancy,
                        "time": e.created.strftime("%H:%M") if e.created else "",
                    })
            except Exception:
                pass
            npcs.append({
                "name": name,
                "tile": list(tile),
                "sector": td.get("sector", ""),
                "arena": td.get("arena", ""),
                "address": p.scratch.act_address or "",
                "action": (p.scratch.act_description or "").replace("\n", " ")[:500],
                "emoji": p.scratch.act_pronunciatio or "🙂",
                "chatting_with": p.scratch.chatting_with,
                "recent_thoughts": recent_thoughts,
                "recent_events": recent_events,
                "importance": getattr(p.scratch, "importance_trigger_curr", 0),
            })
        vs = []
        for uname, v in self.visitors.items():
            td = self.maze.access_tile(v.tile)
            vs.append({
                "name": uname,
                "tile": list(v.tile),
                "sector": td.get("sector", ""),
                "arena": td.get("arena", ""),
                "action": v.act_description,
                "emoji": v.emoji,
            })
        return {
            "type": "state",
            "sim_time": self.sim_time.strftime("%Y-%m-%d %H:%M"),
            "tick": self.tick_count,
            "npcs": npcs,
            "visitors": vs,
            "recent_chats": self.chat_log[-20:],
        }

    def add_listener(self, cb: Callable):
        self.listeners.append(cb)

    def remove_listener(self, cb: Callable):
        if cb in self.listeners:
            self.listeners.remove(cb)

    async def _broadcast(self, payload: dict):
        for cb in list(self.listeners):
            try:
                await cb(payload)
            except Exception as e:
                print(f"[sim] listener error: {e}")

    def add_visitor(self, username: str) -> Visitor:
        spawn = self.maze.get_spawning_tile("南一高中:校门口:校门区")
        v = Visitor(username, spawn)
        self.visitors[username] = v
        return v

    def remove_visitor(self, username: str):
        if username in self.visitors:
            del self.visitors[username]

    def move_visitor(self, username: str, dx: int, dy: int):
        v = self.visitors.get(username)
        if not v:
            return
        nx = max(1, min(self.maze.maze_width - 1, v.tile[0] + dx))
        ny = max(1, min(self.maze.maze_height - 1, v.tile[1] + dy))
        v.tile = (nx, ny)

    def teleport_visitor_to(self, username: str, sector_name: str):
        v = self.visitors.get(username)
        if not v:
            return
        addr = f"南一高中:{sector_name}"
        if addr in self.maze.address_tiles:
            v.tile = next(iter(self.maze.address_tiles[addr]))

    def add_god_command(self, cmd: dict):
        """E.g. {'type': 'force_move', 'target': '钟辰时', 'sector': '图书馆'}"""
        self.god_commands.append(cmd)

    def add_user_message(self, from_user: str, to_npc: str, msg: str):
        pm = {
            "from_user": from_user,
            "to_npc": to_npc,
            "msg": msg,
            "time": self.sim_time.strftime("%H:%M"),
        }
        # Submit to the sim loop's event loop (callable from any thread)
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._handle_user_message(pm), self._loop)

    async def _handle_user_message(self, pm: dict):
        try:
            npc = self.personas.get(pm["to_npc"])
            if not npc:
                return
            # Immediately show the user's message
            entry_in = {
                "time": pm["time"],
                "from": pm["from_user"],
                "to": pm["to_npc"],
                "msg": pm["msg"],
                "kind": "user_to_npc",
            }
            self.chat_log.append(entry_in)
            await self._broadcast({"type": "chat", "entry": entry_in})
            # Generate reply in thread pool (LLM is blocking)
            reply = await asyncio.to_thread(self._generate_npc_reply, npc, pm)
            entry_out = {
                "time": self.sim_time.strftime("%H:%M"),
                "from": pm["to_npc"],
                "to": pm["from_user"],
                "msg": reply,
                "kind": "npc_to_user",
            }
            self.chat_log.append(entry_out)
            await self._broadcast({"type": "chat", "entry": entry_out})
        except Exception as e:
            traceback.print_exc()
            print(f"[user_msg] failed: {e}")

    def _process_god_commands(self):
        """Apply pending god commands before tick."""
        while self.god_commands:
            cmd = self.god_commands.pop(0)
            try:
                if cmd["type"] == "force_move":
                    name = cmd["target"]
                    sector = cmd["sector"]
                    if name in self.personas:
                        addr = f"南一高中:{sector}"
                        if addr in self.maze.address_tiles:
                            tile = next(iter(self.maze.address_tiles[addr]))
                            self.personas[name].scratch.curr_tile = tile
                            print(f"[god] moved {name} → {sector}")
            except Exception as e:
                print(f"[god] cmd failed: {e}")

    # ============================================================
    # Emergent interactions: NPC ↔ NPC chats + visitor greetings
    # ============================================================

    async def _check_emergent_interactions(self):
        """After each tick, check for NPCs that should naturally interact."""
        # 1. NPC <-> NPC: any pair in the same sector
        npc_list = list(self.personas.items())
        for i, (n1, p1) in enumerate(npc_list):
            for n2, p2 in npc_list[i+1:]:
                s1 = self.maze.access_tile(p1.scratch.curr_tile or (0, 0)).get("sector")
                s2 = self.maze.access_tile(p2.scratch.curr_tile or (0, 0)).get("sector")
                if not s1 or s1 != s2:
                    continue
                # Cooldown: same pair shouldn't chat too often
                pair = tuple(sorted([n1, n2]))
                last = self._npc_chat_cooldown.get(pair, -999)
                if self.tick_count - last < self.NPC_CHAT_COOLDOWN_TICKS:
                    continue
                # Skip if either is in a hard-priority activity (sleeping, exam, etc.)
                act1 = (p1.scratch.act_description or "").lower()
                act2 = (p2.scratch.act_description or "").lower()
                if any(k in act1 + act2 for k in ["sleeping", "睡觉", "考试", "上课"]):
                    continue
                self._npc_chat_cooldown[pair] = self.tick_count
                # Run the NPC-NPC chat in thread pool (LLM call)
                asyncio.create_task(self._run_npc_npc_chat(p1, p2, s1))

        # 2. NPC <-> Visitor: NPC notices visitor in same sector → greet
        for vname, v in list(self.visitors.items()):
            v_sector = self.maze.access_tile(v.tile).get("sector")
            if not v_sector:
                continue
            for nname, p in self.personas.items():
                n_sector = self.maze.access_tile(p.scratch.curr_tile or (0, 0)).get("sector")
                if n_sector != v_sector:
                    continue
                pair = (nname, vname)
                last = self._npc_visitor_greeting.get(pair, -999)
                if self.tick_count - last < self.NPC_VISITOR_GREETING_COOLDOWN:
                    continue
                act = (p.scratch.act_description or "").lower()
                if any(k in act for k in ["sleeping", "睡觉", "考试", "上课"]):
                    continue
                self._npc_visitor_greeting[pair] = self.tick_count
                asyncio.create_task(self._run_npc_greet_visitor(p, vname, v_sector))

    async def _run_npc_npc_chat(self, p1: "Persona", p2: "Persona", sector: str):
        """Generate a short multi-turn chat between two NPCs in the same sector."""
        try:
            chat = await asyncio.to_thread(self._gen_npc_npc_dialog, p1, p2, sector)
            time_str = self.sim_time.strftime("%H:%M")
            for speaker, msg in chat:
                target = p2.name if speaker == p1.name else p1.name
                entry = {
                    "time": time_str,
                    "from": speaker,
                    "to": target,
                    "msg": msg,
                    "kind": "npc_to_npc",
                    "sector": sector,
                }
                self.chat_log.append(entry)
                await self._broadcast({"type": "chat", "entry": entry})
        except Exception as e:
            traceback.print_exc()
            print(f"[npc-npc chat] failed: {e}")

    async def _run_npc_greet_visitor(self, npc: "Persona", visitor: str, sector: str):
        """NPC notices a visitor in the same sector and says hi."""
        try:
            greeting = await asyncio.to_thread(self._gen_npc_greeting, npc, visitor, sector)
            entry = {
                "time": self.sim_time.strftime("%H:%M"),
                "from": npc.scratch.name,
                "to": visitor,
                "msg": greeting,
                "kind": "npc_to_user",
                "sector": sector,
                "auto": True,
            }
            self.chat_log.append(entry)
            await self._broadcast({"type": "chat", "entry": entry})
        except Exception as e:
            print(f"[greet] failed: {e}")

    def _gen_npc_npc_dialog(self, p1, p2, sector: str) -> List[tuple]:
        """LLM generates a 3-4 turn chat between p1 and p2."""
        from persona.prompt_template.gpt_structure import ChatGPT_single_request
        import json as _json

        prompt = (
            f"两位高中生在「{sector}」相遇，请生成他们自然的对话（3-4 轮）。\n\n"
            f"=== 角色 1 ===\n"
            f"姓名：{p1.scratch.name}\n性格：{p1.scratch.innate}\n"
            f"背景：{p1.scratch.learned}\n此刻正在：{p1.scratch.act_description or '无'}\n\n"
            f"=== 角色 2 ===\n"
            f"姓名：{p2.scratch.name}\n性格：{p2.scratch.innate}\n"
            f"背景：{p2.scratch.learned}\n此刻正在：{p2.scratch.act_description or '无'}\n\n"
            f"两人的关系：同班同学，平等但有微妙竞争。"
            f"周往叫钟辰时「钟模范」（有讽刺意味），钟辰时叫周往「周同学」。\n\n"
            f"输出 JSON 数组，每个元素是 [说话者姓名, 内容]。"
            f"严格 JSON，不要解释。每句不超过 25 字。\n"
            f"示例：[[\"钟辰时\", \"嗯。\"], [\"周往\", \"切。\"]]"
        )
        try:
            raw = ChatGPT_single_request(prompt).strip()
            # Extract JSON array
            start = raw.find('[')
            end = raw.rfind(']') + 1
            if start >= 0 and end > start:
                arr = _json.loads(raw[start:end])
                return [(item[0], item[1]) for item in arr if len(item) >= 2][:6]
        except Exception as e:
            print(f"[npc-npc gen] {e}")
        # Fallback
        return [
            (p1.scratch.name, "..."),
            (p2.scratch.name, "嗯。"),
        ]

    def _gen_npc_greeting(self, npc, visitor: str, sector: str) -> str:
        """NPC's first greeting to a visitor they just noticed."""
        from persona.prompt_template.gpt_structure import ChatGPT_single_request
        prompt = (
            f"你是高中生 {npc.scratch.name}。\n"
            f"性格：{npc.scratch.innate}\n"
            f"此刻正在：{npc.scratch.act_description or '无'}\n\n"
            f"你刚注意到一个陌生人「{visitor}」也在「{sector}」。"
            f"TA 不是你认识的人。请用一句符合你性格的话主动打招呼或表达注意到 TA 的反应（不超过 20 字）。"
            f"不要使用你对老朋友的特殊昵称。不要加引号。"
        )
        try:
            return ChatGPT_single_request(prompt).strip().strip('"').strip('「').strip('」')[:80]
        except Exception:
            return "（看了一眼，没说话）"

    def _recent_chat_history(self, npc_name: str, user_name: str, limit: int = 6):
        """Get last N messages between this user and this NPC."""
        history = []
        for entry in self.chat_log:
            is_match = (
                (entry.get("from") == user_name and entry.get("to") == npc_name) or
                (entry.get("from") == npc_name and entry.get("to") == user_name)
            )
            if is_match:
                history.append(entry)
        return history[-limit:]

    def _generate_npc_reply(self, npc: Persona, pm: dict) -> str:
        """Generate NPC reply to a user message via LLM."""
        from persona.prompt_template.gpt_structure import ChatGPT_single_request

        user_name = pm["from_user"]
        history = self._recent_chat_history(npc.scratch.name, user_name)
        history_str = ""
        if history:
            for h in history[:-1]:  # exclude the current message (already in pm)
                speaker = h.get("from", "?")
                history_str += f"  {speaker}：{h.get('msg', '')}\n"

        # Identity line: explicitly tell the NPC who the user is.
        # The user is a校外访客 — not a classmate, not 钟辰时, not anyone the NPC knows.
        visitor_id = (
            f"「{user_name}」是一位刚来到南一高中的校外访客（陌生人），"
            f"不是你的同学，也不是你认识的任何人（特别地，不是钟辰时、不是周往、不是陈昔）。"
            f"你之前从未见过 TA。"
        )

        prompt = (
            f"你在扮演高中生角色 {npc.scratch.name}。\n\n"
            f"=== 角色设定 ===\n"
            f"- 姓名：{npc.scratch.name}\n"
            f"- 性格：{npc.scratch.innate}\n"
            f"- 背景：{npc.scratch.learned}\n"
            f"- 当前状态：{npc.scratch.currently}\n"
            f"- 此刻正在：{npc.scratch.act_description or '无'}\n\n"
            f"=== 关于对方 ===\n"
            f"{visitor_id}\n"
            f"因此请勿用你对老朋友或同学的特殊称呼（比如周往叫钟辰时'钟模范'、"
            f"叫女主'牡丹花'）来称呼这位访客——TA 不是那些人。\n\n"
            f"=== 最近的对话 ===\n"
            f"{history_str if history_str else '（这是 TA 第一次跟你说话）'}\n"
            f"=== 当前对话 ===\n"
            f"{user_name}：{pm['msg']}\n\n"
            f"请以 {npc.scratch.name} 的口吻回复一句话（不超过 30 字），"
            f"保持角色性格，不要加引号。"
        )
        try:
            reply = ChatGPT_single_request(prompt).strip().strip('"').strip('「').strip('」')
            return reply or "（沉默）"
        except Exception as e:
            return f"（无法回复：{e}）"

    async def tick_once(self):
        """Run one simulation tick for all NPCs."""
        async with self._lock:
            try:
                self._process_god_commands()

                # Advance NPCs (blocking LLM calls run in thread pool)
                def advance():
                    for name, p in self.personas.items():
                        try:
                            curr_tile = p.scratch.curr_tile
                            execution = p.move(self.maze, self.personas, curr_tile, self.sim_time)
                            next_tile, _, _ = execution
                            self.maze.remove_subject_events_from_tile(name, curr_tile)
                            if p.scratch.act_event:
                                new_event = (*p.scratch.act_event, p.scratch.act_description)
                                self.maze.add_event_from_tile(new_event, next_tile)
                            p.scratch.curr_tile = next_tile
                        except Exception as e:
                            print(f"[tick] {name} failed: {e}")

                await asyncio.to_thread(advance)

                self.tick_count += 1
                self.sim_time += datetime.timedelta(minutes=TICK_MINUTES)

                # Check for emergent interactions (NPC-NPC chats + visitor greetings)
                await self._check_emergent_interactions()

                # Broadcast new state
                await self._broadcast(self.snapshot())
            except Exception as e:
                traceback.print_exc()
                print(f"[tick] failed: {e}")

    async def run_forever(self):
        self._running = True
        self._loop = asyncio.get_running_loop()
        self.load_personas()
        print(f"[sim] started with NPCs: {list(self.personas.keys())}")
        # Initial broadcast so clients see something immediately
        await self._broadcast(self.snapshot())
        while self._running:
            await self.tick_once()
            await asyncio.sleep(self.tick_interval)

    def stop(self):
        self._running = False


# Singleton
_engine: Optional[SimEngine] = None

def get_engine() -> SimEngine:
    global _engine
    if _engine is None:
        _engine = SimEngine()
    return _engine
