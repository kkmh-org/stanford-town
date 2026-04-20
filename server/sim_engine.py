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
            npcs.append({
                "name": name,
                "tile": list(tile),
                "sector": td.get("sector", ""),
                "arena": td.get("arena", ""),
                "address": p.scratch.act_address or "",
                "action": (p.scratch.act_description or "").replace("\n", " ")[:500],
                "emoji": p.scratch.act_pronunciatio or "🙂",
                "chatting_with": p.scratch.chatting_with,
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

    def _generate_npc_reply(self, npc: Persona, pm: dict) -> str:
        """Generate NPC reply to a user message via LLM."""
        from persona.prompt_template.gpt_structure import ChatGPT_single_request
        prompt = (
            f"你在扮演一个高中生角色。\n"
            f"角色设定：\n"
            f"- 姓名：{npc.scratch.name}\n"
            f"- 性格：{npc.scratch.innate}\n"
            f"- 背景：{npc.scratch.learned}\n"
            f"- 当前状态：{npc.scratch.currently}\n"
            f"- 此刻正在：{npc.scratch.act_description or '无'}\n\n"
            f"用户「{pm['from_user']}」对你说：「{pm['msg']}」\n\n"
            f"请以{npc.scratch.name}的口吻回复一句话，保持角色性格，不要超过30个字，不要加引号。"
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
