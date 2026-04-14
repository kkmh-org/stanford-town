#!/usr/bin/env python3
"""
demo_lbs.py  —  LBS Demo 驱动脚本（南一高中 · 钟辰时 × 周往）

不依赖 Django / 原版 reverie.py。直接 tick 驱动两个 Persona，
输出结构化 JSON 日志。

用法:
    cd generative_agents
    python demo_lbs.py                       # 默认跑 1 天 (144 ticks × 10min)
    python demo_lbs.py --ticks 20            # 只跑 20 个 tick
    python demo_lbs.py --start-hour 7        # 从 7:00 开始
"""

import argparse
import datetime
import json
import os
import sys

# ── path setup ──────────────────────────────────────────────────────
BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reverie", "backend_server")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)  # prompt templates use relative paths like "persona/prompt_template/v2/..."

from lbs_maze import LBSMaze
from persona.persona import Persona

# Monkey-patch path_finder so execute.py works without a grid.
import persona.cognitive_modules.execute as _exec_mod
_orig_path_finder = _exec_mod.path_finder
def _lbs_path_finder(maze, start, end, collision_block_char, verbose=False):
    """In a POI world, 'movement' is instant: path = [start, end]."""
    return [start, end]
_exec_mod.path_finder = _lbs_path_finder


# ── config ──────────────────────────────────────────────────────────
SPATIAL_MEM = os.path.join(os.path.dirname(__file__),
                           "lbs_demo_nan_yi", "spatial_memory.json")
PERSONA_ROOT = os.path.join(os.path.dirname(__file__),
                            "environment", "frontend_server", "storage",
                            "base_nan_yi_high", "personas")
PERSONA_NAMES = ["钟辰时", "周往"]
SIM_DATE = datetime.date(2026, 4, 14)  # Tuesday
TICK_MINUTES = 10
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "lbs_demo_nan_yi", "output")


def init_personas(maze):
    personas = {}
    for name in PERSONA_NAMES:
        folder = os.path.join(PERSONA_ROOT, name)
        p = Persona(name, folder_mem_saved=folder)
        spawn_addr = p.scratch.living_area
        tile = maze.get_spawning_tile(spawn_addr)
        p.scratch.curr_tile = tile
        personas[name] = p
    return personas


def run_tick(tick_num, curr_time, maze, personas):
    """Run one simulation tick for all personas. Returns a list of tick logs."""
    tick_logs = []
    for name, persona in personas.items():
        curr_tile = persona.scratch.curr_tile

        try:
            execution = persona.move(maze, personas, curr_tile, curr_time)
            next_tile, pronunciatio, description = execution

            # Update tile events
            maze.remove_subject_events_from_tile(name, curr_tile)
            if persona.scratch.act_event:
                new_event = (*persona.scratch.act_event,
                             persona.scratch.act_description)
                maze.add_event_from_tile(new_event, next_tile)
            persona.scratch.curr_tile = next_tile

            tile_info = maze.access_tile(next_tile)
            log = {
                "tick": tick_num,
                "time": curr_time.strftime("%H:%M"),
                "persona": name,
                "sector": tile_info.get("sector", ""),
                "arena": tile_info.get("arena", ""),
                "action": persona.scratch.act_description,
                "emoji": pronunciatio,
                "address": persona.scratch.act_address,
                "duration_min": persona.scratch.act_duration,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Fallback: keep persona at current tile with the activity from scratch
            act_desc = persona.scratch.act_description or "idle"
            log = {
                "tick": tick_num,
                "time": curr_time.strftime("%H:%M"),
                "persona": name,
                "action": f"(fallback) {act_desc}",
                "sector": maze.access_tile(curr_tile).get("sector", ""),
                "arena": maze.access_tile(curr_tile).get("arena", ""),
                "address": persona.scratch.act_address or persona.scratch.living_area,
                "error_detail": str(e),
            }
        tick_logs.append(log)
        print(f"  [{curr_time.strftime('%H:%M')}] {name}: "
              f"{log.get('action', log.get('error', '?'))}")
    return tick_logs


def main():
    parser = argparse.ArgumentParser(description="LBS Demo runner")
    parser.add_argument("--ticks", type=int, default=144,
                        help="Number of ticks to simulate (default 144 = 24h)")
    parser.add_argument("--start-hour", type=int, default=0,
                        help="Start hour of the day (default 0)")
    args = parser.parse_args()

    print("=== LBS Demo: 南一高中 ===")
    print(f"Simulating {args.ticks} ticks × {TICK_MINUTES} min, "
          f"starting at {args.start_hour}:00\n")

    # Init
    maze = LBSMaze(SPATIAL_MEM)
    print(f"Maze: {maze}")
    personas = init_personas(maze)
    for name, p in personas.items():
        print(f"  Persona '{name}' loaded  |  tile={p.scratch.curr_tile}  |  "
              f"living_area={p.scratch.living_area}")
    print()

    all_logs = []
    start_dt = datetime.datetime(SIM_DATE.year, SIM_DATE.month, SIM_DATE.day,
                                 args.start_hour, 0)

    for tick in range(args.ticks):
        curr_time = start_dt + datetime.timedelta(minutes=tick * TICK_MINUTES)
        print(f"--- Tick {tick} ({curr_time.strftime('%A %H:%M')}) ---")
        logs = run_tick(tick, curr_time, maze, personas)
        all_logs.extend(logs)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "sim_log.json")
    with open(out_path, "w") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(all_logs)} entries written to {out_path}")

    # Summary per persona
    for name in PERSONA_NAMES:
        p = personas[name]
        print(f"\n=== {name} ===")
        print(f"  Schedule summary:\n{p.scratch.get_str_daily_schedule_summary()}")


if __name__ == "__main__":
    main()
