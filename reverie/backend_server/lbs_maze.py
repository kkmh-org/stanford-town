"""
LBSMaze: POI-based maze adapter for the LBS Demo.

Replaces the tile-grid Maze with a lightweight POI graph built from
spatial_memory.json.  Each arena gets a unique virtual tile coordinate
so that the rest of the Generative Agents codebase (perceive, plan,
execute) can run without modification.

Adjacency:  arenas within the same sector are always "nearby".
            arenas in different sectors are nearby if the sectors are
            listed as adjacent in SECTOR_ADJACENCY (defaults to all
            sectors being reachable from all others — small campus).
"""

import json
from collections import defaultdict


class LBSMaze:
    def __init__(self, spatial_memory_path, world_name=None):
        with open(spatial_memory_path) as f:
            raw = json.load(f)

        if world_name is None:
            world_name = list(raw.keys())[0]
        self.world_name = world_name
        self.world_tree = raw[world_name]

        self.maze_width = 200
        self.maze_height = 200

        # Virtual tile grid (sparse — only POI tiles are populated)
        self._tiles = {}          # (x, y) -> tile_details dict
        self.address_tiles = {}   # "world:sector:arena:obj" -> {(x,y), ...}
        self.collision_maze = []  # unused but expected by path_finder

        self._sector_coords = {}  # sector_name -> set of (x,y)

        self._build_virtual_grid()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_virtual_grid(self):
        """Assign a unique (x, y) tile to every game_object, arena, and sector."""
        x, y = 1, 1  # start at (1,1), leave row/col 0 as border

        for sector, arenas in self.world_tree.items():
            sector_tiles = set()

            for arena, objects in arenas.items():
                arena_base_x = x

                if not objects:
                    objects = []

                for obj in objects:
                    td = self._make_tile(sector, arena, obj)
                    self._tiles[(x, y)] = td
                    sector_tiles.add((x, y))
                    self._register_address(td, (x, y))
                    x += 1

                # If arena had no objects, still create one tile for it
                if not objects:
                    td = self._make_tile(sector, arena, "")
                    self._tiles[(x, y)] = td
                    sector_tiles.add((x, y))
                    self._register_address(td, (x, y))
                    x += 1

                # Also register the arena-level address
                arena_addr = f"{self.world_name}:{sector}:{arena}"
                if arena_addr not in self.address_tiles:
                    self.address_tiles[arena_addr] = set()
                self.address_tiles[arena_addr].add((arena_base_x, y))

            # Register sector-level address
            sector_addr = f"{self.world_name}:{sector}"
            if sector_addr not in self.address_tiles:
                self.address_tiles[sector_addr] = set()
            first_tile = next(iter(sector_tiles)) if sector_tiles else (x, y)
            self.address_tiles[sector_addr].add(first_tile)

            self._sector_coords[sector] = sector_tiles
            y += 2  # leave a gap between sectors
            x = 1

    def _make_tile(self, sector, arena, game_object):
        obj_addr = ":".join(
            [self.world_name, sector, arena] + ([game_object] if game_object else [])
        )
        events = set()
        if game_object:
            events.add((obj_addr, None, None, None))
        return {
            "world": self.world_name,
            "sector": sector,
            "arena": arena,
            "game_object": game_object,
            "spawning_location": "",
            "collision": False,
            "events": events,
        }

    def _register_address(self, td, coord):
        w, s, a, g = td["world"], td["sector"], td["arena"], td["game_object"]
        keys = [f"{w}:{s}:{a}"]
        if g:
            keys.append(f"{w}:{s}:{a}:{g}")
        for k in keys:
            if k not in self.address_tiles:
                self.address_tiles[k] = set()
            self.address_tiles[k].add(coord)

    # ------------------------------------------------------------------
    # Maze interface expected by perceive / plan / execute
    # ------------------------------------------------------------------

    def access_tile(self, tile):
        if tile in self._tiles:
            return self._tiles[tile]
        return {
            "world": self.world_name,
            "sector": "",
            "arena": "",
            "game_object": "",
            "spawning_location": "",
            "collision": False,
            "events": set(),
        }

    def get_tile_path(self, tile, level):
        td = self.access_tile(tile)
        path = td["world"]
        if level == "world":
            return path
        path += f":{td['sector']}"
        if level == "sector":
            return path
        path += f":{td['arena']}"
        if level == "arena":
            return path
        path += f":{td['game_object']}"
        return path

    def get_nearby_tiles(self, tile, vision_r):
        """Return all tiles in the same sector + adjacent sectors.

        In a small campus, we treat every sector as adjacent (vision_r
        is ignored in favour of semantic adjacency).
        """
        td = self.access_tile(tile)
        curr_sector = td["sector"]
        result = set()
        if curr_sector and curr_sector in self._sector_coords:
            result |= self._sector_coords[curr_sector]
        # Also include all other sectors (small campus — everything visible)
        for sec, coords in self._sector_coords.items():
            result |= coords
        return list(result)

    # ------ event mutation (used by reverie.py / execute.py) ------

    def add_event_from_tile(self, curr_event, tile):
        td = self._tiles.get(tile)
        if td:
            td["events"].add(curr_event)

    def remove_event_from_tile(self, curr_event, tile):
        td = self._tiles.get(tile)
        if not td:
            return
        td["events"].discard(curr_event)

    def turn_event_from_tile_idle(self, curr_event, tile):
        td = self._tiles.get(tile)
        if not td:
            return
        td["events"].discard(curr_event)
        td["events"].add((curr_event[0], None, None, None))

    def remove_subject_events_from_tile(self, subject, tile):
        td = self._tiles.get(tile)
        if not td:
            return
        td["events"] = {e for e in td["events"] if e[0] != subject}

    # ------ helpers ------

    def get_spawning_tile(self, arena_address):
        """Given an arena address string, return one tile coordinate in it."""
        if arena_address in self.address_tiles:
            return next(iter(self.address_tiles[arena_address]))
        return (1, 1)

    def __repr__(self):
        sectors = list(self.world_tree.keys())
        return f"<LBSMaze world={self.world_name!r} sectors={sectors}>"
