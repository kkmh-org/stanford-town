"""
DB abstraction — supports both SQLite (local dev) and Postgres (production).
Switch via env var:
  DATABASE_URL=postgresql://user:pass@host:port/dbname   → Postgres
  DATABASE_URL=sqlite:///path/to/app.db                  → SQLite (default)
"""
import os
import hashlib
import secrets
from contextlib import contextmanager
from typing import Optional, Any
from urllib.parse import urlparse

_default_sqlite = os.path.join(os.path.dirname(__file__), "app.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_default_sqlite}")


def _is_pg() -> bool:
    return DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")


class DB:
    """Minimal DB adapter with the subset of operations we need."""

    def __init__(self):
        self.is_pg = _is_pg()
        if self.is_pg:
            import psycopg2
            self._pg = psycopg2
            u = urlparse(DATABASE_URL)
            self._conn_kwargs = dict(
                host=u.hostname, port=u.port or 5432,
                user=u.username, password=u.password,
                dbname=u.path.lstrip('/'),
            )
        else:
            import sqlite3
            self._sqlite = sqlite3
            # Parse sqlite:///path
            self._sqlite_path = DATABASE_URL.replace("sqlite:///", "")

    @contextmanager
    def cursor(self):
        if self.is_pg:
            conn = self._pg.connect(**self._conn_kwargs)
        else:
            conn = self._sqlite.connect(self._sqlite_path)
        try:
            cur = conn.cursor()
            yield cur, conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ph(self, n: int = 1) -> str:
        """Placeholder(s). Postgres uses %s, SQLite uses ?"""
        return ",".join(["%s" if self.is_pg else "?"] * n)

    def _q(self, sql: str) -> str:
        """Translate ? placeholders to %s for Postgres."""
        return sql.replace("?", "%s") if self.is_pg else sql

    def init_schema(self):
        sqls_sqlite = [
            """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                pwd_hash TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                sim_time TEXT,
                from_who TEXT,
                to_who TEXT,
                msg TEXT,
                kind TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS persona_snapshot (
                name TEXT PRIMARY KEY,
                scratch_json TEXT NOT NULL,
                tile_x INTEGER, tile_y INTEGER,
                updated_at TEXT NOT NULL
            )""",
        ]
        sqls_pg = [
            """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                pwd_hash TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS chat (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP NOT NULL DEFAULT NOW(),
                sim_time TEXT,
                from_who TEXT,
                to_who TEXT,
                msg TEXT,
                kind TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS persona_snapshot (
                name TEXT PRIMARY KEY,
                scratch_json TEXT NOT NULL,
                tile_x INTEGER, tile_y INTEGER,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )""",
        ]
        sqls = sqls_pg if self.is_pg else sqls_sqlite
        with self.cursor() as (cur, _):
            for s in sqls:
                cur.execute(s)
        # Seed default users
        for u, p in [("admin", "nanyi2026"), ("guest", "nanyi2026"),
                     ("user1", "nanyi2026"), ("user2", "nanyi2026")]:
            self.ensure_user(u, p)

    def ensure_user(self, username: str, password: str):
        pwd = hashlib.sha256(password.encode()).hexdigest()
        with self.cursor() as (cur, _):
            if self.is_pg:
                cur.execute("INSERT INTO users (username, pwd_hash) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (username, pwd))
            else:
                cur.execute("INSERT OR IGNORE INTO users (username, pwd_hash) VALUES (?, ?)",
                            (username, pwd))

    def verify_user(self, username: str, password: str) -> bool:
        pwd = hashlib.sha256(password.encode()).hexdigest()
        with self.cursor() as (cur, _):
            cur.execute(self._q("SELECT pwd_hash FROM users WHERE username = ?"), (username,))
            row = cur.fetchone()
            return row is not None and row[0] == pwd

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self.cursor() as (cur, _):
            if self.is_pg:
                cur.execute("INSERT INTO sessions (token, username) VALUES (%s, %s)",
                            (token, username))
            else:
                cur.execute(
                    "INSERT INTO sessions (token, username, created_at) VALUES (?, ?, datetime('now'))",
                    (token, username))
        return token

    def get_user_from_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        with self.cursor() as (cur, _):
            cur.execute(self._q("SELECT username FROM sessions WHERE token = ?"), (token,))
            row = cur.fetchone()
            return row[0] if row else None

    def delete_session(self, token: Optional[str]):
        if not token:
            return
        with self.cursor() as (cur, _):
            cur.execute(self._q("DELETE FROM sessions WHERE token = ?"), (token,))

    def log_chat(self, entry: dict):
        with self.cursor() as (cur, _):
            if self.is_pg:
                cur.execute("""INSERT INTO chat (sim_time, from_who, to_who, msg, kind)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (entry.get("time"), entry.get("from"), entry.get("to"),
                             entry.get("msg"), entry.get("kind")))
            else:
                cur.execute("""INSERT INTO chat (ts, sim_time, from_who, to_who, msg, kind)
                               VALUES (datetime('now'), ?, ?, ?, ?, ?)""",
                            (entry.get("time"), entry.get("from"), entry.get("to"),
                             entry.get("msg"), entry.get("kind")))

    def get_chat_history(self, limit: int = 50) -> list:
        with self.cursor() as (cur, _):
            cur.execute(self._q(
                "SELECT sim_time, from_who, to_who, msg, kind FROM chat ORDER BY id DESC LIMIT ?"),
                (limit,))
            rows = cur.fetchall()
        return [{"time": r[0], "from": r[1], "to": r[2], "msg": r[3], "kind": r[4]}
                for r in reversed(rows)]

    def save_persona_snapshot(self, name: str, scratch_json: str, tile: tuple):
        with self.cursor() as (cur, _):
            if self.is_pg:
                cur.execute("""INSERT INTO persona_snapshot (name, scratch_json, tile_x, tile_y, updated_at)
                               VALUES (%s, %s, %s, %s, NOW())
                               ON CONFLICT (name) DO UPDATE
                               SET scratch_json = EXCLUDED.scratch_json,
                                   tile_x = EXCLUDED.tile_x, tile_y = EXCLUDED.tile_y,
                                   updated_at = NOW()""",
                            (name, scratch_json, tile[0], tile[1]))
            else:
                cur.execute("""INSERT OR REPLACE INTO persona_snapshot
                               (name, scratch_json, tile_x, tile_y, updated_at)
                               VALUES (?, ?, ?, ?, datetime('now'))""",
                            (name, scratch_json, tile[0], tile[1]))

    def load_persona_snapshot(self, name: str) -> Optional[dict]:
        import json
        with self.cursor() as (cur, _):
            cur.execute(self._q(
                "SELECT scratch_json, tile_x, tile_y FROM persona_snapshot WHERE name = ?"),
                (name,))
            row = cur.fetchone()
        if not row:
            return None
        return {"scratch": json.loads(row[0]), "tile": (row[1], row[2])}


_db = None
def get_db() -> DB:
    global _db
    if _db is None:
        _db = DB()
        _db.init_schema()
    return _db
