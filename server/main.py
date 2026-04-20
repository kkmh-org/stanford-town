"""
FastAPI server for LBS Demo — shared world with user interactions.
Supports both SQLite (dev) and Postgres (prod) via DATABASE_URL env var.
"""
import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Cookie, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.sim_engine import get_engine, NPC_NAMES
from server.db import get_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LBS_DEMO_DIR = os.path.join(ROOT, "lbs_demo_nan_yi")
PHASER_DEMO_DIR = os.path.join(ROOT, "environment", "frontend_server")

# DB helpers (delegated)
def log_chat(entry): get_db().log_chat(entry)
def verify_user(u, p): return get_db().verify_user(u, p)
def create_session(u): return get_db().create_session(u)
def get_user_from_token(t): return get_db().get_user_from_token(t)


# ---------------- Lifespan ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_db()  # init schema + seed users
    engine = get_engine()
    async def _chat_logger(payload):
        if payload.get("type") == "chat":
            try:
                log_chat(payload.get("entry", {}))
            except Exception as e:
                print(f"[db] log_chat failed: {e}")
    engine.add_listener(_chat_logger)
    task = asyncio.create_task(engine.run_forever())
    # Periodic persona snapshot saver
    persist_task = asyncio.create_task(_periodic_persist(engine))
    print(f"[server] sim engine launched (DB: {os.environ.get('DATABASE_URL','sqlite (default)')})")
    yield
    engine.stop()
    task.cancel()
    persist_task.cancel()


async def _periodic_persist(engine):
    """Every 30s save persona snapshots to DB so restart doesn't lose state."""
    while True:
        try:
            await asyncio.sleep(30)
            db = get_db()
            for name, p in engine.personas.items():
                try:
                    # Extract key scratch fields (not everything — datetime etc. doesn't serialize)
                    snap = {
                        "name": p.scratch.name,
                        "curr_tile": list(p.scratch.curr_tile) if p.scratch.curr_tile else None,
                        "act_address": p.scratch.act_address,
                        "act_description": p.scratch.act_description,
                        "act_pronunciatio": p.scratch.act_pronunciatio,
                    }
                    db.save_persona_snapshot(name, json.dumps(snap, ensure_ascii=False),
                                             p.scratch.curr_tile or (0, 0))
                except Exception as e:
                    print(f"[persist] {name} failed: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[persist] loop failed: {e}")


app = FastAPI(lifespan=lifespan)


# ---------------- Pydantic ----------------

class LoginReq(BaseModel):
    username: str
    password: str


class ChatReq(BaseModel):
    to_npc: str
    msg: str


class GodCmdReq(BaseModel):
    type: str            # "force_move"
    target: str          # npc name
    sector: Optional[str] = None


class WorldEventReq(BaseModel):
    event: str
    scope: Optional[str] = "all"  # "all" or sector name like "操场"


class MoveReq(BaseModel):
    dx: int
    dy: int


class TeleportReq(BaseModel):
    sector: str


# ---------------- Auth dep ----------------

def require_user(session: Optional[str] = Cookie(None)) -> str:
    user = get_user_from_token(session)
    if not user:
        raise HTTPException(401, "not logged in")
    return user


# ---------------- HTTP routes ----------------

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "login.html"))


@app.post("/api/login")
def api_login(req: LoginReq):
    if not verify_user(req.username, req.password):
        raise HTTPException(401, "invalid credentials")
    token = create_session(req.username)
    resp = JSONResponse({"ok": True, "username": req.username})
    resp.set_cookie("session", token, httponly=True, max_age=86400 * 7)
    return resp


@app.post("/api/logout")
def api_logout(user: str = Depends(require_user)):
    get_engine().remove_visitor(user)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp


@app.get("/api/me")
def api_me(user: str = Depends(require_user)):
    return {"username": user}


@app.get("/api/state")
def api_state(_user: str = Depends(require_user)):
    return get_engine().snapshot()


@app.post("/api/chat")
def api_chat(req: ChatReq, user: str = Depends(require_user)):
    if req.to_npc not in NPC_NAMES:
        raise HTTPException(400, f"unknown npc: {req.to_npc}")
    get_engine().add_user_message(user, req.to_npc, req.msg)
    return {"ok": True}


@app.post("/api/god")
def api_god(req: GodCmdReq, user: str = Depends(require_user)):
    # Any logged-in user can issue god commands (internal tool)
    engine = get_engine()
    if req.type == "force_move":
        if req.target not in NPC_NAMES:
            raise HTTPException(400, f"unknown npc: {req.target}")
        engine.add_god_command({"type": "force_move", "target": req.target, "sector": req.sector})
    else:
        raise HTTPException(400, f"unknown cmd type: {req.type}")
    return {"ok": True}


@app.post("/api/world_event")
def api_world_event(req: WorldEventReq, user: str = Depends(require_user)):
    if not req.event.strip():
        raise HTTPException(400, "event text is required")
    get_engine().add_world_event(req.event.strip(), req.scope or "all", from_user=user)
    return {"ok": True}


@app.post("/api/visitor/join")
def api_visitor_join(user: str = Depends(require_user)):
    get_engine().add_visitor(user)
    return {"ok": True}


@app.post("/api/visitor/leave")
def api_visitor_leave(user: str = Depends(require_user)):
    get_engine().remove_visitor(user)
    return {"ok": True}


@app.post("/api/visitor/move")
def api_visitor_move(req: MoveReq, user: str = Depends(require_user)):
    get_engine().move_visitor(user, req.dx, req.dy)
    return {"ok": True}


@app.post("/api/visitor/teleport")
def api_visitor_teleport(req: TeleportReq, user: str = Depends(require_user)):
    get_engine().teleport_visitor_to(user, req.sector)
    return {"ok": True}


@app.get("/api/chat_history")
def api_chat_history(_user: str = Depends(require_user), limit: int = 50):
    return {"history": get_db().get_chat_history(limit)}


# ---------------- WebSocket ----------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, session: Optional[str] = Cookie(None)):
    user = get_user_from_token(session)
    if not user:
        await ws.close(code=4401)
        return
    await ws.accept()
    engine = get_engine()

    async def send_payload(payload: dict):
        try:
            await ws.send_json(payload)
        except Exception:
            pass

    engine.add_listener(send_payload)
    # Initial snapshot
    await send_payload(engine.snapshot())
    try:
        while True:
            # We don't really need client messages here — everything is HTTP.
            # But keep the connection alive.
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        engine.remove_listener(send_payload)


# ---------------- Static ----------------

# Serve the LBS demo output folder for sim_log.json if needed
app.mount("/lbs_static", StaticFiles(directory=LBS_DEMO_DIR), name="lbs_static")
# Serve Phaser static assets
app.mount("/phaser_static", StaticFiles(directory=os.path.join(PHASER_DEMO_DIR, "static_dirs")), name="phaser_static")
# Server's own static (index.html, login.html, JS)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8090, reload=False)
