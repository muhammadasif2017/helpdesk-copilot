"""Acme Helpdesk — a small ticketing app with an embedded LLM support assistant."""

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from . import approvals, assistant, store, tools  # noqa: E402  (needs env loaded first)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Acme Helpdesk")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

def asset_version() -> int:
    """Cache-buster for the one script tag.

    StaticFiles sends no Cache-Control, so browsers fall back to heuristic
    caching and keep serving an old chat.js after it changes — a shipped fix
    then never reaches the page. Stamping the mtime makes the URL change with
    the file.
    """
    return int((BASE_DIR / "static" / "chat.js").stat().st_mtime)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"tickets": store.TICKETS, "asset_version": asset_version()},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """No icon to serve, but a 404 on every page load is noise in the console
    that makes real errors easier to miss."""
    return Response(status_code=204)


@app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    question = str(payload.get("question", "")).strip()
    if not question:
        return StreamingResponse(iter(["data: [DONE]\n\n"]), media_type="text/event-stream")

    chunks, events = assistant.answer(question, payload.get("ticket_id"))

    def sse():
        sources = [{"source": c.source, "heading": c.heading} for c in chunks]
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/api/approve")
async def approve(request: Request):
    """Execute a proposed write after a human approves it.

    The request carries only a proposal id. The tool name and arguments come
    from the server's own record of what was proposed, so approving cannot be
    turned into "run any tool with any arguments".
    """
    payload = await request.json()
    proposal = approvals.take(str(payload.get("proposal_id", "")))
    if proposal is None:
        return JSONResponse(
            {"error": "That approval is unknown or has already been used."}, status_code=404
        )

    return {
        "tool": proposal.tool,
        "arguments": proposal.arguments,
        "result": tools.execute(proposal.tool, proposal.arguments, proposal.scope),
    }


@app.post("/api/decline")
async def decline(request: Request):
    payload = await request.json()
    proposal = approvals.take(str(payload.get("proposal_id", "")))
    if proposal is None:
        return JSONResponse(
            {"error": "That approval is unknown or has already been used."}, status_code=404
        )
    return {"tool": proposal.tool, "declined": True}
