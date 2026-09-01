"""Acme Helpdesk — a small ticketing app with an embedded LLM support assistant."""

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from . import assistant  # noqa: E402  (needs env loaded first)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Acme Helpdesk")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Demo tickets — in a real product these come from a database.
TICKETS = [
    {"id": 101, "customer": "Dana R.", "subject": "Wrong size shoes, want to return", "status": "open"},
    {"id": 102, "customer": "Leo M.", "subject": "Package marked delivered but missing", "status": "open"},
    {"id": 103, "customer": "Priya K.", "subject": "Can't reset account password", "status": "pending"},
    {"id": 104, "customer": "Sam T.", "subject": "Charged twice for one order", "status": "open"},
]


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"tickets": TICKETS}
    )


@app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    question = str(payload.get("question", "")).strip()
    if not question:
        return StreamingResponse(iter(["data: [DONE]\n\n"]), media_type="text/event-stream")

    chunks, stream = assistant.answer(question)

    def sse():
        sources = [{"source": c.source, "heading": c.heading} for c in chunks]
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        for delta in stream:
            yield f"data: {json.dumps({'delta': delta})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
