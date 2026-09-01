"""Acme Helpdesk — a small ticketing app with an embedded LLM support assistant."""

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from . import assistant, store  # noqa: E402  (needs env loaded first)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Acme Helpdesk")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"tickets": store.TICKETS}
    )


@app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    question = str(payload.get("question", "")).strip()
    if not question:
        return StreamingResponse(iter(["data: [DONE]\n\n"]), media_type="text/event-stream")

    chunks, events = assistant.answer(question)

    def sse():
        sources = [{"source": c.source, "heading": c.heading} for c in chunks]
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
