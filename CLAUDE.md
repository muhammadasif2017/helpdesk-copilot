# Project: Helpdesk Copilot

A helpdesk product with an **embedded** LLM support assistant — RAG-grounded
answers, citations, guardrails, and a two-tier eval suite. The point of the repo
is demonstrating LLM integration *inside a product*, not a standalone chatbot.
Changes that make it look more like a generic chatbot demo work against that.

## Tech Stack

- Python 3.12 (pinned in `.python-version`), managed with **uv**
- FastAPI + Jinja2 templates + vanilla JS (no React/Streamlit/Gradio — see Boundaries)
- Retrieval: **fastembed** (`BAAI/bge-small-en-v1.5`, 384-dim ONNX, CPU) + **sqlite-vec**
- LLM: any OpenAI-compatible endpoint via the `openai` SDK; backend chosen by env vars
- Local default: Ollama running `qwen2.5:3b`

## Commands

```bash
uv sync --group dev                  # install (dev group holds pytest + httpx)
uv run python scripts/ingest.py      # build data/kb.db — REQUIRED before tests or first run
uv run uvicorn app.main:app --reload # serve on :8000
uv run pytest -m "not llm"           # fast suite, ~4s, LLM stubbed
uv run pytest -m llm                 # live-model suite, ~2min on CPU
uv run pytest                        # everything, 20 tests
```

## Architecture

```
app/main.py       FastAPI routes; /api/chat emits SSE: sources event → delta events → [DONE]
app/assistant.py  orchestration: retrieve → build grounded prompt → stream
app/rag.py        chunking, embedding, sqlite-vec search; one SQLite file, no services
app/llm.py        OpenAI-compatible client; backend from LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
kb/*.md           knowledge base, chunked one-per-`##`-section
```

## Conventions

- **No orchestration framework.** Agent/RAG logic is written directly against the
  SDK. This is deliberate — it's the part reviewers read. Don't introduce
  LangChain/LlamaIndex to "clean it up".
- Backend swaps happen through env vars only. No provider branching in code.
- Modules import as `from . import llm` so `llm.stream_chat` stays monkeypatchable
  by name in tests. Don't switch to `from .llm import stream_chat`.
- Comments explain constraints and non-obvious *why*, never what the next line does.
- Tests are named as sentences describing the property under test
  (`test_every_citation_names_a_document_that_was_actually_retrieved`).

## Eval Conventions

The eval suite is the repo's main credibility signal. Treat it as product code.

- **Split by cost.** Anything needing a live model gets `@pytest.mark.llm`. The
  unmarked suite must stay fast enough to run on every commit.
- **Never assert LLM behavior with a single-shot boolean.** Output is
  non-deterministic; sample and assert a rate. See
  `test_declines_when_the_answer_is_not_in_the_knowledge_base`:

  ```python
  answers = [ask(question) for _ in range(REFUSAL_SAMPLES)]
  rate = len([a for a in answers if is_refusal(a)]) / len(answers)
  assert rate >= REFUSAL_THRESHOLD, f"...non-refusals: {...}"
  ```

- **Test at the temperature the product ships** (0.2). Testing at temperature 0
  measures a configuration users never get.
- **Don't widen a string-match list to make a red test green.** That converts a
  real signal into a tautology. Either the property is real and worth a better
  assertion, or the test should go.
- Substring matching cannot verify *absence of fabrication*. That needs an LLM
  judge (v3). Don't fake it with negative substring checks — a correct refusal to
  a question about X legitimately contains the word X.

## Boundaries

- **Never commit `.env`** (holds backend config), `data/` (regenerable index), or `.venv/`.
- **Never put a real KB filename in `SYSTEM_PROMPT` as a format example.** A live
  filename in the prompt on every request biases citations toward it; small models
  copy format examples verbatim. Use `[filename.md]`. This was a real bug here.
- **Ask before editing `SYSTEM_PROMPT` rules.** The live evals assert on the
  behavior those rules produce; changing them silently breaks tests in ways that
  look like model regressions.
- **No GPU on the target machine** (Intel UHD 620, 16GB RAM). Don't add torch,
  sentence-transformers, or anything pulling CUDA. fastembed's ONNX runtime is
  chosen specifically to avoid this.
- **No reasoning models** (Qwen3, DeepSeek-R1) as the local default — thinking
  tokens add roughly a minute per answer on CPU. Plain instruct models only.
- Keep the dependency list short. Every added dep is one more thing a reviewer
  has to accept.

## Gotchas

- `data/kb.db` is gitignored, so a fresh clone **fails all retrieval tests until
  `scripts/ingest.py` runs**. This is the first thing to check on unexplained
  test failures.
- **Uvicorn orphans hold port 8000.** It runs as a child process, so killing the
  process you launched leaves the child alive still holding the socket. The next
  start then fails with `WinError 10013`, which reads like a permissions error
  but isn't. Running `.venv\Scripts\python.exe -m uvicorn` instead of `uv run`
  does *not* avoid this — there is still a child.
  Confusingly, `netstat`/`Get-NetTCPConnection` report the socket as owned by the
  dead parent PID, so the PID they name may not exist. Find the real holder by
  parent:

  ```powershell
  Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
      Select-Object ProcessId, ParentProcessId, CommandLine
  ```

  Always kill the whole tree: `taskkill /PID <pid> /T /F`. Verify the port is
  genuinely free by binding it, not by reading the connection table.
- First LLM call after boot is much slower than the rest — Ollama is loading the
  model into RAM. Roughly 15-20s per call after that.
- Windows converts LF→CRLF on checkout; files are committed as LF.
- **Never gate a merge on `pytest ... | tail`.** A shell pipeline reports the
  *last* command's status, so a failing suite piped through `tail` exits 0 and
  looks green. Read the summary line, or use `${PIPESTATUS[0]}`. This nearly put
  a red suite on `main`.
