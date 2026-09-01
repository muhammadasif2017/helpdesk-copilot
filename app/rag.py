"""RAG store: fastembed embeddings + sqlite-vec vector search.

One SQLite file holds both the chunk text and the vector index — no external
services, which keeps the whole stack runnable on a laptop.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "kb.db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ONNX, CPU-friendly
EMBED_DIM = 384

_embedder: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def init_schema(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        " id INTEGER PRIMARY KEY, source TEXT NOT NULL, heading TEXT, body TEXT NOT NULL)"
    )
    db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{EMBED_DIM}])"
    )


@dataclass
class Chunk:
    source: str
    heading: str
    body: str
    distance: float = 0.0


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file into one chunk per ## section (intro text included)."""
    chunks: list[Chunk] = []
    heading = path.stem
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if lines and "".join(lines).strip():
                chunks.append(Chunk(path.name, heading, "\n".join(lines).strip()))
            heading = line[3:].strip()
            lines = []
        else:
            lines.append(line)
    if lines and "".join(lines).strip():
        chunks.append(Chunk(path.name, heading, "\n".join(lines).strip()))
    return chunks


def ingest(kb_dir: Path, db_path: Path = DB_PATH) -> int:
    db = connect(db_path)
    init_schema(db)
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM vec_chunks")

    chunks: list[Chunk] = []
    for md in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_markdown(md))

    texts = [f"{c.heading}\n{c.body}" for c in chunks]
    embeddings = list(get_embedder().embed(texts))

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        db.execute(
            "INSERT INTO chunks (id, source, heading, body) VALUES (?, ?, ?, ?)",
            (i, chunk.source, chunk.heading, chunk.body),
        )
        db.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
            (i, sqlite_vec.serialize_float32(emb.tolist())),
        )
    db.commit()
    db.close()
    return len(chunks)


def search(query: str, k: int = 4, db_path: Path = DB_PATH) -> list[Chunk]:
    db = connect(db_path)
    query_emb = next(iter(get_embedder().embed([query])))
    rows = db.execute(
        "SELECT c.source, c.heading, c.body, v.distance"
        " FROM vec_chunks v JOIN chunks c ON c.id = v.rowid"
        " WHERE v.embedding MATCH ? AND v.k = ?"
        " ORDER BY v.distance",
        (sqlite_vec.serialize_float32(query_emb.tolist()), k),
    ).fetchall()
    db.close()
    return [Chunk(source=r[0], heading=r[1], body=r[2], distance=r[3]) for r in rows]
