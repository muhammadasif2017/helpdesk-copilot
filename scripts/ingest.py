"""Build (or rebuild) the knowledge-base vector index from kb/*.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag

if __name__ == "__main__":
    kb_dir = Path(__file__).resolve().parent.parent / "kb"
    n = rag.ingest(kb_dir)
    print(f"Ingested {n} chunks from {kb_dir} into {rag.DB_PATH}")
