"""Build (or rebuild) the ChromaDB index from the knowledge base file.

Usage:  python scripts/build_index.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.retriever import KnowledgeBase  # noqa: E402

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    kb = KnowledgeBase(Settings.from_env())
    print(f"Index ready: {kb.size} chunks in {kb.settings.chroma_dir}")
