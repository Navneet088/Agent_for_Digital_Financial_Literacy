"""ChromaDB-backed retriever with automatic (re)indexing when the knowledge base changes."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .chunker import parse_knowledge_base
from .config import Settings

log = logging.getLogger(__name__)
BATCH = 128


@dataclass(frozen=True)
class Hit:
    text: str
    part: str
    title: str
    urls: tuple[str, ...]
    distance: float  # cosine distance: 0 = identical, higher = less similar


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]        # hits within MAX_DISTANCE (used for answering)
    candidates: list[Hit]  # all top-k candidates (useful for debugging/tuning)


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embed = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        self._collection = self._ensure_index()

    # ------------------------------------------------------------------ indexing
    def _fingerprint(self, raw_text: str) -> str:
        s = self.settings
        payload = f"{s.embedding_model}|{s.chunk_max_chars}|{raw_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_index(self):
        s = self.settings
        if not s.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {s.kb_path}")
        raw = s.kb_path.read_text(encoding="utf-8")
        fingerprint = self._fingerprint(raw)
        fp_file = s.chroma_dir / ".kb_fingerprint"

        try:
            collection = self._client.get_collection(s.collection_name, embedding_function=self._embed)
            up_to_date = (
                collection.count() > 0
                and fp_file.exists()
                and fp_file.read_text().strip() == fingerprint
            )
            if up_to_date:
                log.info("Reusing existing index (%d chunks).", collection.count())
                return collection
        except Exception:  # collection does not exist yet
            pass

        log.info("Building vector index from %s ...", s.kb_path.name)
        try:
            self._client.delete_collection(s.collection_name)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=s.collection_name,
            embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )
        chunks = parse_knowledge_base(raw, max_chars=s.chunk_max_chars)
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "part": c.part,
                        "title": c.title,
                        "language": c.language,
                        "urls": " | ".join(c.urls),
                    }
                    for c in batch
                ],
            )
        fp_file.write_text(fingerprint)
        log.info("Indexed %d chunks.", len(chunks))
        return collection

    # ------------------------------------------------------------------ search
    @property
    def size(self) -> int:
        return self._collection.count()

    def search(self, query: str) -> SearchResult:
        return self.search_multi([query])

    def search_multi(self, queries: list[str]) -> SearchResult:
        """Search with several phrasings of the same question (e.g. original + English
        translation), merge by chunk keeping the best distance, return the overall top-k."""
        s = self.settings
        n = min(s.top_k, max(self._collection.count(), 1))
        best: dict[str, Hit] = {}
        for query in dict.fromkeys(q for q in queries if q.strip()):
            res = self._collection.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                urls = tuple(u for u in (meta.get("urls") or "").split(" | ") if u)
                hit = Hit(text=doc, part=meta.get("part", ""), title=meta.get("title", ""), urls=urls, distance=float(dist))
                if doc not in best or hit.distance < best[doc].distance:
                    best[doc] = hit
        candidates = sorted(best.values(), key=lambda h: h.distance)[: s.top_k]
        hits = [h for h in candidates if h.distance <= s.max_distance]
        return SearchResult(hits=hits, candidates=candidates)
