"""Structure-aware chunking of the ArthSaathi knowledge base.

The knowledge base uses Markdown-like headings:
    "# PART X: ..."  -> a part
    "## Question"    -> a section (one question / topic)

Each section becomes one chunk (long sections are split on paragraph/sentence
boundaries). Every chunk is prefixed with "<part> | <title>" so the embedding
carries the topic context. This module is pure Python (no ML dependencies).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

URL_RE = re.compile(r"https?://[^\s)\]>,;]+")
HINDI_PART_RE = re.compile(r"HINDI|HINGLISH", re.IGNORECASE)
MIN_BODY_CHARS = 20


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str  # text that is embedded and shown to the LLM
    part: str
    title: str
    language: str  # "hi" for Hindi/Hinglish Q&A parts, otherwise "en"
    urls: tuple[str, ...]


def _extract_urls(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in URL_RE.findall(text):
        url = raw.rstrip(".,;)")
        if url not in seen:
            seen.append(url)
    return tuple(seen)


def _pack(units: list[str], max_chars: int, sep: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for unit in units:
        if cur and len(cur) + len(sep) + len(unit) > max_chars:
            out.append(cur)
            cur = unit
        else:
            cur = f"{cur}{sep}{unit}" if cur else unit
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into pieces <= max_chars (best effort) on natural boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n", para) if s.strip()]
            units.extend(_pack(sentences, max_chars, " "))
    return _pack(units, max_chars, "\n\n")


def parse_knowledge_base(raw_text: str, max_chars: int = 1400) -> list[Chunk]:
    sections: list[tuple[str, str, list[str]]] = []  # (part, title, lines)
    part, title, lines = "General", "Overview", []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if len(body) >= MIN_BODY_CHARS:
            sections.append((part, title, [body]))

    for line in raw_text.splitlines():
        if line.startswith("## "):
            flush()
            title, lines = line[3:].strip(), []
        elif line.startswith("# "):
            flush()
            part, title, lines = line[2:].strip(), "Overview", []
        else:
            lines.append(line)
    flush()

    chunks: list[Chunk] = []
    for idx, (sec_part, sec_title, body_list) in enumerate(sections):
        body = re.sub(r"\n{3,}", "\n\n", body_list[0])
        language = "hi" if HINDI_PART_RE.search(sec_part) else "en"
        pieces = split_text(body, max_chars)
        for n, piece in enumerate(pieces):
            suffix = f" (part {n + 1}/{len(pieces)})" if len(pieces) > 1 else ""
            header = f"{sec_part} | {sec_title}{suffix}"
            text = f"{header}\n{piece}"
            cid = hashlib.sha1(f"{idx}|{n}|{sec_part}|{sec_title}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=cid,
                    text=text,
                    part=sec_part,
                    title=sec_title,
                    language=language,
                    urls=_extract_urls(piece),
                )
            )
    return chunks
