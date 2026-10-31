"""Central configuration. Values come from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


@dataclass(frozen=True)
class Settings:
    # IBM watsonx.ai
    ibm_api_key: str
    ibm_project_id: str
    ibm_url: str
    model_id: str
    # Retrieval
    embedding_model: str
    kb_path: Path
    chroma_dir: Path
    collection_name: str
    top_k: int
    max_distance: float
    chunk_max_chars: int
    # Generation
    max_new_tokens: int
    temperature: float
    history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ibm_api_key=_env("IBM_API_KEY"),
            ibm_project_id=_env("IBM_PROJECT_ID"),
            ibm_url=_env("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
            model_id=_env("IBM_MODEL_ID", "ibm/granite-3-3-8b-instruct"),
            embedding_model=_env("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            kb_path=_resolve(_env("KB_PATH", "data/ArthSaathi_Combined_Knowledge_Base_Final.txt")),
            chroma_dir=_resolve(_env("CHROMA_DIR", "chroma_db")),
            collection_name=_env("COLLECTION_NAME", "arthsaathi_kb"),
            top_k=int(_env("TOP_K", "4")),
            max_distance=float(_env("MAX_DISTANCE", "0.70")),
            chunk_max_chars=int(_env("CHUNK_MAX_CHARS", "1400")),
            max_new_tokens=int(_env("MAX_NEW_TOKENS", "700")),
            temperature=float(_env("TEMPERATURE", "0.1")),
            history_turns=int(_env("HISTORY_TURNS", "3")),
        )

    @property
    def has_ibm_credentials(self) -> bool:
        return bool(self.ibm_api_key and self.ibm_project_id)

    def with_ibm_credentials(self, api_key: str, project_id: str) -> "Settings":
        return replace(self, ibm_api_key=api_key.strip(), ibm_project_id=project_id.strip())
