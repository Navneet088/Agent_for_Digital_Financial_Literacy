"""Verify IBM watsonx.ai credentials and list the Granite models available to you.

Usage:  python scripts/check_ibm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.llm import GraniteClient, LLMError  # noqa: E402


def list_granite_models(settings: Settings) -> list[str]:
    from ibm_watsonx_ai import APIClient, Credentials

    client = APIClient(
        credentials=Credentials(url=settings.ibm_url, api_key=settings.ibm_api_key),
        project_id=settings.ibm_project_id,
    )
    specs = client.foundation_models.get_model_specs() or {}
    ids = [r.get("model_id", "") for r in specs.get("resources", [])]
    return sorted(i for i in ids if "granite" in i.lower())


if __name__ == "__main__":
    s = Settings.from_env()
    if not s.has_ibm_credentials:
        sys.exit("IBM_API_KEY / IBM_PROJECT_ID missing in .env")

    print(f"Region URL : {s.ibm_url}")
    print(f"Model      : {s.model_id}")
    try:
        print("\nGranite models available:")
        for mid in list_granite_models(s):
            print("  -", mid)
    except Exception as exc:  # noqa: BLE001
        print(f"(could not list models: {exc})")

    print("\nTest call ...")
    try:
        reply = GraniteClient(s).chat([{"role": "user", "content": "Say 'ready' in one word."}])
        print("OK ->", reply)
    except LLMError as exc:
        sys.exit(f"FAILED -> {exc}")
