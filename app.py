"""ArthSaathi - AI Agent for Digital Financial Literacy (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import logging

import streamlit as st

from src.config import Settings
from src.llm import GraniteClient, LLMError
from src.pipeline import KB_DATE, FinancialLiteracyAgent
from src.retriever import KnowledgeBase

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="ArthSaathi | Financial Literacy AI", page_icon="💰", layout="centered")

LANGUAGES = {"Auto-detect": "auto", "English": "en", "हिन्दी": "hi", "Hinglish": "hinglish"}

QUICK_QUESTIONS = {
    "💸 UPI": [
        "How do I send money via UPI?",
        "UPI se paise kaise bhejte hain?",
        "Galat UPI ID par paise chale gaye, kya karein?",
    ],
    "🛡️ Scams": [
        "Mujhe WhatsApp par APK file aayi hai, kya karun?",
        "What is a digital arrest scam?",
        "Someone asked me to scan a QR code to receive a refund. Is it safe?",
    ],
    "🏦 Loans & Interest": [
        "What is a safe interest rate for a loan?",
        "Flat rate aur reducing rate mein kya fark hai?",
        "How is EMI calculated?",
    ],
    "📊 Budget & Savings": [
        "How do I create a monthly budget?",
        "What is the 50/30/20 rule?",
        "How much should I keep as an emergency fund?",
    ],
}


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading knowledge base (the first time may take a little while)...")
def load_kb(cache_key: str, _settings: Settings) -> KnowledgeBase:
    _ = cache_key  # Used by Streamlit to invalidate the cache when settings change.
    return KnowledgeBase(_settings)


@st.cache_resource(show_spinner="Connecting to IBM Granite...")
def load_llm(cache_key: str, _settings: Settings) -> GraniteClient:
    _ = cache_key  # Used by Streamlit to invalidate the cache when settings change.
    return GraniteClient(_settings)


def _queue_question(q: str) -> None:
    st.session_state["queued"] = q


# --------------------------------------------------------------------------- settings / credentials
settings = Settings.from_env()

with st.sidebar:
    st.header("⚙️ Settings")
    language_label = st.radio("Reply language", list(LANGUAGES), index=0)
    language = LANGUAGES[language_label]

    st.divider()
    if settings.has_ibm_credentials:
        # Credentials are in .env -> no input needed.
        st.success("IBM credentials loaded from .env ")
    else:
        st.subheader("🔑 IBM Cloud credentials")
        st.caption("Set `IBM_API_KEY` and `IBM_PROJECT_ID` in `.env` to hide this form.")
        key_in = st.text_input("IBM Cloud API Key", type="password")
        proj_in = st.text_input("watsonx.ai Project ID")
        if key_in and proj_in:
            settings = settings.with_ibm_credentials(key_in, proj_in)

    st.divider()
    show_debug = st.toggle("Show retrieval details", value=False, help="Displays retrieved chunks and similarity distances.")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    st.divider()
    st.subheader("💡 Try asking")
    for group, questions in QUICK_QUESTIONS.items():
        with st.expander(group):
            for i, q in enumerate(questions):
                st.button(q, key=f"{group}-{i}", on_click=_queue_question, args=(q,), use_container_width=True)

    st.caption(f"Model: `{settings.model_id}`  \nKnowledge base updated: {KB_DATE}")

st.title("💰 ArthSaathi")
st.caption("Digital financial literacy assistant · RAG + IBM Granite · English / हिन्दी / Hinglish")

if not settings.has_ibm_credentials:
    st.warning("⚠️ Add the IBM API Key and Project ID to `.env` or enter them in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- load engine
try:
    kb = load_kb(f"{settings.kb_path}|{settings.embedding_model}|{settings.chunk_max_chars}", settings)
    key_hash = hashlib.sha256(settings.ibm_api_key.encode()).hexdigest()[:12]
    llm = load_llm(f"{settings.ibm_url}|{settings.model_id}|{settings.ibm_project_id}|{key_hash}", settings)
except FileNotFoundError as exc:
    st.error(f"Knowledge base file not found: {exc}")
    st.stop()
except LLMError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Startup error: {exc}")
    st.stop()

agent = FinancialLiteracyAgent(settings, kb, llm)

# --------------------------------------------------------------------------- chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 🙏 I am **ArthSaathi**. You can ask me about UPI, online fraud, loans and interest rates, budgeting, savings, "
                "insurance, or government schemes in English, Hindi, or Hinglish.\n\n"
                "_Never share your OTP, PIN, or password here._"
            ),
        }
    ]


def render_extras(msg: dict) -> None:
    sources = msg.get("sources") or []
    if sources:
        with st.expander("📚 Sources / official links"):
            seen_urls: set[str] = set()
            for s in sources:
                st.markdown(f"- **{s['title']}** · _{s['part']}_")
                for url in s["urls"]:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        st.markdown(f"  - {url}")
    if show_debug and msg.get("debug"):
        with st.expander("🔎 Retrieval details"):
            for d in msg["debug"]:
                flag = "✅" if d["used"] else "❌"
                st.markdown(f"{flag} `{d['distance']:.3f}` · {d['part']} › {d['title']}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

# --------------------------------------------------------------------------- handle input
question = st.chat_input("Ask Any Question... (e.g. How do I send money via UPI?)")
if not question and st.session_state.get("queued"):
    question = st.session_state.pop("queued")

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[1:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the official knowledge base..."):
            try:
                answer = agent.ask(question, history=history, language=language)
            except LLMError as exc:
                st.error(f"IBM watsonx.ai error: {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.stop()

        msg = {
            "role": "assistant",
            "content": answer.text,
            "sources": [
                {"title": s.title, "part": s.part, "urls": list(s.urls)} for s in answer.sources
            ],
            "debug": [
                {"part": h.part, "title": h.title, "distance": h.distance,
                 "used": h.distance <= settings.max_distance}
                for h in answer.debug
            ],
        }
        st.markdown(msg["content"])
        render_extras(msg)
    st.session_state.messages.append(msg)
