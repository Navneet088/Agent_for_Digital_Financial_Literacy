"""RAG pipeline: guardrails -> retrieval -> grounded Granite answer -> post-processing."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .guardrails import (
    STRINGS,
    detect_language,
    is_urgent_fraud,
    mask_sensitive,
    needs_disclaimer,
)
from .llm import GraniteClient
from .retriever import Hit, KnowledgeBase

KB_DATE = "30 October 2026"

SYSTEM_PROMPT = """You are ArthSaathi, a friendly and trustworthy AI assistant for digital financial literacy in India.
You help ordinary people (students, farmers, shopkeepers, homemakers, senior citizens, first-time earners) understand UPI and digital payments, online fraud, interest rates and loans, budgeting, savings, insurance, investing basics and government schemes.

RULES (follow strictly):
1. Answer ONLY from the KNOWLEDGE below. If it does not contain the answer, say clearly that you do not have reliable information and suggest the relevant official source. Never invent facts, numbers, rates, limits, dates or scheme conditions.
2. The user's question language decides the reply language: ALWAYS reply in {language}, even if the KNOWLEDGE below is written in English, Hindi or Hinglish. Translate the facts faithfully; keep numbers, amounts, URLs, helpline numbers and terms such as UPI, OTP, PIN, EMI, KYC, APK unchanged. Use short sentences and simple words. Use a numbered list for steps. Keep the answer focused and under about 200 words unless the user asks for detail.
3. Never ask for, or accept, an OTP, UPI PIN, ATM PIN, CVV, password or full card number. When relevant, remind the user that these must never be shared and that a PIN is never needed to RECEIVE money.
4. For rates, limits, fees, deadlines and rules, mention that they can change and should be verified on the named official website or with the bank. Knowledge base last updated: {kb_date}.
5. Never promise returns or safety of an investment, never recommend a specific share, fund or product, and never guarantee a refund or loan approval.
6. Be kind and non-judgmental. Fraud victims must never be blamed.
7. Do not mention "knowledge", "context" or "documents" in your reply. Name official sources (for example RBI, NPCI, cybercrime.gov.in) naturally when helpful.

KNOWLEDGE:
{context}"""


@dataclass
class Source:
    title: str
    part: str
    distance: float
    urls: tuple[str, ...]


@dataclass
class Answer:
    text: str
    language: str
    grounded: bool
    urgent: bool
    sources: list[Source] = field(default_factory=list)
    debug: list[Hit] = field(default_factory=list)


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))


class FinancialLiteracyAgent:
    def __init__(self, settings: Settings, kb: KnowledgeBase, llm: GraniteClient):
        self.settings = settings
        self.kb = kb
        self.llm = llm

    @staticmethod
    def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
        """Short follow-ups ("and the fees?") are expanded with the previous user question."""
        if len(question.split()) <= 4:
            for msg in reversed(history):
                if msg["role"] == "user":
                    return f"{msg['content']} {question}"
        return question

    def _to_english(self, text: str) -> str:
        """Translate a Hindi/Hinglish question into a short English search query.
        Falls back to '' on any failure so retrieval still works with the original text."""
        try:
            out = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's question about Indian personal finance, banking, UPI, "
                            "scams, loans or insurance into one clear English sentence. Keep terms like "
                            "UPI, OTP, EMI, KYC, APK, NEFT as they are. Output ONLY the translation."
                        ),
                    },
                    {"role": "user", "content": text[:500]},
                ]
            )
            return out.strip().strip('"')[:300]
        except Exception:  # noqa: BLE001
            return ""

    def _history_messages(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        keep = history[-(self.settings.history_turns * 2):]
        return [{"role": m["role"], "content": m["content"][:800]} for m in keep]

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        language: str = "auto",
    ) -> Answer:
        history = history or []
        safe_question, masked = mask_sensitive(question.strip())
        lang = detect_language(safe_question) if language == "auto" else language
        ui = STRINGS[lang]

        prefix: list[str] = []
        if masked:
            prefix.append(ui["sensitive"].format(items=", ".join(masked)))
        urgent = is_urgent_fraud(safe_question)
        if urgent:
            prefix.append(ui["urgent"])

        # Cross-lingual retrieval: the knowledge base mixes English, Hindi and Hinglish, so a
        # Hindi/Hinglish question is also searched in English (and vice-versa matches still work
        # through the multilingual embedding model).
        search_text = self._retrieval_query(safe_question, history)
        queries = [search_text]
        if lang != "en":
            english = self._to_english(search_text)
            if english:
                queries.append(english)
        result = self.kb.search_multi(queries)

        if not result.hits:
            body = ui["not_found"]
            text = "\n\n".join(prefix + [body])
            return Answer(text=text, language=lang, grounded=False, urgent=urgent, debug=result.candidates)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    language=ui["llm_lang"], kb_date=KB_DATE, context=_format_context(result.hits)
                ),
            },
            *self._history_messages(history),
            {"role": "user", "content": safe_question},
        ]
        reply = self.llm.chat(messages)
        if not reply:
            reply = ui["not_found"]

        parts = prefix + [reply]
        if needs_disclaimer(safe_question):
            parts.append(ui["disclaimer"])

        sources = [Source(h.title, h.part, h.distance, h.urls) for h in result.hits]
        return Answer(
            text="\n\n".join(parts),
            language=lang,
            grounded=True,
            urgent=urgent,
            sources=sources,
            debug=result.candidates,
        )
