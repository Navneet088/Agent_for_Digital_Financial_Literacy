import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import parse_knowledge_base, split_text  # noqa: E402
from src.guardrails import detect_language, is_urgent_fraud, mask_sensitive  # noqa: E402

KB = Path(__file__).resolve().parent.parent / "data" / "ArthSaathi_Combined_Knowledge_Base_Final.txt"

SAMPLE = """# PART A: DEMO
## A1. How do I send money via UPI?
Source: NPCI (https://www.npci.org.in/).
1. Open the app. 2. Enter the amount.

## A2. Tiny
x
# PART O: HINDI AND HINGLISH QUESTIONS AND ANSWERS
## UPI PIN kya hota hai?
UPI PIN ek secret code hai jisse aap payment authorize karte hain.
"""


def test_chunk_structure():
    chunks = parse_knowledge_base(SAMPLE)
    assert len(chunks) == 2  # "Tiny" is dropped (too short)
    first, second = chunks
    assert first.part == "PART A: DEMO" and "UPI" in first.title
    assert first.urls == ("https://www.npci.org.in/",)
    assert first.language == "en" and second.language == "hi"
    assert first.text.startswith("PART A: DEMO | A1.")


def test_long_section_is_split():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(10))
    pieces = split_text(text, 800)
    assert len(pieces) > 1 and all(len(p) <= 900 for p in pieces)


def test_real_knowledge_base():
    chunks = parse_knowledge_base(KB.read_text(encoding="utf-8"))
    assert len(chunks) > 150
    assert len({c.id for c in chunks}) == len(chunks)
    assert any("1930" in c.text for c in chunks)


def test_language_detection():
    assert detect_language("How do I send money via UPI?") == "en"
    assert detect_language("UPI se paise kaise bhejte hain?") == "hinglish"
    assert detect_language("यूपीआई से पैसे कैसे भेजें?") == "hi"


def test_masking():
    masked, found = mask_sensitive("my otp is 482913 and card 4111 1111 1111 1111")
    assert "482913" not in masked and "4111" not in masked
    assert "OTP/PIN/CVV" in found and "card number" in found


def test_urgent_detection():
    assert is_urgent_fraud("mere account se paise kat gaye")
    assert is_urgent_fraud("I shared my OTP with a caller")
    assert not is_urgent_fraud("How do I avoid fraud?")
