"""Check retrieval quality and tune MAX_DISTANCE (no IBM credentials needed).

Usage:  python scripts/evaluate_retrieval.py

In-scope questions should land on the expected topic with a small distance;
out-of-scope questions should have a large distance (above MAX_DISTANCE).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.retriever import KnowledgeBase  # noqa: E402

IN_SCOPE = [
    ("How do I send money via UPI?", "send money"),
    ("UPI se paise kaise bhejte hain?", "upi"),
    ("What is a safe interest rate for a loan?", "safe interest rate"),
    ("Flat rate aur reducing rate mein kya fark hai?", "flat"),
    ("How is EMI calculated?", "emi"),
    ("What is a digital arrest scam?", "digital arrest"),
    ("Mujhe WhatsApp par APK file aayi hai", "apk"),
    ("What is the 50/30/20 rule?", "50/30/20"),
    ("How do I check my credit report for free?", "credit report"),
    ("How much is my bank deposit insured?", "deposit"),
    ("Mudra loan kya hota hai?", "mudra"),
    ("What is the call forwarding USSD scam?", "call forwarding"),
]
OUT_OF_SCOPE = [
    "Who won the cricket world cup?",
    "Give me a recipe for paneer butter masala",
    "Does the campus have a swimming pool?",
    "Write a python script to sort a list",
]

if __name__ == "__main__":
    s = Settings.from_env()
    kb = KnowledgeBase(s)
    print(f"Chunks: {kb.size} | MAX_DISTANCE={s.max_distance}\n")

    hits_ok = 0
    print("== IN SCOPE ==")
    for q, keyword in IN_SCOPE:
        res = kb.search(q)
        top = res.candidates[0]
        ok = any(keyword.lower() in (h.title + h.text).lower() for h in res.candidates)
        hits_ok += ok
        print(f"[{'OK ' if ok else 'MISS'}] d={top.distance:.3f} | {q}\n        -> {top.part} > {top.title}")
    print(f"\nHit@{s.top_k}: {hits_ok}/{len(IN_SCOPE)}")

    print("\n== OUT OF SCOPE (distance should exceed MAX_DISTANCE) ==")
    for q in OUT_OF_SCOPE:
        top = kb.search(q).candidates[0]
        verdict = "rejected" if top.distance > s.max_distance else "WOULD ANSWER (raise strictness)"
        print(f"d={top.distance:.3f} | {verdict} | {q}")
