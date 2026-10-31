"""Safety layer: language detection, sensitive-data masking, fraud-emergency detection
and localized static messages. Everything here is deterministic (no LLM involved)."""
from __future__ import annotations

import re

# ---------------------------------------------------------------- language
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_HINGLISH_TOKENS = {
    "kaise", "kya", "hai", "hain", "karein", "karun", "kare", "nahi", "nahin", "mera", "meri",
    "mere", "paise", "paisa", "bhejte", "bhejna", "bhejein", "mein", "ko", "ka", "ki", "ke", "se",
    "aap", "kyun", "kab", "kitna", "kitne", "chahiye", "batao", "bataiye", "hota", "hoti", "hua",
    "gaya", "gaye", "raha", "rahi", "mujhe", "wala", "wale", "dena", "dein", "liya", "lena",
    "sakta", "sakte", "koi", "aur", "toh", "samjhao", "batayein", "kitni", "abhi", "turant",
}


def detect_language(text: str) -> str:
    """Return 'hi' (Devanagari), 'hinglish' (romanized Hindi) or 'en'."""
    letters = re.findall(r"[A-Za-zऀ-ॿ]", text)
    if letters:
        deva = len(_DEVANAGARI_RE.findall(text))
        if deva / len(letters) > 0.3:
            return "hi"
    tokens = re.findall(r"[a-z]+", text.lower())
    if tokens:
        hits = sum(1 for t in tokens if t in _HINGLISH_TOKENS)
        if hits >= 2 and hits / len(tokens) >= 0.2:
            return "hinglish"
    return "en"


# ---------------------------------------------------------------- sensitive data
_SENSITIVE_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("card number", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD-NUMBER-REMOVED]"),
    ("Aadhaar number", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[AADHAAR-REMOVED]"),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN-REMOVED]"),
    (
        "OTP/PIN/CVV",
        re.compile(r"(?i)\b(otp|pin|cvv|password|passcode)\b(\s*(?:is|:|=|-|hai)?\s*)\d{3,8}\b"),
        r"\1 [REMOVED]",
    ),
]


def mask_sensitive(text: str) -> tuple[str, list[str]]:
    """Mask secrets before the text is sent to the LLM. Returns (masked_text, labels_found)."""
    found: list[str] = []
    for label, pattern, repl in _SENSITIVE_RULES:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(repl, text)
    return text, found


# ---------------------------------------------------------------- fraud emergency
_URGENT_PATTERNS = [
    r"\b(paise|paisa|rupay\w*|amount|money)\b.*\b(kat|cut|deduct\w*|chale gaye|chala gaya|gayab|stolen|lost|transferred)\b",
    r"\b(fraud|scam|thagi|thag|dhokha|dhoka|cheat\w*|hack\w*|phishing)\b.*\b(ho gaya|ho gayi|hua|happened|victim|mere saath|with me|mujhe)\b",
    r"\b(otp|pin|cvv|password)\b.*\b(share kar|bata diya|de diya|bhej diya|shared|gave|told|disclosed)\b",
    r"\b(i|already|accidentally)\b.*\b(shared|gave|told|disclosed|sent)\b.*\b(otp|pin|cvv|password)\b",
    r"\b(apk|app)\b.*\b(install kar (diya|di|liya)|installed|download kar (diya|liya))\b",
    r"\bunauthori[sz]ed\b",
    r"\bwithout (my )?(permission|knowledge)\b",
    r"bina permission",
    r"\bscammed\b|\bcheated\b|\bdefrauded\b",
    r"paise kat gaye|paise chale gaye|account se paise",
    r"(धोखा|ठगी|फ्रॉड|हैक|पैसे कट|पैसे चले गए)",
    r"(ओटीपी|पिन).*(दे दिया|बता दिया|शेयर कर दिया)",
]
_URGENT_RE = re.compile("|".join(f"(?:{p})" for p in _URGENT_PATTERNS), re.IGNORECASE)


def is_urgent_fraud(text: str) -> bool:
    return bool(_URGENT_RE.search(text))


# ---------------------------------------------------------------- disclaimers
_ADVICE_RE = re.compile(
    r"invest|share|stock|mutual fund|\bsip\b|crypto|\btax\b|\bitr\b|loan|insurance|policy|retire|"
    r"निवेश|शेयर|कर\b|लोन|बीमा|nivesh|tax|bima",
    re.IGNORECASE,
)


def needs_disclaimer(text: str) -> bool:
    return bool(_ADVICE_RE.search(text))


# ---------------------------------------------------------------- localized strings
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "urgent": (
            "🚨 **If money has been lost or you shared an OTP/PIN, act right now:**\n"
            "1. Call your bank on its **official number** (back of your card / bank app) and block the card, UPI and net banking.\n"
            "2. Call **1930** (National Cyber Crime Helpline) and file a report at **cybercrime.gov.in**.\n"
            "3. Do **not** pay any \"recovery agent\" and do not share OTP/PIN with anyone.\n"
            "4. Save screenshots, transaction ID/UTR and the scammer's phone numbers.\n"
            "5. Reporting quickly improves the chance of action, but recovery is never guaranteed."
        ),
        "not_found": (
            "I could not find reliable information about this in my knowledge base. "
            "Please check the official source (rbi.org.in, your bank's official app/branch) or the concerned provider's helpline.\n\n"
            "I can help with: UPI and digital payments, online scams and fraud reporting, loans and EMI, interest rates, "
            "budgeting and savings, insurance, investing basics, credit score and government schemes."
        ),
        "sensitive": "🔒 I removed sensitive details ({items}) from your message. Never share OTP, PIN, CVV, passwords or full card numbers with anyone, including this assistant.",
        "disclaimer": "_ℹ️ This is general educational information, not personal financial, tax or legal advice. Rates, limits and rules change, so verify them on the official website or with your bank._",
        "llm_lang": "English",
    },
    "hi": {
        "urgent": (
            "🚨 **अगर पैसे कट गए हैं या आपने OTP/PIN शेयर कर दिया है, तो तुरंत ये करें:**\n"
            "1. अपने बैंक को **आधिकारिक नंबर** (कार्ड के पीछे / बैंक ऐप) पर कॉल करें और कार्ड, UPI व नेट बैंकिंग ब्लॉक करवाएँ।\n"
            "2. **1930** (राष्ट्रीय साइबर क्राइम हेल्पलाइन) पर कॉल करें और **cybercrime.gov.in** पर शिकायत दर्ज करें।\n"
            "3. किसी \"रिकवरी एजेंट\" को पैसे **न दें** और OTP/PIN किसी को न बताएँ।\n"
            "4. स्क्रीनशॉट, ट्रांज़ैक्शन ID/UTR और ठग के फ़ोन नंबर सुरक्षित रखें।\n"
            "5. जल्दी शिकायत करने से कार्रवाई की संभावना बढ़ती है, पर पैसे वापस मिलने की गारंटी नहीं होती।"
        ),
        "not_found": (
            "मुझे इस बारे में अपने नॉलेज बेस में भरोसेमंद जानकारी नहीं मिली। "
            "कृपया आधिकारिक स्रोत (rbi.org.in, अपने बैंक का ऐप/ब्रांच) या संबंधित संस्था की हेल्पलाइन देखें।\n\n"
            "मैं इन विषयों में मदद कर सकता हूँ: UPI और डिजिटल पेमेंट, ऑनलाइन ठगी और रिपोर्टिंग, लोन और EMI, ब्याज दर, "
            "बजट और बचत, बीमा, निवेश की बुनियादी बातें, क्रेडिट स्कोर और सरकारी योजनाएँ।"
        ),
        "sensitive": "🔒 मैंने आपके संदेश से संवेदनशील जानकारी ({items}) हटा दी है। OTP, PIN, CVV, पासवर्ड या पूरा कार्ड नंबर किसी को भी न बताएँ, इस असिस्टेंट को भी नहीं।",
        "disclaimer": "_ℹ️ यह सामान्य शैक्षणिक जानकारी है, व्यक्तिगत वित्तीय, कर या कानूनी सलाह नहीं। दरें, सीमाएँ और नियम बदलते रहते हैं, इसलिए आधिकारिक वेबसाइट या अपने बैंक से जाँच लें।_",
        "llm_lang": "Hindi (Devanagari script)",
    },
    "hinglish": {
        "urgent": (
            "🚨 **Agar paise kat gaye hain ya aapne OTP/PIN share kar diya hai, to turant ye karein:**\n"
            "1. Apne bank ko **official number** (card ke peeche / bank app) par call karke card, UPI aur net banking block karwayein.\n"
            "2. **1930** (National Cyber Crime Helpline) par call karein aur **cybercrime.gov.in** par complaint karein.\n"
            "3. Kisi \"recovery agent\" ko paise **na dein** aur OTP/PIN kisi ko na batayein.\n"
            "4. Screenshots, transaction ID/UTR aur thag ke phone number save rakhein.\n"
            "5. Jaldi report karne se action ke chance badhte hain, par paise wapas milne ki guarantee nahi hoti."
        ),
        "not_found": (
            "Mujhe iske baare mein apne knowledge base mein bharosemand jaankari nahi mili. "
            "Kripya official source (rbi.org.in, apne bank ka official app/branch) ya related company ki helpline check karein.\n\n"
            "Main in topics mein madad kar sakta hoon: UPI aur digital payments, online fraud aur reporting, loan aur EMI, interest rate, "
            "budget aur savings, insurance, investing basics, credit score aur sarkari schemes."
        ),
        "sensitive": "🔒 Maine aapke message se sensitive details ({items}) hata di hain. OTP, PIN, CVV, password ya poora card number kisi ko bhi na dein, is assistant ko bhi nahi.",
        "disclaimer": "_ℹ️ Ye general educational jaankari hai, personal financial, tax ya legal advice nahi. Rates, limits aur rules badalte rehte hain, isliye official website ya apne bank se verify kar lein._",
        "llm_lang": "Hinglish (Hindi written in Roman/English letters)",
    },
}
