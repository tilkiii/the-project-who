"""
Text cleaning, Turkish-aware Unicode normalization, speaker token parsing, and slug sanitization.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, List, Set, Tuple

# Explicit Turkish transliteration map (handles dotless 'ı' and dotted 'İ' before NFKD)
_TR_SLUG_MAP = str.maketrans({
    "ç": "c",
    "Ç": "c",
    "ğ": "g",
    "Ğ": "g",
    "ı": "i",
    "I": "i",
    "İ": "i",
    "ö": "o",
    "Ö": "o",
    "ş": "s",
    "Ş": "s",
    "ü": "u",
    "Ü": "u",
})

_TR_CHARS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_WORD_TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+")

# Common Turkish words starting with 'I' (dotless) that contain no other diacritics
_TR_DOTLESS_I_PREFIXES = (
    "ISIK", "ISIL", "ISIN", "ISIT", "ILIK", "ILIM", "ILIC", "IRMAK",
    "IRK", "ISPARTA", "IGDIR", "IHLAMUR", "ISRAR", "ISLAH", "ISLIK",
)

TURKISH_STOPWORDS: Set[str] = {
    "acaba", "ama", "ancak", "artık", "aslında", "ayrıca", "bana", "bazı", "belki",
    "ben", "benden", "beni", "benim", "beri", "bile", "bir", "biraz", "birçok",
    "biri", "birkaç", "birşey", "biz", "bizden", "bize", "bizi", "bizim", "böyle",
    "böylece", "bu", "buna", "bundan", "bunlar", "bunları", "bunların", "bunu",
    "bunun", "burada", "bütün", "çoğu", "çok", "çünkü", "da", "daha", "dahi", "de",
    "defa", "değil", "diğer", "diye", "dolayı", "dolayısıyla", "elbette", "en",
    "fakat", "falan", "felan", "filan", "gene", "gereği", "gibi", "göre", "halen",
    "hangi", "hani", "hatta", "hem", "henüz", "hep", "hepsi", "her", "herhangi",
    "herkes", "hiç", "hiçbir", "için", "içinde", "ile", "ilgili", "ise", "işte",
    "itibaren", "itibariyle", "kadar", "karşın", "kendi", "kendilerine", "kendini",
    "kendisi", "kez", "ki", "kim", "kime", "kimi", "kimse", "lakin", "madem", "mi",
    "mı", "mu", "mü", "nasıl", "ne", "neden", "nedeniyle", "nerde", "nerede",
    "nereye", "nesi", "neyse", "niçin", "niye", "o", "olan", "olarak", "oldu",
    "olduğu", "olduğunu", "olduklarını", "olmadı", "olmadığı", "olmak", "olması",
    "olmayan", "olmaz", "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "ona",
    "ondan", "onlar", "onlara", "onlardan", "onları", "onların", "onu", "onun",
    "orada", "oysa", "oysaki", "öbür", "ön", "önce", "ötürü", "öyle", "pek",
    "rağmen", "sadece", "sanki", "sen", "senden", "seni", "senin", "siz", "sizden",
    "size", "sizi", "sizin", "sonra", "şayet", "şey", "şeyden", "şeyi", "şeyler",
    "şimdi", "şöyle", "şu", "şuna", "şunları", "şunu", "tarafından", "trilyon",
    "tüm", "üzerine", "üzere", "var", "vardı", "ve", "veya", "veyahut", "ya",
    "ya da", "yani", "yapacak", "yapılan", "yapılması", "yapıyor", "yapmak",
    "yaptı", "yaptığı", "yaptığını", "yaptıkları", "yerine", "yine", "yoksa",
    "zaten", "zira", "target_primary", "interlocutor_context",
}

ENGLISH_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "that", "this", "these", "those", "it", "its", "as", "by", "from", "about",
    "can", "could", "will", "would", "should", "not", "their", "they", "them", "we",
    "us", "our", "you", "your", "i", "me", "my", "he", "him", "his", "she", "her",
    "into", "than", "then", "there", "here", "when", "where", "which", "who", "whom",
    "what", "while", "also", "more", "most", "some", "such", "only", "other", "over",
    "target_primary", "interlocutor_context",
}

BILINGUAL_STOPWORDS: Set[str] = TURKISH_STOPWORDS | ENGLISH_STOPWORDS


def tr_lower(text: str) -> str:
    """
    Bilingual-safe (Turkish + English) lowercase normalization.
    - Prevents Python's `'İ'.lower() -> 'i\\u0307'` bug that splits capitalized Turkish words
      (e.g., 'İçin' -> 'için', 'İstanbul' -> 'istanbul', never 'i' + 'çin').
    - Converts uppercase 'I' to Turkish dotless 'ı' inside words containing Turkish characters
      or known Turkish stems (e.g., 'Işık' -> 'ışık', 'Kanıtlar Işığında' -> 'kanıtlar ışığında',
      'ŞÜPHESİZ' -> 'şüphesiz'), while preserving standalone English pronoun 'I' -> 'i' and
      English vocabulary ('Inquiry' -> 'inquiry').
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", text)
    # Always map Turkish dotted capital İ -> i and strip stray combining dot above (U+0307)
    norm = norm.replace("İ", "i").replace("\u0307", "")

    def _lower_token(match: re.Match[str]) -> str:
        word = match.group(0)
        if "I" in word:
            upper_ascii = word.translate(_TR_SLUG_MAP).upper()
            if _TR_CHARS_RE.search(word) or upper_ascii.startswith(_TR_DOTLESS_I_PREFIXES):
                word = word.replace("I", "ı")
        return word.lower()

    return re.sub(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", _lower_token, norm).replace("\u0307", "")


def tokenize_words(text: str, min_len: int = 1) -> List[str]:
    """Tokenize bilingual Turkish/English text cleanly after Turkish-aware lowercasing."""
    if not text:
        return []
    lowered = tr_lower(text)
    tokens = [w.strip("'") for w in _WORD_TOKEN_RE.findall(lowered)]
    return [t for t in tokens if len(t) >= min_len]


def compute_mattr(words: List[str], window_size: int = 50) -> float:
    """
    Compute Moving-Average Type-Token Ratio (MATTR) to measure lexical diversity
    independently of total document length.
    """
    if not words:
        return 0.0
    if len(words) <= window_size:
        return round(len(set(words)) / len(words), 3)

    window_unique = len(set(words[:window_size]))
    counts: Dict[str, int] = {}
    for w in words[:window_size]:
        counts[w] = counts.get(w, 0) + 1

    ratios = [window_unique / window_size]
    for idx in range(window_size, len(words)):
        outgoing = words[idx - window_size]
        incoming = words[idx]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[incoming] = counts.get(incoming, 0) + 1
        ratios.append(len(counts) / window_size)

    return round(sum(ratios) / len(ratios), 3)


def extract_target_primary_text(raw_content: str) -> str:
    """
    Strip YAML frontmatter and isolate strictly [TARGET_PRIMARY] segments,
    filtering out [INTERLOCUTOR_CONTEXT] lines and scrape failure placeholders.
    """
    if not raw_content:
        return ""
    text = raw_content.strip()
    # Strip YAML frontmatter if present
    if text.startswith("---"):
        end_idx = text.find("\n---", 3)
        if end_idx != -1:
            text = text[end_idx + 4 :].strip()

    if "# Scrape Failure" in text or "[Content extraction failed]" in text:
        return ""

    segments = parse_diarized_segments(text)
    target_parts = [seg["text"] for seg in segments if seg.get("role") == "target" and seg.get("text")]
    return "\n\n".join(target_parts).strip()


def sanitize_slug(name: str) -> str:
    """Convert any target name (including Turkish 'ı'/'İ'/'ş'/'ğ') into a filesystem, ChromaDB, and URL-safe slug."""
    if not name or not name.strip():
        return "unnamed_target"
    # Transliterate Turkish characters explicitly before NFKD so dotless 'ı' becomes 'i' (not deleted)
    transliterated = name.translate(_TR_SLUG_MAP)
    text = unicodedata.normalize("NFKD", transliterated)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s_]+", "_", text).strip("_")
    if slug:
        return slug
    # Deterministic fallback for non-Latin scripts so distinct targets never collide
    digest = hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:8]
    return f"target_{digest}"


def clean_text(text: str) -> str:
    """Normalize whitespace, preserve Turkish NFC characters, remove control chars, and clean artifacts."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_diarized_segments(transcript: str) -> List[Dict[str, str]]:
    """
    Parse a transcript containing speaker attribution tokens:
    e.g., [TARGET_PRIMARY]: "..." or [INTERLOCUTOR_CONTEXT]: "..."
    Returns a list of dictionaries with 'speaker', 'role', and 'text'.
    """
    pattern = re.compile(
        r"\[(TARGET_PRIMARY|INTERLOCUTOR_CONTEXT|[\w\s_çğıöşüÇĞİÖŞÜ-]+)\]:\s*([^\n]+(?:\n(?!\[[\w\s_çğıöşüÇĞİÖŞÜ-]+\]:)[^\n]+)*)",
        re.IGNORECASE,
    )
    segments = []
    matches = list(pattern.finditer(transcript))

    if matches:
        for m in matches:
            raw_speaker = m.group(1).strip()
            content = m.group(2).strip()
            spk_low = tr_lower(raw_speaker)
            role = "target" if "target" in spk_low or "primary" in spk_low else "interlocutor"
            segments.append({
                "speaker": raw_speaker,
                "role": role,
                "text": content,
            })
    else:
        cleaned = clean_text(transcript)
        if cleaned:
            segments.append({
                "speaker": "TARGET_PRIMARY",
                "role": "target",
                "text": cleaned,
            })
    return segments


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    metadata: Dict[str, str] | None = None,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks, preserving sentence boundaries where possible.
    Returns list of chunk objects with content and inherited metadata.
    """
    meta = metadata or {}
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > chunk_size and current_chunk:
            chunk_str = " ".join(current_chunk)
            chunks.append({
                "text": chunk_str,
                **meta,
            })
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) < chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += s_len

    if current_chunk:
        chunk_str = " ".join(current_chunk)
        chunks.append({
            "text": chunk_str,
            **meta,
        })
    return chunks


def redact_pii(text: str) -> str:
    """
    Scrub personally identifiable information (PII) from private chat logs:
    - Email addresses
    - IBANs and 16-digit credit card numbers
    - National ID numbers (TCKN 11-digit — checked BEFORE phone numbers)
    - Phone numbers (Turkish & International)
    - Private WhatsApp/Telegram invite links
    """
    if not text:
        return ""

    sanitized = text

    # 1. Email Addresses
    sanitized = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
        sanitized,
    )

    # 2. IBAN & Credit Cards
    sanitized = re.sub(
        r"\bTR\d{2}(?:\s*\d{4}){5}\s*\d{2}\b",
        "[REDACTED_IBAN]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "[REDACTED_CARD]",
        sanitized,
    )

    # 3. WhatsApp & Telegram Invite Links
    sanitized = re.sub(
        r"https?://chat\.whatsapp\.com/[A-Za-z0-9]+",
        "[REDACTED_CHAT_LINK]",
        sanitized,
    )
    sanitized = re.sub(
        r"https?://t\.me/[A-Za-z0-9_+]+",
        "[REDACTED_TELEGRAM_LINK]",
        sanitized,
    )

    # 4. National ID numbers (11 digits starting 1-9, ending in even digit per TCKN rule, or preceded by TC/TCKN)
    sanitized = re.sub(
        r"(?:\b(?:tc|tckn|kimlik)\s*:?\s*[1-9]\d{10}\b|\b[1-9]\d{9}[02468]\b)",
        "[REDACTED_ID]",
        sanitized,
        flags=re.IGNORECASE,
    )

    # 5. Phone Numbers (Turkish +90 5xx / 05xx / 5xx and formatted international numbers)
    sanitized = re.sub(
        r"(?:\+90[-.\s]?|\b0|\b)\(?5\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b|\+\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b",
        "[REDACTED_PHONE]",
        sanitized,
    )

    return sanitized


def extract_quotes(text: str, query: str, max_quotes: int = 3) -> List[str]:
    """Find exact or highly relevant sentences in text matching query keywords (Turkish-aware)."""
    if not text or not query:
        return []
    keywords = [k for k in tokenize_words(query, min_len=3) if k not in BILINGUAL_STOPWORDS]
    if not keywords:
        keywords = tokenize_words(query, min_len=2)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored: List[Tuple[int, str]] = []

    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 20:
            continue
        s_lower = tr_lower(s_clean)
        score = sum(1 for kw in keywords if kw in s_lower)
        if score > 0:
            scored.append((score, s_clean))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:max_quotes]]
