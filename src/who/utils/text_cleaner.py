"""
Text cleaning, normalization, speaker token parsing, and slug sanitization.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Tuple


_TURKISH_SLUG_TRANS = str.maketrans({
    "ı": "i",
    "İ": "I",
    "ğ": "g",
    "Ğ": "G",
    "ş": "s",
    "Ş": "S",
    "ç": "c",
    "Ç": "C",
    "ö": "o",
    "Ö": "O",
    "ü": "u",
    "Ü": "U",
})


def sanitize_slug(name: str) -> str:
    """Convert any target name or string into a filesystem and URL-safe slug."""
    if not name:
        return "unnamed_target"
    text = name.translate(_TURKISH_SLUG_TRANS)
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alphanumeric with underscores
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", text)
    return slug or "unnamed_target"


def clean_text(text: str) -> str:
    """Normalize whitespace, remove control chars, and clean markdown/html artifacts."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newlines/tabs
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Condense horizontal whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Condense excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_diarized_segments(transcript: str) -> List[Dict[str, str]]:
    """
    Parse a transcript containing speaker attribution tokens:
    e.g., [TARGET_PRIMARY]: "..." or [INTERLOCUTOR_CONTEXT]: "..."
    Returns a list of dictionaries with 'speaker', 'role', and 'text'.
    """
    pattern = re.compile(r"\[(TARGET_PRIMARY|INTERLOCUTOR_CONTEXT|[\w\s_-]+)\]:\s*([^\n]+(?:\n(?!\[[\w\s_-]+\]:)[^\n]+)*)", re.IGNORECASE)
    segments = []
    matches = list(pattern.finditer(transcript))

    if matches:
        for m in matches:
            raw_speaker = m.group(1).strip()
            content = m.group(2).strip()
            role = "target" if "target" in raw_speaker.lower() or "primary" in raw_speaker.lower() else "interlocutor"
            segments.append({
                "speaker": raw_speaker,
                "role": role,
                "text": content,
            })
    else:
        # If no speaker markers exist, treat entire text as target document
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
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > chunk_size and current_chunk:
            chunk_str = " ".join(current_chunk)
            chunks.append({
                "text": chunk_str,
                **meta,
            })
            # Overlap handling
            overlap_sentences = []
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
    - Phone numbers (Turkish & International)
    - Email addresses
    - IBANs and 16-digit credit card numbers
    - National ID numbers (TCKN 11-digit)
    - Private WhatsApp/Telegram invite links
    """
    if not text:
        return ""

    sanitized = text

    # 1. Email Addresses
    sanitized = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
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

    # 4. Phone Numbers (International and Turkish formats)
    # +90 5xx xxx xx xx, 05xx xxx xx xx, 5xx xxx xx xx
    sanitized = re.sub(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b",
        "[REDACTED_PHONE]",
        sanitized,
    )

    # 5. National ID numbers (11 digits preceded by keyword or standalone)
    sanitized = re.sub(
        r"\b(?:tc|tckn|kimlik)?\s*([1-9]\d{10})\b",
        "[REDACTED_ID]",
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized


def is_boilerplate_or_junk(text: str) -> bool:
    """Detect whether a text snippet is web footer, copyright boilerplate, cookie policy, or UI junk."""
    if not text:
        return True
    t_clean = text.strip().lower()
    if len(t_clean) < 25:
        return True

    junk_patterns = [
        r"all rights reserved",
        r"tüm hakları saklıdır",
        r"aboneliğiniz başarıyla tamamlandı",
        r"haber bültenine abone",
        r"çerez politikası",
        r"cookie policy",
        r"gizlilik politikası",
        r"privacy policy",
        r"kullanım koşulları",
        r"terms of use",
        r"terms of service",
        r"copyright\s*©",
        r"©\s*\d{4}",
        r"site haritası",
        r"bizi takip edin",
        r"follow us on",
        r"click here to subscribe",
        r"tebrikler\s*!",
    ]
    for pat in junk_patterns:
        if re.search(pat, t_clean):
            return True

    letters = sum(1 for c in t_clean if c.isalpha())
    if letters / max(len(t_clean), 1) < 0.4:
        return True

    return False


def extract_quotes(text: str, query: str, max_quotes: int = 3) -> List[str]:
    """Find exact or highly relevant sentences in text matching query keywords."""
    if not text or not query:
        return []
    keywords = [k.lower() for k in query.split() if len(k) > 3]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored: List[Tuple[int, str]] = []

    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 20:
            continue
        s_lower = s_clean.lower()
        score = sum(1 for kw in keywords if kw in s_lower)
        if score > 0:
            scored.append((score, s_clean))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:max_quotes]]


def search_raw_corpus_paragraphs(
    raw_data_dir: Path,
    query: str,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Fallback lexical paragraph search across raw_data/*.md|*.txt when no vector index is present."""
    if not raw_data_dir or not raw_data_dir.exists() or not query.strip():
        return []

    q_tokens = set(re.findall(r"\b\w{4,}\b", query.lower()))
    if not q_tokens:
        return []

    files = sorted(list(raw_data_dir.glob("*.md")) + list(raw_data_dir.glob("*.txt")))
    scored: List[Tuple[int, Dict[str, Any]]] = []
    seen_texts: set[str] = set()

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for raw_block in re.split(r"\n+", content):
            block = raw_block.strip()
            if not block or block.startswith("#") or block.startswith("---"):
                continue
            block = re.sub(r"^(?:>|[-*]|\d+\.)\s*", "", block).strip()
            if len(block) < 25 or is_boilerplate_or_junk(block):
                continue
            if block in seen_texts:
                continue
            seen_texts.add(block)
            block_lower = block.lower()
            hits = sum(1 for tok in q_tokens if tok in block_lower)
            if hits > 0:
                score = round(min(0.85, 0.45 + (0.10 * hits)), 3)
                scored.append((
                    hits,
                    {
                        "text": block,
                        "metadata": {"source_id": fpath.name},
                        "similarity_score": score,
                    },
                ))

    scored.sort(key=lambda item: (item[0], item[1]["similarity_score"]), reverse=True)
    return [item[1] for item in scored[:top_k]]
