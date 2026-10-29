"""
Text cleaning, normalization, speaker token parsing, and slug sanitization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple


def sanitize_slug(name: str) -> str:
    """Convert any target name or string into a filesystem and URL-safe slug."""
    if not name:
        return "unnamed_target"
    # Normalize unicode
    text = unicodedata.normalize("NFKD", name)
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
