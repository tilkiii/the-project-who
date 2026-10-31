"""
Strict Entity Resolution & Zero-Substitution Disambiguation Engine for who??.
Enforces the mandatory rule (AGENTS.md Rule #1):
- Exact token matching for target identities.
- Rejects candidate expansion (e.g. adding middle/prefixed names like 'Muhammet' to 'Mucteba Ozdemir').
- Prevents False Entity Resolution / Identity Conflation across OSINT, academic, and web harvesting.
- Zero third-party data persistence: rejected records are discarded before touching profile storage.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

DISAMBIGUATION_NOTICE_TR = (
    "İnternette benzer isimde başka kayıtlar çıktı ancak senin verdiğin kimlikle uyuşmadığı için yok sayıldı."
)


def normalize_name(name: str) -> str:
    """Normalize text by Turkish-safe lowercasing, stripping diacritics/accents, and trimming whitespace."""
    if not name:
        return ""
    # Explicitly map Turkish dotted/dotless I variants before lower() + NFKD so 'İ' never leaves '\u0307'
    pre_cleaned = (
        name.strip()
        .replace("İ", "i")
        .replace("I", "i")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("Ğ", "g")
        .replace("ü", "u")
        .replace("Ü", "u")
        .replace("ş", "s")
        .replace("Ş", "s")
        .replace("ö", "o")
        .replace("Ö", "o")
        .replace("ç", "c")
        .replace("Ç", "c")
    )
    normalized = unicodedata.normalize("NFKD", pre_cleaned.lower())
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w\s]", "", without_accents)
    return " ".join(cleaned.split())


def is_strict_entity_match(target_name: str, candidate_name: str) -> Tuple[bool, str]:
    """
    Enforces the Zero-Substitution Protocol:
    1. Returns (True, "EXACT_MATCH") if tokens match identically.
    2. Rejects if candidate has unexpected extra given names (e.g. 'Muhammet' added to 'Mucteba Ozdemir').
    3. Rejects if names differ or candidate is a partial mismatch.
    """
    norm_target = normalize_name(target_name)
    norm_candidate = normalize_name(candidate_name)

    if not norm_target or not norm_candidate:
        return False, "EMPTY_NAME"

    if norm_target == norm_candidate:
        return True, "EXACT_MATCH"

    target_tokens = norm_target.split()
    candidate_tokens = norm_candidate.split()

    # If candidate has more tokens than target, extra names were injected
    if len(candidate_tokens) > len(target_tokens):
        extra_tokens = set(candidate_tokens) - set(target_tokens)
        return False, f"UNAUTHORIZED_NAME_EXPANSION: candidate contains extra names {extra_tokens}"

    if len(target_tokens) > len(candidate_tokens):
        missing_tokens = set(target_tokens) - set(candidate_tokens)
        return False, f"MISSING_TOKENS: candidate lacks required target tokens {missing_tokens}"

    # Same number of tokens but different
    if target_tokens != candidate_tokens:
        return False, f"TOKEN_MISMATCH: target '{norm_target}' != candidate '{norm_candidate}'"

    return True, "EXACT_MATCH"


# Honorifics and common sentence-initial words that are not personal given-name expansions
_NON_NAME_PREFIXES = {
    "dr", "prof", "doc", "sayin", "mr", "mrs", "ms", "sir", "av", "muh", "uzm",
    "ars", "gor", "by", "author", "yazar", "editor", "speaker", "konusmaci",
    "bugun", "dun", "yarin", "ancak", "fakat", "lakin", "ayrica", "ornegin",
    "ozellikle", "burada", "simdi", "sonra", "once", "the", "when", "while",
    "where", "according", "here", "this", "that", "with", "from", "about",
    "public", "dossier", "profile", "target", "live", "interview", "article",
}


def detect_unauthorized_name_expansion(
    target_name: str,
    text: str,
    declared_author: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Inspect an article/document's declared author and body text for Zero-Substitution violations.
    Returns (is_contaminated, reason):
    - True if declared_author fails strict entity match against target_name.
    - True if an explicit Author/Yazar byline fails strict entity match.
    - True if contiguous capitalized proper-noun tokens expand target_name (e.g. 'Muhammet Mücteba Özdemir'
      for target 'Mucteba Özdemir') with no standalone exact-identity occurrence.
    """
    norm_target = normalize_name(target_name)
    if not norm_target:
        return False, "NO_TARGET_CONSTRAINT"

    # 1. Explicit author metadata check
    if declared_author and declared_author.strip():
        matched, reason = is_strict_entity_match(target_name, declared_author.strip())
        if not matched:
            return True, f"AUTHOR_MISMATCH ({reason})"

    if not text:
        return False, "CLEAN"

    # 2. Check explicit byline patterns inside markdown/article text
    byline_pattern = re.compile(
        r"(?:^|\n)\s*(?:\*\*?)?(?:Author|Yazar|Hazırlayan)(?:\*\*?)?\s*:\s*([^\n\r|]+)",
        re.IGNORECASE,
    )
    for m in byline_pattern.finditer(text):
        byline_name = m.group(1).strip().strip("*_")
        if byline_name:
            matched, reason = is_strict_entity_match(target_name, byline_name)
            if not matched:
                return True, f"BYLINE_MISMATCH ({reason})"

    # 3. Contiguous whitespace-separated capitalized word sequence check (never crosses punctuation)
    target_tokens = norm_target.split()
    t_len = len(target_tokens)
    if t_len >= 2:
        exact_standalone_hits = 0
        expanded_hits = 0
        # Split into punctuation-bounded clauses so we only inspect contiguous word runs
        clauses = re.split(r"[.!?,;:\n\r()\[\]\"'|—–\-]+", text)
        for clause in clauses:
            words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", clause)
            if len(words) < t_len:
                continue
            norm_words = [normalize_name(w) for w in words]
            for idx in range(len(norm_words) - t_len + 1):
                if norm_words[idx : idx + t_len] == target_tokens:
                    prev_raw = words[idx - 1] if idx > 0 else ""
                    prev_norm = norm_words[idx - 1] if idx > 0 else ""
                    is_cap_prefix = bool(prev_raw) and prev_raw[0].isupper() and len(prev_norm) >= 3
                    if is_cap_prefix and prev_norm not in _NON_NAME_PREFIXES:
                        expanded_hits += 1
                    else:
                        exact_standalone_hits += 1

        if expanded_hits > 0 and exact_standalone_hits == 0:
            return True, "TEXT_PREFIXED_NAME_EXPANSION"

    return False, "CLEAN"


def filter_verified_entities(
    target_name: str, candidates: List[Dict[str, Any]], name_key: str = "name"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split a list of candidate records into:
    - verified: strictly matched entities only
    - rejected: ambiguous namesakes or expanded names that MUST NOT be ingested into the vault
    """
    verified: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for item in candidates:
        cand_name = item.get(name_key, "")
        is_match, reason = is_strict_entity_match(target_name, cand_name)
        record = dict(item)
        record["validation_reason"] = reason

        if is_match:
            verified.append(record)
        else:
            rejected.append(record)

    return verified, rejected
