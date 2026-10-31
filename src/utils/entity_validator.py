"""
Strict Entity Resolution & Zero-Substitution Disambiguation Engine for who??.
Enforces the mandatory rule:
- Exact token matching for target identities.
- Rejects candidate expansion (e.g. adding middle names or different compounds).
- Prevents False Entity Resolution / Identity Conflation across OSINT and web harvesting.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def normalize_name(name: str) -> str:
    """Normalize text by lowercasing, stripping diacritics/accents, and trimming whitespace."""
    if not name:
        return ""
    # Normalize unicode (decompose accented characters)
    normalized = unicodedata.normalize("NFKD", name.strip().lower())
    # Remove combining diacritical marks
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    # Replace common Turkish letter variations
    without_accents = (
        without_accents.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )
    # Clean non-alphanumeric except whitespace
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
