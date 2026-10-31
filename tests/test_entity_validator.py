"""Unit tests for Strict Entity Resolution & Zero-Substitution Disambiguation."""

import pytest
from who.utils.entity_validator import (
    normalize_name,
    is_strict_entity_match,
    filter_verified_entities,
)


def test_normalize_name():
    assert normalize_name("Mücteba Özdemir") == "mucteba ozdemir"
    assert normalize_name("Mucteba Özdemir") == "mucteba ozdemir"
    assert normalize_name("  Remzi   Sanver  ") == "remzi sanver"


def test_strict_match_accepts_exact_names():
    is_match, reason = is_strict_entity_match("Mucteba Özdemir", "Mücteba Özdemir")
    assert is_match is True
    assert reason == "EXACT_MATCH"


def test_strict_match_rejects_unauthorized_expansion():
    # E.g. injecting 'Muhammet' into 'Mucteba Ozdemir' must be strictly rejected!
    is_match, reason = is_strict_entity_match("Mucteba Özdemir", "Muhammet Mücteba Özdemir")
    assert is_match is False
    assert "UNAUTHORIZED_NAME_EXPANSION" in reason


def test_strict_match_rejects_partial_or_different_names():
    is_match, reason = is_strict_entity_match("Remzi Sanver", "Remzi")
    assert is_match is False
    assert "MISSING_TOKENS" in reason

    is_match2, reason2 = is_strict_entity_match("Remzi Sanver", "Ahmet Sanver")
    assert is_match2 is False
    assert "TOKEN_MISMATCH" in reason2


def test_filter_verified_entities():
    candidates = [
        {"name": "Mücteba Özdemir", "url": "https://instagram.com/muctebell"},
        {"name": "Muhammet Mücteba Özdemir", "url": "https://adalet.gov.tr/sample"},
        {"name": "Mücteba Can", "url": "https://sample.com/other"},
    ]
    verified, rejected = filter_verified_entities("Mucteba Özdemir", candidates)

    assert len(verified) == 1
    assert verified[0]["name"] == "Mücteba Özdemir"
    assert len(rejected) == 2
    assert "Muhammet Mücteba Özdemir" in [r["name"] for r in rejected]
