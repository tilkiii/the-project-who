"""Unit tests for Strict Entity Resolution & Zero-Substitution Disambiguation."""

import pytest
from who.utils.entity_validator import (
    normalize_name,
    is_strict_entity_match,
    filter_verified_entities,
)


def test_normalize_name():
    assert normalize_name("Özgür Çelik") == "ozgur celik"
    assert normalize_name("Ozgur Celik") == "ozgur celik"
    assert normalize_name("  Target   Alpha  ") == "target alpha"


def test_strict_match_accepts_exact_names():
    is_match, reason = is_strict_entity_match("Ozgur Celik", "Özgür Çelik")
    assert is_match is True
    assert reason == "EXACT_MATCH"


def test_strict_match_rejects_unauthorized_expansion():
    # Rejects injecting an unexpected extra given name into the target identity
    is_match, reason = is_strict_entity_match("Ozgur Celik", "Demir Özgür Çelik")
    assert is_match is False
    assert "UNAUTHORIZED_NAME_EXPANSION" in reason


def test_strict_match_rejects_partial_or_different_names():
    is_match, reason = is_strict_entity_match("Target Alpha", "Target")
    assert is_match is False
    assert "MISSING_TOKENS" in reason

    is_match2, reason2 = is_strict_entity_match("Target Alpha", "Beta Alpha")
    assert is_match2 is False
    assert "TOKEN_MISMATCH" in reason2


def test_filter_verified_entities():
    candidates = [
        {"name": "Özgür Çelik", "url": "https://example.com/ozgurcelik"},
        {"name": "Demir Özgür Çelik", "url": "https://example.com/sample"},
        {"name": "Özgür Can", "url": "https://example.com/other"},
    ]
    verified, rejected = filter_verified_entities("Ozgur Celik", candidates)

    assert len(verified) == 1
    assert verified[0]["name"] == "Özgür Çelik"
    assert len(rejected) == 2
    assert "Demir Özgür Çelik" in [r["name"] for r in rejected]
