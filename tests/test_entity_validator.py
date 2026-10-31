"""Unit tests for Strict Entity Resolution & Zero-Substitution Disambiguation."""

import pytest
from src.utils.entity_validator import (
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


def test_academic_harvester_zero_substitution_semantic_scholar(tmp_path, monkeypatch):
    import httpx
    from src.harvest.academic_harvester import AcademicHarvester

    class DummyResp:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {
                        "name": "Muhammet Mücteba Özdemir",
                        "papers": [
                            {"title": "Contaminated Namesake Paper", "year": 2024, "abstract": "Wrong person", "citationCount": 12}
                        ],
                    },
                    {
                        "name": "Mücteba Özdemir",
                        "papers": [
                            {"title": "Verified Primary Target Paper", "year": 2025, "abstract": "Exact match", "citationCount": 5}
                        ],
                    },
                ]
            }

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, *a, **kw: DummyResp())

    harvester = AcademicHarvester(output_dir=tmp_path)
    pubs = harvester.search_author_publications("Mucteba Özdemir")

    assert len(pubs) == 1
    assert pubs[0]["title"] == "Verified Primary Target Paper"
    assert "Contaminated Namesake Paper" not in [p["title"] for p in pubs]
    saved_files = list(tmp_path.glob("academic_*.md"))
    assert len(saved_files) == 1


def test_academic_harvester_zero_substitution_arxiv(tmp_path, monkeypatch):
    import httpx
    from src.harvest.academic_harvester import AcademicHarvester

    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Namesake Arxiv Preprint</title>
        <summary>Should be blocked by Disambiguation Gate.</summary>
        <published>2024-05-01T00:00:00Z</published>
        <author><name>Muhammet Mücteba Özdemir</name></author>
      </entry>
      <entry>
        <title>Verified Target Arxiv Preprint</title>
        <summary>Authored by exact target identity.</summary>
        <published>2025-09-15T00:00:00Z</published>
        <author><name>Mücteba Özdemir</name></author>
        <author><name>Remzi Sanver</name></author>
      </entry>
    </feed>"""

    class DummyResp:
        def __init__(self, url: str):
            self.status_code = 200
            self._url = url
            self.text = arxiv_xml

        def json(self):
            return {"data": []}

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, *a, **kw: DummyResp(url))

    harvester = AcademicHarvester(output_dir=tmp_path)
    pubs = harvester.search_author_publications("Mucteba Özdemir")

    assert len(pubs) == 1
    assert pubs[0]["title"] == "Verified Target Arxiv Preprint"
    assert pubs[0]["authors"] == ["Mücteba Özdemir", "Remzi Sanver"]
    assert harvester.rejected_namesakes_count == 1


def test_web_harvester_disambiguation_gate_blocks_namesake_expansion(tmp_path, monkeypatch):
    import httpx
    from src.harvest.web_harvester import WebHarvester
    from src.utils.entity_validator import detect_unauthorized_name_expansion

    # Sentence-initial Turkish word ('Bugün') must NOT trigger false positive
    is_bad, _ = detect_unauthorized_name_expansion(
        "Mucteba Özdemir",
        "Bugün Mücteba Özdemir yeni bilişsel mimari makalesini yayınladı.",
    )
    assert is_bad is False

    # Prefixed namesake ('Muhammet Mücteba Özdemir') MUST be blocked
    is_bad2, reason2 = detect_unauthorized_name_expansion(
        "Mucteba Özdemir",
        "Kararı imzalayan Muhammet Mücteba Özdemir görevine başladı.",
    )
    assert is_bad2 is True
    assert "EXPANSION" in reason2

    class DummyAsyncResp:
        def __init__(self, html: str):
            self.status_code = 200
            self.text = html

    async def fake_async_get(self, url, **kwargs):
        if "namesake" in url:
            return DummyAsyncResp(
                "<html><head><title>Muhammet Mücteba Özdemir Atama Kararı</title></head>"
                "<body><p>Muhammet Mücteba Özdemir yeni görevine atandı.</p></body></html>"
            )
        return DummyAsyncResp(
            "<html><head><title>Mücteba Özdemir Makalesi</title></head>"
            "<body><p>Mücteba Özdemir olasılıksal dijital ikiz mimarisini açıkladı.</p></body></html>"
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_async_get)

    wh = WebHarvester(output_dir=tmp_path, target_name="Mucteba Özdemir")
    results = wh.harvest_urls(["https://example.org/namesake", "https://example.org/verified"])

    assert len(results) == 1
    assert wh.rejected_namesakes_count == 1
    assert "olasılıksal dijital ikiz" in results[0]["content"]
    # Zero third-party namesake files written to disk
    saved_files = list(tmp_path.glob("web_*.md"))
    assert len(saved_files) == 1
    assert "Muhammet" not in saved_files[0].read_text(encoding="utf-8")
