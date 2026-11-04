"""
Global pytest fixtures and test isolation configuration for who??.
"""

from pathlib import Path
import pytest
from who.config import settings


@pytest.fixture(autouse=True)
def isolate_test_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Ensure all unit and integration tests run against an isolated temporary
    profiles directory, preventing test target pollution in the main workspace.
    """
    test_profiles = tmp_path / "test_profiles"
    test_profiles.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "profiles_dir", test_profiles)
    monkeypatch.setenv("WHO_PROVIDER", "offline")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    # Mock external network queries for fast, deterministic offline testing
    try:
        from who.harvest.academic_harvester import AcademicHarvester
        monkeypatch.setattr(AcademicHarvester, "search_author_publications", lambda self, author, max_results=5: [])
    except Exception:
        pass

    try:
        from who.learn.autonomous_researcher import AutonomousAcademicResearcher
        monkeypatch.setattr(AutonomousAcademicResearcher, "query_arxiv", lambda self, q, max_results=1: [])
        monkeypatch.setattr(AutonomousAcademicResearcher, "query_pubmed", lambda self, q, max_results=1: [])
        monkeypatch.setattr(AutonomousAcademicResearcher, "query_semantic_scholar", lambda self, q, max_results=1: [])
        monkeypatch.setattr(AutonomousAcademicResearcher, "query_crossref", lambda self, q, max_results=1: [])
    except Exception:
        pass

    yield
