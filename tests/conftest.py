"""
Global pytest fixtures and test isolation configuration for who??.
"""

from pathlib import Path
import pytest
from src.config import settings


@pytest.fixture(autouse=True)
def isolate_test_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Ensure all unit and integration tests run against an isolated temporary
    profiles directory, preventing test target pollution in the main workspace.
    """
    test_profiles = tmp_path / "test_profiles"
    test_profiles.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "profiles_dir", test_profiles)
    yield
