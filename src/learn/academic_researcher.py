"""
Academic researcher forwarder for backward compatibility.
Points directly to AutonomousAcademicResearcher in autonomous_researcher.py.
"""

from src.learn.autonomous_researcher import (
    AutonomousAcademicResearcher,
    AcademicResearcher,
    fisher_z,
    inv_fisher_z,
)

__all__ = [
    "AutonomousAcademicResearcher",
    "AcademicResearcher",
    "fisher_z",
    "inv_fisher_z",
]
