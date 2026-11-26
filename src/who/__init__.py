"""
who - Evidence-grounded cognitive digital twin and behavioral modeling toolkit for AI agents.
"""

__version__ = "1.0.1"
__author__ = "Mucteba Özdemir"

from who.agent.sdk import (
    AcademicResearcher,
    CognitiveTwin,
    get_target,
    list_targets,
)
from who.mcp_server import get_tool_schemas

__all__ = [
    "CognitiveTwin",
    "AcademicResearcher",
    "list_targets",
    "get_target",
    "get_tool_schemas",
    "__version__",
]
