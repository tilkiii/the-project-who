"""
Rich-based logger and terminal UI formatting for the who?? framework.
Enforces epistemic formatting and probabilistic evidence presentation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def get_logger(name: str = "who") -> logging.Logger:
    """Return a configured logger with RichHandler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def print_banner() -> None:
    """Print the ASCII banner and version for who??."""
    title = Text()
    title.append(" █     █░ ██░ ██  ▒█████   ██▀███      ▒███████▒▒███████▒\n", style="bold cyan")
    title.append("▓█░ █ ░█░▓██░ ██▒▒██▒  ██▒▓██ ▒ ██▒    ▒ ▒ ▒ ▄▀░▒ ▒ ▒ ▄▀░\n", style="bold cyan")
    title.append("▒█░ █ ░█ ▒██▀▀██░▒██░  ██▒▓██ ░▄█ ▒    ░ ▒ ▄▀▒░ ░ ▒ ▄▀▒░ \n", style="bold blue")
    title.append("░█░ █ ░█ ░▓█ ░██ ▒██   ██░▒██▀▀█▄        ▄▀▒   ░  ▄▀▒   ░\n", style="bold magenta")
    title.append("░░██▒██▓ ░▓█▒░██▓░ ████▓▒░░██▓ ▒██▒    ▒███████▒▒███████▒\n", style="bold magenta")
    title.append(" █ ▐ ▒   ▒ ░░▒░▒░ ▒░▒░▒░ ░ ▒▓ ░▒▓░    ░▒▒ ▓░▒░▒░▒▒ ▓░▒░▒\n", style="dim")
    console.print(title)
    console.print(
        Panel(
            "[bold white]who - Cognitive Digital Twin & Behavioral Modeling Toolkit[/bold white]\n"
            "[dim cyan]Semantic Vector Memory • Knowledge Graph • Agent Tools[/dim cyan]",
            border_style="cyan",
            expand=False,
        )
    )


def print_probabilistic_claim(
    claim: str,
    confidence_score: float,
    sources: List[str],
    quotes: List[str],
    graph_path: Optional[List[str]] = None,
) -> None:
    """
    Format and display a behavioral finding with strict epistemological rigor:
    - Probabilistic phrasing only
    - Explicit confidence percentage
    - Direct primary quote citations
    - Optional graph trace path
    """
    score_pct = int(confidence_score * 100)
    color = "green" if confidence_score >= 0.8 else ("yellow" if confidence_score >= 0.6 else "red")

    table = Table(title="[bold cyan]Probabilistic Behavioral Finding[/bold cyan]", show_header=False)
    table.add_column("Field", style="bold white", width=18)
    table.add_column("Details", style="white")

    table.add_row("Observed Pattern", f"The data suggests a plausible alignment with: [italic]{claim}[/italic]")
    table.add_row(
        "Confidence Score",
        f"[{color}]{score_pct}%[/{color}] (Derived from {len(quotes)} primary quotes and {len(sources)} verified sources)",
    )

    if sources:
        table.add_row("Evidence Sources", "\n".join(f"• Based on observed evidence from {s}" for s in sources))

    if quotes:
        quotes_formatted = "\n".join(f'  [dim]"{q}"[/dim]' for q in quotes[:5])
        table.add_row("Primary Quotes", quotes_formatted)

    if graph_path:
        table.add_row("Graph Trace Path", " ➔ ".join(graph_path))

    console.print(table)
