"""
Typer-based Modular CLI for who??.
Enterprise Evidence-Based Cognitive Digital Twin & Behavioral Modeling Framework.
"""

from __future__ import annotations

import typer
from src.utils.logger import console, print_banner

# Subcommand implementations
from src.cli.commands.harvest import anchor, archaeology, code_osint, cyber, harvest, hunt, monitor, vision
from src.cli.commands.network import network
from src.cli.commands.system import providers, purge, targets, ui
from src.cli.commands.twin import (
    arena,
    ask,
    backtest,
    chat,
    compose,
    evaluate_profile,
    explain,
    export_dossier,
    negotiate,
    psych_insights,
    simulate,
)
from src.cli.commands.vault import (
    arguments,
    blindspots,
    build_vault,
    diff,
    graph,
    graph_export,
    learn,
    snapshot,
    stylometry,
    topics,
)

app = typer.Typer(
    name="who",
    help="who?? - Evidence-Based Cognitive Digital Twin & Behavioral Modeling Framework",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Print banner if no subcommand given."""
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("[dim]Use [bold cyan]who --help[/bold cyan] to view available commands.[/dim]\n")


# Register Harvest & OSINT Commands
app.command(name="harvest")(harvest)
app.command(name="hunt")(hunt)
app.command(name="cyber")(cyber)
app.command(name="monitor")(monitor)
app.command(name="vision")(vision)
app.command(name="anchor")(anchor)
app.command(name="archaeology")(archaeology)
app.command(name="code-osint")(code_osint)

# Register Vault & Autonomous Learning Commands
app.command(name="learn")(learn)
app.command(name="topics")(topics)
app.command(name="stylometry")(stylometry)
app.command(name="blindspots")(blindspots)
app.command(name="arguments")(arguments)
app.command(name="build-vault")(build_vault)
app.command(name="graph")(graph)
app.command(name="graph-export")(graph_export)
app.command(name="snapshot")(snapshot)
app.command(name="diff")(diff)

# Register Digital Twin & Reasoning Commands
app.command(name="evaluate-profile")(evaluate_profile)
app.command(name="explain")(explain)
app.command(name="psych-insights")(psych_insights)
app.command(name="ask")(ask)
app.command(name="simulate")(simulate)
app.command(name="negotiate")(negotiate)
app.command(name="backtest")(backtest)
app.command(name="compose")(compose)
app.command(name="chat")(chat)
app.command(name="arena")(arena)
app.command(name="export-dossier")(export_dossier)

# Register Social Network Commands
app.command(name="network")(network)

# Register System & UI Commands
app.command(name="providers")(providers)
app.command(name="targets")(targets)
app.command(name="purge")(purge)
app.command(name="ui")(ui)


def main() -> None:
    """Entry point for who CLI."""
    app()


__all__ = ["app", "main"]
