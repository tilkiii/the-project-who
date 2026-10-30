"""
Multi-Target Social Graph and Entity Network CLI Commands.
"""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from src.utils.logger import console, print_banner
from src.vault.social_graph import SocialGraphAnalyzer


def network(
    targets: str = typer.Option(..., "--targets", "-t", help="Comma-separated target names to cross-reference (e.g. 'Remzi Sanver, Test Target')"),
    export_html: bool = typer.Option(True, "--export-html/--no-html", help="Generate standalone D3.js interactive HTML visualizer"),
) -> None:
    """Perform multi-target entity resolution, detect bridge broker nodes, and build social graphs."""
    print_banner()
    t_list = [t.strip() for t in targets.split(",") if t.strip()]
    if len(t_list) < 2:
        console.print("[bold red]Error:[/bold red] Social network analysis requires at least 2 targets to map shared ties.")
        raise typer.Exit(1)

    console.print(f"[bold green]Constructing Unified Social Network Graph for:[/bold green] [bold cyan]{' & '.join(t_list)}[/bold cyan]\n")

    analyzer = SocialGraphAnalyzer(t_list)
    net_res = analyzer.build_multi_target_network()

    table = Table(title="🌐 Social Network Link Analysis", header_style="bold cyan")
    table.add_column("Metric / Dimension", style="bold white", width=28)
    table.add_column("Value / Finding", style="green")

    table.add_row("Total Network Nodes", str(net_res["total_nodes"]))
    table.add_row("Total Relationships & Edges", str(net_res["total_edges"]))
    table.add_row("Cross-Target Shared Bridges", f"{len(net_res['shared_bridges'])} shared concept/entity bridge(s)")

    top_brokers = ", ".join(f"{b['node']} ({b['connected_target_count']} bağ)" for b in net_res["broker_nodes_ranking"][:3]) or "None"
    table.add_row("Top Betweenness Brokers", top_brokers)

    console.print(table)

    if net_res["shared_bridges"]:
        console.print(Panel(
            "\n".join(f"- [cyan]{b['bridge_concept']}[/cyan] connects targets: [yellow]{', '.join(b['connected_targets'])}[/yellow]" for b in net_res["shared_bridges"]),
            title="[bold yellow]🔗 Detected Shared Affiliations & Bridges[/bold yellow]",
            border_style="yellow"
        ))

    html_path = analyzer.output_dir / f"social_network_{'_'.join(''.join(c if c.isalnum() else '' for c in t) for t in t_list[:3])}.html"
    console.print(f"\n[bold green]✔ Interactive D3.js Social Graph rendered:[/bold green] [cyan]{html_path}[/cyan]\n")
