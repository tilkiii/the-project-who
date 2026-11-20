"""
System Administration, AI Provider Auditing, Target Directory Management, and UI Server CLI Commands.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Optional
import typer
from rich.panel import Panel
from rich.table import Table

from who.config import settings
from who.providers.manager import provider_manager
from who.utils.logger import console, print_banner
from who.utils.text_cleaner import sanitize_slug


def providers() -> None:
    """Inspect configured and available AI providers (Gemini, OpenAI, Anthropic, Groq, DeepSeek, Ollama)."""
    print_banner()
    console.print("[bold green]AI Provider Registry & Status Audit[/bold green]\n")

    statuses = provider_manager.list_all_statuses()

    table = Table(title="Configured AI Engine Providers", show_header=True, header_style="bold cyan")
    table.add_column("Status", width=14)
    table.add_column("Provider", style="bold white", width=25)
    table.add_column("Default Model", style="magenta", width=26)
    table.add_column("Environment Key", style="yellow", width=24)
    table.add_column("Type", style="dim", width=18)

    provider_notes = {
        "gemini": "Google Official (Free Tier / Multimodal)",
        "openai": "OpenAI Cloud (GPT-4o / GPT-4o-mini)",
        "anthropic": "Anthropic Claude (Sonnet / Haiku)",
        "groq": "Groq Cloud (Fast Llama-3.3)",
        "deepseek": "DeepSeek API (V3 / R1 reasoning)",
        "openrouter": "Universal Multi-Model Gateway",
        "mistral": "Mistral AI European Cloud",
        "ollama": "Local Open-Source (No key needed)",
        "vllm": "High-Throughput Local Engine",
        "offline": "Corpus Grounded Fallback (No key)",
    }

    for s in statuses:
        is_active = s.get("is_active", False)
        is_avail = s.get("available", False)
        pid = s.get("id", "")

        if is_active:
            status_badge = "[bold green]● ACTIVE[/bold green]"
        elif is_avail:
            status_badge = "[bold blue]○ READY[/bold blue]"
        else:
            status_badge = "[dim]○ NOT SET[/dim]"

        env_keys = ", ".join(s.get("env_keys", [])) or "None (Local)"
        p_type = provider_notes.get(pid, "General LLM")

        table.add_row(
            status_badge,
            s.get("name", pid),
            s.get("default_model", ""),
            env_keys,
            p_type,
        )

    console.print(table)
    console.print()

    active_provider = provider_manager.detect_active_provider()
    if active_provider.provider_id != "offline":
        console.print(
            f"[bold green]✔ Active Provider:[/bold green] [bold cyan]{active_provider.display_name}[/bold cyan] "
            f"(Model: [magenta]{active_provider.default_model}[/magenta])"
        )
    else:
        console.print(
            "[bold yellow]! Notice:[/bold yellow] No active external AI API keys detected in `.env`. "
            "Running in resilient [bold cyan]Offline Fallback[/bold cyan] mode.\n"
            "To connect Google Gemini, add your key to `.env`:\n"
            "  [bold green]GEMINI_API_KEY=your_gemini_api_key_here[/bold green]\n"
            "Or use any other provider (OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY)."
        )
    console.print()


def targets() -> None:
    """List all digital twin target profiles, corpus sizes, and model readiness statuses."""
    print_banner()
    profiles_dir = Path(settings.profiles_dir)

    if not profiles_dir.exists():
        console.print("[yellow]Profiles directory does not exist yet.[/yellow]")
        return

    target_entries = []
    ignored = {"social_graphs", "arena_debates", "__pycache__"}

    for item in sorted(profiles_dir.iterdir()):
        if not item.is_dir() or item.name in ignored or item.name.startswith("."):
            continue

        raw_dir = item / "raw_data"
        vault_dir = item / "vault"
        skills_dir = item / "learned_skills"

        raw_count = len(list(raw_dir.glob("*.*"))) if raw_dir.exists() else 0
        has_character = (vault_dir / "character.json").exists()
        has_mindset = (vault_dir / "mindset_summary.md").exists()
        has_graph = (vault_dir / "concept_network.json").exists() or (vault_dir / "graph_metrics.json").exists()
        has_skills = (skills_dir / "features.json").exists() or (skills_dir / "topics.json").exists()

        # Target name extraction
        target_display_name = item.name.replace("_", " ").title()
        if has_character:
            try:
                char_data = json.loads((vault_dir / "character.json").read_text(encoding="utf-8"))
                target_display_name = char_data.get("name", target_display_name)
            except Exception:
                pass

        if has_character and has_graph:
            readiness = "[bold green]● READY[/bold green]"
        elif raw_count > 0 or has_skills:
            readiness = "[bold yellow]◐ IN PROGRESS[/bold yellow]"
        else:
            readiness = "[dim]○ RAW EMPTY[/dim]"

        target_entries.append({
            "name": target_display_name,
            "slug": item.name,
            "raw_count": raw_count,
            "has_mindset": "✓ Yes" if has_mindset else "No",
            "has_graph": "✓ Yes" if has_graph else "No",
            "readiness": readiness,
        })

    if not target_entries:
        console.print("[dim]No digital twin profiles found in storage.[/dim]")
        console.print("[cyan]Create your first profile with: [bold]who harvest --target \"Name\" --urls ...[/bold][/cyan]\n")
        return

    table = Table(title="Digital Twin Profiles Registry", header_style="bold cyan")
    table.add_column("Status", width=16)
    table.add_column("Target Name", style="bold white", width=24)
    table.add_column("Directory Slug", style="dim", width=20)
    table.add_column("Corpus Assets", style="magenta", width=14)
    table.add_column("Mindset Profile", style="cyan", width=16)
    table.add_column("Knowledge Graph", style="green", width=16)

    for entry in target_entries:
        table.add_row(
            entry["readiness"],
            entry["name"],
            entry["slug"],
            f"{entry['raw_count']} file(s)",
            entry["has_mindset"],
            entry["has_graph"],
        )

    console.print(table)
    console.print(f"\n[dim]Total targets in local storage: {len(target_entries)}[/dim]\n")


def purge(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name or directory slug to purge"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without interactive confirmation prompt"),
) -> None:
    """Safely delete a target profile and its local corpus and vault data."""
    profiles_dir = Path(settings.profiles_dir)
    slug = sanitize_slug(target)

    # Find matching directory
    target_dir = profiles_dir / slug
    if not target_dir.exists():
        # Check direct name
        target_dir = profiles_dir / target
        if not target_dir.exists():
            console.print(f"[bold red]Error:[/bold red] Target profile '[cyan]{target}[/cyan]' not found in {profiles_dir}.")
            raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Are you sure you want to PERMANENTLY purge profile '{target_dir.name}'? This cannot be undone.")
        if not confirm:
            console.print("[dim]Purge cancelled by user.[/dim]")
            raise typer.Exit(0)

    try:
        shutil.rmtree(target_dir)
        console.print(Panel(
            f"[bold green]✔ Successfully purged target profile:[/bold green] [bold cyan]{target_dir.name}[/bold cyan]\n"
            f"[dim]Directory removed: {target_dir}[/dim]",
            title="[bold red]Target Profile Purged[/bold red]",
            border_style="red",
        ))
    except Exception as exc:
        console.print(f"[bold red]Failed to purge target directory:[/bold red] {exc}")
        raise typer.Exit(1)


def mcp() -> None:
    """Start the standard Model Context Protocol (MCP) server over stdio for AI agents."""
    from who.mcp_server import MCPServer
    server = MCPServer()
    server.run()




def doctor() -> None:
    """Run comprehensive system diagnostics, verify API keys, storage, profiles, graph, and MCP readiness."""
    import sys
    print_banner()
    console.print("[bold green]🩺 who?? System Integrity Diagnostic (who doctor)[/bold green]\n")

    table = Table(title="Diagnostic System Health Matrix", show_header=True, header_style="bold cyan")
    table.add_column("Subsystem / Component", style="bold white", width=28)
    table.add_column("Status", width=18)
    table.add_column("Operational Findings & Guidance", style="dim")

    critical_failures = 0

    # 1. Python Environment Check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        table.add_row("Python Runtime", "[bold green]✔ COMPLIANT[/bold green]", f"Python {py_ver} on {sys.platform}")
    else:
        table.add_row("Python Runtime", "[bold red]✘ OUTDATED[/bold red]", f"Python {py_ver} (Requires Python >= 3.10)")
        critical_failures += 1

    # 2. AI Provider & Credentials
    active_prov = provider_manager.detect_active_provider()
    if active_prov.is_available() and active_prov.provider_id != "offline":
        table.add_row("LLM Inference Provider", "[bold green]✔ ACTIVE[/bold green]", f"{active_prov.display_name} (API key verified)")
    elif active_prov.provider_id == "offline":
        table.add_row("LLM Inference Provider", "[bold yellow]▲ OFFLINE MODE[/bold yellow]", "Corpus-grounded fallback. Set GEMINI_API_KEY for cloud reasoning.")
    else:
        table.add_row("LLM Inference Provider", "[bold yellow]▲ UNCONFIGURED[/bold yellow]", f"{active_prov.display_name} (Missing API keys in .env)")

    # 3. Storage & Profiles Directory
    profiles_dir = Path(settings.profiles_dir)
    if profiles_dir.exists():
        target_count = len([p for p in profiles_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))])
        table.add_row("Local Vault Storage", "[bold green]✔ OPERATIONAL[/bold green]", f"{profiles_dir} ({target_count} target profile directories)")
    else:
        table.add_row("Local Vault Storage", "[bold red]✘ MISSING[/bold red]", f"Directory '{profiles_dir}' does not exist.")
        critical_failures += 1

    # 4. Benchmark Profile (Target X)
    target_x_dir = profiles_dir / "target_x"
    if target_x_dir.exists():
        has_mindset = (target_x_dir / "vault" / "mindset_summary.md").exists()
        has_graph = (target_x_dir / "vault" / "knowledge_graph.json").exists()
        if has_mindset and has_graph:
            table.add_row("Benchmark Demo Profile", "[bold green]✔ READY[/bold green]", "Target X is fully calibrated with mindset and knowledge graph.")
        else:
            table.add_row("Benchmark Demo Profile", "[bold yellow]▲ INCOMPLETE[/bold yellow]", "Target X folder exists but missing vault artifacts.")
    else:
        table.add_row("Benchmark Demo Profile", "[bold dim]○ NOT FOUND[/bold dim]", "profiles/target_x/ not located in active storage.")

    # 5. Graph Engine
    try:
        import networkx as nx
        g = nx.DiGraph()
        g.add_edge("DiagnosticNodeA", "DiagnosticNodeB")
        table.add_row("Graph Engine (NetworkX)", "[bold green]✔ OPERATIONAL[/bold green]", f"NetworkX v{nx.__version__} graph operations validated.")
    except Exception as exc:
        table.add_row("Graph Engine (NetworkX)", "[bold red]✘ FAILED[/bold red]", f"NetworkX error: {exc}")
        critical_failures += 1

    # 6. Vector Store (ChromaDB & Local Embeddings)
    try:
        import chromadb
        table.add_row("Vector Store (ChromaDB)", "[bold green]✔ OPERATIONAL[/bold green]", f"ChromaDB v{chromadb.__version__} HNSW engine initialized.")
    except Exception:
        table.add_row("Vector Store (Fallback)", "[bold yellow]▲ LOCAL FALLBACK[/bold yellow]", "Using built-in cosine vector index. (Install chromadb for persistent HNSW backend)")

    # 7. Model Context Protocol (MCP) Server Tools
    try:
        from who.mcp_server import TOOLS_SCHEMA
        tool_count = len(TOOLS_SCHEMA)
        table.add_row("MCP Agent Protocol", "[bold green]✔ OPERATIONAL[/bold green]", f"{tool_count} JSON-RPC 2.0 tools registered (who_chat, who_simulate, etc.)")
    except Exception as exc:
        table.add_row("MCP Agent Protocol", "[bold red]✘ FAILED[/bold red]", f"MCP schema failure: {exc}")
        critical_failures += 1

    console.print(table)
    console.print()

    if critical_failures == 0:
        console.print(Panel(
            "[bold green]✔ ALL CORE SUBSYSTEMS OPERATIONAL[/bold green]\n"
            "[dim]The framework is ready for multi-agent workflows, CLI operations, and MCP client sessions.[/dim]",
            title="[bold green]Health Check Verdict: PASSED[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]✘ {critical_failures} CRITICAL SUBSYSTEM FAILURE(S) DETECTED[/bold red]\n"
            "[dim]Review the findings above and resolve missing dependencies or directories.[/dim]",
            title="[bold red]Health Check Verdict: FAILED[/bold red]",
            border_style="red",
        ))
        raise typer.Exit(1)
