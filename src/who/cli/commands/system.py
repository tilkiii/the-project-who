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
