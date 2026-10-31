"""
Cognitive Digital Twin, Behavioral Simulation, Multi-Agent Arena, and Reasoning CLI Commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.config import settings
from src.engine.arena import SwarmArena
from src.engine.composer import Composer
from src.engine.dossier_exporter import DossierExporter
from src.engine.evaluator import ProfileEvaluator
from src.engine.explainability_engine import ExplainabilityEngine
from src.engine.oracle import Oracle
from src.engine.psych_insights import PsychInsights
from src.engine.simulator import Simulator
from src.engine.twin_chat import TwinChat
from src.utils.logger import console, print_banner, print_probabilistic_claim


def evaluate_profile(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Run DSPy-driven evaluation of retrieval quality, profile consistency, and hallucination rate."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    evaluator = ProfileEvaluator(target, dirs)
    results = evaluator.evaluate_profile()

    score_pct = results.get("overall_score_pct")
    metrics = results.get("metrics", {})

    table = Table(title=f"Profile Scientific Evaluation Audit: {target}", header_style="bold cyan")
    table.add_column("Evaluation Dimension", style="bold white", width=25)
    table.add_column("Score", style="green", width=12)
    table.add_column("Audit Findings", style="dim")

    table.add_row("Citation Validity", f"{int(metrics.get('citation_validity', {}).get('score', 0) * 100)}%", f"Verified in corpus: {metrics.get('citation_validity', {}).get('verified_in_corpus')} / {metrics.get('citation_validity', {}).get('quotes_checked')}")
    table.add_row("Profile Consistency", f"{int(metrics.get('profile_consistency', {}).get('score', 0) * 100)}%", "Cross-layer verification across skill, mindset, and graph")
    table.add_row("Retrieval Quality", f"{int(metrics.get('retrieval_quality', {}).get('score', 0) * 100)}%", f"Indexed vector chunks: {metrics.get('retrieval_quality', {}).get('chunks_indexed')}")
    table.add_row("Hallucination Rate", f"{round(metrics.get('hallucination_rate', {}).get('score', 0) * 100, 1)}%", f"Status: {metrics.get('hallucination_rate', {}).get('status')}")

    console.print(table)
    console.print(f"\n[bold]Overall Scientific Integrity Rating:[/bold] [bold green]{score_pct}[/bold green] ([cyan]{results.get('evaluation_verdict')}[/cyan])\n")


def explain(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    claim: str = typer.Option(..., "--claim", "-c", help="Claim to trace and verify"),
) -> None:
    """Trace any claim back to primary quotes, Neo4j graph paths, and confidence scores."""
    dirs = settings.get_target_dirs(target)
    engine = ExplainabilityEngine(target, dirs)
    res = engine.explain_claim(claim)

    graph_path_str = [" -> ".join(p) for p in res.get("graph_trace_paths", [])]
    first_path = graph_path_str[0].split(" -> ") if graph_path_str else None

    print_probabilistic_claim(
        claim=res.get("calibrated_probabilistic_statement", claim),
        confidence_score=res.get("confidence_score", 0.70),
        sources=res.get("evidence_sources", []),
        quotes=res.get("primary_quotes", []),
        graph_path=first_path,
    )

    dissonance = res.get("cognitive_dissonance_analysis", {})
    if dissonance.get("has_dissonance_tension"):
        console.print(Panel(
            f"[bold red]⚠️ {dissonance.get('probabilistic_warning')}[/bold red]",
            title="[bold yellow]⚡ Graph-Driven Cognitive Dissonance Alert[/bold yellow]",
            border_style="red"
        ))


def psych_insights(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Generate behavioral report citing both target quotes AND academic paper DOIs."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    reporter = PsychInsights(target, dirs)
    report = reporter.generate_report()

    console.print(Panel(report.get("report_markdown", ""), title=f"[bold cyan]Scientific Behavioral Report: {target}[/bold cyan]", border_style="cyan"))

    # Render dedicated table for Innovative Psychoinformatic Metrics
    table = Table(title=f"🔬 İnovatif Psikoinformatik & Epistemik Metrikler: {target}", header_style="bold magenta")
    table.add_column("Yetenek / Analiz Boyutu", style="bold white", width=34)
    table.add_column("Olasılıksal Çıkarım & Temellendirilmiş Bulgu", style="cyan")

    table.add_row(
        "Cognitive Chronomics (Sirkadyen)",
        report.get("chronomics_statement", "Hedefin gece ve gündüz metinleri arasında sirkadyen sapma analizi yapıldı.")
    )
    table.add_row(
        "Graph Cognitive Dissonance",
        report.get("dissonance_statement", "Graf üzerinde karşıt kavram kümelenmeleri ve [CONTRADICTS] kenarları denetlendi.")
    )
    table.add_row(
        "Epistemic Manipulation Index",
        report.get("manipulation_statement", "Dogmatik ve kanıtsız mutlakiyetçi dil eğilimi ölçüldü.")
    )

    console.print(table)
    console.print()

    # Render Autonomous Academic Research Agent Table (Calibrated Scientific Weights & Reasoning Notes)
    weights = report.get("scientific_weights", [])
    if weights:
        w_table = Table(title=f"📚 Autonomous Academic Research Agent: Calibrated Scientific Weights ({target})", header_style="bold green")
        w_table.add_column("Davranışsal Belirti (Marker)", style="bold white", width=25)
        w_table.add_column("Korele Psikometrik Boyut", style="cyan", width=24)
        w_table.add_column("Korelasyon (r)", style="green", width=14)
        w_table.add_column("Güven Skoru", style="yellow", width=12)
        w_table.add_column("Kanıt Kaynağı (DOI)", style="magenta", width=28)
        w_table.add_column("Ajanın Bilimsel Muhakeme Notu", style="dim")

        for w in weights[:6]:
            w_table.add_row(
                w.get("behavioral_marker", ""),
                w.get("correlated_psychometric", ""),
                f"r={w.get('correlation_coefficient_r', 0.0)}",
                f"%{int(w.get('confidence_score', 0.8) * 100)}",
                w.get("evidence_source", ""),
                w.get("agent_reasoning_note", ""),
            )
        console.print(w_table)
        console.print()


def ask(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Question to ask about the target"),
) -> None:
    """Evidence-backed Q&A answering questions about the target using Graph + Vector memory."""
    dirs = settings.get_target_dirs(target)
    oracle = Oracle(target, dirs)
    res = oracle.ask(prompt)

    console.print(f"\n[bold cyan]Question:[/bold cyan] {prompt}")
    console.print(Panel(res.get("answer", ""), title=f"[bold green]Oracle Answer ({int(res.get('confidence_score', 0.8) * 100)}% Confidence)[/bold green]", border_style="green"))

    if res.get("primary_quotes"):
        console.print("[bold white]Supporting Primary Quotes:[/bold white]")
        for q in res.get("primary_quotes", [])[:2]:
            console.print(f"  [dim]• \"{q}\"[/dim]")
    console.print()


def simulate(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario to simulate"),
) -> None:
    """Simulate plausible reactions and counter-moves under explicit scenario assumptions."""
    dirs = settings.get_target_dirs(target)
    sim = Simulator(target, dirs)
    res = sim.simulate_scenario(scenario)

    console.print(f"\n[bold cyan]Simulated Scenario:[/bold cyan] {scenario}")
    console.print(Panel(res.get("simulation_analysis", ""), title=f"[bold magenta]Plausible Reaction Simulation ({int(res.get('confidence_score', 0.75) * 100)}% Confidence)[/bold magenta]", border_style="magenta"))
    console.print(f"[dim italic]{res.get('epistemic_qualification')}[/dim italic]\n")


def compose(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    goal: str = typer.Option(..., "--goal", "-g", help="Communication or outreach goal"),
    context: Optional[str] = typer.Option(None, "--context", help="Additional context"),
) -> None:
    """Generate hyper-tailored outreach letters with adversarial rejection stress-testing."""
    dirs = settings.get_target_dirs(target)
    composer = Composer(target, dirs)
    res = composer.compose_outreach(goal, context=context)

    console.print(f"\n[bold cyan]Outreach Goal:[/bold cyan] {goal}\n")
    console.print(Panel(res.get("initial_draft", ""), title="[dim]Initial Calibrated Draft[/dim]", border_style="blue"))
    console.print(Panel(res.get("adversarial_rejection_analysis", ""), title="[bold red]Adversarial Rejection Stress-Test[/bold red]", border_style="red"))
    console.print(Panel(res.get("optimized_final_pitch", ""), title="[bold green]Optimized Final Pitch[/bold green]", border_style="green"))


def chat(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    mode: str = typer.Option("autodetect", "--mode", "-m", help="Tone mode: autodetect, formal, informal"),
) -> None:
    """Interactive CLI digital twin chat with tone toggling."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    twin = TwinChat(target, dirs, mode=mode)

    console.print(f"[bold green]Starting Digital Twin Interactive Session for:[/bold green] [bold cyan]{target}[/bold cyan]")
    console.print(f"[dim]Mode: {mode.upper()} • Type 'exit' or 'quit' to end session.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            if not user_input.strip():
                continue
            if user_input.lower().strip() in ("exit", "quit", "q"):
                console.print("[dim]Ending digital twin chat session.[/dim]")
                break

            response = twin.respond(user_input)
            console.print(f"\n[bold cyan]{target}[/bold cyan]: {response}\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session interrupted. Exiting.[/dim]")
            break


def arena(
    targets: str = typer.Option(..., "--targets", "-t", help="Comma-separated target participant names (e.g. 'Remzi Sanver, Test Target')"),
    topic: str = typer.Option(..., "--topic", "-p", help="Debate proposition, dilemma, or policy question"),
    rounds: int = typer.Option(2, "--rounds", "-r", help="Number of multi-agent debate rounds (1-5)"),
) -> None:
    """Convene multiple digital twins into an interactive debate arena tracking consensus divergence."""
    print_banner()
    participant_list = [p.strip() for p in targets.split(",") if p.strip()]
    if len(participant_list) < 2:
        console.print("[bold red]Error:[/bold red] Swarm Arena requires at least 2 target participants.")
        raise typer.Exit(1)

    console.print(f"[bold magenta]⚔️ Initiating Swarm Arena Debate:[/bold magenta] [bold cyan]{' vs '.join(participant_list)}[/bold cyan]")
    console.print(f"[bold yellow]Proposition:[/bold yellow] [italic]{topic}[/italic]\n")

    swarm = SwarmArena(participant_names=participant_list, topic=topic, rounds=rounds)
    debate_res = swarm.run_debate()

    # Display turns in terminal
    for turn in debate_res.get("transcript", []):
        console.print(Panel(
            turn["statement"],
            title=f"[bold cyan]Tur {turn['round']}: {turn['speaker']}[/bold cyan]",
            border_style="cyan"
        ))

    synth = debate_res.get("synthesis", {})
    common_pts = "\n".join(f"- {c}" for c in synth.get("common_ground_points", []))
    friction_pts = "\n".join(f"- {f}" for f in synth.get("irreconcilable_frictions", []))

    console.print(Panel(
        f"[bold]Ortak Konsensüs Oranı:[/bold] [green]%{synth.get('consensus_score_pct', 55)}[/green] | "
        f"[bold]Fikirsel Ayrışma / Sürtünme:[/bold] [red]%{synth.get('consensus_divergence_pct', 45)}[/red]\n\n"
        f"[bold yellow]Ortak Payda:[/bold yellow]\n{common_pts}\n\n"
        f"[bold red]Temel Çelişkiler:[/bold red]\n{friction_pts}\n\n"
        f"[bold green]Sentez Çıkarımı:[/bold green] {synth.get('emergent_synthesis', '')}",
        title="[bold magenta]📊 Swarm Arena Konsensüs ve Epistemik Sentez Raporu[/bold magenta]",
        border_style="magenta"
    ))


def export_dossier(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Custom output HTML file path"),
) -> None:
    """Compile and export a standalone, self-contained HTML intelligence dossier for the target."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    exporter = DossierExporter(target, dirs)
    out_file = exporter.export_to_file(Path(output) if output else None)
    console.print(
        Panel(
            f"[bold green]✔ Standalone HTML Intelligence Dossier exported![/bold green]\n\n"
            f"• Target: [bold cyan]{target}[/bold cyan]\n"
            f"• File Path: [cyan]{out_file}[/cyan]",
            title="[bold cyan]📁 Intelligence Dossier Export[/bold cyan]",
            border_style="green",
        )
    )
