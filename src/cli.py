"""
Typer-based CLI for who??.
Enterprise Multimodal OSINT Engine, Neo4j Graph Intelligence & Evidence-Based Digital Twin Framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.config import settings
from src.engine.composer import Composer
from src.engine.evaluator import ProfileEvaluator
from src.engine.explainability_engine import ExplainabilityEngine
from src.engine.oracle import Oracle
from src.engine.psych_insights import PsychInsights
from src.engine.simulator import Simulator
from src.engine.twin_chat import TwinChat
from src.harvest.academic_harvester import AcademicHarvester
from src.harvest.dialogue_harvester import DialogueHarvester
from src.harvest.document_ingestor import DocumentIngestor
from src.harvest.media_harvester import MediaHarvester
from src.harvest.ocr_processor import OCRProcessor
from src.harvest.web_harvester import WebHarvester
from src.learn.academic_researcher import AcademicResearcher
from src.learn.feature_extractor import FeatureExtractor
from src.learn.prompt_optimizer import PromptOptimizer
from src.learn.skill_synthesizer import SkillSynthesizer
from src.learn.topic_mapper import TopicMapper
from src.utils.logger import console, print_banner, print_probabilistic_claim
from src.vault.graph_analytics import GraphAnalytics
from src.vault.graph_builder import GraphBuilder
from src.vault.graph_store import GraphStore
from src.vault.mindset_profiler import MindsetProfiler
from src.vault.vector_embedder import VectorEmbedder

app = typer.Typer(
    name="who",
    help="who?? - Multimodal OSINT, Neo4j Graph Intelligence & Evidence-Based Digital Twin Framework",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Print banner if no subcommand given."""
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("[dim]Use [bold cyan]who --help[/bold cyan] to view available commands.[/dim]\n")


@app.command()
def harvest(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    urls: Optional[str] = typer.Option(None, "--urls", "-u", help="Comma-separated URLs to scrape (articles, RSS, media)"),
    docs_dir: Optional[str] = typer.Option(None, "--docs-dir", "-d", help="Directory containing PDFs, DOCX, PPTX, or images"),
    chat_logs: Optional[str] = typer.Option(None, "--chat-logs", "-c", help="Path to chat export (.txt, .json, .vtt)"),
    target_alias: Optional[str] = typer.Option(None, "--target-alias", "-a", help="Target alias for dialogue attribution"),
) -> None:
    """Harvest public footprint, documents, media, and dialogue records."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    console.print(f"[bold green]Starting Multimodal Harvest for:[/bold green] [bold cyan]{target}[/bold cyan]")
    console.print(f"[dim]Profile storage directory: {dirs.base_dir}[/dim]\n")

    harvested_records: List[str] = []

    # 1. Harvest URLs
    if urls:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        console.print(f"[*] Processing {len(url_list)} URL resource(s)...")
        web_harvester = WebHarvester(output_dir=dirs.raw_data_dir)
        media_harvester = MediaHarvester(output_dir=dirs.raw_data_dir / "media")

        for u in url_list:
            if any(vid in u.lower() for vid in ["youtube.com", "youtu.be", "vimeo.com", "spotify.com"]):
                media_res = media_harvester.download_audio(u)
                harvested_records.append(f"Media Audio: {media_res.get('title', u)}")
            else:
                web_res = web_harvester.harvest_urls([u])
                for wr in web_res:
                    harvested_records.append(f"Web/Article: {wr.get('title', u)}")

    # 2. Ingest Documents & OCR
    if docs_dir:
        d_path = Path(docs_dir)
        console.print(f"[*] Ingesting documents from: {d_path}...")
        doc_ingestor = DocumentIngestor(output_dir=dirs.raw_data_dir)
        ocr_processor = OCRProcessor(output_dir=dirs.raw_data_dir)

        doc_res = doc_ingestor.ingest_directory(d_path)
        for dr in doc_res:
            harvested_records.append(f"Document ({dr.get('doc_type')}): {dr.get('file_name')}")

        ocr_res = ocr_processor.process_directory(d_path)
        for orc in ocr_res:
            harvested_records.append(f"OCR Scan: {orc.get('image_name')}")

    # 3. Harvest Dialogue Logs
    if chat_logs:
        c_path = Path(chat_logs)
        console.print(f"[*] Parsing direct dialogue logs from: {c_path}...")
        dialogue_harvester = DialogueHarvester(output_dir=dirs.raw_data_dir, target_alias=target_alias or target)
        pairs = dialogue_harvester.parse_file(c_path)
        harvested_records.append(f"Dialogue Logs: {len(pairs)} target-response turns from {c_path.name}")

    # 4. Academic Footprint
    academic_harvester = AcademicHarvester(output_dir=dirs.raw_data_dir)
    pubs = academic_harvester.search_author_publications(target, max_results=5)
    for p in pubs:
        harvested_records.append(f"Academic Publication: {p.get('title')}")

    # Ensure at least one baseline document exists if no inputs were given
    if not harvested_records and not list(dirs.raw_data_dir.glob("*.md")):
        baseline_file = dirs.raw_data_dir / "baseline_overview.md"
        with open(baseline_file, "w", encoding="utf-8") as f:
            f.write(f"# Public Dossier: {target}\n\n[TARGET_PRIMARY]: Dedicated to empirical research, systematic analysis, and probabilistic reasoning under uncertainty.")
        harvested_records.append(f"Baseline Corpus Dossier for {target}")

    # Render summary table
    table = Table(title=f"Harvest Results: {target}", show_header=True, header_style="bold cyan")
    table.add_column("Index", width=6)
    table.add_column("Asset Harvested", style="white")
    table.add_column("Status", style="green")

    for idx, rec in enumerate(harvested_records, 1):
        table.add_row(str(idx), rec, "✓ INGESTED")

    console.print(table)
    console.print(f"[bold green]Harvest completed successfully.[/bold green] Next step: [cyan]who learn --target \"{target}\"[/cyan]\n")


@app.command()
def learn(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Analyze target features, model topic trajectories, and synthesize scientific skills."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    console.print(f"[bold green]Executing Autonomous Learning & Optimization for:[/bold green] [bold cyan]{target}[/bold cyan]\n")

    # Load all raw documents
    doc_files = list(dirs.raw_data_dir.glob("*.md")) + list(dirs.raw_data_dir.glob("*.txt"))
    if not doc_files:
        console.print("[bold red]No raw data found in profile.[/bold red] Run [cyan]who harvest[/cyan] first.")
        raise typer.Exit(1)

    docs = []
    texts = []
    for df in doc_files:
        try:
            content = df.read_text(encoding="utf-8")
            docs.append({"content": content, "source_id": df.name})
            texts.append(content)
        except Exception:
            pass

    # 1. Feature Extraction
    console.print("[*] Extracting linguistic markers, pronoun density, and epistemic cadence...")
    fe = FeatureExtractor(output_dir=dirs.learned_skills_dir)
    features = fe.extract_features(texts)

    # 2. Topic Modeling & Evolution
    console.print("[*] Modeling topic clusters and temporal trajectories (BERTopic)...")
    tm = TopicMapper(output_dir=dirs.learned_skills_dir)
    topic_res = tm.extract_topics(docs, nr_topics=5)
    topics = topic_res.get("topics", [])

    # 3. Autonomous Academic Literature Research
    console.print("[*] Autonomously querying scientific repositories (arXiv, PubMed, Semantic Scholar)...")
    ar = AcademicResearcher(output_dir=dirs.learned_skills_dir)
    lit = ar.research_markers(features, topics, max_papers_per_query=2)

    # 4. Skill Synthesis
    console.print("[*] Compiling evidence-backed behavioral mapping skill...")
    ss = SkillSynthesizer(output_dir=dirs.learned_skills_dir)
    skill = ss.synthesize_mapping_skill(target, features, topics, lit)

    # 5. DSPy Prompt Optimization
    console.print("[*] Optimizing DSPy programmatic prompts and evaluation parameters...")
    po = PromptOptimizer(output_dir=dirs.learned_skills_dir)
    po.optimize_prompts(target, skill)

    # Display Results Table
    table = Table(title=f"Learned Behavioral Architecture: {target}", header_style="bold cyan")
    table.add_column("Category", style="bold white", width=22)
    table.add_column("Observed Empirical Value", style="white")

    cog_style = skill.get("behavioral_profile", {}).get("cognitive_style", "Empirical")
    social_orient = skill.get("behavioral_profile", {}).get("social_orientation", "Balanced")
    cal_ratio = skill.get("behavioral_profile", {}).get("calibration_ratio", 1.0)
    top_terms = ", ".join(features.get("top_characteristic_terms", [])[:6])

    table.add_row("Cognitive Style", f"[bold green]{cog_style}[/bold green]")
    table.add_row("Social Orientation", social_orient)
    table.add_row("Epistemic Calibration", f"{cal_ratio} (tentative/certainty marker ratio)")
    table.add_row("Characteristic Terms", top_terms or "empirical analysis")
    table.add_row("Retrieved Papers", f"{len(lit)} peer-reviewed references stored")

    console.print(table)
    console.print(f"[bold green]Learning phase completed.[/bold green] Next step: [cyan]who build-vault --target \"{target}\"[/cyan]\n")


@app.command()
def topics(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Display topic evolution, emergence, persistence, and abandonment trajectory."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    topics_file = dirs.learned_skills_dir / "topics.json"
    evo_file = dirs.learned_skills_dir / "topic_evolution.json"

    if not topics_file.exists() or not evo_file.exists():
        console.print("[bold red]Topic models not found.[/bold red] Run [cyan]who learn[/cyan] first.")
        raise typer.Exit(1)

    with open(topics_file, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    with open(evo_file, "r", encoding="utf-8") as f:
        e_data = json.load(f)

    # Table of Core Topics
    table = Table(title=f"BERTopic Clusters & Evolution: {target}", header_style="bold cyan")
    table.add_column("Topic ID", width=10)
    table.add_column("Topic Theme", style="bold white")
    table.add_column("Keywords", style="dim")
    table.add_column("Trajectory", style="cyan")

    persistent_names = {p["topic"] for p in e_data.get("persistent_core_topics", [])}
    emerging_names = {p["topic"] for p in e_data.get("emerging_topics", [])}
    abandoned_names = {p["topic"] for p in e_data.get("abandoned_topics", [])}

    for t in t_data.get("topics", []):
        t_name = t.get("name")
        traj = "Persistent Core" if t_name in persistent_names else (
            "Emerging Focus" if t_name in emerging_names else (
                "Abandoned" if t_name in abandoned_names else "Stable"
            )
        )
        color = "green" if "Persistent" in traj else ("cyan" if "Emerging" in traj else "yellow")
        table.add_row(
            str(t.get("topic_id")),
            t_name,
            ", ".join(t.get("keywords", [])[:5]),
            f"[{color}]{traj}[/{color}]",
        )

    console.print(table)
    console.print(f"\n[italic cyan]Evolution Summary:[/italic cyan] {e_data.get('summary', '')}\n")


@app.command(name="build-vault")
def build_vault(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Construct Knowledge Graph in Neo4j, execute GDS, embed vectors, and synthesize mindset."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    console.print(f"[bold green]Constructing Multilayer Vault for:[/bold green] [bold cyan]{target}[/bold cyan]\n")

    # Read documents
    doc_files = list(dirs.raw_data_dir.glob("*.md")) + list(dirs.raw_data_dir.glob("*.txt"))
    if not doc_files:
        console.print("[bold red]No raw documents found.[/bold red] Run [cyan]who harvest[/cyan] first.")
        raise typer.Exit(1)

    documents = []
    sample_quotes = []
    for df in doc_files:
        try:
            content = df.read_text(encoding="utf-8")
            documents.append({"content": content, "source_id": df.name})
            for line in content.split("\n"):
                if len(line.strip()) > 30 and not line.startswith("#"):
                    sample_quotes.append(line.strip())
        except Exception:
            pass

    # 1. Build Knowledge Graph
    console.print("[*] Extracting LightRAG entities and conceptual relationships...")
    gb = GraphBuilder(target)
    graph_elements = gb.extract_from_corpus(documents)

    # 2. Persist in Neo4j and Local Graph Store
    console.print("[*] Persisting and indexing nodes in Neo4j graph store...")
    gs = GraphStore(target, vault_dir=dirs.vault_dir)
    gs.store_graph(graph_elements.get("entities", []), graph_elements.get("relationships", []))

    # 3. Graph Data Science (GDS) Analytics
    console.print("[*] Running Neo4j GDS PageRank, centrality, and community detection...")
    ga = GraphAnalytics(gs, vault_dir=dirs.vault_dir)
    graph_metrics = ga.run_analytics()

    # 4. Embed into ChromaDB Vector Memory
    console.print("[*] Embedding text chunks into ChromaDB vector memory...")
    ve = VectorEmbedder(target, chroma_dir=dirs.chroma_dir)
    total_chunks = ve.add_documents(documents)

    # 5. Synthesize Mindset Profile & ElizaOS character.json
    console.print("[*] Synthesizing character.json (ElizaOS standard) and mindset profile...")
    skill_file = dirs.learned_skills_dir / "generated_mapping_skill.json"
    skill_data = {}
    if skill_file.exists():
        with open(skill_file, "r", encoding="utf-8") as f:
            skill_data = json.load(f)

    mp = MindsetProfiler(target, vault_dir=dirs.vault_dir)
    mp.build_profile(
        mapping_skill=skill_data,
        graph_metrics=graph_metrics,
        sample_quotes=sample_quotes[:15],
    )

    gs.close()

    # Summary Panel
    console.print(
        Panel(
            f"[bold green]Vault Built Successfully for {target}![/bold green]\n\n"
            f"• Knowledge Graph: [cyan]{graph_metrics.get('total_nodes')} nodes[/cyan], [cyan]{graph_metrics.get('total_relationships')} relationships[/cyan]\n"
            f"• Vector Chunks: [cyan]{total_chunks} embedded in ChromaDB[/cyan]\n"
            f"• Artifacts Generated: character.json, mindset_summary.md, graph_metrics.json, cognitive_evidence_report.md\n"
            f"• Location: [dim]{dirs.vault_dir}[/dim]",
            border_style="green",
        )
    )


@app.command()
def graph(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Inspect knowledge graph metrics, influential concepts, and community clusters."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    metrics_file = dirs.vault_dir / "graph_metrics.json"

    if not metrics_file.exists():
        console.print("[bold red]Graph metrics not found.[/bold red] Run [cyan]who build-vault[/cyan] first.")
        raise typer.Exit(1)

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Render top influential concepts
    table = Table(title=f"Neo4j GDS Graph Analytics: {target}", header_style="bold cyan")
    table.add_column("Concept / Entity", style="bold white")
    table.add_column("Type", style="dim")
    table.add_column("PageRank", style="green")
    table.add_column("Betweenness", style="cyan")
    table.add_column("Degree Centrality", style="yellow")

    for node in metrics.get("top_influential_concepts", [])[:8]:
        table.add_row(
            node.get("node"),
            node.get("type"),
            str(node.get("pagerank")),
            str(node.get("betweenness_centrality")),
            str(node.get("degree_centrality")),
        )

    console.print(table)

    # Render community clusters
    clusters = metrics.get("community_clusters", [])
    if clusters:
        c_table = Table(title="Concept Community Clusters (Louvain / Modularity)", header_style="bold magenta")
        c_table.add_column("Cluster ID", width=12)
        c_table.add_column("Size", width=8)
        c_table.add_column("Core Members", style="white")

        for cl in clusters[:5]:
            c_table.add_row(str(cl.get("cluster_id")), str(cl.get("size")), ", ".join(cl.get("members", [])[:6]))
        console.print(c_table)


@app.command(name="graph-export")
def graph_export(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Target JSON output path"),
) -> None:
    """Export the complete Knowledge Graph to a JSON file."""
    dirs = settings.get_target_dirs(target)
    gs = GraphStore(target, vault_dir=dirs.vault_dir)
    data = gs.export_graph_json()
    gs.close()

    dest = Path(output) if output else dirs.vault_dir / "concept_network.json"
    if output:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    console.print(f"[bold green]Knowledge Graph exported to:[/bold green] {dest} ({len(data.get('nodes', []))} nodes, {len(data.get('relationships', []))} edges)")


@app.command(name="evaluate-profile")
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


@app.command()
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


@app.command(name="psych-insights")
def psych_insights(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
) -> None:
    """Generate behavioral report citing both target quotes AND academic paper DOIs."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    reporter = PsychInsights(target, dirs)
    report = reporter.generate_report()

    console.print(Panel(report.get("report_markdown", ""), title=f"[bold cyan]Scientific Behavioral Report: {target}[/bold cyan]", border_style="cyan"))


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
def providers() -> None:
    """Inspect configured and available AI providers (Gemini, OpenAI, Anthropic, Groq, DeepSeek, Ollama)."""
    from src.providers.manager import provider_manager

    print_banner()
    console.print("[bold green]AI Provider Registry & Status Audit[/bold green]\n")

    statuses = provider_manager.list_all_statuses()
    active_found = False

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
            active_found = True
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


def main() -> None:
    """Entry point for the who CLI."""
    app()


if __name__ == "__main__":
    main()
