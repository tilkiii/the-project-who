"""
Vault, Knowledge Graph, Vector Embedding, and Autonomous Learning CLI Commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.learn.academic_researcher import AcademicResearcher
from src.learn.feature_extractor import FeatureExtractor
from src.learn.prompt_optimizer import PromptOptimizer
from src.learn.skill_synthesizer import SkillSynthesizer
from src.learn.topic_mapper import TopicMapper
from src.utils.logger import console, print_banner
from src.vault.graph_analytics import GraphAnalytics
from src.vault.graph_builder import GraphBuilder
from src.vault.graph_store import GraphStore
from src.vault.mindset_profiler import MindsetProfiler
from src.vault.vector_embedder import VectorEmbedder


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
    console.print("[*] Extracting linguistic markers, pronoun density, epistemic cadence, and chronomics...")
    fe = FeatureExtractor(output_dir=dirs.learned_skills_dir)
    features = fe.extract_features(texts, documents=docs)

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
    table.add_column("Category", style="bold white", width=25)
    table.add_column("Observed Empirical Value", style="white")

    cog_style = skill.get("behavioral_profile", {}).get("cognitive_style", "Empirical")
    social_orient = skill.get("behavioral_profile", {}).get("social_orientation", "Balanced")
    cal_ratio = skill.get("behavioral_profile", {}).get("calibration_ratio", 1.0)
    top_terms = ", ".join(features.get("top_characteristic_terms", [])[:6])

    manip = features.get("epistemic_manipulation_index", {})
    chron = features.get("chronomics", {}).get("circadian_drift_summary", {})

    table.add_row("Cognitive Style", f"[bold green]{cog_style}[/bold green]")
    table.add_row("Social Orientation", social_orient)
    table.add_row("Epistemic Calibration", f"{cal_ratio} (tentative/certainty marker ratio)")
    table.add_row("Manipulation Index", f"{manip.get('score_pct', '0%')} ({manip.get('risk_tier', 'EMPIRICALLY_CALIBRATED')})")
    if chron:
        table.add_row(
            "Circadian Epistemic Drift",
            f"Night epistemic filter shift: {chron.get('epistemic_filter_decay_pct', 0)}%, TTR shift: {chron.get('lexical_diversity_decay_pct', 0)}%"
        )
    table.add_row("Characteristic Terms", top_terms or "empirical analysis")
    table.add_row("Academic Research Agent", f"{len(lit)} papers parsed • dynamic scientific_weights.json calibrated")

    console.print(table)
    console.print(f"[bold green]Learning phase completed.[/bold green] Next step: [cyan]who build-vault --target \"{target}\"[/cyan]\n")


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

    # 2. Render Cognitive Chronomics (Circadian Drift)
    chron_file = dirs.learned_skills_dir / "chronomics.json"
    chron_data = {}
    if chron_file.exists():
        try:
            with open(chron_file, "r", encoding="utf-8") as f:
                chron_data = json.load(f)
        except Exception:
            pass
    elif (dirs.learned_skills_dir / "features.json").exists():
        try:
            with open(dirs.learned_skills_dir / "features.json", "r", encoding="utf-8") as f:
                chron_data = json.load(f).get("chronomics", {})
        except Exception:
            pass

    if chron_data and chron_data.get("metrics_by_window"):
        c_table = Table(title=f"⏰ Cognitive Chronomics: Circadian Drift Analysis ({target})", header_style="bold yellow")
        c_table.add_column("Sirkadyen Pencere", style="bold white", width=18)
        c_table.add_column("Saat Aralığı", style="dim", width=16)
        c_table.add_column("Kelime Çeşitliliği (TTR)", style="green", width=22)
        c_table.add_column("Epistemik Kalibrasyon", style="cyan", width=22)
        c_table.add_column("Kesinlik / 1k", style="magenta", width=16)
        c_table.add_column("Eleştirel Ton / 1k", style="red", width=18)

        hour_labels = {
            "NIGHT": "00:00 - 05:00",
            "MORNING": "05:00 - 09:00",
            "DAY": "09:00 - 18:00",
            "EVENING": "18:00 - 24:00",
        }
        for win, metrics in chron_data.get("metrics_by_window", {}).items():
            if metrics.get("total_words", 0) > 0:
                c_table.add_row(
                    win,
                    hour_labels.get(win, ""),
                    str(metrics.get("type_token_ratio", 0.0)),
                    str(metrics.get("epistemic_calibration_ratio", 1.0)),
                    str(metrics.get("certainty_density", 0.0)),
                    str(metrics.get("critical_density", 0.0)),
                )
        console.print(c_table)

        if chron_data.get("probabilistic_statement"):
            console.print(Panel(
                f"[yellow]{chron_data.get('probabilistic_statement')}[/yellow]",
                title="[bold yellow]Sirkadyen Bilişsel Sapma Raporu[/bold yellow]",
                border_style="yellow"
            ))
        console.print()


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

    fe_file = dirs.learned_skills_dir / "features.json"
    features_data = {}
    if fe_file.exists():
        try:
            with open(fe_file, "r", encoding="utf-8") as f:
                features_data = json.load(f)
        except Exception:
            pass

    mp = MindsetProfiler(target, vault_dir=dirs.vault_dir)
    mp.build_profile(
        mapping_skill=skill_data,
        graph_metrics=graph_metrics,
        sample_quotes=sample_quotes[:15],
        features=features_data,
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

    # Render Cognitive Dissonance Clusters ([CONTRADICTS])
    cog_diss = metrics.get("cognitive_dissonance", {})
    diss_clusters = cog_diss.get("dissonance_clusters", [])
    if diss_clusters:
        d_table = Table(title=f"⚡ Graph-Driven Cognitive Dissonance Tensions ([CONTRADICTS]): {target}", header_style="bold red")
        d_table.add_column("Küme ID", width=10)
        d_table.add_column("Çelişen Kavramlar / Düğümler", style="bold white", width=34)
        d_table.add_column("Çelişki Skoru", style="red", width=16)
        d_table.add_column("Çelişki Açıklaması", style="dim")

        for cl in diss_clusters:
            d_table.add_row(
                str(cl.get("cluster_id")),
                " ⚡ ".join(cl.get("concepts", [])[:3]),
                f"%{int(cl.get('mean_dissonance_score', 0) * 100)}",
                cl.get("primary_conflict", ""),
            )
        console.print(d_table)

    if cog_diss.get("probabilistic_statement"):
        console.print(Panel(
            f"[bold red]{cog_diss.get('probabilistic_statement')}[/bold red]",
            title="[bold red]Bilişsel Çelişki Kümelenmesi & Gerilim Özeti[/bold red]",
            border_style="red"
        ))
    console.print()


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
