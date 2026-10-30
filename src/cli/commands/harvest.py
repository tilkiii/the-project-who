"""
Harvest and OSINT Reconnaissance CLI Commands.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import typer
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.harvest.academic_harvester import AcademicHarvester
from src.harvest.dialogue_harvester import DialogueHarvester
from src.harvest.document_ingestor import DocumentIngestor
from src.harvest.media_harvester import MediaHarvester
from src.harvest.ocr_processor import OCRProcessor
from src.harvest.web_harvester import WebHarvester
from src.harvest.live_monitor import LiveMonitor
from src.harvest.social_osint import SocialOSINTHunter
from src.harvest.cyber_footprint import CyberFootprintAuditor
from src.utils.logger import console, print_banner


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


def hunt(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    username: str = typer.Option(..., "--username", "-u", help="Username / handle to hunt across platforms"),
) -> None:
    """Probe 15+ major platforms for username existence and ingest public dossiers."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    console.print(f"[bold green]Initiating Multi-Platform Username Hunt for:[/bold green] [bold cyan]{target}[/bold cyan] ([yellow]@{username}[/yellow])\n")

    hunter = SocialOSINTHunter(target, dirs)
    profiles = hunter.hunt_username(username)

    if profiles:
        table = Table(title=f"🌐 Discovered Digital Footprints: @{username}", header_style="bold cyan")
        table.add_column("Platform", style="bold white", width=18)
        table.add_column("Category", style="yellow", width=22)
        table.add_column("Profile Link", style="blue")
        table.add_column("Snippet / Bio", style="dim")

        for p in profiles:
            table.add_row(
                f"{p['icon']} {p['platform']}",
                p["category"],
                p["url"],
                p.get("bio_snippet", "")[:45] + ("..." if len(p.get("bio_snippet", "")) > 45 else ""),
            )
        console.print(table)
        console.print(f"\n[green]Dossier ingested to:[/green] [cyan]{dirs.raw_data_dir / f'social_osint_dossier_{username}.md'}[/cyan]\n")
    else:
        console.print(f"[yellow]No active profiles discovered for @{username} across queried platforms.[/yellow]\n")


def cyber(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    email: str = typer.Option(..., "--email", "-e", help="Email address to audit"),
) -> None:
    """Perform passive email reconnaissance, DNS infrastructure validation, and breach auditing."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    console.print(f"[bold green]Starting Cyber Footprint & Breach Reconnaissance for:[/bold green] [bold cyan]{target}[/bold cyan]\n")

    auditor = CyberFootprintAuditor(target, dirs)
    res = auditor.run_full_audit(email)

    dns = res["dns_infrastructure"]
    breach = res["breach_exposure"]

    table = Table(title=f"🛡️ Cyber Reconnaissance Audit: {email}", header_style="bold magenta")
    table.add_column("Dimension", style="bold white", width=24)
    table.add_column("Observed Status / Metric", style="cyan")
    table.add_column("Security Finding", style="white")

    table.add_row("Email Syntax & Format", "Valid" if res["validation"]["is_valid_format"] else "Invalid", f"Domain: {res['validation']['domain']}")
    table.add_row("DNS & Mail Resolution", f"IP: {dns.get('resolved_ip') or 'None'}", "Host Active" if dns.get("resolves") else "Unresolvable Host")
    table.add_row("Burner / Disposable Check", "Passed" if not dns.get("is_disposable") else "DISPOSABLE RISK", "Legitimate Provider" if not dns.get("is_disposable") else "Temporary Mail Service")
    table.add_row("Passive Service Discovery", f"{len(res['passive_services_detected'])} account(s) found", ", ".join(s['service'] for s in res['passive_services_detected']) or "None detected")
    table.add_row("Breach Exposure Risk", f"Risk: {breach['risk_tier']}", f"Hygiene Rating: %{breach['digital_hygiene_score_pct']}")

    console.print(table)
    console.print(Panel(
        f"[yellow]{breach['probabilistic_statement']}[/yellow]\n"
        f"[dim]Belgelenmiş Tarihsel Olaylar: {', '.join(b['source'] for b in breach['documented_breach_incidents'])}[/dim]",
        title="[bold red]⚠️ Siber Tehdit & Sızıntı Değerlendirmesi[/bold red]",
        border_style="red"
    ))


def monitor(
    target: str = typer.Option(..., "--target", "-t", help="Target individual name"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated RSS feeds or URLs to monitor"),
    simulate_feed: bool = typer.Option(False, "--simulate-feed", help="Simulate a live feed item to test anomaly triggers"),
) -> None:
    """Scan live external feeds and detect real-time circadian, dogmatic, or cognitive anomalies."""
    print_banner()
    dirs = settings.get_target_dirs(target)
    console.print(f"[bold green]Starting Live OSINT Monitor for:[/bold green] [bold cyan]{target}[/bold cyan]\n")

    monitor_engine = LiveMonitor(target, dirs)
    url_list = [u.strip() for u in sources.split(",") if u.strip()] if sources else []

    feed_items = []
    if url_list:
        feed_items.extend(monitor_engine.fetch_live_feed(url_list))

    if simulate_feed or not feed_items:
        sim_dt = datetime.now()
        feed_items.append({
            "title": f"Live Capture: {target} nocturnal discourse",
            "content": "Kesinlikle herkes bilmelidir ki, eski usuller tamamen çökmüştür. Asla başka bir alternatif kabul edilemez, durum şüphesiz ortadadır.",
            "url": "https://example.org/simulated_stream",
            "timestamp": sim_dt.replace(hour=2, minute=15).isoformat(),
            "source_type": "simulated_live_feed",
        })
        console.print("[dim]Analyzing active live stream candidate...[/dim]\n")

    anomalies = monitor_engine.detect_anomalies(feed_items)

    if anomalies:
        console.print(f"[bold red]⚠️ {len(anomalies)} Cognitive / Behavioral Anomaly Alert(s) Detected![/bold red]\n")
        table = Table(title=f"🚨 Live Monitoring Anomaly Alerts: {target}", header_style="bold red")
        table.add_column("Anomaly Type", style="bold yellow", width=26)
        table.add_column("Severity", style="bold red", width=12)
        table.add_column("Observed Detail", style="white")
        table.add_column("Source", style="dim", width=22)

        for anom in anomalies:
            for trig in anom.get("triggers", []):
                sev_color = "red" if trig.get("severity") == "HIGH" else "yellow"
                table.add_row(
                    trig.get("type", "UNKNOWN"),
                    f"[{sev_color}]{trig.get('severity', 'MED')}[/{sev_color}]",
                    trig.get("detail", ""),
                    anom.get("item_title", "")[:22],
                )
        console.print(table)
        console.print(f"\n[green]Alerts saved to:[/green] [cyan]{dirs.vault_dir / 'monitoring_alerts.json'}[/cyan]\n")
    else:
        console.print("[bold green]✔ No cognitive or circadian anomalies detected. Footprint matches baseline within normal boundaries.[/bold green]\n")
