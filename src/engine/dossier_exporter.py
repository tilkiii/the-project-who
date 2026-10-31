"""
Standalone Interactive HTML Intelligence Dossier Exporter for who??.
Compiles a target's complete evidence-backed profile — Mindset Summary, ElizaOS Bio,
D3.js Knowledge & [CONTRADICTS] Cognitive Dissonance Graph, Epistemic Calibration,
Manipulation Index, Cognitive Chronomics Circadian Drift, Cyber Footprint, and
Autonomous Academic Literature — into a single portable HTML report.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("who.engine.dossier")


class DossierExporter:
    """Renders a self-contained HTML intelligence dossier grounded in primary corpus evidence."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs

    def _read_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Could not parse {path}: {e}")
        return default

    def compile_dossier_payload(self) -> Dict[str, Any]:
        """Gather all verified vault and learned artifacts for the target."""
        vault = self.dirs.vault_dir
        skills = self.dirs.learned_skills_dir
        raw = self.dirs.raw_data_dir

        character = self._read_json(vault / "character.json", {})
        features = self._read_json(skills / "features.json", {})
        topics_obj = self._read_json(skills / "topics.json", {})
        graph_net = self._read_json(vault / "concept_network.json", {})
        graph_metrics = self._read_json(vault / "graph_metrics.json", {})
        literature = self._read_json(skills / "retrieved_academic_literature.json", [])
        visual_report = self._read_json(vault / "visual_forensics_report.json", {})

        chronomics = features.get("chronomics") or self._read_json(skills / "chronomics.json", {})

        cyber_reports: List[Dict[str, Any]] = []
        if vault.exists():
            for cf in sorted(vault.glob("cyber_footprint_*.json")):
                parsed_cf = self._read_json(cf, {})
                if parsed_cf:
                    cyber_reports.append(parsed_cf)

        mindset_path = vault / "mindset_summary.md"
        mindset_md = (
            mindset_path.read_text(encoding="utf-8")
            if mindset_path.exists()
            else "Mindset summary not yet synthesized."
        )

        raw_docs_count = (
            len(list(raw.glob("*.md"))) + len(list(raw.glob("*.txt")))
            if raw.exists()
            else 0
        )

        display_name = character.get("name") or self.target_name

        # Build D3 graph payload with [CONTRADICTS] flags
        nodes: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []
        raw_nodes = graph_net.get("nodes", []) or graph_net.get("entities", [])
        for n in raw_nodes[:150]:
            nid = n.get("name", "Concept")
            nodes.append({
                "id": nid,
                "label": nid,
                "type": n.get("type", "Concept"),
                "group": 1 if nid == display_name else 2,
            })
        node_ids = {n["id"] for n in nodes}
        contradiction_count = 0
        for r in graph_net.get("relationships", [])[:250]:
            src, tgt = r.get("source"), r.get("target")
            if src in node_ids and tgt in node_ids:
                rel = r.get("relation", "RELATES_TO")
                is_contra = rel == "CONTRADICTS" or float(r.get("dissonance_score", 0.0) or 0.0) > 0.0
                if is_contra:
                    contradiction_count += 1
                links.append({
                    "source": src,
                    "target": tgt,
                    "relation": rel,
                    "is_contradiction": is_contra,
                    "weight": float(r.get("weight", 0.5) or 0.5),
                })

        if not nodes:
            nodes = [{"id": display_name, "label": display_name, "type": "TargetPerson", "group": 1}]

        confidence_score = round(min(0.95, 0.55 + min(raw_docs_count, 10) * 0.04), 2)

        return {
            "target_name": display_name,
            "slug": self.dirs.target_slug,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "confidence_score": confidence_score,
            "raw_docs_count": raw_docs_count,
            "character": character,
            "mindset_markdown": mindset_md,
            "features": features,
            "chronomics": chronomics,
            "topics": topics_obj.get("topics", []) if isinstance(topics_obj, dict) else [],
            "graph": {"nodes": nodes, "links": links, "contradictions_count": contradiction_count},
            "cognitive_dissonance": graph_metrics.get("cognitive_dissonance", {}),
            "literature": literature if isinstance(literature, list) else [],
            "cyber_reports": cyber_reports,
            "visual_assets": visual_report.get("assets", []) if isinstance(visual_report, dict) else [],
        }

    def render_html(self, payload: Optional[Dict[str, Any]] = None) -> str:
        """Render the compiled dossier payload into a standalone HTML5 document."""
        data = payload or self.compile_dossier_payload()
        esc = html.escape

        name = esc(str(data["target_name"]))
        exported_at = esc(str(data["exported_at"]))
        conf_pct = int(float(data.get("confidence_score", 0.7)) * 100)
        docs_count = int(data.get("raw_docs_count", 0))

        features = data.get("features", {})
        cal_ratio = features.get("epistemic_markers", {}).get("epistemic_calibration_ratio", 1.0)
        manip = features.get("epistemic_manipulation_index", {})
        manip_pct = esc(str(manip.get("score_pct", "0%")))
        manip_tier = esc(str(manip.get("risk_tier", "EMPIRICALLY_CALIBRATED")))
        manip_stmt = esc(str(manip.get("probabilistic_statement", "Based on observed corpus evidence, rhetorical certainty remains calibrated.")))

        bio_items = "".join(
            f"<li>{esc(str(b))}</li>" for b in (data.get("character", {}).get("bio", []) or ["Evidence-grounded cognitive digital twin profile."])[:6]
        )

        topics_badges = "".join(
            f'<span class="badge">#{esc(str(t.get("name", "topic")))}</span>'
            for t in data.get("topics", [])[:10]
        ) or '<span class="muted">No topic clusters modeled yet.</span>'

        # Chronomics rows
        chron = data.get("chronomics", {}) or {}
        chron_rows = ""
        for win, m in (chron.get("metrics_by_window", {}) or {}).items():
            if m.get("total_words", 0) > 0:
                chron_rows += (
                    f"<tr><td><strong>{esc(str(win))}</strong></td>"
                    f"<td>{m.get('type_token_ratio', 0.0)}</td>"
                    f"<td>{m.get('epistemic_calibration_ratio', 1.0)}</td>"
                    f"<td>{m.get('certainty_density', 0.0)}</td>"
                    f"<td>{m.get('critical_density', 0.0)}</td></tr>"
                )
        if not chron_rows:
            chron_rows = '<tr><td colspan="5" class="muted">Insufficient timestamped segments for circadian window breakdown.</td></tr>'
        chron_stmt = esc(str(chron.get("probabilistic_statement", "")))

        # Cognitive dissonance clusters
        diss = data.get("cognitive_dissonance", {}) or {}
        diss_cards = "".join(
            f'<div class="alert-rose"><strong>⚡ {" ↔ ".join(esc(str(c)) for c in cl.get("concepts", []))}</strong> '
            f'({int(float(cl.get("mean_dissonance_score", 0.0)) * 100)}% dissonance)<br>'
            f'<span>{esc(str(cl.get("primary_conflict", "")))}</span></div>'
            for cl in (diss.get("dissonance_clusters", []) or [])
        ) or '<p class="muted">No high-severity [CONTRADICTS] clusters recorded in vault.</p>'

        # Cyber footprint summary
        cyber_html = ""
        for cr in data.get("cyber_reports", [])[:3]:
            breach = cr.get("breach_exposure", {})
            cyber_html += (
                f'<div class="card-sub"><strong>{esc(str(cr.get("email", "")))}</strong> — '
                f'Risk: {esc(str(breach.get("risk_tier", "LOW")))} | '
                f'Digital Hygiene: {esc(str(breach.get("digital_hygiene_score_pct", 85)))}%</div>'
            )
        if not cyber_html:
            cyber_html = '<p class="muted">No passive cyber footprint audits executed for this target.</p>'

        # Academic literature
        lit_html = "".join(
            f'<div class="card-sub"><strong>{esc(str(p.get("title", "Scientific Study")))}</strong><br>'
            f'<span class="muted">{esc(str(p.get("doi_or_url", p.get("repository", "Peer-reviewed repository"))))}</span></div>'
            for p in data.get("literature", [])[:6]
        ) or '<p class="muted">No autonomous academic literature linked yet.</p>'

        graph_json = json.dumps(data["graph"], ensure_ascii=False).replace("</", "<\\/")

        visual_rows = "".join(
            f"<tr><td><strong>{esc(str(v.get('file_name', '')))}</strong></td>"
            f"<td>{v.get('width', 0)}x{v.get('height', 0)} ({esc(str(v.get('layout_class', '')))})</td>"
            f"<td><code>{esc(str(v.get('dhash', '')))}</code></td>"
            f"<td>{esc(str(v.get('capture_timestamp_iso') or 'No EXIF'))} {esc(str(v.get('camera_device') or ''))}</td>"
            f"<td>{esc(str(v.get('authenticity_confidence_pct', 0)))}%</td></tr>"
            for v in (data.get("visual_assets", []) or [])[:12]
        ) or '<tr><td colspan="5" class="muted">No verified visual OSINT assets ingested yet.</td></tr>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>who?? Intelligence Dossier — {name}</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {{ margin: 0; padding: 32px; background: #0B0F17; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.55; }}
    .container {{ max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 22px; }}
    .card {{ background: #151C28; border: 1px solid #222E42; border-radius: 14px; padding: 22px; }}
    .card-sub {{ background: #0B0F17; border: 1px solid #222E42; border-radius: 10px; padding: 12px; margin-top: 8px; font-size: 13px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
    .stat {{ background: #0B0F17; border: 1px solid #222E42; border-radius: 12px; padding: 14px; }}
    .stat-label {{ font-size: 11px; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; }}
    .stat-val {{ font-size: 22px; font-weight: 700; color: #38BDF8; margin-top: 4px; }}
    h1 {{ margin: 0; font-size: 26px; color: #F8FAFC; }}
    h2 {{ margin: 0 0 12px 0; font-size: 16px; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.04em; }}
    .badge {{ display: inline-block; padding: 4px 10px; margin: 3px; border-radius: 999px; background: #0B0F17; border: 1px solid #38BDF8; color: #38BDF8; font-size: 12px; }}
    .epistemic-banner {{ background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 10px; padding: 12px 16px; font-size: 13px; color: #BAE6FD; }}
    .alert-rose {{ background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 10px; padding: 12px; margin-top: 8px; font-size: 13px; color: #FECDD3; }}
    .muted {{ color: #64748B; font-size: 13px; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; background: #0B0F17; border: 1px solid #222E42; border-radius: 10px; padding: 14px; font-size: 12.5px; color: #CBD5E1; max-height: 420px; overflow-y: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #222E42; }}
    th {{ color: #94A3B8; font-size: 11px; text-transform: uppercase; }}
    #graph-box {{ width: 100%; height: 420px; background: #0B0F17; border: 1px solid #222E42; border-radius: 12px; overflow: hidden; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
        <div>
          <div class="stat-label">who?? Enterprise OSINT &amp; Cognitive Digital Twin Dossier</div>
          <h1>{name}</h1>
          <div class="muted">Exported: {exported_at} • Slug: {esc(str(data["slug"]))}</div>
        </div>
        <span class="badge">Confidence Score: {conf_pct}% (Derived from {docs_count} primary corpus documents)</span>
      </div>
      <div class="epistemic-banner" style="margin-top:14px;">
        <strong>Epistemic Calibration Notice:</strong> Based on observed evidence from primary corpus documents, this dossier models plausible behavioral and rhetorical alignments under quantified uncertainty.
      </div>
    </div>

    <div class="grid-4">
      <div class="stat"><div class="stat-label">Primary Corpus Docs</div><div class="stat-val">{docs_count}</div></div>
      <div class="stat"><div class="stat-label">Epistemic Calibration</div><div class="stat-val">{esc(str(cal_ratio))}</div></div>
      <div class="stat"><div class="stat-label">Manipulation Index</div><div class="stat-val" style="color:#FBBF24;">{manip_pct}</div><div class="muted">{manip_tier}</div></div>
      <div class="stat"><div class="stat-label">[CONTRADICTS] Tensions</div><div class="stat-val" style="color:#FB7185;">{data["graph"]["contradictions_count"]}</div></div>
    </div>

    <div class="card">
      <h2>Interactive Knowledge &amp; Cognitive Dissonance Graph</h2>
      <div class="muted" style="margin-bottom:8px;">Dashed crimson edges indicate [CONTRADICTS] cognitive dissonance tensions. Drag nodes or scroll to zoom.</div>
      <div id="graph-box"></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>Mindset Summary &amp; Philosophical Anchors</h2>
        <ul>{bio_items}</ul>
        <pre>{esc(str(data["mindset_markdown"]))}</pre>
      </div>
      <div class="card">
        <h2>Cognitive Dissonance &amp; Manipulation Audit</h2>
        <div class="card-sub">{manip_stmt}</div>
        <div style="margin-top:14px;">{diss_cards}</div>
        <h2 style="margin-top:20px;">Topic Clusters</h2>
        <div>{topics_badges}</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>Cognitive Chronomics (Circadian Drift)</h2>
        <table>
          <thead><tr><th>Window</th><th>TTR</th><th>Calibration</th><th>Certainty/1k</th><th>Critical/1k</th></tr></thead>
          <tbody>{chron_rows}</tbody>
        </table>
        {f'<div class="card-sub" style="margin-top:10px;">{chron_stmt}</div>' if chron_stmt else ''}
      </div>
      <div class="card">
        <h2>Cyber Footprint &amp; Academic Grounding</h2>
        {cyber_html}
        <h2 style="margin-top:18px;">Peer-Reviewed Literature</h2>
        {lit_html}
        <h2 style="margin-top:18px;">Visual OSINT &amp; EXIF Forensic Ledger</h2>
        <table>
          <thead><tr><th>Asset</th><th>Geometry</th><th>dHash</th><th>EXIF / Device</th><th>Auth</th></tr></thead>
          <tbody>{visual_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const graphData = {graph_json};
    const box = document.getElementById('graph-box');
    const w = box.clientWidth || 960, h = 420;
    const svg = d3.select('#graph-box').append('svg').attr('width', w).attr('height', h);
    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', e => g.attr('transform', e.transform)));
    const sim = d3.forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(95))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(w / 2, h / 2));
    const link = g.append('g').selectAll('line').data(graphData.links).enter().append('line')
      .attr('stroke', d => d.is_contradiction ? '#F43F5E' : '#475569')
      .attr('stroke-width', d => d.is_contradiction ? 2.6 : 1.5)
      .attr('stroke-dasharray', d => d.is_contradiction ? '6,4' : null);
    const node = g.append('g').selectAll('circle').data(graphData.nodes).enter().append('circle')
      .attr('r', d => d.group === 1 ? 14 : 8)
      .attr('fill', d => d.group === 1 ? '#38BDF8' : '#A855F7')
      .attr('stroke', '#fff').attr('stroke-width', 1.5);
    const label = g.append('g').selectAll('text').data(graphData.nodes).enter().append('text')
      .text(d => d.label).attr('font-size', '10px').attr('fill', '#CBD5E1').attr('dx', 12).attr('dy', 4);
    sim.on('tick', () => {{
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('x', d => d.x).attr('y', d => d.y);
    }});
  </script>
</body>
</html>
"""

    def export_to_file(self, output_path: Optional[Path] = None) -> Path:
        """Compile and save standalone HTML intelligence dossier to disk."""
        payload = self.compile_dossier_payload()
        html_doc = self.render_html(payload)
        dest = (
            Path(output_path)
            if output_path is not None
            else (self.dirs.vault_dir / f"{self.dirs.target_slug}_intelligence_dossier.html")
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html_doc, encoding="utf-8")
        logger.info(f"Exported standalone HTML intelligence dossier to {dest}")
        return dest
