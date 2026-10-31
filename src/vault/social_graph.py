"""
Entity Resolution & Multi-Target Social Network Graph Engine for who??.
Performs cross-target linkage, detects shared affiliations and bridge/broker nodes,
and generates interactive standalone D3.js force-directed network visualizations.
Corresponds to Layer 7 (Relational Graph Memory).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("who.vault.social_graph")


class SocialGraphAnalyzer:
    """Constructs multi-target relational graphs, calculates bridge centrality, and renders visual link analysis."""

    def __init__(self, target_names: List[str], output_dir: Optional[Path] = None):
        self.target_names = [t.strip() for t in target_names if t.strip()]
        self.output_dir = output_dir or Path("profiles/social_graphs")

    def build_multi_target_network(self, export_html: bool = True) -> Dict[str, Any]:
        """Merge individual target knowledge graphs into a unified multi-target social network."""
        logger.info(f"Building unified social graph across {len(self.target_names)} targets: {self.target_names}")

        merged_nodes: Dict[str, Dict[str, Any]] = {}
        merged_edges: List[Dict[str, Any]] = []
        target_concept_map: Dict[str, Set[str]] = {}

        for t_name in self.target_names:
            dirs = settings.get_target_dirs(t_name)
            merged_nodes[t_name] = {
                "id": t_name,
                "label": t_name,
                "type": "TargetPerson",
                "group": 1,
                "radius": 18,
            }
            target_concept_map[t_name] = set()

            # Load concept network or graph store
            concept_file = dirs.vault_dir / "concept_network.json"
            if concept_file.exists():
                try:
                    with open(concept_file, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
                        raw_entities = c_data.get("nodes", []) or c_data.get("entities", [])
                        for entity in raw_entities:
                            eid = entity.get("name", "").strip()
                            if not eid:
                                continue
                            if eid not in merged_nodes:
                                merged_nodes[eid] = {
                                    "id": eid,
                                    "label": eid,
                                    "type": entity.get("type", "Concept"),
                                    "group": 2,
                                    "radius": 10,
                                }
                            target_concept_map[t_name].add(eid)

                        for rel in c_data.get("relationships", []):
                            src_node = rel.get("source")
                            tgt_node = rel.get("target")
                            if src_node and tgt_node:
                                if src_node not in merged_nodes:
                                    merged_nodes[src_node] = {"id": src_node, "label": src_node, "type": "Concept", "group": 2, "radius": 10}
                                if tgt_node not in merged_nodes:
                                    merged_nodes[tgt_node] = {"id": tgt_node, "label": tgt_node, "type": "Concept", "group": 2, "radius": 10}
                                merged_edges.append({
                                    "source": src_node,
                                    "target": tgt_node,
                                    "relation": rel.get("relation", "RELATES_TO"),
                                    "weight": rel.get("weight", 0.5),
                                    "quote": rel.get("quote", ""),
                                })
                except Exception as e:
                    logger.debug(f"Failed loading concept network for {t_name}: {e}")

        # Compute Cross-Target Shared Affiliations & Bridges
        shared_bridges: List[Dict[str, Any]] = []
        bridge_nodes: Dict[str, int] = {}

        if len(self.target_names) >= 2:
            all_concepts = set()
            for c_set in target_concept_map.values():
                all_concepts.update(c_set)

            for concept in all_concepts:
                connected_targets = [t for t, c_set in target_concept_map.items() if concept in c_set]
                if len(connected_targets) >= 2:
                    bridge_nodes[concept] = len(connected_targets)
                    shared_bridges.append({
                        "bridge_concept": concept,
                        "connected_targets": connected_targets,
                        "connection_type": "SHARED_AFFILIATION_OR_PRINCIPLE",
                    })
                    # Add explicit bridge edges
                    for i in range(len(connected_targets)):
                        for j in range(i + 1, len(connected_targets)):
                            merged_edges.append({
                                "source": connected_targets[i],
                                "target": connected_targets[j],
                                "relation": "SHARED_AFFILIATION",
                                "via": concept,
                                "weight": 0.8,
                            })

        # Calculate Betweenness Broker ranking
        broker_ranking = sorted(
            [{"node": k, "connected_target_count": v} for k, v in bridge_nodes.items()],
            key=lambda x: x["connected_target_count"],
            reverse=True,
        )

        network_payload = {
            "targets": self.target_names,
            "total_nodes": len(merged_nodes),
            "total_edges": len(merged_edges),
            "shared_bridges": shared_bridges,
            "broker_nodes_ranking": broker_ranking[:10],
            "nodes": list(merged_nodes.values()),
            "edges": merged_edges,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Generate D3 HTML Visualizer
        html_content = self._render_d3_html(network_payload) if export_html else None
        self._persist_network(network_payload, html_content)

        return network_payload

    def _render_d3_html(self, network_data: Dict[str, Any]) -> str:
        """Generate a self-contained, responsive D3.js force-directed graph HTML visualizer."""
        node_ids = {n["id"] for n in network_data["nodes"]}
        valid_links = [
            {"source": e["source"], "target": e["target"], "relation": e["relation"], "weight": e.get("weight", 0.5)}
            for e in network_data["edges"]
            if e.get("source") in node_ids and e.get("target") in node_ids
        ]

        data_json = json.dumps({
            "nodes": network_data["nodes"],
            "links": valid_links,
        }, ensure_ascii=False)

        targets_str = ", ".join(self.target_names)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>who?? Social Network Intelligence Graph: {targets_str}</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {{
      margin: 0;
      background-color: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
    }}
    #header {{
      position: absolute;
      top: 15px;
      left: 20px;
      z-index: 10;
      background: rgba(22, 27, 34, 0.85);
      backdrop-filter: blur(8px);
      padding: 12px 20px;
      border-radius: 8px;
      border: 1px solid #30363d;
    }}
    h1 {{ margin: 0 0 6px 0; font-size: 18px; color: #58a6ff; }}
    p {{ margin: 0; font-size: 12px; color: #8b949e; }}
    #tooltip {{
      position: absolute;
      visibility: hidden;
      background: #161b22;
      border: 1px solid #58a6ff;
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 12px;
      color: #f0f6fc;
      pointer-events: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    .link {{ stroke-opacity: 0.6; stroke-width: 1.5px; }}
    .link.contradicts {{ stroke: #ff7b72; stroke-dasharray: 4; }}
    .link.shared {{ stroke: #d2a8ff; stroke-width: 2.5px; }}
  </style>
</head>
<body>
  <div id="header">
    <h1>🌐 who?? Social Network Graph Intelligence</h1>
    <p>Targets: <strong>{targets_str}</strong> | Nodes: {network_data['total_nodes']} | Links: {network_data['total_edges']}</p>
    <p style="margin-top: 4px; font-size: 11px; color: #d2a8ff;">Purple edges represent Cross-Target Shared Affiliations & Broker Bridges</p>
  </div>
  <div id="tooltip"></div>
  <svg id="network-svg" width="100%" height="100vh"></svg>

  <script>
    const data = {data_json};
    const width = window.innerWidth;
    const height = window.innerHeight;

    const svg = d3.select("#network-svg");
    const container = svg.append("g");

    // Zoom and pan
    svg.call(d3.zoom().scaleExtent([0.1, 4]).on("zoom", (event) => {{
      container.attr("transform", event.transform);
    }}));

    const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id(d => d.id).distance(90))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => (d.radius || 10) + 12));

    const link = container.append("g")
      .selectAll("line")
      .data(data.links)
      .enter().append("line")
      .attr("class", d => "link " + (d.relation === "CONTRADICTS" ? "contradicts" : (d.relation === "SHARED_AFFILIATION" ? "shared" : "")))
      .attr("stroke", d => d.relation === "CONTRADICTS" ? "#ff7b72" : (d.relation === "SHARED_AFFILIATION" ? "#d2a8ff" : "#30363d"));

    const tooltip = d3.select("#tooltip");

    const node = container.append("g")
      .selectAll("g")
      .data(data.nodes)
      .enter().append("g")
      .call(d3.drag()
        .on("start", (event, d) => {{
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        }})
        .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
        .on("end", (event, d) => {{
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        }}));

    node.append("circle")
      .attr("r", d => d.radius || 10)
      .attr("fill", d => d.type === "TargetPerson" ? "#58a6ff" : (d.type === "Organization" ? "#7ee787" : "#ffa657"))
      .attr("stroke", "#ffffff")
      .attr("stroke-width", d => d.type === "TargetPerson" ? 2.5 : 1)
      .on("mouseover", (event, d) => {{
        tooltip.style("visibility", "visible")
          .html("<strong>" + d.label + "</strong><br>Type: " + d.type);
      }})
      .on("mousemove", (event) => {{
        tooltip.style("top", (event.pageY + 10) + "px").style("left", (event.pageX + 10) + "px");
      }})
      .on("mouseout", () => tooltip.style("visibility", "hidden"));

    node.append("text")
      .text(d => d.label)
      .attr("x", 14)
      .attr("y", 4)
      .attr("fill", "#c9d1d9")
      .attr("font-size", "11px");

    simulation.on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});
  </script>
</body>
</html>"""

    def _persist_network(self, network_data: Dict[str, Any], html_content: Optional[str] = None) -> None:
        """Persist JSON and interactive HTML report to social graphs directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = "_".join("".join(c if c.isalnum() else "" for c in t) for t in self.target_names[:3])
        out_json = self.output_dir / f"social_network_{safe_prefix}.json"
        out_html = self.output_dir / f"social_network_{safe_prefix}.html"

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(network_data, f, indent=2, ensure_ascii=False)

        if html_content is not None:
            with open(out_html, "w", encoding="utf-8") as f_html:
                f_html.write(html_content)
            logger.info(f"Saved Social Network JSON ({out_json}) and HTML visualizer ({out_html})")
        else:
            logger.info(f"Saved Social Network Intelligence JSON to {out_json}")



