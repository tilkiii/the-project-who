"""
Interactive Web Dashboard 2.0 & Visualization Server for who?? Cognitive Digital Twin Framework.
Runs a zero-dependency, multi-threaded HTTP server providing REST APIs and an ultra-modern D3.js web interface
with Twin Chat, Scenario Simulator, D3 [CONTRADICTS] Cognitive Dissonance Graph, Oracle Q&A,
Adversarial Outreach Composer, and Multi-Agent Swarm Arena Debate Room.
"""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import urllib.parse
from typing import Any, Dict, List, Optional
import webbrowser

from src.config import settings
from src.utils.logger import get_logger
from src.utils.text_cleaner import sanitize_slug

logger = get_logger("who.ui.server")


def get_all_target_names() -> List[str]:
    """Scan profiles directory for active target profiles."""
    profiles_dir = Path(settings.profiles_dir)
    if not profiles_dir.exists():
        return []
    targets = []
    for item in profiles_dir.iterdir():
        if item.is_dir() and item.name not in ("social_graphs", "arena_debates", "__pycache__"):
            targets.append(item.name)
    return sorted(targets)


def get_target_profile_summary(target_slug: str) -> Dict[str, Any]:
    """Compile structured summary data for a given target profile."""
    target_dir = Path(settings.profiles_dir) / target_slug
    vault_dir = target_dir / "vault"
    skills_dir = target_dir / "learned_skills"
    raw_dir = target_dir / "raw_data"

    char_file = vault_dir / "character.json"
    mindset_file = vault_dir / "mindset_summary.md"
    metrics_file = vault_dir / "graph_metrics.json"
    features_file = skills_dir / "features.json"
    topics_file = skills_dir / "topics.json"
    lit_file = skills_dir / "retrieved_academic_literature.json"
    weights_file = skills_dir / "scientific_weights.json"

    character_data: Dict[str, Any] = {}
    if char_file.exists():
        try:
            with open(char_file, "r", encoding="utf-8") as f:
                character_data = json.load(f)
        except Exception:
            pass

    features_data: Dict[str, Any] = {}
    if features_file.exists():
        try:
            with open(features_file, "r", encoding="utf-8") as f:
                features_data = json.load(f)
        except Exception:
            pass

    graph_metrics: Dict[str, Any] = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                graph_metrics = json.load(f)
        except Exception:
            pass

    topics_data: List[Dict[str, Any]] = []
    if topics_file.exists():
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                topics_data = json.load(f).get("topics", [])
        except Exception:
            pass

    literature_data: List[Dict[str, Any]] = []
    if lit_file.exists():
        try:
            with open(lit_file, "r", encoding="utf-8") as f:
                literature_data = json.load(f)
        except Exception:
            pass

    mindset_md = mindset_file.read_text(encoding="utf-8") if mindset_file.exists() else "Mindset profile not synthesized yet."
    raw_docs_count = len(list(raw_dir.glob("*.md"))) + len(list(raw_dir.glob("*.txt"))) if raw_dir.exists() else 0

    return {
        "slug": target_slug,
        "name": character_data.get("name") or target_slug.replace("_", " ").title(),
        "bio": character_data.get("bio", []),
        "raw_docs_count": raw_docs_count,
        "mindset_ready": mindset_file.exists(),
        "mindset_markdown": mindset_md[:2500],
        "features": features_data,
        "cognitive_dissonance": graph_metrics.get("cognitive_dissonance", {}),
        "topics": topics_data[:8],
        "academic_literature_count": len(literature_data),
        "academic_literature_sample": literature_data[:4],
        "has_weights": weights_file.exists(),
    }


def get_target_graph_data(target_slug: str) -> Dict[str, Any]:
    """Fetch Knowledge Graph nodes and edges (including [CONTRADICTS] dissonance metadata) for D3.js visualization."""
    target_dir = Path(settings.profiles_dir) / target_slug
    concept_file = target_dir / "vault" / "concept_network.json"

    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    contradiction_nodes: Dict[str, int] = {}
    contradictions_count = 0

    if concept_file.exists():
        try:
            with open(concept_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_nodes = data.get("nodes", []) or data.get("entities", [])
                for n in raw_nodes[:150]:
                    node_name = n.get("name", "Concept")
                    nodes.append({
                        "id": node_name,
                        "label": node_name,
                        "type": n.get("type", "Concept"),
                        "description": n.get("description", ""),
                        "group": 1 if node_name == data.get("target_name") else 2,
                        "has_contradiction": False,
                        "contradiction_count": 0,
                    })
                node_ids = {n["id"] for n in nodes}
                for r in data.get("relationships", [])[:250]:
                    src = r.get("source")
                    tgt = r.get("target")
                    if src in node_ids and tgt in node_ids:
                        rel_type = r.get("relation", "RELATES_TO")
                        is_contradiction = (rel_type == "CONTRADICTS") or float(r.get("dissonance_score", 0.0) or 0.0) > 0.0
                        if is_contradiction:
                            contradictions_count += 1
                            contradiction_nodes[src] = contradiction_nodes.get(src, 0) + 1
                            contradiction_nodes[tgt] = contradiction_nodes.get(tgt, 0) + 1

                        links.append({
                            "source": src,
                            "target": tgt,
                            "relation": rel_type,
                            "weight": float(r.get("weight", 0.5) or 0.5),
                            "quote": r.get("quote", ""),
                            "is_contradiction": is_contradiction,
                            "dissonance_score": float(r.get("dissonance_score", 0.85 if rel_type == "CONTRADICTS" else 0.0) or 0.0),
                            "contradiction_summary": r.get("contradiction_summary", ""),
                            "quote_a": r.get("quote_a", ""),
                            "quote_b": r.get("quote_b", ""),
                        })

                for n in nodes:
                    c_count = contradiction_nodes.get(n["id"], 0)
                    if c_count > 0:
                        n["has_contradiction"] = True
                        n["contradiction_count"] = c_count
        except Exception as e:
            logger.debug(f"Graph read note: {e}")

    if not nodes:
        t_name = target_slug.replace("_", " ").title()
        nodes = [{
            "id": t_name,
            "label": t_name,
            "type": "TargetPerson",
            "description": "Primary digital twin target",
            "group": 1,
            "has_contradiction": False,
            "contradiction_count": 0,
        }]

    return {
        "nodes": nodes,
        "links": links,
        "contradictions_count": contradictions_count,
    }


def _resolve_target_display_name(target_ref: str) -> str:
    """Resolve human-readable target name from slug or raw target input."""
    slug = sanitize_slug(target_ref)
    char_file = Path(settings.profiles_dir) / slug / "vault" / "character.json"
    if char_file.exists():
        try:
            data = json.loads(char_file.read_text(encoding="utf-8"))
            if data.get("name"):
                return str(data["name"])
        except Exception:
            pass
    if "_" in target_ref and target_ref == slug:
        return target_ref.replace("_", " ").title()
    return target_ref


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>who?? Cognitive Digital Twin Dashboard 2.0</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 400: '#38bdf8', 500: '#0ea5e9', 600: '#0284c7', 900: '#0c4a6e' },
            darkbg: '#0B0F17',
            darkpanel: '#151C28',
            darkborder: '#222E42'
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #0B0F17; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .glass-card { background: rgba(21, 28, 40, 0.85); backdrop-filter: blur(12px); border: 1px solid #222E42; }
    .node-target { fill: #38bdf8; stroke: #fff; stroke-width: 2.5px; }
    .node-concept { fill: #a855f7; stroke: #222E42; stroke-width: 1.5px; }
    .node-dissonance { fill: #fb7185; stroke: #fda4af; stroke-width: 2.5px; filter: drop-shadow(0 0 6px rgba(244, 63, 94, 0.7)); }
    .link-line { stroke: #475569; stroke-opacity: 0.65; }
    .link-contradicts { stroke: #f43f5e; stroke-opacity: 0.95; stroke-dasharray: 6, 4; filter: drop-shadow(0 0 4px rgba(244, 63, 94, 0.55)); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- Navigation Bar -->
  <header class="border-b border-darkborder bg-darkpanel/80 backdrop-blur sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center font-black text-white text-lg shadow-lg shadow-brand-500/20">
        ?
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <h1 class="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-400 via-indigo-400 to-purple-400">
            who?? Digital Twin Console
          </h1>
          <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-brand-600/20 text-brand-400 border border-brand-500/30">v2.0</span>
        </div>
        <p class="text-xs text-slate-400">Cognitive Modeling • Epistemic Grounding • Dissonance Graph • Swarm Arena</p>
      </div>
    </div>

    <!-- Active Target Selector -->
    <div class="flex items-center space-x-3">
      <span id="providerBadge" class="hidden md:inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg bg-darkbg border border-darkborder text-slate-400">
        <i class="fa-solid fa-microchip text-emerald-400"></i> <span id="providerName">provider</span>
      </span>
      <div class="flex items-center space-x-2 bg-darkbg px-3 py-1.5 rounded-lg border border-darkborder">
        <i class="fa-solid fa-user-astronaut text-brand-400 text-sm"></i>
        <span class="text-xs text-slate-400">Target:</span>
        <select id="targetSelector" onchange="switchTarget()" class="bg-transparent text-sm font-semibold text-slate-200 outline-none cursor-pointer">
          <option value="" disabled selected>Loading...</option>
        </select>
      </div>
      <button onclick="exportCurrentDossier()" title="Export Standalone HTML Dossier" class="px-3 py-1.5 rounded-lg bg-brand-600/20 border border-brand-500/40 text-brand-400 hover:bg-brand-600 hover:text-white text-xs font-semibold transition flex items-center space-x-1.5">
        <i class="fa-solid fa-file-export"></i> <span class="hidden sm:inline">Export Dossier</span>
      </button>
      <button onclick="refreshData()" title="Refresh Profile Data" class="p-2 rounded-lg bg-darkbg border border-darkborder text-slate-400 hover:text-white transition">
        <i class="fa-solid fa-rotate-right"></i>
      </button>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="bg-darkpanel/40 border-b border-darkborder px-6 flex overflow-x-auto space-x-6 text-sm font-medium">
    <button onclick="showTab('chat')" id="tab-chat" class="py-3 border-b-2 border-brand-500 text-brand-400 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-regular fa-comment-dots"></i> <span>Twin Chat & Simulation</span>
    </button>
    <button onclick="showTab('graph')" id="tab-graph" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-solid fa-diagram-project"></i> <span>D3 Knowledge Graph</span>
      <span id="navDissonanceBadge" class="hidden text-[10px] px-1.5 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold">0 ⚡</span>
    </button>
    <button onclick="showTab('oracle')" id="tab-oracle" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-solid fa-eye"></i> <span>Oracle Evidence Q&A</span>
    </button>
    <button onclick="showTab('composer')" id="tab-composer" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-solid fa-feather-pointed"></i> <span>Outreach Composer</span>
    </button>
    <button onclick="showTab('arena')" id="tab-arena" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-solid fa-people-arrows"></i> <span>Swarm Arena</span>
    </button>
    <button onclick="showTab('mindset')" id="tab-mindset" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-solid fa-brain"></i> <span>Mindset & Epistemics</span>
    </button>
    <button onclick="showTab('targets')" id="tab-targets" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition">
      <i class="fa-solid fa-users-viewfinder"></i> <span>Targets Directory</span>
    </button>
  </div>

  <!-- Content Container -->
  <main class="flex-1 p-6 max-w-7xl w-full mx-auto">

    <!-- TAB 1: TWIN CHAT & SCENARIO SIMULATION -->
    <div id="content-chat" class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Chat Box -->
      <div class="lg:col-span-2 glass-card rounded-2xl p-5 flex flex-col h-[650px] shadow-xl">
        <div class="flex items-center justify-between pb-4 border-b border-darkborder mb-4">
          <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-full bg-slate-800 border border-brand-500/50 flex items-center justify-center font-bold text-brand-400" id="avatarName">
              ?
            </div>
            <div>
              <h2 class="font-bold text-slate-100" id="twinDisplayName">Digital Twin</h2>
              <span class="text-xs text-emerald-400 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Grounded in primary corpus</span>
            </div>
          </div>
          <!-- Tone Selector -->
          <div class="flex items-center space-x-2 bg-darkbg p-1 rounded-lg border border-darkborder text-xs">
            <button id="tone-autodetect" onclick="setTone('autodetect')" class="px-2.5 py-1 rounded bg-brand-600 text-white font-medium">Auto</button>
            <button id="tone-formal" onclick="setTone('formal')" class="px-2.5 py-1 rounded text-slate-400 hover:text-white">Formal Keynote</button>
            <button id="tone-informal" onclick="setTone('informal')" class="px-2.5 py-1 rounded text-slate-400 hover:text-white">1-on-1 Dialogue</button>
          </div>
        </div>

        <!-- Message Feed -->
        <div id="chatFeed" class="flex-1 overflow-y-auto space-y-4 pr-2 text-sm">
          <div class="bg-darkbg/60 border border-darkborder/60 p-4 rounded-xl text-slate-300">
            👋 Hello! I am the calibrated cognitive digital twin of <strong class="text-brand-400" id="welcomeTarget">Target</strong>.
            Ask me about my philosophies, decision principles, or test hypothetical scenarios!
          </div>
        </div>

        <!-- Chat Input -->
        <div class="pt-4 mt-2 border-t border-darkborder flex space-x-3">
          <input type="text" id="chatInput" placeholder="Ask a question or propose a collaboration..." onkeydown="if(event.key==='Enter') sendChatMessage()"
                 class="flex-1 bg-darkbg border border-darkborder rounded-xl px-4 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-brand-500 transition">
          <button onclick="sendChatMessage()" class="bg-brand-600 hover:bg-brand-500 text-white font-semibold px-5 py-2.5 rounded-xl text-sm transition flex items-center space-x-2">
            <span>Send</span> <i class="fa-solid fa-paper-plane text-xs"></i>
          </button>
        </div>
      </div>

      <!-- Quick Scenario Simulator & Core Signals -->
      <div class="space-y-6">
        <div class="glass-card rounded-2xl p-5 shadow-xl">
          <h3 class="font-bold text-slate-100 flex items-center space-x-2 mb-3">
            <i class="fa-solid fa-chess-knight text-purple-400"></i>
            <span>Plausible Reaction Simulator</span>
          </h3>
          <p class="text-xs text-slate-400 mb-3">Simulate target reaction under explicit assumptions:</p>
          <textarea id="scenarioInput" rows="3" placeholder="e.g. A consortium asks target to endorse a closed-source protocol..."
                    class="w-full bg-darkbg border border-darkborder rounded-xl p-3 text-xs text-slate-200 outline-none focus:border-purple-400 transition mb-3"></textarea>
          <button onclick="runSimulation()" class="w-full bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold py-2.5 rounded-xl transition flex items-center justify-center space-x-2">
            <i class="fa-solid fa-wand-magic-sparkles"></i> <span>Simulate Reaction</span>
          </button>
          <div id="simulationOutput" class="hidden mt-3 p-3 bg-darkbg/80 border border-purple-500/30 rounded-xl text-xs text-slate-300 space-y-2 max-h-64 overflow-y-auto"></div>
        </div>

        <!-- Mindset Summary Card -->
        <div class="glass-card rounded-2xl p-5 shadow-xl">
          <h3 class="font-bold text-slate-100 flex items-center space-x-2 mb-2">
            <i class="fa-solid fa-fingerprint text-brand-400"></i>
            <span>Twin Core Signals</span>
          </h3>
          <div class="space-y-3 text-xs mt-3">
            <div class="flex justify-between border-b border-darkborder/50 pb-2">
              <span class="text-slate-400">Primary Corpus Documents:</span>
              <span class="font-semibold text-slate-200" id="statDocs">0</span>
            </div>
            <div class="flex justify-between border-b border-darkborder/50 pb-2">
              <span class="text-slate-400">Peer-Reviewed Papers Linked:</span>
              <span class="font-semibold text-brand-400" id="statPapers">0</span>
            </div>
            <div class="flex justify-between border-b border-darkborder/50 pb-2">
              <span class="text-slate-400">Epistemic Calibration:</span>
              <span class="font-semibold text-emerald-400" id="statCalibration">Calibrated</span>
            </div>
            <div class="flex justify-between border-b border-darkborder/50 pb-2">
              <span class="text-slate-400">Manipulation Risk Index:</span>
              <span class="font-semibold text-amber-400" id="statManipulation">0%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: D3 KNOWLEDGE GRAPH WITH [CONTRADICTS] COGNITIVE DISSONANCE VISUALIZATION -->
    <div id="content-graph" class="hidden glass-card rounded-2xl p-5 shadow-xl flex flex-col h-[760px]">
      <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-darkborder mb-3">
        <div>
          <h2 class="font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-circle-nodes text-brand-400"></i>
            <span>Interactive Knowledge & Cognitive Dissonance Graph</span>
          </h2>
          <p class="text-xs text-slate-400">Click or hover nodes & dashed crimson <code class="text-rose-400">[CONTRADICTS]</code> edges to inspect epistemic tensions and primary quotes</p>
        </div>
        <div class="flex flex-wrap items-center gap-4 text-xs text-slate-300">
          <label class="flex items-center gap-2 bg-darkbg px-3 py-1.5 rounded-lg border border-rose-500/30 cursor-pointer">
            <input type="checkbox" id="filterContradictionsOnly" onchange="renderD3Graph()" class="accent-rose-500">
            <span class="text-rose-300 font-medium">⚡ Only [CONTRADICTS] Tensions</span>
          </label>
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-brand-400 inline-block"></span> Target</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-purple-500 inline-block"></span> Concept</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-rose-400 inline-block shadow shadow-rose-500"></span> Dissonance Node</span>
          <span class="flex items-center gap-1.5"><span class="w-5 border-t-2 border-dashed border-rose-500 inline-block"></span> [CONTRADICTS]</span>
        </div>
      </div>

      <div class="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-4 min-h-0">
        <div id="d3Container" class="lg:col-span-3 relative overflow-hidden rounded-xl bg-darkbg/90 border border-darkborder"></div>
        <div class="glass-card rounded-xl p-4 flex flex-col justify-between overflow-y-auto">
          <div>
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center justify-between">
              <span>Graph Evidence Inspector</span>
              <span id="graphStatsBadge" class="text-[11px] text-brand-400 font-mono">0N / 0E</span>
            </h3>
            <div id="graphInspector" class="text-xs text-slate-300 space-y-2.5 bg-darkbg/70 p-3 rounded-lg border border-darkborder">
              <p class="text-slate-400">Select any node or relationship edge in the force-directed graph to inspect primary evidence quotes and cognitive dissonance scores.</p>
            </div>
          </div>
          <div id="dissonanceQuickList" class="mt-4 space-y-2 text-xs"></div>
        </div>
      </div>
    </div>

    <!-- TAB 3: ORACLE EVIDENCE Q&A -->
    <div id="content-oracle" class="hidden grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2 glass-card rounded-2xl p-6 shadow-xl space-y-5">
        <div>
          <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-eye text-brand-400"></i>
            <span>Oracle Analytical OSINT Engine</span>
          </h2>
          <p class="text-xs text-slate-400 mt-1">Query the target's vector memory and knowledge graph with calibrated probabilistic confidence scoring (Epistemic Rule #3).</p>
        </div>

        <div class="flex space-x-3">
          <input type="text" id="oracleQueryInput" placeholder="e.g. What are the target's core arguments regarding institutional autonomy vs centralization?"
                 onkeydown="if(event.key==='Enter') runOracleQuery()"
                 class="flex-1 bg-darkbg border border-darkborder rounded-xl px-4 py-3 text-sm text-slate-100 focus:outline-none focus:border-brand-500 transition">
          <button onclick="runOracleQuery()" class="bg-brand-600 hover:bg-brand-500 text-white font-semibold px-6 py-3 rounded-xl text-sm transition flex items-center space-x-2">
            <i class="fa-solid fa-magnifying-glass-chart"></i> <span>Consult Oracle</span>
          </button>
        </div>

        <div id="oracleAnswerPanel" class="bg-darkbg/70 border border-darkborder rounded-xl p-5 space-y-4">
          <p class="text-xs text-slate-400">Submit an analytical question above to synthesize an evidence-backed probabilistic assessment.</p>
        </div>
      </div>

      <div class="space-y-6">
        <div class="glass-card rounded-2xl p-5 shadow-xl space-y-3">
          <h3 class="text-sm font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-scale-balanced text-emerald-400"></i>
            <span>Epistemic Confidence Meter</span>
          </h3>
          <div class="w-full bg-darkbg rounded-full h-3 border border-darkborder overflow-hidden">
            <div id="oracleConfidenceBar" class="bg-gradient-to-r from-brand-500 to-emerald-400 h-full transition-all duration-500" style="width: 0%"></div>
          </div>
          <div class="flex justify-between text-xs">
            <span class="text-slate-400">Calibrated Score:</span>
            <span id="oracleConfidenceLabel" class="font-bold text-emerald-400">—</span>
          </div>
        </div>

        <div class="glass-card rounded-2xl p-5 shadow-xl space-y-3">
          <h3 class="text-sm font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-quote-left text-indigo-400"></i>
            <span>Primary Supporting Quotes & Graph Paths</span>
          </h3>
          <div id="oracleEvidenceList" class="space-y-2 text-xs text-slate-300 max-h-[380px] overflow-y-auto">
            <p class="text-slate-500">No query executed yet.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: ADVERSARIAL OUTREACH COMPOSER -->
    <div id="content-composer" class="hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2">
              <i class="fa-solid fa-feather-pointed text-indigo-400"></i>
              <span>Adversarial Outreach & Persuasion Composer</span>
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">Drafts a tailored proposal aligned with target rhetoric, stress-tests it against simulated rejection reflexes, and outputs a hardened final pitch.</p>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label class="block text-xs font-semibold text-slate-300 mb-1.5">Communication Goal *</label>
            <input type="text" id="composerGoalInput" placeholder="e.g. Invite target as keynote speaker on verifiable governance protocols"
                   class="w-full bg-darkbg border border-darkborder rounded-xl px-4 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-400 transition">
          </div>
          <div>
            <label class="block text-xs font-semibold text-slate-300 mb-1.5">Additional Context (Optional)</label>
            <input type="text" id="composerContextInput" placeholder="e.g. Academic symposium in Istanbul, peer-reviewed audience, zero commercial sponsors"
                   class="w-full bg-darkbg border border-darkborder rounded-xl px-4 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-400 transition">
          </div>
        </div>

        <div class="flex items-center justify-between">
          <div id="composerTermsRow" class="flex flex-wrap gap-1.5 text-xs"></div>
          <button onclick="runComposer()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-2.5 rounded-xl text-sm transition flex items-center space-x-2">
            <i class="fa-solid fa-bolt"></i> <span>Synthesize & Stress-Test Pitch</span>
          </button>
        </div>
      </div>

      <div id="composerResultsGrid" class="hidden grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="glass-card rounded-2xl p-5 shadow-xl flex flex-col">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
            <span class="w-5 h-5 rounded-full bg-slate-700 text-slate-200 inline-flex items-center justify-center text-[10px]">1</span>
            <span>Initial Calibrated Draft</span>
          </h3>
          <div id="composerInitialDraft" class="flex-1 text-xs leading-relaxed text-slate-300 whitespace-pre-wrap bg-darkbg/70 p-4 rounded-xl border border-darkborder max-h-96 overflow-y-auto"></div>
        </div>

        <div class="glass-card rounded-2xl p-5 shadow-xl flex flex-col border-rose-500/30">
          <h3 class="text-xs font-bold uppercase tracking-wider text-rose-400 mb-3 flex items-center gap-2">
            <span class="w-5 h-5 rounded-full bg-rose-500/20 text-rose-300 inline-flex items-center justify-center text-[10px]">2</span>
            <span>Simulated Rejection & Skepticism Critique</span>
          </h3>
          <div id="composerCritique" class="flex-1 text-xs leading-relaxed text-rose-100/90 whitespace-pre-wrap bg-rose-950/20 p-4 rounded-xl border border-rose-500/30 max-h-96 overflow-y-auto"></div>
        </div>

        <div class="glass-card rounded-2xl p-5 shadow-xl flex flex-col border-emerald-500/30">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-2">
              <span class="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-300 inline-flex items-center justify-center text-[10px]">3</span>
              <span>Hardened Final Pitch</span>
            </h3>
            <button onclick="copyFinalPitch()" class="text-[11px] px-2.5 py-1 rounded bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 border border-emerald-500/30 transition">
              <i class="fa-regular fa-copy"></i> Copy
            </button>
          </div>
          <div id="composerFinalPitch" class="flex-1 text-xs leading-relaxed text-emerald-50 whitespace-pre-wrap bg-emerald-950/20 p-4 rounded-xl border border-emerald-500/30 max-h-96 overflow-y-auto"></div>
        </div>
      </div>
    </div>

    <!-- TAB 5: MULTI-AGENT SWARM ARENA DEBATE ROOM -->
    <div id="content-arena" class="hidden grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="glass-card rounded-2xl p-6 shadow-xl space-y-5 h-fit">
        <div>
          <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-people-arrows text-amber-400"></i>
            <span>Swarm Arena Configuration</span>
          </h2>
          <p class="text-xs text-slate-400 mt-1">Convene 2 or more digital twins into a structured multi-round intellectual debate.</p>
        </div>

        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-2">Select Debate Participants (Min 2) *</label>
          <div id="arenaParticipantsList" class="space-y-2 max-h-48 overflow-y-auto bg-darkbg/70 p-3 rounded-xl border border-darkborder text-xs"></div>
        </div>

        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Debate Proposition / Topic *</label>
          <textarea id="arenaTopicInput" rows="3" placeholder="e.g. Akademik otonomi ile merkezi kurumsal yönetim arasındaki denge nasıl kurulmalıdır?"
                    class="w-full bg-darkbg border border-darkborder rounded-xl p-3 text-xs text-slate-100 focus:outline-none focus:border-amber-400 transition"></textarea>
        </div>

        <div class="flex items-center justify-between">
          <span class="text-xs text-slate-300 font-semibold">Debate Rounds:</span>
          <select id="arenaRoundsSelect" class="bg-darkbg border border-darkborder rounded-lg px-3 py-1.5 text-xs text-slate-200 outline-none">
            <option value="1">1 Round (Fast)</option>
            <option value="2" selected>2 Rounds (Standard)</option>
            <option value="3">3 Rounds (Deep)</option>
          </select>
        </div>

        <button onclick="runArenaDebate()" class="w-full bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white font-semibold py-3 rounded-xl text-sm transition flex items-center justify-center space-x-2 shadow-lg shadow-amber-600/20">
          <i class="fa-solid fa-gavel"></i> <span>Launch Swarm Arena Debate</span>
        </button>
      </div>

      <div class="lg:col-span-2 space-y-6">
        <!-- Consensus & Divergence Synthesis Card -->
        <div id="arenaSynthesisCard" class="hidden glass-card rounded-2xl p-6 shadow-xl space-y-4">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <h3 class="text-sm font-bold text-slate-100 flex items-center space-x-2">
              <i class="fa-solid fa-scale-unbalanced text-amber-400"></i>
              <span>Chief Epistemic Moderator Synthesis</span>
            </h3>
            <div class="flex items-center space-x-3 text-xs">
              <span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold" id="arenaConsensusBadge">Consensus: 0%</span>
              <span class="px-2.5 py-1 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30 font-semibold" id="arenaDivergenceBadge">Divergence: 0%</span>
            </div>
          </div>
          <p id="arenaEmergentSynthesis" class="text-xs text-slate-200 leading-relaxed bg-darkbg/70 p-3.5 rounded-xl border border-darkborder"></p>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div class="bg-emerald-950/20 border border-emerald-500/30 rounded-xl p-3.5">
              <div class="font-bold text-emerald-400 mb-1.5">✓ Common Ground Axes</div>
              <ul id="arenaCommonGround" class="list-disc list-inside space-y-1 text-slate-300"></ul>
            </div>
            <div class="bg-rose-950/20 border border-rose-500/30 rounded-xl p-3.5">
              <div class="font-bold text-rose-400 mb-1.5">⚡ Irreconcilable Epistemic Frictions</div>
              <ul id="arenaFrictions" class="list-disc list-inside space-y-1 text-slate-300"></ul>
            </div>
          </div>
        </div>

        <!-- Debate Transcript -->
        <div class="glass-card rounded-2xl p-6 shadow-xl space-y-4">
          <h3 class="text-sm font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-comments text-brand-400"></i>
            <span>Arena Debate Transcript</span>
          </h3>
          <div id="arenaTranscriptFeed" class="space-y-4 max-h-[520px] overflow-y-auto pr-1 text-xs">
            <p class="text-slate-400">Select at least two digital twins on the left and launch a debate to observe multi-agent epistemic dynamics.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 6: MINDSET & EPISTEMICS -->
    <div id="content-mindset" class="hidden grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="glass-card rounded-2xl p-6 shadow-xl space-y-4">
        <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2">
          <i class="fa-solid fa-scroll text-amber-400"></i>
          <span>Mindset Summary & Philosophical Profile</span>
        </h2>
        <div id="mindsetText" class="text-xs leading-relaxed text-slate-300 font-mono whitespace-pre-wrap bg-darkbg/60 p-4 rounded-xl border border-darkborder max-h-[540px] overflow-y-auto"></div>
      </div>

      <div class="space-y-6">
        <!-- Cognitive Dissonance Summary -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2 mb-2">
            <i class="fa-solid fa-bolt text-rose-400"></i>
            <span>Cognitive Dissonance & Internal Tensions</span>
          </h2>
          <div id="mindsetDissonanceBox" class="text-xs text-slate-300 space-y-2"></div>
        </div>

        <!-- Academic Literature Grounding -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2 mb-3">
            <i class="fa-solid fa-graduation-cap text-indigo-400"></i>
            <span>Autonomous Academic Literature Grounding</span>
          </h2>
          <p class="text-xs text-slate-400 mb-3">Peer-reviewed correlation studies parsed to calibrate twin weights:</p>
          <div id="academicList" class="space-y-2 text-xs"></div>
        </div>

        <!-- Topic Evolution -->
        <div class="glass-card rounded-2xl p-6 shadow-xl">
          <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2 mb-3">
            <i class="fa-solid fa-chart-line text-emerald-400"></i>
            <span>BERTopic Clusters</span>
          </h2>
          <div id="topicList" class="flex flex-wrap gap-2 text-xs"></div>
        </div>
      </div>
    </div>

    <!-- TAB 7: TARGETS DIRECTORY -->
    <div id="content-targets" class="hidden glass-card rounded-2xl p-6 shadow-xl space-y-4">
      <div class="flex items-center justify-between mb-2">
        <h2 class="font-bold text-slate-100 text-base">Local Profiles Directory</h2>
        <span class="text-xs text-slate-400">Target profiles stored locally under profiles/</span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-slate-300">
          <thead class="bg-darkbg text-slate-400 uppercase font-semibold border-b border-darkborder">
            <tr>
              <th class="p-3">Target Name</th>
              <th class="p-3">Harvested Documents</th>
              <th class="p-3">Mindset Status</th>
              <th class="p-3">Academic Grounding</th>
              <th class="p-3">Actions</th>
            </tr>
          </thead>
          <tbody id="targetsTableBody" class="divide-y divide-darkborder/50"></tbody>
        </table>
      </div>
    </div>

  </main>

  <script>
    let currentTarget = '';
    let currentTone = 'autodetect';
    let allTargetsSummary = [];
    let graphData = {nodes: [], links: [], contradictions_count: 0};

    const ALL_TABS = ['chat', 'graph', 'oracle', 'composer', 'arena', 'mindset', 'targets'];

    function escapeHtml(value) {
      if (value === null || value === undefined) return '';
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    async function init() {
      await loadSystemStatus();
      await loadTargets();
      await loadTargetsTable();
      if (currentTarget) {
        await loadTargetDetails();
      }
    }

    async function loadSystemStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.active_provider) {
          document.getElementById('providerName').innerText = data.active_provider;
        }
      } catch (e) {
        console.error(e);
      }
    }

    async function loadTargets() {
      try {
        const res = await fetch('/api/targets');
        const data = await res.json();
        const selector = document.getElementById('targetSelector');
        selector.innerHTML = '';
        (data.targets || []).forEach(t => {
          const opt = document.createElement('option');
          opt.value = t;
          opt.innerText = t.replace(/_/g, ' ').toUpperCase();
          selector.appendChild(opt);
        });
        if ((data.targets || []).length > 0) {
          currentTarget = data.targets[0];
          selector.value = currentTarget;
        }
      } catch (e) {
        console.error(e);
      }
    }

    function switchTarget() {
      currentTarget = document.getElementById('targetSelector').value;
      loadTargetDetails();
    }

    async function loadTargetDetails() {
      if (!currentTarget) return;
      try {
        const res = await fetch('/api/target/' + encodeURIComponent(currentTarget));
        const data = await res.json();

        document.getElementById('twinDisplayName').innerText = data.name;
        document.getElementById('welcomeTarget').innerText = data.name;
        document.getElementById('avatarName').innerText = (data.name || '?').charAt(0);
        document.getElementById('statDocs').innerText = data.raw_docs_count;
        document.getElementById('statPapers').innerText = data.academic_literature_count;
        document.getElementById('mindsetText').innerText = data.mindset_markdown;

        const features = data.features || {};
        const calRatio = (features.epistemic_markers || {}).epistemic_calibration_ratio;
        document.getElementById('statCalibration').innerText = calRatio !== undefined ? (calRatio + ' ratio') : 'Calibrated';

        const manip = features.epistemic_manipulation_index || {};
        document.getElementById('statManipulation').innerText = manip.score_pct || '0%';

        // Cognitive dissonance box on Mindset tab
        const dissBox = document.getElementById('mindsetDissonanceBox');
        const cogDiss = data.cognitive_dissonance || {};
        const clusters = cogDiss.dissonance_clusters || [];
        if (clusters.length > 0) {
          dissBox.innerHTML = clusters.map(c =>
            `<div class="p-3 bg-rose-950/20 border border-rose-500/30 rounded-xl space-y-1">
               <div class="font-semibold text-rose-300">⚡ ${(c.concepts || []).map(escapeHtml).join(' ↔ ')} (${Math.round((c.mean_dissonance_score || 0) * 100)}%)</div>
               <div class="text-slate-300 text-[11px]">${escapeHtml(c.primary_conflict || '')}</div>
             </div>`
          ).join('');
        } else {
          dissBox.innerHTML = '<p class="text-slate-400">No high-severity [CONTRADICTS] clusters detected in current vault.</p>';
        }

        // Topics
        const topicList = document.getElementById('topicList');
        topicList.innerHTML = '';
        (data.topics || []).forEach(t => {
          const span = document.createElement('span');
          span.className = 'px-3 py-1 bg-darkbg border border-darkborder rounded-full text-brand-400 font-medium';
          span.innerText = '#' + (t.name || 'topic');
          topicList.appendChild(span);
        });

        // Academic literature
        const academicList = document.getElementById('academicList');
        academicList.innerHTML = '';
        (data.academic_literature_sample || []).forEach(p => {
          const d = document.createElement('div');
          d.className = 'p-3 bg-darkbg/70 border border-darkborder rounded-xl space-y-1';
          d.innerHTML = `<div class="font-semibold text-slate-200">${escapeHtml(p.title || 'Scientific Study')}</div>
                         <div class="text-slate-400 text-[11px]">${escapeHtml(p.doi_or_url || 'arXiv / PubMed')}</div>`;
          academicList.appendChild(d);
        });

        await loadGraphData();
        await loadTargetsTable();
      } catch (e) {
        console.error(e);
      }
    }

    async function loadTargetsTable() {
      try {
        const res = await fetch('/api/targets_summary');
        allTargetsSummary = await res.json();
        const tbody = document.getElementById('targetsTableBody');
        tbody.innerHTML = '';

        const arenaList = document.getElementById('arenaParticipantsList');
        const previouslyChecked = new Set(
          Array.from(document.querySelectorAll('.arena-participant-cb:checked')).map(cb => cb.value)
        );
        arenaList.innerHTML = '';

        allTargetsSummary.forEach((item, idx) => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-darkbg/40 transition';
          tr.innerHTML = `
            <td class="p-3 font-semibold text-white">${escapeHtml(item.name)}</td>
            <td class="p-3">${item.raw_docs_count} docs</td>
            <td class="p-3">${item.mindset_ready ? '<span class="text-emerald-400">✓ Ready</span>' : '<span class="text-amber-400">Pending</span>'}</td>
            <td class="p-3">${item.academic_literature_count} papers</td>
            <td class="p-3 space-x-2">
              <button onclick="selectAndShow('${escapeHtml(item.slug)}', 'chat')" class="text-brand-400 hover:underline">Chat</button>
              <button onclick="selectAndShow('${escapeHtml(item.slug)}', 'graph')" class="text-purple-400 hover:underline">Graph</button>
              <button onclick="selectAndShow('${escapeHtml(item.slug)}', 'oracle')" class="text-emerald-400 hover:underline">Oracle</button>
            </td>
          `;
          tbody.appendChild(tr);

          const isChecked = previouslyChecked.size > 0 ? previouslyChecked.has(item.slug) : (idx < 2);
          const label = document.createElement('label');
          label.className = 'flex items-center justify-between p-2 rounded-lg hover:bg-darkpanel/70 cursor-pointer border border-transparent hover:border-darkborder';
          label.innerHTML = `
            <span class="flex items-center space-x-2">
              <input type="checkbox" value="${escapeHtml(item.slug)}" class="arena-participant-cb accent-amber-500" ${isChecked ? 'checked' : ''}>
              <span class="font-medium text-slate-200">${escapeHtml(item.name)}</span>
            </span>
            <span class="text-[10px] text-slate-400">${item.raw_docs_count} docs</span>
          `;
          arenaList.appendChild(label);
        });
      } catch (e) {
        console.error(e);
      }
    }

    function selectAndShow(slug, tab) {
      currentTarget = slug;
      document.getElementById('targetSelector').value = slug;
      loadTargetDetails();
      showTab(tab);
    }

    function showTab(tabName) {
      ALL_TABS.forEach(t => {
        const contentEl = document.getElementById('content-' + t);
        const tabEl = document.getElementById('tab-' + t);
        if (contentEl) contentEl.classList.add('hidden');
        if (tabEl) tabEl.className = 'py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 whitespace-nowrap transition';
      });
      document.getElementById('content-' + tabName).classList.remove('hidden');
      document.getElementById('tab-' + tabName).className = 'py-3 border-b-2 border-brand-500 text-brand-400 flex items-center space-x-2 whitespace-nowrap transition';
      if (tabName === 'graph') {
        renderD3Graph();
      }
    }

    function setTone(tone) {
      currentTone = tone;
      ['autodetect', 'formal', 'informal'].forEach(t => {
        document.getElementById('tone-' + t).className = 'px-2.5 py-1 rounded text-slate-400 hover:text-white';
      });
      document.getElementById('tone-' + tone).className = 'px-2.5 py-1 rounded bg-brand-600 text-white font-medium';
    }

    async function sendChatMessage() {
      const input = document.getElementById('chatInput');
      const text = input.value.trim();
      if (!text || !currentTarget) return;
      input.value = '';

      const feed = document.getElementById('chatFeed');
      const userBubble = document.createElement('div');
      userBubble.className = 'flex justify-end';
      userBubble.innerHTML = `<div class="bg-brand-600 text-white p-3 rounded-2xl rounded-tr-none max-w-lg shadow">${escapeHtml(text)}</div>`;
      feed.appendChild(userBubble);
      feed.scrollTop = feed.scrollHeight;

      const botBubble = document.createElement('div');
      botBubble.className = 'flex justify-start';
      botBubble.innerHTML = `<div class="bg-darkbg border border-darkborder text-slate-200 p-3 rounded-2xl rounded-tl-none max-w-lg shadow animate-pulse">Thinking...</div>`;
      feed.appendChild(botBubble);
      feed.scrollTop = feed.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({target: currentTarget, message: text, mode: currentTone})
        });
        const data = await res.json();
        botBubble.innerHTML = `<div class="bg-darkbg border border-darkborder text-slate-200 p-3 rounded-2xl rounded-tl-none max-w-lg shadow leading-relaxed whitespace-pre-wrap">${escapeHtml(data.response || data.error || '')}</div>`;
      } catch (e) {
        botBubble.innerHTML = `<div class="bg-red-950/40 border border-red-500/40 text-red-200 p-3 rounded-2xl text-xs">Error generating twin response.</div>`;
      }
      feed.scrollTop = feed.scrollHeight;
    }

    async function runSimulation() {
      const scenario = document.getElementById('scenarioInput').value.trim();
      if (!scenario || !currentTarget) return;
      const out = document.getElementById('simulationOutput');
      out.classList.remove('hidden');
      out.innerHTML = '<span class="text-purple-400 animate-pulse">Simulating behavioral reaction trajectory...</span>';

      try {
        const res = await fetch('/api/simulate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({target: currentTarget, scenario: scenario})
        });
        const data = await res.json();
        out.innerHTML = `<div class="font-semibold text-purple-300">Plausible Reaction (Confidence: ${Math.round((data.confidence_score || 0.7) * 100)}%):</div>
                         <div class="whitespace-pre-wrap">${escapeHtml(data.simulation_analysis || '')}</div>
                         <div class="text-[11px] text-slate-400 italic">${escapeHtml(data.epistemic_qualification || '')}</div>`;
      } catch (e) {
        out.innerHTML = '<span class="text-red-400">Simulation failed.</span>';
      }
    }

    async function runOracleQuery() {
      const query = document.getElementById('oracleQueryInput').value.trim();
      if (!query || !currentTarget) return;

      const panel = document.getElementById('oracleAnswerPanel');
      const evidenceList = document.getElementById('oracleEvidenceList');
      panel.innerHTML = '<div class="text-brand-400 animate-pulse text-sm">Querying vector embeddings & knowledge graph paths...</div>';

      try {
        const res = await fetch('/api/oracle', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({target: currentTarget, query: query})
        });
        const data = await res.json();
        if (!res.ok) {
          panel.innerHTML = `<div class="text-rose-400 text-xs">${escapeHtml(data.error || 'Oracle query failed.')}</div>`;
          return;
        }

        const pct = Math.round((data.confidence_score || 0) * 100);
        document.getElementById('oracleConfidenceBar').style.width = pct + '%';
        document.getElementById('oracleConfidenceLabel').innerText = pct + '% (Corpus-Calibrated)';

        panel.innerHTML = `
          <div class="flex items-center justify-between border-b border-darkborder pb-2">
            <span class="text-xs font-bold uppercase tracking-wider text-brand-400">Probabilistic OSINT Assessment</span>
            <span class="text-[11px] text-slate-400">Sources: ${escapeHtml((data.sources || []).join(', ') || 'local corpus')}</span>
          </div>
          <div class="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">${escapeHtml(data.answer || '')}</div>
        `;

        const quotesHtml = (data.primary_quotes || []).map(q =>
          `<div class="p-2.5 bg-darkbg/80 border border-indigo-500/30 rounded-lg text-slate-300 italic">"${escapeHtml(q)}"</div>`
        ).join('');
        const pathsHtml = (data.graph_paths || []).map(p =>
          `<div class="p-2 bg-darkbg/60 border border-darkborder rounded-lg font-mono text-[11px] text-brand-400">${escapeHtml(p)}</div>`
        ).join('');

        evidenceList.innerHTML = (quotesHtml + pathsHtml) || '<p class="text-slate-500">No direct quotes or graph paths matched.</p>';
      } catch (e) {
        panel.innerHTML = '<div class="text-rose-400 text-xs">Error connecting to Oracle engine.</div>';
      }
    }

    async function runComposer() {
      const goal = document.getElementById('composerGoalInput').value.trim();
      const context = document.getElementById('composerContextInput').value.trim();
      if (!goal || !currentTarget) return;

      const grid = document.getElementById('composerResultsGrid');
      grid.classList.remove('hidden');
      document.getElementById('composerInitialDraft').innerText = 'Drafting initial message calibrated to target lexical profile...';
      document.getElementById('composerCritique').innerText = 'Simulating adversarial target skepticism...';
      document.getElementById('composerFinalPitch').innerText = 'Optimizing hardened final outreach...';

      try {
        const res = await fetch('/api/compose', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({target: currentTarget, goal: goal, context: context})
        });
        const data = await res.json();
        if (!res.ok) {
          document.getElementById('composerInitialDraft').innerText = data.error || 'Composer failed.';
          return;
        }

        document.getElementById('composerInitialDraft').innerText = data.initial_draft || '';
        document.getElementById('composerCritique').innerText = data.adversarial_rejection_analysis || '';
        document.getElementById('composerFinalPitch').innerText = data.optimized_final_pitch || '';

        const termsRow = document.getElementById('composerTermsRow');
        termsRow.innerHTML = (data.calibrated_terms_used || []).map(t =>
          `<span class="px-2.5 py-0.5 rounded-full bg-indigo-500/15 border border-indigo-500/30 text-indigo-300">⚓ ${escapeHtml(t)}</span>`
        ).join('');
      } catch (e) {
        document.getElementById('composerInitialDraft').innerText = 'Error running Outreach Composer.';
      }
    }

    function copyFinalPitch() {
      const text = document.getElementById('composerFinalPitch').innerText;
      if (text && navigator.clipboard) {
        navigator.clipboard.writeText(text);
      }
    }

    async function runArenaDebate() {
      const selected = Array.from(document.querySelectorAll('.arena-participant-cb:checked')).map(cb => cb.value);
      const topic = document.getElementById('arenaTopicInput').value.trim();
      const rounds = parseInt(document.getElementById('arenaRoundsSelect').value || '2', 10);
      const feed = document.getElementById('arenaTranscriptFeed');

      if (selected.length < 2) {
        feed.innerHTML = '<div class="p-3 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-300">Please select at least 2 digital twin participants.</div>';
        return;
      }
      if (!topic) {
        feed.innerHTML = '<div class="p-3 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-300">Please enter a debate topic or proposition.</div>';
        return;
      }

      document.getElementById('arenaSynthesisCard').classList.add('hidden');
      feed.innerHTML = '<div class="p-4 rounded-xl bg-darkbg/70 border border-amber-500/30 text-amber-300 animate-pulse">Convening Swarm Arena participants and synthesizing multi-round turns...</div>';

      try {
        const res = await fetch('/api/arena', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({targets: selected, topic: topic, rounds: rounds})
        });
        const data = await res.json();
        if (!res.ok) {
          feed.innerHTML = `<div class="p-3 rounded-xl bg-rose-950/30 border border-rose-500/40 text-rose-300">${escapeHtml(data.error || 'Arena simulation failed.')}</div>`;
          return;
        }

        const syn = data.synthesis || {};
        document.getElementById('arenaSynthesisCard').classList.remove('hidden');
        document.getElementById('arenaConsensusBadge').innerText = `Consensus: ${syn.consensus_score_pct ?? 50}%`;
        document.getElementById('arenaDivergenceBadge').innerText = `Divergence: ${syn.consensus_divergence_pct ?? 50}%`;
        document.getElementById('arenaEmergentSynthesis').innerText = syn.emergent_synthesis || '';

        document.getElementById('arenaCommonGround').innerHTML = (syn.common_ground_points || []).map(pt =>
          `<li>${escapeHtml(pt)}</li>`
        ).join('') || '<li>Empirical dialogue established</li>';

        document.getElementById('arenaFrictions').innerHTML = (syn.irreconcilable_frictions || []).map(pt =>
          `<li>${escapeHtml(pt)}</li>`
        ).join('') || '<li>Methodological weighting differences</li>';

        feed.innerHTML = (data.transcript || []).map(turn => `
          <div class="p-4 rounded-xl bg-darkbg/80 border border-darkborder space-y-2">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="w-7 h-7 rounded-full bg-amber-500/20 border border-amber-500/40 text-amber-300 font-bold flex items-center justify-center">
                  ${escapeHtml((turn.speaker || '?').charAt(0))}
                </span>
                <span class="font-bold text-slate-100 text-sm">${escapeHtml(turn.speaker)}</span>
              </div>
              <span class="px-2 py-0.5 rounded bg-darkpanel border border-darkborder text-[10px] text-amber-400 font-mono">Round ${turn.round}</span>
            </div>
            <div class="text-slate-300 leading-relaxed whitespace-pre-wrap">${escapeHtml(turn.statement)}</div>
          </div>
        `).join('');
      } catch (e) {
        feed.innerHTML = '<div class="text-rose-400">Error executing Swarm Arena simulation.</div>';
      }
    }

    async function loadGraphData() {
      if (!currentTarget) return;
      try {
        const res = await fetch('/api/target/' + encodeURIComponent(currentTarget) + '/graph');
        graphData = await res.json();
        const cCount = graphData.contradictions_count || 0;
        const navBadge = document.getElementById('navDissonanceBadge');
        if (cCount > 0) {
          navBadge.innerText = cCount + ' ⚡';
          navBadge.classList.remove('hidden');
        } else {
          navBadge.classList.add('hidden');
        }
        if (!document.getElementById('content-graph').classList.contains('hidden')) {
          renderD3Graph();
        }
      } catch (e) {
        console.error(e);
      }
    }

    function inspectGraphElement(htmlContent) {
      document.getElementById('graphInspector').innerHTML = htmlContent;
    }

    function renderD3Graph() {
      const container = document.getElementById('d3Container');
      container.innerHTML = '';
      const width = container.clientWidth || 850;
      const height = container.clientHeight || 650;

      const onlyContradictions = document.getElementById('filterContradictionsOnly').checked;
      const rawLinks = (graphData.links || []).map(l => ({
        ...l,
        source: typeof l.source === 'object' ? l.source.id : l.source,
        target: typeof l.target === 'object' ? l.target.id : l.target
      }));

      const activeLinks = onlyContradictions ? rawLinks.filter(l => l.is_contradiction) : rawLinks;
      const connectedIds = new Set();
      activeLinks.forEach(l => { connectedIds.add(l.source); connectedIds.add(l.target); });

      const activeNodes = (graphData.nodes || [])
        .filter(n => !onlyContradictions || connectedIds.has(n.id) || n.group === 1)
        .map(n => ({...n}));

      document.getElementById('graphStatsBadge').innerText = `${activeNodes.length}N / ${activeLinks.length}E`;

      // Populate quick list of contradiction edges
      const quickList = document.getElementById('dissonanceQuickList');
      const contradictionLinks = rawLinks.filter(l => l.is_contradiction);
      if (contradictionLinks.length > 0) {
        quickList.innerHTML = `<div class="text-[11px] font-bold uppercase text-rose-400">⚡ Detected [CONTRADICTS] Edges (${contradictionLinks.length})</div>` +
          contradictionLinks.slice(0, 5).map(cl => `
            <div class="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/30 text-[11px] space-y-1">
              <div class="font-semibold text-rose-200">${escapeHtml(cl.source)} ⚡ ${escapeHtml(cl.target)}</div>
              <div class="text-rose-300/80">Dissonance: ${Math.round((cl.dissonance_score || 0.85) * 100)}%</div>
              ${cl.contradiction_summary ? `<div class="text-slate-300">${escapeHtml(cl.contradiction_summary)}</div>` : ''}
            </div>
          `).join('');
      } else {
        quickList.innerHTML = '<div class="text-[11px] text-slate-500">0 [CONTRADICTS] edges in current graph.</div>';
      }

      const svg = d3.select('#d3Container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

      const g = svg.append('g');

      svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => {
        g.attr('transform', e.transform);
      }));

      const simulation = d3.forceSimulation(activeNodes)
        .force('link', d3.forceLink(activeLinks).id(d => d.id).distance(d => d.is_contradiction ? 125 : 85))
        .force('charge', d3.forceManyBody().strength(-210))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide().radius(28));

      const link = g.append('g')
        .selectAll('line')
        .data(activeLinks)
        .enter().append('line')
        .attr('class', d => d.is_contradiction ? 'link-contradicts cursor-pointer' : 'link-line cursor-pointer')
        .attr('stroke-width', d => d.is_contradiction ? 2.8 : Math.max(1.2, (d.weight || 0.5) * 2.2))
        .on('click', (event, d) => {
          const srcName = typeof d.source === 'object' ? d.source.id : d.source;
          const tgtName = typeof d.target === 'object' ? d.target.id : d.target;
          inspectGraphElement(`
            <div class="font-bold ${d.is_contradiction ? 'text-rose-400' : 'text-brand-400'}">
              ${escapeHtml(srcName)} —[${escapeHtml(d.relation)}]→ ${escapeHtml(tgtName)}
            </div>
            <div class="text-[11px] text-slate-400">Weight: ${d.weight} ${d.is_contradiction ? `| ⚡ Dissonance: ${Math.round((d.dissonance_score || 0.85) * 100)}%` : ''}</div>
            ${d.contradiction_summary ? `<div class="p-2 rounded bg-rose-950/30 border border-rose-500/30 text-rose-200">${escapeHtml(d.contradiction_summary)}</div>` : ''}
            ${d.quote ? `<div class="italic text-slate-300">"${escapeHtml(d.quote)}"</div>` : ''}
            ${d.quote_a ? `<div class="italic text-slate-300">Quote A: "${escapeHtml(d.quote_a)}"</div>` : ''}
            ${d.quote_b ? `<div class="italic text-slate-300">Quote B: "${escapeHtml(d.quote_b)}"</div>` : ''}
          `);
        });

      const edgeLabels = g.append('g')
        .selectAll('text')
        .data(activeLinks.filter(l => l.is_contradiction))
        .enter().append('text')
        .text('⚡ CONTRADICTS')
        .attr('font-size', '9px')
        .attr('font-weight', 'bold')
        .attr('fill', '#fb7185')
        .attr('text-anchor', 'middle');

      const node = g.append('g')
        .selectAll('circle')
        .data(activeNodes)
        .enter().append('circle')
        .attr('r', d => d.group === 1 ? 16 : (d.has_contradiction ? 11 : 8))
        .attr('class', d => d.group === 1 ? 'node-target cursor-pointer' : (d.has_contradiction ? 'node-dissonance cursor-pointer' : 'node-concept cursor-pointer'))
        .on('click', (event, d) => {
          inspectGraphElement(`
            <div class="font-bold text-slate-100 text-sm">${escapeHtml(d.label)}</div>
            <div class="text-[11px] text-brand-400">Type: ${escapeHtml(d.type || 'Concept')}</div>
            ${d.has_contradiction ? `<div class="text-[11px] text-rose-400 font-semibold">⚡ Involved in ${d.contradiction_count} [CONTRADICTS] tension(s)</div>` : ''}
            ${d.description ? `<div class="text-slate-300">${escapeHtml(d.description)}</div>` : ''}
          `);
        })
        .call(d3.drag()
          .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
        );

      const labels = g.append('g')
        .selectAll('text')
        .data(activeNodes)
        .enter().append('text')
        .text(d => d.label)
        .attr('font-size', '10px')
        .attr('fill', d => d.has_contradiction ? '#fda4af' : '#94a3b8')
        .attr('dx', 13)
        .attr('dy', 4);

      simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);

        edgeLabels
          .attr('x', d => (d.source.x + d.target.x) / 2)
          .attr('y', d => (d.source.y + d.target.y) / 2 - 4);

        node
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);

        labels
          .attr('x', d => d.x)
          .attr('y', d => d.y);
      });
    }

    function exportCurrentDossier() {
      if (!currentTarget) return;
      window.open('/api/target/' + encodeURIComponent(currentTarget) + '/dossier', '_blank');
    }

    function refreshData() {
      loadTargetDetails();
    }

    window.onload = init;
  </script>
</body>
</html>
"""


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Handles REST API and Web Dashboard 2.0 UI requests."""

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html", "/ui"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        if path == "/api/status":
            self._send_json({
                "status": "ok",
                "service": "who-engine",
                "version": "2.0",
                "active_provider": settings.active_provider,
                "targets_count": len(get_all_target_names()),
            })
            return

        if path == "/api/targets":
            targets = get_all_target_names()
            self._send_json({"targets": targets})
            return

        if path == "/api/targets_summary":
            targets = get_all_target_names()
            summary = [get_target_profile_summary(t) for t in targets]
            self._send_json(summary)
            return

        if path.startswith("/api/target/") and path.endswith("/dossier"):
            parts = path.split("/")
            target_slug = sanitize_slug(urllib.parse.unquote(parts[3]))
            display_name = _resolve_target_display_name(target_slug)
            target_dirs = settings.get_target_dirs(target_slug)
            from src.engine.dossier_exporter import DossierExporter

            html_doc = DossierExporter(display_name, target_dirs).render_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{target_slug}_intelligence_dossier.html"')
            self.end_headers()
            self.wfile.write(html_doc.encode("utf-8"))
            return

        if path.startswith("/api/target/") and path.endswith("/graph"):
            parts = path.split("/")
            target_slug = sanitize_slug(urllib.parse.unquote(parts[3]))
            graph = get_target_graph_data(target_slug)
            self._send_json(graph)
            return

        if path.startswith("/api/target/"):
            target_slug = sanitize_slug(urllib.parse.unquote(path.split("/")[3]))
            summary = get_target_profile_summary(target_slug)
            self._send_json(summary)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if path == "/api/chat":
            target = str(payload.get("target", "")).strip()
            message = str(payload.get("message", "")).strip()
            mode = str(payload.get("mode", "autodetect")).strip()
            if not target or not message:
                self._send_json({"error": "Both 'target' and 'message' are required."}, status=400)
                return

            display_name = _resolve_target_display_name(target)
            target_dirs = settings.get_target_dirs(target)
            from src.engine.twin_chat import TwinChat

            twin = TwinChat(display_name, target_dirs, mode=mode)
            reply = twin.respond(message)
            self._send_json({"response": reply})
            return

        if path == "/api/simulate":
            target = str(payload.get("target", "")).strip()
            scenario = str(payload.get("scenario", "")).strip()
            if not target or not scenario:
                self._send_json({"error": "Both 'target' and 'scenario' are required."}, status=400)
                return

            display_name = _resolve_target_display_name(target)
            target_dirs = settings.get_target_dirs(target)
            from src.engine.simulator import Simulator

            sim = Simulator(display_name, target_dirs)
            res = sim.simulate_scenario(scenario)
            self._send_json(res)
            return

        if path == "/api/oracle":
            target = str(payload.get("target", "")).strip()
            query = str(payload.get("query", "")).strip()
            if not target or not query:
                self._send_json({"error": "Both 'target' and 'query' are required."}, status=400)
                return

            display_name = _resolve_target_display_name(target)
            target_dirs = settings.get_target_dirs(target)
            from src.engine.oracle import Oracle

            oracle = Oracle(display_name, target_dirs)
            try:
                res = oracle.ask(query)
            finally:
                oracle.graph_store.close()
            self._send_json(res)
            return

        if path == "/api/compose":
            target = str(payload.get("target", "")).strip()
            goal = str(payload.get("goal", "")).strip()
            context = str(payload.get("context", "")).strip() or None
            if not target or not goal:
                self._send_json({"error": "Both 'target' and 'goal' are required."}, status=400)
                return

            display_name = _resolve_target_display_name(target)
            target_dirs = settings.get_target_dirs(target)
            from src.engine.composer import Composer

            composer = Composer(display_name, target_dirs)
            res = composer.compose_outreach(goal=goal, context=context)
            self._send_json(res)
            return

        if path == "/api/arena":
            raw_targets = payload.get("targets", [])
            if isinstance(raw_targets, str):
                target_list = [t.strip() for t in raw_targets.split(",") if t.strip()]
            elif isinstance(raw_targets, list):
                target_list = [str(t).strip() for t in raw_targets if str(t).strip()]
            else:
                target_list = []

            topic = str(payload.get("topic", "")).strip()
            try:
                rounds = int(payload.get("rounds", 2))
            except (TypeError, ValueError):
                rounds = 2

            if len(target_list) < 2:
                self._send_json({"error": "Swarm Arena requires at least 2 participants."}, status=400)
                return
            if not topic:
                self._send_json({"error": "Debate 'topic' is required."}, status=400)
                return

            resolved_participants = [_resolve_target_display_name(t) for t in target_list]
            from src.engine.arena import SwarmArena

            arena = SwarmArena(participant_names=resolved_participants, topic=topic, rounds=rounds)
            debate_res = arena.run_debate()
            self._send_json(debate_res)
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy request logging."""
        pass


def run_dashboard(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Launch the Web Dashboard server."""
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    url = f"http://{host}:{port}"
    print(f"\n🚀 who?? Cognitive Digital Twin Dashboard 2.0 running at: {url}")
    print("Press Ctrl+C to stop the dashboard server.\n")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()


# Function aliases for seamless compatibility
start_dashboard = run_dashboard
