"""
Interactive Web Dashboard & Visualization Server for who?? Cognitive Digital Twin Framework.
Runs a zero-dependency, multi-threaded HTTP server providing REST APIs and an ultra-modern D3.js web interface.
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
    features_file = skills_dir / "features.json"
    topics_file = skills_dir / "topics.json"
    lit_file = skills_dir / "retrieved_academic_literature.json"
    weights_file = skills_dir / "scientific_weights.json"

    character_data = {}
    if char_file.exists():
        try:
            with open(char_file, "r", encoding="utf-8") as f:
                character_data = json.load(f)
        except Exception:
            pass

    features_data = {}
    if features_file.exists():
        try:
            with open(features_file, "r", encoding="utf-8") as f:
                features_data = json.load(f)
        except Exception:
            pass

    topics_data = []
    if topics_file.exists():
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                topics_data = json.load(f).get("topics", [])
        except Exception:
            pass

    literature_data = []
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
        "mindset_markdown": mindset_md[:1500],
        "features": features_data,
        "topics": topics_data[:6],
        "academic_literature_count": len(literature_data),
        "academic_literature_sample": literature_data[:4],
        "has_weights": weights_file.exists(),
    }


def get_target_graph_data(target_slug: str) -> Dict[str, Any]:
    """Fetch Knowledge Graph nodes and edges for D3.js visualization."""
    target_dir = Path(settings.profiles_dir) / target_slug
    concept_file = target_dir / "vault" / "concept_network.json"

    nodes = []
    links = []

    if concept_file.exists():
        try:
            with open(concept_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_nodes = data.get("nodes", []) or data.get("entities", [])
                for n in raw_nodes[:150]:
                    nodes.append({
                        "id": n.get("name", "Concept"),
                        "label": n.get("name", "Concept"),
                        "type": n.get("type", "Concept"),
                        "group": 1 if n.get("name") == data.get("target_name") else 2,
                    })
                node_ids = {n["id"] for n in nodes}
                for r in data.get("relationships", [])[:200]:
                    if r.get("source") in node_ids and r.get("target") in node_ids:
                        links.append({
                            "source": r.get("source"),
                            "target": r.get("target"),
                            "relation": r.get("relation", "RELATES_TO"),
                            "weight": r.get("weight", 0.5),
                        })
        except Exception as e:
            logger.debug(f"Graph read note: {e}")

    if not nodes:
        # Default placeholder node
        t_name = target_slug.replace("_", " ").title()
        nodes = [{"id": t_name, "label": t_name, "type": "TargetPerson", "group": 1}]

    return {"nodes": nodes, "links": links}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>who?? Cognitive Digital Twin Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#38bdf8', 600: '#0284c7', 900: '#0c4a6e' },
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
    .node-target { fill: #38bdf8; stroke: #fff; stroke-width: 2px; }
    .node-concept { fill: #a855f7; stroke: #222E42; stroke-width: 1.5px; }
    .link-line { stroke: #334155; stroke-opacity: 0.6; }
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
        <h1 class="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-500 via-indigo-400 to-purple-400">
          who?? Digital Twin Console
        </h1>
        <p class="text-xs text-slate-400">Cognitive Modeling • Epistemic Grounding • Swarm Intelligence</p>
      </div>
    </div>

    <!-- Active Target Selector -->
    <div class="flex items-center space-x-4">
      <div class="flex items-center space-x-2 bg-darkbg px-3 py-1.5 rounded-lg border border-darkborder">
        <i class="fa-solid fa-user-astronaut text-brand-500 text-sm"></i>
        <span class="text-xs text-slate-400">Target:</span>
        <select id="targetSelector" onchange="switchTarget()" class="bg-transparent text-sm font-semibold text-slate-200 outline-none cursor-pointer">
          <option value="" disabled selected>Loading...</option>
        </select>
      </div>
      <button onclick="refreshData()" class="p-2 rounded-lg bg-darkbg border border-darkborder text-slate-400 hover:text-white transition">
        <i class="fa-solid fa-rotate-right"></i>
      </button>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="bg-darkpanel/40 border-b border-darkborder px-6 flex space-x-6 text-sm font-medium">
    <button onclick="showTab('chat')" id="tab-chat" class="py-3 border-b-2 border-brand-500 text-brand-400 flex items-center space-x-2 transition">
      <i class="fa-regular fa-comment-dots"></i> <span>Twin Chat & Simulation</span>
    </button>
    <button onclick="showTab('graph')" id="tab-graph" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 transition">
      <i class="fa-solid fa-diagram-project"></i> <span>D3 Knowledge Graph</span>
    </button>
    <button onclick="showTab('mindset')" id="tab-mindset" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 transition">
      <i class="fa-solid fa-brain"></i> <span>Mindset & Epistemics</span>
    </button>
    <button onclick="showTab('targets')" id="tab-targets" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 transition">
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

      <!-- Quick Scenario Simulator & Adversarial Pitch -->
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
          <div id="simulationOutput" class="hidden mt-3 p-3 bg-darkbg/80 border border-purple-500/30 rounded-xl text-xs text-slate-300 space-y-2"></div>
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
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: D3 KNOWLEDGE GRAPH -->
    <div id="content-graph" class="hidden glass-card rounded-2xl p-5 shadow-xl flex flex-col h-[750px]">
      <div class="flex items-center justify-between pb-3 border-b border-darkborder mb-2">
        <div>
          <h2 class="font-bold text-slate-100 flex items-center space-x-2">
            <i class="fa-solid fa-circle-nodes text-brand-500"></i>
            <span>Interactive Knowledge & Concept Graph</span>
          </h2>
          <p class="text-xs text-slate-400">Physics-driven force simulation of target nodes and relationship weights</p>
        </div>
        <div class="text-xs text-slate-400 flex items-center space-x-4">
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-brand-400 inline-block"></span> Target Person</span>
          <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-purple-500 inline-block"></span> Concept / Principle</span>
        </div>
      </div>
      <div id="d3Container" class="flex-1 relative overflow-hidden rounded-xl bg-darkbg/90 border border-darkborder"></div>
    </div>

    <!-- TAB 3: MINDSET & EPISTEMICS -->
    <div id="content-mindset" class="hidden grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="glass-card rounded-2xl p-6 shadow-xl space-y-4">
        <h2 class="text-base font-bold text-slate-100 flex items-center space-x-2">
          <i class="fa-solid fa-scroll text-amber-400"></i>
          <span>Mindset Summary & Philosophical Profile</span>
        </h2>
        <div id="mindsetText" class="text-xs leading-relaxed text-slate-300 font-mono whitespace-pre-wrap bg-darkbg/60 p-4 rounded-xl border border-darkborder max-h-[500px] overflow-y-auto"></div>
      </div>

      <div class="space-y-6">
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

    <!-- TAB 4: TARGETS DIRECTORY -->
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

    async function init() {
      await loadTargets();
      if (currentTarget) {
        await loadTargetDetails();
      }
    }

    async function loadTargets() {
      try {
        const res = await fetch('/api/targets');
        const data = await res.json();
        const selector = document.getElementById('targetSelector');
        selector.innerHTML = '';
        data.targets.forEach(t => {
          const opt = document.createElement('option');
          opt.value = t;
          opt.innerText = t.replace('_', ' ').toUpperCase();
          selector.appendChild(opt);
        });
        if (data.targets.length > 0) {
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
        const res = await fetch('/api/target/' + currentTarget);
        const data = await res.json();
        
        document.getElementById('twinDisplayName').innerText = data.name;
        document.getElementById('welcomeTarget').innerText = data.name;
        document.getElementById('avatarName').innerText = data.name.charAt(0);
        document.getElementById('statDocs').innerText = data.raw_docs_count;
        document.getElementById('statPapers').innerText = data.academic_literature_count;
        document.getElementById('mindsetText').innerText = data.mindset_markdown;

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
          d.innerHTML = `<div class="font-semibold text-slate-200">${p.title || 'Scientific Study'}</div>
                         <div class="text-slate-400 text-[11px]">${p.doi_or_url || 'arXiv / PubMed'}</div>`;
          academicList.appendChild(d);
        });

        loadGraphData();
        loadTargetsTable();
      } catch (e) {
        console.error(e);
      }
    }

    async function loadTargetsTable() {
      const res = await fetch('/api/targets_summary');
      const data = await res.json();
      const tbody = document.getElementById('targetsTableBody');
      tbody.innerHTML = '';
      data.forEach(item => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-darkbg/40 transition';
        tr.innerHTML = `
          <td class="p-3 font-semibold text-white">${item.name}</td>
          <td class="p-3">${item.raw_docs_count} docs</td>
          <td class="p-3">${item.mindset_ready ? '<span class="text-emerald-400">✓ Ready</span>' : '<span class="text-amber-400">Pending</span>'}</td>
          <td class="p-3">${item.academic_literature_count} papers</td>
          <td class="p-3">
            <button onclick="selectAndShow('${item.slug}', 'chat')" class="text-brand-400 hover:underline mr-3">Chat</button>
            <button onclick="selectAndShow('${item.slug}', 'graph')" class="text-purple-400 hover:underline">Graph</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    function selectAndShow(slug, tab) {
      currentTarget = slug;
      document.getElementById('targetSelector').value = slug;
      loadTargetDetails();
      showTab(tab);
    }

    function showTab(tabName) {
      ['chat', 'graph', 'mindset', 'targets'].forEach(t => {
        document.getElementById('content-' + t).classList.add('hidden');
        document.getElementById('tab-' + t).className = 'py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center space-x-2 transition';
      });
      document.getElementById('content-' + tabName).classList.remove('hidden');
      document.getElementById('tab-' + tabName).className = 'py-3 border-b-2 border-brand-500 text-brand-400 flex items-center space-x-2 transition';
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
      userBubble.innerHTML = `<div class="bg-brand-600 text-white p-3 rounded-2xl rounded-tr-none max-w-lg shadow">${text}</div>`;
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
        botBubble.innerHTML = `<div class="bg-darkbg border border-darkborder text-slate-200 p-3 rounded-2xl rounded-tl-none max-w-lg shadow leading-relaxed">${data.response}</div>`;
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
        out.innerHTML = `<div class="font-semibold text-purple-300">Plausible Reaction:</div>
                         <div>${data.simulation_analysis}</div>
                         <div class="text-[11px] text-slate-400 italic">${data.epistemic_qualification || ''}</div>`;
      } catch (e) {
        out.innerHTML = '<span class="text-red-400">Simulation failed.</span>';
      }
    }

    let graphData = {nodes: [], links: []};
    async function loadGraphData() {
      if (!currentTarget) return;
      try {
        const res = await fetch('/api/target/' + currentTarget + '/graph');
        graphData = await res.json();
      } catch(e) {
        console.error(e);
      }
    }

    function renderD3Graph() {
      const container = document.getElementById('d3Container');
      container.innerHTML = '';
      const width = container.clientWidth || 900;
      const height = container.clientHeight || 650;

      const svg = d3.select('#d3Container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

      const g = svg.append('g');

      svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', (e) => {
        g.attr('transform', e.transform);
      }));

      const simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-180))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide().radius(25));

      const link = g.append('g')
        .selectAll('line')
        .data(graphData.links)
        .enter().append('line')
        .attr('class', 'link-line')
        .attr('stroke-width', d => Math.max(1, (d.weight || 0.5) * 2));

      const node = g.append('g')
        .selectAll('circle')
        .data(graphData.nodes)
        .enter().append('circle')
        .attr('r', d => d.group === 1 ? 16 : 8)
        .attr('class', d => d.group === 1 ? 'node-target cursor-pointer' : 'node-concept cursor-pointer')
        .call(d3.drag()
          .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
        );

      const labels = g.append('g')
        .selectAll('text')
        .data(graphData.nodes)
        .enter().append('text')
        .text(d => d.label)
        .attr('font-size', '10px')
        .attr('fill', '#94a3b8')
        .attr('dx', 12)
        .attr('dy', 4);

      simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);

        node
          .attr('cx', d => d.x)
          .attr('cy', d => d.y);

        labels
          .attr('x', d => d.x)
          .attr('y', d => d.y);
      });
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
    """Handles REST API and UI requests."""

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html", "/ui"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
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

        if path.startswith("/api/target/") and path.endswith("/graph"):
            parts = path.split("/")
            target_slug = parts[3]
            graph = get_target_graph_data(target_slug)
            self._send_json(graph)
            return

        if path.startswith("/api/target/"):
            target_slug = path.split("/")[3]
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
            target = payload.get("target", "")
            message = payload.get("message", "")
            mode = payload.get("mode", "autodetect")

            target_dirs = settings.get_target_dirs(target)
            from src.engine.twin_chat import TwinChat

            twin = TwinChat(target, target_dirs, mode=mode)
            reply = twin.respond(message)
            self._send_json({"response": reply})
            return

        if path == "/api/simulate":
            target = payload.get("target", "")
            scenario = payload.get("scenario", "")
            target_dirs = settings.get_target_dirs(target)
            from src.engine.simulator import Simulator

            sim = Simulator(target, target_dirs)
            res = sim.simulate_scenario(scenario)
            self._send_json(res)
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: Any) -> None:
        self.send_response(200)
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
    print(f"\n🚀 who?? Cognitive Digital Twin Dashboard running at: {url}")
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
