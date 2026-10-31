"""
Standard Model Context Protocol (MCP) Server for who??.
Exposes who?? Cognitive Digital Twin, OSINT & Behavioral Analysis as tools to any AI Agent (Antigravity, Claude Desktop, Cursor, etc.).
Runs over standard JSON-RPC 2.0 stdio transport.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from who.agent.sdk import AcademicResearcher, CognitiveTwin, list_targets
from who.utils.logger import get_logger

logger = get_logger("who.mcp")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "who-twin-mcp"
SERVER_VERSION = "4.5.0"

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "who_chat",
        "description": "Chat directly in-character with a target's cognitive digital twin using verified primary memory, behavioral boundaries, and active operational rules.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name or directory slug of the target profile (e.g. 'remzi_sanver').",
                },
                "message": {
                    "type": "string",
                    "description": "User question, statement, or dialogic prompt for the digital twin.",
                },
                "tone": {
                    "type": "string",
                    "enum": ["autodetect", "formal", "informal"],
                    "default": "autodetect",
                    "description": "Dialogue register: 'formal' for keynote/academic, 'informal' for 1-on-1, or 'autodetect'.",
                },
            },
            "required": ["target", "message"],
        },
    },
    {
        "name": "who_simulate",
        "description": "Simulate target's plausible counter-move and reaction to a hypothetical proposition, dilemma, or adversarial stress-test.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name or directory slug of the target profile.",
                },
                "scenario": {
                    "type": "string",
                    "description": "Hypothetical proposition or scenario (e.g. 'Kendisine kurumsal sadakatini bozması karşılığında asimetrik bir menfaat vaat edilirse nasıl tepki verir?').",
                },
            },
            "required": ["target", "scenario"],
        },
    },
    {
        "name": "who_get_profile",
        "description": "Retrieve target's comprehensive psychometric profile, 5 grand dimensions (Machiavellianism, Need for Closure, Loss Aversion, Affective Volatility, Moral Foundations), active operational rules, and empirical boundaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name or directory slug of the target profile.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "who_explain",
        "description": "Trace any claim or behavioral assertion back to primary corpus quotes, Neo4j graph relationships, and epistemic confidence scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name or directory slug of the target profile.",
                },
                "claim": {
                    "type": "string",
                    "description": "Statement or claim to verify against the target's primary corpus.",
                },
            },
            "required": ["target", "claim"],
        },
    },
    {
        "name": "who_academic_research",
        "description": "Search peer-reviewed literature across arXiv, PubMed, Semantic Scholar, and OpenAlex. Evaluates findings using the 0-100 AI Scientific Referee Rigor Index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Academic topic or psychological construct to research.",
                },
                "target": {
                    "type": "string",
                    "description": "Optional target name to associate and optionally ingest literature into.",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of papers to retrieve.",
                },
                "ingest": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to automatically ingest verified papers into the target's academic corpus.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "who_compose_outreach",
        "description": "Generate hyper-tailored communication drafts optimized for the target individual with adversarial rejection stress-testing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name or directory slug of the target profile.",
                },
                "goal": {
                    "type": "string",
                    "description": "Outreach objective (e.g. 'Invite target to keynote speech on consensus protocols').",
                },
                "context": {
                    "type": "string",
                    "description": "Optional background or situational context.",
                },
            },
            "required": ["target", "goal"],
        },
    },
    {
        "name": "who_list_targets",
        "description": "List all profiled individuals and organizations available in the local vault with document counts and readiness status.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class MCPServer:
    """Implements standard Model Context Protocol (MCP) over JSON-RPC 2.0 stdio."""

    def __init__(self) -> None:
        self.running = True

    def run(self) -> None:
        """Main stdio loop reading JSON-RPC requests."""
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                # Handle optional Content-Length header prefix if host uses HTTP-style framing
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":")[1].strip())
                    # Read empty line
                    sys.stdin.readline()
                    # Read exact bytes
                    body = sys.stdin.read(length)
                    req = json.loads(body)
                else:
                    req = json.loads(line)

                response = self.handle_request(req)
                if response is not None:
                    self._send_response(response)
            except Exception as e:
                logger.error(f"MCP server loop error: {e}")

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": False,
                        },
                    },
                },
            }

        if method in ("notifications/initialized", "initialized"):
            # Client confirmed initialization
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": TOOLS_SCHEMA,
                },
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result_content = self.execute_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result_content if isinstance(result_content, str) else json.dumps(result_content, ensure_ascii=False, indent=2),
                            }
                        ],
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": str(e),
                    },
                }

        # Unknown method
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found.",
                },
            }
        return None

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name == "who_chat":
            target = args["target"]
            message = args["message"]
            tone = args.get("tone", "autodetect")
            twin = CognitiveTwin(target, mode=tone)
            return twin.chat(message, tone=tone)

        if name == "who_simulate":
            target = args["target"]
            scenario = args["scenario"]
            twin = CognitiveTwin(target)
            return twin.simulate_scenario(scenario)

        if name == "who_get_profile":
            target = args["target"]
            twin = CognitiveTwin(target)
            return twin.get_profile()

        if name == "who_explain":
            target = args["target"]
            claim = args["claim"]
            twin = CognitiveTwin(target)
            return twin.explain(claim)

        if name == "who_academic_research":
            topic = args["topic"]
            target = args.get("target")
            limit = int(args.get("limit", 5))
            ingest = bool(args.get("ingest", False))
            researcher = AcademicResearcher(target_name=target)
            return researcher.research_topic(topic, limit=limit, ingest=ingest)

        if name == "who_compose_outreach":
            target = args["target"]
            goal = args["goal"]
            context = args.get("context")
            twin = CognitiveTwin(target)
            return twin.compose_outreach(goal, context=context)

        if name == "who_list_targets":
            return list_targets()

        raise ValueError(f"Unknown tool name: {name}")

    def _send_response(self, response: Dict[str, Any]) -> None:
        payload = json.dumps(response, ensure_ascii=False)
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()


def main() -> None:
    """Run standard stdio MCP Server."""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
