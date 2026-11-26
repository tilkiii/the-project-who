"""
Knowledge graph storage and query engine.
Supports Neo4j Cypher queries with an in-memory NetworkX fallback.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx

from who.config import settings
from who.utils.logger import get_logger

logger = get_logger("who.vault.store")


class GraphStore:
    """Manages persistent entity and relationship graphs in Neo4j with offline NetworkX fallback."""

    def __init__(self, target_name: str, vault_dir: Optional[Path] = None):
        self.target_name = target_name
        self.vault_dir = vault_dir
        self.driver = None
        self.connected_to_neo4j = False
        self.nx_graph = nx.DiGraph()

        self._init_neo4j()
        self._load_local_graph()

    def _init_neo4j(self) -> None:
        """Attempt to connect to Neo4j only when NEO4J_URI is explicitly configured."""
        uri = (os.getenv("NEO4J_URI") or settings.neo4j_uri or "").strip()
        if not uri:
            logger.debug("NEO4J_URI not configured; using local NetworkX graph store.")
            self.connected_to_neo4j = False
            return

        try:
            from neo4j import GraphDatabase
            try:
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                    connection_timeout=1.5,
                )
            except TypeError:
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
            # Verify connectivity
            self.driver.verify_connectivity()
            self.connected_to_neo4j = True
            logger.info(f"Connected to Neo4j at {uri}")
            self._setup_schema()
        except Exception as e:
            logger.info("Neo4j instance unreachable. Operating seamlessly in local NetworkX graph mode.")
            logger.debug(f"Neo4j connection detail: {e}")
            self.connected_to_neo4j = False

    def _setup_schema(self) -> None:
        """Setup uniqueness constraints and indexes in Neo4j."""
        if not self.connected_to_neo4j or not self.driver:
            return
        try:
            with self.driver.session(database=settings.neo4j_database) as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        except Exception as e:
            logger.debug(f"Schema setup warning: {e}")

    def store_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
        """Persist entities and relationships into Neo4j and local store."""
        logger.info(f"Storing {len(entities)} entities and {len(relationships)} relationships.")

        # 1. Store in local NetworkX graph
        for ent in entities:
            name = ent.get("name", "").strip()
            if name:
                self.nx_graph.add_node(
                    name,
                    type=ent.get("type", "Concept"),
                    description=ent.get("description", ""),
                    version=self.nx_graph.nodes[name].get("version", 0) + 1 if name in self.nx_graph else 1,
                    target_owner=self.target_name,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )

        for rel in relationships:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            if src and tgt:
                edge_props = {
                    "relation": rel.get("relation", "RELATES_TO"),
                    "weight": float(rel.get("weight", 0.8)),
                    "quote": rel.get("quote", ""),
                    "source_doc": rel.get("source_doc", ""),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                if rel.get("relation") == "CONTRADICTS" or "dissonance_score" in rel:
                    edge_props["dissonance_score"] = float(rel.get("dissonance_score", 0.85))
                    edge_props["contradiction_summary"] = rel.get("contradiction_summary", "")
                    edge_props["quote_a"] = rel.get("quote_a", "")
                    edge_props["quote_b"] = rel.get("quote_b", "")

                self.nx_graph.add_edge(src, tgt, **edge_props)

        # 2. Store in Neo4j if online
        if self.connected_to_neo4j and self.driver:
            try:
                with self.driver.session(database=settings.neo4j_database) as session:
                    # Merge entities
                    for ent in entities:
                        session.run(
                            """
                            MERGE (e:Entity {name: $name})
                            ON CREATE SET e.type = $type, e.description = $description,
                                          e.version = 1, e.target_owner = $target,
                                          e.created_at = datetime()
                            ON MATCH SET e.type = $type, e.description = $description,
                                         e.version = coalesce(e.version, 0) + 1,
                                         e.updated_at = datetime()
                            """,
                            name=ent.get("name"),
                            type=ent.get("type", "Concept"),
                            description=ent.get("description", ""),
                            target=self.target_name,
                        )

                    # Merge relationships
                    for rel in relationships:
                        rel_type = rel.get("relation", "RELATES_TO").upper().replace(" ", "_")
                        query = f"""
                        MATCH (s:Entity {{name: $source}})
                        MATCH (t:Entity {{name: $target}})
                        MERGE (s)-[r:{rel_type}]->(t)
                        SET r.weight = $weight, r.quote = $quote,
                            r.source_doc = $source_doc,
                            r.dissonance_score = $dissonance_score,
                            r.contradiction_summary = $contradiction_summary,
                            r.quote_a = $quote_a,
                            r.quote_b = $quote_b,
                            r.updated_at = datetime()
                        """
                        session.run(
                            query,
                            source=rel.get("source"),
                            target=rel.get("target"),
                            weight=float(rel.get("weight", 0.8)),
                            quote=rel.get("quote", ""),
                            source_doc=rel.get("source_doc", ""),
                            dissonance_score=float(rel.get("dissonance_score", 0.0)),
                            contradiction_summary=rel.get("contradiction_summary", ""),
                            quote_a=rel.get("quote_a", ""),
                            quote_b=rel.get("quote_b", ""),
                        )
                logger.info("Synchronized graph to Neo4j successfully.")
            except Exception as e:
                logger.error(f"Failed to persist graph to Neo4j: {e}")

        # 3. Save local JSON backup
        self.export_graph_json()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Retrieve all nodes with attributes."""
        nodes = []
        for n, data in self.nx_graph.nodes(data=True):
            nodes.append({"name": n, **data})
        return nodes

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Retrieve all relationships with attributes."""
        rels = []
        for u, v, data in self.nx_graph.edges(data=True):
            rels.append({"source": u, "target": v, **data})
        return rels

    def export_graph_json(self) -> Dict[str, Any]:
        """Export current graph state to concept_network.json."""
        payload = {
            "target_name": self.target_name,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "nodes": self.get_all_nodes(),
            "relationships": self.get_all_relationships(),
        }
        if self.vault_dir:
            self.vault_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.vault_dir / "concept_network.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        return payload

    def _load_local_graph(self) -> None:
        """Load graph state from local concept_network.json or knowledge_graph.json if present."""
        if not self.vault_dir:
            return
        for filename in ("concept_network.json", "knowledge_graph.json"):
            net_file = self.vault_dir / filename
            if not net_file.exists():
                continue
            try:
                with open(net_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                nodes = data.get("nodes", []) or data.get("entities", [])
                for node in nodes:
                    name = node.get("name") or node.get("id")
                    if name:
                        attrs = {k: v for k, v in node.items() if k not in ("name", "id")}
                        self.nx_graph.add_node(name, **attrs)
                rels = data.get("relationships") or data.get("edges") or []
                for rel in rels:
                    src = rel.get("source")
                    tgt = rel.get("target")
                    if src and tgt:
                        attrs = {k: v for k, v in rel.items() if k not in ("source", "target")}
                        attrs.setdefault("relation", "RELATES_TO")
                        self.nx_graph.add_edge(src, tgt, **attrs)
                return
            except Exception as e:
                logger.debug(f"Could not load existing {filename}: {e}")

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass
