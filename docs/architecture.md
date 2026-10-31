# Architecture & System Design

The `who??` engine operates across three major operational pipelines: **Harvesting & Ingestion**, **Cognitive Modeling**, and **Agent Interface Delivery**.

```mermaid
graph TD
    subgraph Data Layer
        H[OSINT Harvesters] --> R[Raw Data Corpus]
        R --> V[Vector Embedder - ChromaDB]
        R --> G[Knowledge Graph - NetworkX / Neo4j]
    end

    subgraph Cognitive Layer
        V & G --> P[Psychometric Profiler]
        P --> S[Learned Skills & Decision Boundaries]
        S --> C[Cognitive Twin Engine]
    end

    subgraph Delivery Layer
        C --> SDK[Python SDK]
        C --> CLI[Terminal CLI]
        C --> MCP[Model Context Protocol Server]
    end
```

## 1. Zero-Substitution Entity Resolution
In strict OSINT intelligence, false attribution destroys model validity. The ingestion pipeline enforces exact token matching and disallows hallucinated aliases or name conflation.

## 2. 5-Dimension Behavioral Psychometrics
Profiles are analyzed along five calibrated dimensions:
1. **Machiavellianism & Power Orientation:** Pragmatic utility vs. moralistic emotional framing.
2. **Need for Cognitive Closure (NFC):** Tolerance for ambiguity vs. demand for structured decisiveness.
3. **Prospect Risk Calibration:** Loss aversion ratio ($L/G$) and asymmetric risk preference.
4. **Affective Volatility:** Composure under adversarial challenge vs. emotional escalation.
5. **Moral Foundations Matrix:** In-group loyalty, authority, fairness, and care priorities.

## 3. Active Behavioral Boundaries
Derived rules (e.g. `[TACTICAL_TRANSACTIONALISM]`, `[STOIC_EQUANIMITY]`) actively shape LLM system prompts and prevent persona drift during multi-turn exchanges.
