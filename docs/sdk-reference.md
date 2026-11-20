# Python Agent SDK Reference

The `who` package provides an idiomatic Python interface designed for AI agents, multi-agent frameworks, and autonomous workflows.

## Top-Level Exports

```python
from who import (
    CognitiveTwin,
    AcademicResearcher,
    list_targets,
)
```

---

## `list_targets()`

Retrieves all targets currently available in the local vault.

```python
targets = list_targets()
for t in targets:
    print(t["slug"], t["is_ready"], t["raw_documents_count"])
```

**Returns:** `List[Dict[str, Any]]`
- `slug`: Directory name identifier.
- `name`: Formatted title name.
- `raw_documents_count`: Count of ingested documents.
- `is_ready`: Boolean indicating if vault and knowledge graph exist.
- `has_skills`: Boolean indicating if learned skills are generated.

---

## `CognitiveTwin`

Interface to an individual target's cognitive digital twin.

### Constructor
```python
twin = CognitiveTwin(target_name: str, mode: str = "autodetect")
```
- `target_name`: Name or slug of the profiled target.
- `mode`: Tone mode (`"autodetect"`, `"formal"`, `"informal"`).

### Methods

#### `chat(message: str, tone: Optional[str] = None) -> str`
Sends an interactive message and receives an in-character response calibrated to the target's persona, memory, and guardrails.

#### `simulate_scenario(scenario: str) -> Dict[str, Any]`
Simulates how the target would react to a hypothetical or adversarial proposition. Returns:
- `simulation_analysis`: Detailed strategic breakdown.
- `confidence_score`: Calibrated probability score.
- `cognitive_style_assumed`: Active reasoning pattern.

#### `explain(claim: str) -> Dict[str, Any]`
Traces any claim or behavioral assertion back to primary quotes, graph paths, and epistemic confidence.

#### `get_profile() -> Dict[str, Any]`
Returns full target metadata including psychometrics and active decision boundaries.

#### `get_operational_rules() -> List[str]`
Returns the active guardrails (e.g., `["[TACTICAL_TRANSACTIONALISM] ...", "[STOIC_EQUANIMITY] ..."]`).
