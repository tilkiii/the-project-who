"""
Temporal Digital Twin Snapshot & Cognitive Drift Diff Engine for who??.
Captures point-in-time behavioral/graph snapshots and quantifies temporal shifts in
epistemic calibration, dogmatic manipulation index, lexical diversity, concept nodes,
and [CONTRADICTS] cognitive dissonance tensions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.text_cleaner import sanitize_slug

logger = get_logger("who.vault.snapshot_diff")


class SnapshotManager:
    """Manages point-in-time profile snapshots and computes temporal cognitive drift."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs
        self.snapshots_dir: Path = self.dirs.base_dir / "snapshots"

    def _read_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug(f"Snapshot JSON read note ({path}): {e}")
        return default

    def capture_current_state(self, label: str = "live") -> Dict[str, Any]:
        """Compile a normalized point-in-time cognitive state dictionary from current vault & skills."""
        features = self._read_json(self.dirs.learned_skills_dir / "features.json", {})
        topics_obj = self._read_json(self.dirs.learned_skills_dir / "topics.json", {})
        graph_net = self._read_json(self.dirs.vault_dir / "concept_network.json", {})

        corpus_stats = features.get("corpus_stats", {})
        epistemic = features.get("epistemic_markers", {})
        manip = features.get("epistemic_manipulation_index", {})

        raw_nodes = graph_net.get("nodes", []) or graph_net.get("entities", [])
        concepts = sorted({str(n.get("name", "")).strip() for n in raw_nodes if str(n.get("name", "")).strip()})

        contradictions_set = set()
        for rel in graph_net.get("relationships", []):
            rel_type = rel.get("relation", "")
            diss_score = float(rel.get("dissonance_score", 0.0) or 0.0)
            if rel_type == "CONTRADICTS" or diss_score > 0.0:
                src = str(rel.get("source", "")).strip()
                tgt = str(rel.get("target", "")).strip()
                if src and tgt:
                    contradictions_set.add(f"{src} ⚡ {tgt}")

        raw_topics = topics_obj.get("topics", []) if isinstance(topics_obj, dict) else []
        topic_names = sorted({str(t.get("name", "")).strip() for t in raw_topics if str(t.get("name", "")).strip()})

        return {
            "target_name": self.target_name,
            "label": label,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "total_words": int(corpus_stats.get("total_words", 0) or 0),
                "type_token_ratio": float(corpus_stats.get("type_token_ratio", 0.0) or 0.0),
                "epistemic_calibration_ratio": float(epistemic.get("epistemic_calibration_ratio", 1.0) or 0.0),
                "manipulation_score": float(manip.get("score", 0.0) or 0.0),
            },
            "concepts": concepts,
            "contradictions": sorted(contradictions_set),
            "topics": topic_names,
        }

    def create_snapshot(self, label: Optional[str] = None) -> Path:
        """Persist a timestamped snapshot of the target's current cognitive vault."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        effective_label = sanitize_slug(label) if label else "checkpoint"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_%f")
        out_path = self.snapshots_dir / f"snapshot_{ts}_{effective_label}.json"

        state = self.capture_current_state(label=label or "checkpoint")
        out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved cognitive twin snapshot to {out_path}")
        return out_path

    def list_snapshots(self) -> List[Path]:
        """Return chronologically ordered snapshot files for the target."""
        if not self.snapshots_dir.exists():
            return []
        return sorted(self.snapshots_dir.glob("snapshot_*.json"))

    def diff_states(self, older: Dict[str, Any], newer: Dict[str, Any]) -> Dict[str, Any]:
        """Compute quantitative and structural drift between two cognitive snapshot states."""
        m_old = older.get("metrics", {})
        m_new = newer.get("metrics", {})

        cal_delta = round(float(m_new.get("epistemic_calibration_ratio", 1.0)) - float(m_old.get("epistemic_calibration_ratio", 1.0)), 3)
        manip_delta = round(float(m_new.get("manipulation_score", 0.0)) - float(m_old.get("manipulation_score", 0.0)), 3)
        ttr_delta = round(float(m_new.get("type_token_ratio", 0.0)) - float(m_old.get("type_token_ratio", 0.0)), 3)

        old_concepts = set(older.get("concepts", []))
        new_concepts = set(newer.get("concepts", []))
        added_concepts = sorted(new_concepts - old_concepts)
        removed_concepts = sorted(old_concepts - new_concepts)

        old_contra = set(older.get("contradictions", []))
        new_contra = set(newer.get("contradictions", []))
        new_contradictions = sorted(new_contra - old_contra)
        resolved_contradictions = sorted(old_contra - new_contra)

        old_topics = set(older.get("topics", []))
        new_topics = set(newer.get("topics", []))
        emerging_topics = sorted(new_topics - old_topics)
        dropped_topics = sorted(old_topics - new_topics)

        combined_words = int(m_old.get("total_words", 0)) + int(m_new.get("total_words", 0))
        confidence_score = round(min(0.94, 0.58 + min(combined_words, 2000) / 2000.0 * 0.34), 2)

        cal_trend = "artış (daha ihtiyatlı/kanıt odaklı üslup)" if cal_delta > 0 else (
            "düşüş (daha kesinlikçi üslup)" if cal_delta < 0 else "durağan seyir"
        )
        probabilistic_statement = (
            f"Zamansal Bilişsel Kayma Analizi: Gözlemlenen iki anlık görüntü ('{older.get('label', 'old')}' -> '{newer.get('label', 'new')}') "
            f"karşılaştırıldığında, epistemik kalibrasyon oranında {cal_delta:+.3f} yönünde olası bir {cal_trend} ve "
            f"manipülasyon endeksinde {manip_delta:+.3f} değişim gözlemlenmiştir. "
            f"Bilgi grafiğine {len(added_concepts)} yeni kavram ile {len(new_contradictions)} yeni [CONTRADICTS] gerilimi eklenmiştir "
            f"(Güven Skoru: %{int(confidence_score * 100)})."
        )

        return {
            "target_name": self.target_name,
            "from_label": older.get("label", "older"),
            "to_label": newer.get("label", "newer"),
            "from_timestamp": older.get("captured_at", ""),
            "to_timestamp": newer.get("captured_at", ""),
            "epistemic_calibration_delta": cal_delta,
            "manipulation_score_delta": manip_delta,
            "lexical_ttr_delta": ttr_delta,
            "added_concepts": added_concepts,
            "removed_concepts": removed_concepts,
            "new_contradictions": new_contradictions,
            "resolved_contradictions": resolved_contradictions,
            "emerging_topics": emerging_topics,
            "dropped_topics": dropped_topics,
            "confidence_score": confidence_score,
            "probabilistic_drift_statement": probabilistic_statement,
        }

    def compare_snapshots(
        self,
        snap_a_path: Optional[Path] = None,
        snap_b_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Compare two explicit snapshot files, the two latest saved snapshots,
        or a single saved baseline snapshot against the live vault state.
        """
        if snap_a_path and snap_b_path:
            older = self._read_json(Path(snap_a_path), {})
            newer = self._read_json(Path(snap_b_path), {})
            return self.diff_states(older, newer)

        snaps = self.list_snapshots()
        if len(snaps) >= 2:
            older = self._read_json(snaps[-2], {})
            newer = self._read_json(snaps[-1], {})
            return self.diff_states(older, newer)
        if len(snaps) == 1:
            older = self._read_json(snaps[0], {})
            newer = self.capture_current_state(label="current_live")
            return self.diff_states(older, newer)

        raise ValueError(
            f"No saved snapshots found for '{self.target_name}'. Run 'who snapshot --target \"{self.target_name}\"' first."
        )
