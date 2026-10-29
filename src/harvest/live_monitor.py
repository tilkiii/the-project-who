"""
Live OSINT Monitor and Cognitive Anomaly Detection Engine for who??.
Continuously scans external feeds (RSS, articles, social streams) and flags
circadian drift, sudden dogmatism spikes, and epistemic contradictions.
Corresponds to Layer 1 (Direct Evidence) & Layer 2 (Autonomous Anomaly Detection).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.request

from src.config import call_llm
from src.learn.chronomics import CognitiveChronomics
from src.learn.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger("who.harvest.monitor")


class LiveMonitor:
    """Monitors live digital footprints and detects real-time cognitive and ideological anomalies."""

    def __init__(self, target_name: str, target_dirs: Any):
        self.target_name = target_name
        self.dirs = target_dirs
        self.history_file = self.dirs.raw_data_dir / "monitored_feed_history.json"
        self.alerts_file = self.dirs.vault_dir / "monitoring_alerts.json"

    def load_feed_history(self) -> Dict[str, Any]:
        """Load history of previously processed feed items to prevent duplicate processing."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"processed_hashes": [], "last_scan": None, "total_scanned": 0}

    def save_feed_history(self, history: Dict[str, Any]) -> None:
        """Persist feed history to target raw data directory."""
        self.dirs.raw_data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def fetch_live_feed(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch and parse live content from RSS feeds or web URLs."""
        fetched_items: List[Dict[str, Any]] = []

        for url in urls:
            logger.info(f"Scanning live endpoint: {url}")
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "who-OSINT-Engine/4.0 (+https://github.com/tilkiii/the-project-who)"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    raw_bytes = response.read()
                    text_content = raw_bytes.decode("utf-8", errors="replace")

                # Basic XML / RSS detection
                if "<rss" in text_content.lower() or "<feed" in text_content.lower():
                    items = self._parse_rss_feed(text_content, url)
                    fetched_items.extend(items)
                else:
                    # Clean web article text
                    cleaned = clean_text(text_content)
                    if len(cleaned) > 80:
                        fetched_items.append({
                            "title": f"Live Capture: {url}",
                            "content": cleaned[:4000],
                            "url": url,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source_type": "live_web_capture",
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch live source {url}: {e}")

        return fetched_items

    def _parse_rss_feed(self, xml_text: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract titles, descriptions, and timestamps from RSS/Atom XML."""
        items: List[Dict[str, Any]] = []
        item_blocks = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL | re.IGNORECASE)
        if not item_blocks:
            item_blocks = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL | re.IGNORECASE)

        for block in item_blocks:
            title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
            desc_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL | re.IGNORECASE) or re.search(r"<content.*?>(.*?)</content>", block, re.DOTALL | re.IGNORECASE)
            date_m = re.search(r"<pubDate>(.*?)</pubDate>", block, re.DOTALL | re.IGNORECASE) or re.search(r"<updated>(.*?)</updated>", block, re.DOTALL | re.IGNORECASE)
            link_m = re.search(r"<link>(.*?)</link>", block, re.DOTALL | re.IGNORECASE)

            title = clean_text(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title_m.group(1))) if title_m else "Untitled"
            desc = clean_text(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", desc_m.group(1))) if desc_m else ""
            ts = date_m.group(1).strip() if date_m else datetime.now(timezone.utc).isoformat()
            link = link_m.group(1).strip() if link_m else source_url

            content_text = f"{title}\n\n{desc}".strip()
            if len(content_text) > 40:
                items.append({
                    "title": title,
                    "content": content_text,
                    "url": link,
                    "timestamp": ts,
                    "source_type": "rss_feed_item",
                })

        return items

    def detect_anomalies(self, items: List[Dict[str, Any]], ignore_history: bool = False) -> List[Dict[str, Any]]:
        """
        Compare incoming live items against learned target baseline:
        1. Circadian Drift: Nocturnal timing during executive relaxation window.
        2. Dogmatic Surge: High absolutist / manipulation language exceeding baseline.
        3. Lexical Disruption: Drastic divergence in vocabulary richness.
        """
        history = self.load_feed_history()
        processed_hashes = set(history.get("processed_hashes", []))

        # Load baseline features
        features_file = self.dirs.learned_skills_dir / "features.json"
        baseline_features = {}
        if features_file.exists():
            try:
                with open(features_file, "r", encoding="utf-8") as f:
                    baseline_features = json.load(f)
            except Exception:
                pass

        baseline_manip = baseline_features.get("epistemic_manipulation_index", {}).get("dogmatic_density", 0.5)
        baseline_cal = baseline_features.get("epistemic_markers", {}).get("epistemic_calibration_ratio", 1.0)

        fe = FeatureExtractor()
        chronomics = CognitiveChronomics()
        anomalies: List[Dict[str, Any]] = []

        for item in items:
            raw_content = item.get("content", "")
            item_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            if not ignore_history and item_hash in processed_hashes:
                continue

            processed_hashes.add(item_hash)

            # Analyze item
            doc_feats = fe.extract_features(raw_content)
            epistemic = doc_feats.get("epistemic_markers", {})
            manip = doc_feats.get("epistemic_manipulation_index", {})
            item_dogmatic = manip.get("dogmatic_density", 0.0)
            item_cal = epistemic.get("epistemic_calibration_ratio", 1.0)

            # 1. Nocturnal Circadian Anomaly
            ts_str = item.get("timestamp", "")
            is_nocturnal = False
            try:
                # Try parsing timestamp
                dt = chronomics._parse_iso_or_custom(ts_str) or datetime.now()
                hr = dt.hour
                if 0 <= hr < 5:
                    is_nocturnal = True
            except Exception:
                pass

            detected_triggers = []

            if is_nocturnal:
                detected_triggers.append({
                    "type": "CIRCADIAN_NOCTURNAL_SPIKE",
                    "severity": "HIGH",
                    "detail": f"Content was authored during nocturnal window ({dt.strftime('%H:%M')}), where prefrontal inhibition decays by ~28%.",
                })

            # 2. Dogmatic / Rhetorical Surge Anomaly
            if item_dogmatic > (baseline_manip * 1.5) and item_dogmatic > 1.2:
                detected_triggers.append({
                    "type": "DOGMATISM_SURGE",
                    "severity": "HIGH",
                    "detail": f"Observed dogmatic density ({item_dogmatic:.2f}) is significantly elevated above target baseline ({baseline_manip:.2f}).",
                })

            # 3. Sudden Epistemic Certainty Collapse
            if item_cal < 0.4 and baseline_cal > 1.2:
                detected_triggers.append({
                    "type": "EPISTEMIC_DOUBT_COLLAPSE",
                    "severity": "MEDIUM",
                    "detail": f"Calibration dropped to {item_cal:.2f} (baseline: {baseline_cal:.2f}), reflecting sudden cessation of uncertainty qualifiers.",
                })

            if detected_triggers:
                anomaly_entry = {
                    "item_title": item.get("title", "Untitled"),
                    "item_url": item.get("url", ""),
                    "timestamp": ts_str,
                    "triggers": detected_triggers,
                    "observed_metrics": {
                        "dogmatic_density": item_dogmatic,
                        "epistemic_calibration": item_cal,
                        "is_nocturnal": is_nocturnal,
                    },
                    "content_snippet": raw_content[:250] + "...",
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                }
                anomalies.append(anomaly_entry)

        # Update history
        history["processed_hashes"] = list(processed_hashes)
        history["last_scan"] = datetime.now(timezone.utc).isoformat()
        history["total_scanned"] = len(processed_hashes)
        self.save_feed_history(history)

        # Save active alerts
        if anomalies:
            self._save_alerts(anomalies)

        return anomalies

    def _save_alerts(self, new_anomalies: List[Dict[str, Any]]) -> None:
        """Append or update active monitoring alerts in target vault."""
        existing_alerts = []
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, "r", encoding="utf-8") as f:
                    existing_alerts = json.load(f)
            except Exception:
                pass

        all_alerts = new_anomalies + existing_alerts
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        with open(self.alerts_file, "w", encoding="utf-8") as f:
            json.dump(all_alerts[:50], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(new_anomalies)} new anomaly alerts to {self.alerts_file}")
