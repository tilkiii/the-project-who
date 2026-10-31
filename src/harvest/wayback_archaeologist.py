"""
Wayback Digital Archaeology & Temporal Discourse Revision Detector for who??.
Queries the Internet Archive CDX & Wayback endpoints (and supports offline temporal snapshot comparison)
to uncover silent biographical deletions, removed affiliations, and softened/hardened stance revisions.
Enforces AGENTS.md Rule #1 (Zero-Substitution Gate) & Rule #3 (Zero-Hallucination).
"""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

from src.config import settings
from src.utils.entity_validator import DISAMBIGUATION_NOTICE_TR, detect_unauthorized_name_expansion
from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_text, tokenize_words, tr_lower

logger = get_logger("who.harvest.archaeology")


class WaybackDigitalArchaeologist:
    """Detects historical revisions, silent deletions, and rhetorical shifts across archival web snapshots."""

    CERTAINTY_MARKERS = {
        "kesinlikle", "asla", "mutlaka", "şüphesiz", "tartışmasız",
        "definitely", "never", "always", "undoubtedly", "unquestionably", "committed",
    }
    HEDGING_MARKERS = {
        "belki", "olası", "muhtemelen", "koşullu", "araştırıyoruz",
        "perhaps", "plausible", "potentially", "exploring", "tentatively", "might",
    }

    def __init__(self, target_name: str, target_dirs: Optional[Any] = None):
        self.target_name = target_name
        self.dirs = target_dirs or settings.get_target_dirs(target_name)
        self.rejected_namesakes_count = 0

    def _html_to_clean_sentences(self, html_or_text: str) -> List[str]:
        """Extract clean readable sentences from HTML or plain text."""
        if not html_or_text:
            return []
        if "<html" in html_or_text.lower() or "<body" in html_or_text.lower() or "<p" in html_or_text.lower():
            soup = BeautifulSoup(html_or_text, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer", "svg"]):
                tag.decompose()
            text = clean_text(soup.get_text(separator="\n"))
        else:
            text = clean_text(html_or_text)

        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences: List[str] = []
        seen = set()
        for part in raw_parts:
            s = part.strip(" -•*\t")
            if len(s) >= 20:
                norm_key = tr_lower(s)
                if norm_key not in seen:
                    seen.add(norm_key)
                    sentences.append(s)
        return sentences

    def fetch_cdx_snapshots(self, url: str, max_snapshots: int = 4) -> List[Dict[str, str]]:
        """Query Wayback Machine CDX API for distinct content-digest HTTP 200 snapshots across time."""
        cdx_url = (
            "https://web.archive.org/cdx/search/cdx?"
            f"url={urllib.parse.quote(url)}&output=json"
            "&fl=timestamp,original,statuscode,digest&filter=statuscode:200&collapse=digest&limit=30"
        )
        try:
            req = urllib.request.Request(cdx_url, headers={"User-Agent": "who-OSINT-Archaeologist/4.5"})
            with urllib.request.urlopen(req, timeout=8) as res:
                if res.getcode() != 200:
                    return []
                rows = json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"Wayback CDX query skipped/unreachable for {url}: {exc}")
            return []

        if not rows or len(rows) < 2:
            return []

        headers = rows[0]
        records = [dict(zip(headers, r)) for r in rows[1:]]
        if len(records) <= max_snapshots:
            selected = records
        else:
            # Evenly sample from earliest to latest to capture multi-year trajectory
            step = (len(records) - 1) / float(max_snapshots - 1)
            selected = [records[round(i * step)] for i in range(max_snapshots)]

        snapshots: List[Dict[str, str]] = []
        for rec in selected:
            ts = rec.get("timestamp", "")
            orig = rec.get("original", url)
            snap_url = f"https://web.archive.org/web/{ts}id_/{orig}"
            try:
                req = urllib.request.Request(snap_url, headers={"User-Agent": "who-OSINT-Archaeologist/4.5"})
                with urllib.request.urlopen(req, timeout=8) as sres:
                    if sres.getcode() == 200:
                        body = sres.read().decode("utf-8", errors="replace")
                        # Enforce Zero-Substitution Gate before accepting snapshot
                        is_bad, reason = detect_unauthorized_name_expansion(self.target_name, body)
                        if is_bad:
                            self.rejected_namesakes_count += 1
                            logger.warning(f"[Zero-Substitution Gate] Ignored Wayback snapshot {ts}: {reason}")
                            continue
                        iso_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ts
                        snapshots.append({
                            "timestamp": iso_date,
                            "wayback_url": f"https://web.archive.org/web/{ts}/{orig}",
                            "content": body,
                        })
            except Exception as exc:
                logger.debug(f"Snapshot fetch failed for {snap_url}: {exc}")

        return snapshots

    def compare_snapshots(
        self,
        snapshots: List[Dict[str, str]],
        source_url: str = "archival_corpus",
    ) -> Dict[str, Any]:
        """
        Compare ordered temporal snapshots (each dict having 'timestamp' and 'content')
        to identify silent deletions, newly added claims, and stance softening/hardening.
        """
        valid_snaps = []
        for s in snapshots:
            content = s.get("content", "")
            is_bad, _ = detect_unauthorized_name_expansion(self.target_name, content)
            if is_bad:
                self.rejected_namesakes_count += 1
                continue
            valid_snaps.append({
                "timestamp": s.get("timestamp", "unknown_epoch"),
                "wayback_url": s.get("wayback_url", source_url),
                "sentences": self._html_to_clean_sentences(content),
            })

        valid_snaps.sort(key=lambda x: x["timestamp"])

        if len(valid_snaps) < 2:
            report = {
                "target_name": self.target_name,
                "source_url": source_url,
                "status": "INSUFFICIENT_TEMPORAL_SNAPSHOTS",
                "snapshots_analyzed": len(valid_snaps),
                "rejected_namesakes_count": self.rejected_namesakes_count,
                "silent_deletions": [],
                "newly_added_claims": [],
                "modified_revisions": [],
                "lexical_drift_pct": 0.0,
                "probabilistic_statement": (
                    "Dijital Arkeoloji Notu: Karşılaştırma yapabilmek için en az 2 farklı zamansal anlık görüntü "
                    "(Wayback kaydı veya tarihsel sürüm) gerekmektedir. Hiçbir varsayımsal revizyon üretilmemiştir."
                ),
                "audited_at": datetime.now(timezone.utc).isoformat(),
            }
            self._persist_report(report)
            return report

        earliest = valid_snaps[0]
        latest = valid_snaps[-1]

        early_map = {tr_lower(s): s for s in earliest["sentences"]}
        late_map = {tr_lower(s): s for s in latest["sentences"]}

        removed_keys = [k for k in early_map if k not in late_map]
        added_keys = [k for k in late_map if k not in early_map]

        modified_revisions: List[Dict[str, Any]] = []
        matched_added: set[str] = set()
        pure_deletions: List[Dict[str, Any]] = []

        for rk in removed_keys:
            best_match_key = None
            best_ratio = 0.0
            for ak in added_keys:
                if ak in matched_added:
                    continue
                ratio = difflib.SequenceMatcher(None, rk, ak).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match_key = ak

            if best_match_key and best_ratio >= 0.45:
                matched_added.add(best_match_key)
                old_sent = early_map[rk]
                new_sent = late_map[best_match_key]
                shift_type = self._classify_stance_shift(old_sent, new_sent)
                modified_revisions.append({
                    "from_epoch": earliest["timestamp"],
                    "to_epoch": latest["timestamp"],
                    "before_quote": old_sent,
                    "after_quote": new_sent,
                    "similarity_ratio": round(best_ratio, 2),
                    "revision_type": shift_type,
                })
            else:
                pure_deletions.append({
                    "present_in_epoch": earliest["timestamp"],
                    "absent_in_epoch": latest["timestamp"],
                    "deleted_quote": early_map[rk],
                })

        newly_added = [
            {
                "introduced_in_epoch": latest["timestamp"],
                "added_quote": late_map[ak],
            }
            for ak in added_keys
            if ak not in matched_added
        ]

        early_vocab = set(tokenize_words(" ".join(earliest["sentences"]), min_len=4))
        late_vocab = set(tokenize_words(" ".join(latest["sentences"]), min_len=4))
        union_vocab = early_vocab | late_vocab
        jaccard = (len(early_vocab & late_vocab) / len(union_vocab)) if union_vocab else 1.0
        lexical_drift_pct = round((1.0 - jaccard) * 100.0, 1)

        confidence = round(min(0.95, 0.70 + 0.08 * len(valid_snaps)), 2)

        report = {
            "target_name": self.target_name,
            "source_url": source_url,
            "status": "ARCHIVAL_DIFF_COMPLETED",
            "snapshots_analyzed": len(valid_snaps),
            "epoch_span": {"earliest": earliest["timestamp"], "latest": latest["timestamp"]},
            "rejected_namesakes_count": self.rejected_namesakes_count,
            "silent_deletions": pure_deletions[:15],
            "newly_added_claims": newly_added[:15],
            "modified_revisions": modified_revisions[:15],
            "lexical_drift_pct": lexical_drift_pct,
            "confidence_score": confidence,
            "disambiguation_notice": DISAMBIGUATION_NOTICE_TR if self.rejected_namesakes_count > 0 else None,
            "probabilistic_statement": (
                f"Dijital Arkeoloji Bulgusu ({earliest['timestamp']} -> {latest['timestamp']}): "
                f"Doğrulanmış arşiv sürümleri arasında {len(pure_deletions)} sessizce kaldırılmış ifade, "
                f"{len(modified_revisions)} revize edilmiş önerme ve %{lexical_drift_pct} kavramsal sözlük kayması "
                f"gözlemlenmiştir (Güven Skoru: %{int(confidence * 100)})."
            ),
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def audit_url_history(self, url: str, max_snapshots: int = 4) -> Dict[str, Any]:
        """Fetch archival snapshots from Wayback Machine for a URL and run full revision archaeology."""
        snapshots = self.fetch_cdx_snapshots(url, max_snapshots=max_snapshots)
        return self.compare_snapshots(snapshots, source_url=url)

    def _classify_stance_shift(self, old_sent: str, new_sent: str) -> str:
        """Classify whether a revised sentence softened, hardened, or reframed an earlier claim."""
        old_toks = set(tokenize_words(old_sent))
        new_toks = set(tokenize_words(new_sent))

        old_cert = len(old_toks & self.CERTAINTY_MARKERS)
        new_cert = len(new_toks & self.CERTAINTY_MARKERS)
        old_hedge = len(old_toks & self.HEDGING_MARKERS)
        new_hedge = len(new_toks & self.HEDGING_MARKERS)

        if old_cert > new_cert or new_hedge > old_hedge:
            return "STANCE_SOFTENED_OR_HEDGED"
        if new_cert > old_cert:
            return "STANCE_HARDENED"
        return "CONCEPTUAL_REFRAMING"

    def _persist_report(self, report: Dict[str, Any]) -> None:
        """Persist digital archaeology report to vault."""
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.dirs.vault_dir / "wayback_archaeology.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        md_path = self.dirs.vault_dir / "wayback_archaeology_report.md"
        lines = [
            f"# 🏛️ Wayback Digital Archaeology Report: {self.target_name}",
            f"*Source:* `{report.get('source_url')}`",
            f"*Snapshots Analyzed:* `{report.get('snapshots_analyzed')}` | *Lexical Drift:* `%{report.get('lexical_drift_pct')}`\n",
            f"> {report.get('probabilistic_statement')}\n",
            "## 1. Silent Deletions (Kaldırılan İfadeler / Ortaklıklar)",
        ]
        if report["silent_deletions"]:
            for d in report["silent_deletions"]:
                lines.append(f"- **[{d['present_in_epoch']} -> Silindi {d['absent_in_epoch']}]** \"{d['deleted_quote']}\"")
        else:
            lines.append("- Kaldırılan birincil ifade tespit edilmedi.")

        lines.append("\n## 2. Modified & Reframed Statements (Söylem Revizyonları)")
        if report["modified_revisions"]:
            for m in report["modified_revisions"]:
                lines.append(
                    f"- **[{m['revision_type']}]** ({m['from_epoch']} -> {m['to_epoch']})\n"
                    f"  - *Önceki:* \"{m['before_quote']}\"\n"
                    f"  - *Sonraki:* \"{m['after_quote']}\""
                )
        else:
            lines.append("- Doğrudan cümle revizyonu tespit edilmedi.")

        lines.append("\n## 3. Newly Introduced Claims (Yeni Eklenen İfadeler)")
        if report["newly_added_claims"]:
            for a in report["newly_added_claims"][:8]:
                lines.append(f"- **[{a['introduced_in_epoch']}]** \"{a['added_quote']}\"")
        else:
            lines.append("- Yeni eklenen ifade tespit edilmedi.")

        md_path.write_text("\n".join(lines), encoding="utf-8")

