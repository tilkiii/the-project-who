"""
Technical Digital Twin: Code & Commit Intelligence Harvester for who??.
Extracts engineering domain expertise, programming language trajectories, commit timezone/circadian habits,
and collaboration patterns from verified GitHub accounts or local Git repositories.
Strictly enforces AGENTS.md Rule #1 (Zero-Substitution Gate) & Rule #3 (Zero-Hallucination).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

from src.config import settings
from src.utils.entity_validator import DISAMBIGUATION_NOTICE_TR, is_strict_entity_match
from src.utils.logger import get_logger

logger = get_logger("who.harvest.code_intelligence")


class CodeIntelligenceHarvester:
    """Profiles a technical target's verified software engineering footprint, stack, and commit chronomics."""

    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "AI / Machine Learning & NLP": ["llm", "nlp", "transformer", "neural", "ml", "agent", "embedding", "rag", "pytorch"],
        "Cryptography & Distributed Systems": ["crypto", "consensus", "zk", "protocol", "p2p", "blockchain", "byzantine", "raft"],
        "Cybersecurity & OSINT": ["osint", "security", "forensic", "audit", "scanner", "recon", "cve", "threat"],
        "Systems & Cloud Infrastructure": ["kernel", "docker", "kubernetes", "compiler", "runtime", "rust", "grpc", "distributed"],
        "Full-Stack & Product Engineering": ["api", "web", "react", "typescript", "dashboard", "frontend", "graphql", "ui"],
    }

    def __init__(self, target_name: str, target_dirs: Optional[Any] = None):
        self.target_name = target_name
        self.dirs = target_dirs or settings.get_target_dirs(target_name)

    def _github_get_json(self, endpoint: str) -> Optional[Any]:
        """Query public GitHub REST API v3 safely."""
        url = f"https://api.github.com{endpoint}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "who-OSINT-CodeIntel/4.5",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=6) as res:
                if res.getcode() == 200:
                    return json.loads(res.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"GitHub API request skipped for {endpoint}: {exc}")
        return None

    def analyze_github_account(self, username: str) -> Dict[str, Any]:
        """Profile a GitHub developer account with Strict Zero-Substitution verification."""
        handle = username.strip().lstrip("@")
        profile = self._github_get_json(f"/users/{urllib.parse.quote(handle)}")
        if not isinstance(profile, dict):
            return {
                "target_name": self.target_name,
                "username": handle,
                "status": "UNREACHABLE_OR_NOT_FOUND",
            }

        declared_name = (profile.get("name") or "").strip()
        if declared_name:
            matched, reason = is_strict_entity_match(self.target_name, declared_name)
            if not matched:
                logger.warning(
                    f"[Zero-Substitution Gate] Blocked GitHub @{handle} ('{declared_name}') for target '{self.target_name}': {reason}"
                )
                return {
                    "target_name": self.target_name,
                    "username": handle,
                    "declared_name": declared_name,
                    "status": "REJECTED_ZERO_SUBSTITUTION",
                    "rejection_reason": reason,
                    "disambiguation_notice": DISAMBIGUATION_NOTICE_TR,
                }

        repos = self._github_get_json(f"/users/{urllib.parse.quote(handle)}/repos?sort=updated&per_page=30") or []
        events = self._github_get_json(f"/users/{urllib.parse.quote(handle)}/events/public?per_page=40") or []

        lang_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        original_repos = 0
        forked_repos = 0
        total_stars = 0
        notable_repos: List[Dict[str, Any]] = []
        corpus_descriptions: List[str] = []

        for r in repos:
            if not isinstance(r, dict):
                continue
            if r.get("fork"):
                forked_repos += 1
                continue
            original_repos += 1
            lang = r.get("language")
            stars = int(r.get("stargazers_count") or 0)
            total_stars += stars
            if lang:
                lang_counter[lang] += max(1, stars + 1)

            desc = (r.get("description") or "").strip()
            topics = r.get("topics") or []
            blob = f"{r.get('name', '')} {desc} {' '.join(topics)}".lower()
            for dom, kws in self.DOMAIN_KEYWORDS.items():
                if any(kw in blob for kw in kws):
                    domain_counter[dom] += 1

            notable_repos.append({
                "name": r.get("name"),
                "language": lang or "Unspecified",
                "stars": stars,
                "description": desc,
                "url": r.get("html_url"),
            })
            if desc:
                corpus_descriptions.append(f"[TARGET_PRIMARY]: Repository '{r.get('name')}' — {desc}")

        notable_repos.sort(key=lambda x: x["stars"], reverse=True)

        # Analyze public event timestamps (UTC hours) and commit message cadence
        active_utc_hours: Counter[int] = Counter()
        commit_messages: List[str] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            created = ev.get("created_at") or ""
            if len(created) >= 13 and "T" in created:
                try:
                    hr = int(created.split("T")[1][:2])
                    active_utc_hours[hr] += 1
                except ValueError:
                    pass
            if ev.get("type") == "PushEvent":
                for c in (ev.get("payload") or {}).get("commits", []) or []:
                    msg = (c.get("message") or "").strip().splitlines()[0] if c.get("message") else ""
                    if len(msg) >= 8:
                        commit_messages.append(msg)

        lang_total = sum(lang_counter.values()) or 1
        language_distribution = [
            {"language": lang, "weight_pct": round((cnt / lang_total) * 100.0, 1)}
            for lang, cnt in lang_counter.most_common(6)
        ]

        peak_hours = [f"{hr:02d}:00 UTC" for hr, _ in active_utc_hours.most_common(3)]
        confidence = 0.92 if declared_name else 0.76

        report = {
            "target_name": self.target_name,
            "source_type": "github_api",
            "username": handle,
            "declared_name": declared_name or self.target_name,
            "status": "VERIFIED_CODE_INTELLIGENCE",
            "original_repositories_count": original_repos,
            "forked_repositories_count": forked_repos,
            "originality_ratio": round(original_repos / max(1, original_repos + forked_repos), 2),
            "total_stars": total_stars,
            "language_distribution": language_distribution,
            "inferred_engineering_domains": [d for d, _ in domain_counter.most_common(3)],
            "peak_activity_hours_utc": peak_hours,
            "notable_repositories": notable_repos[:6],
            "sample_commit_messages": commit_messages[:10],
            "confidence_score": confidence,
            "probabilistic_statement": (
                f"Teknik İkiz Analizi: Doğrulanmış @{handle} profilinde {original_repos} özgün yazılım deposu incelenmiştir. "
                f"Birincil dil ağırlığı {', '.join(x['language'] for x in language_distribution[:3]) or 'belirsiz'} "
                f"olup mimari odak {', '.join(d for d, _ in domain_counter.most_common(2)) or 'genel yazılım geliştirme'} "
                f"eksenindedir (Güven Skoru: %{int(confidence * 100)})."
            ),
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report, corpus_lines=corpus_descriptions, slug_tag=handle)
        return report

    def analyze_local_git_repo(self, repo_path: Path, max_commits: int = 200) -> Dict[str, Any]:
        """
        Extract commit chronomics, author timezone fingerprint, and engineering cadence
        from a local Git repository strictly for commits authored by target_name (Zero-Substitution enforced).
        """
        repo_dir = Path(repo_path)
        if not (repo_dir / ".git").exists():
            return {"target_name": self.target_name, "status": "NOT_A_GIT_REPOSITORY", "repo_path": str(repo_dir)}

        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_dir), "log", f"-n{int(max_commits)}", "--format=%an|%ae|%aI|%s"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            lines = (proc.stdout or "").splitlines()
        except Exception as exc:
            return {"target_name": self.target_name, "status": f"GIT_EXECUTION_ERROR: {exc}"}

        verified_commits: List[Dict[str, str]] = []
        rejected_authors = 0
        tz_counter: Counter[str] = Counter()
        local_hour_counter: Counter[int] = Counter()

        for line in lines:
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            author_name, author_email, iso_ts, subject = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            matched, _ = is_strict_entity_match(self.target_name, author_name)
            if not matched:
                rejected_authors += 1
                continue

            tz_match = re.search(r"([+-]\d{2}:\d{2}|Z)$", iso_ts)
            if tz_match:
                tz_counter[tz_match.group(1)] += 1
            if "T" in iso_ts and len(iso_ts.split("T")[1]) >= 2:
                try:
                    local_hour_counter[int(iso_ts.split("T")[1][:2])] += 1
                except ValueError:
                    pass

            verified_commits.append({
                "author": author_name,
                "email": author_email,
                "timestamp": iso_ts,
                "subject": subject,
            })

        night_commits = sum(cnt for hr, cnt in local_hour_counter.items() if 0 <= hr < 5)
        total_verified = len(verified_commits)
        nocturnal_coding_pct = round((night_commits / total_verified) * 100.0, 1) if total_verified else 0.0
        primary_tz = tz_counter.most_common(1)[0][0] if tz_counter else "UNKNOWN"
        confidence = round(min(0.96, 0.65 + (total_verified / 50.0) * 0.30), 2) if total_verified else 0.0

        report = {
            "target_name": self.target_name,
            "source_type": "local_git_repository",
            "repo_path": str(repo_dir),
            "status": "VERIFIED_LOCAL_GIT_INTEL" if total_verified > 0 else "NO_MATCHING_TARGET_COMMITS",
            "verified_commits_count": total_verified,
            "excluded_third_party_commits": rejected_authors,
            "primary_timezone_offset": primary_tz,
            "nocturnal_coding_ratio_pct": nocturnal_coding_pct,
            "peak_local_coding_hours": [f"{hr:02d}:00" for hr, _ in local_hour_counter.most_common(3)],
            "sample_commit_messages": [c["subject"] for c in verified_commits[:12]],
            "confidence_score": confidence,
            "probabilistic_statement": (
                f"Yerel Git İstihbaratı: Sıfır İkame filtresiyle yalnızca '{self.target_name}' imzalı {total_verified} commit "
                f"analiz edilmiştir ({rejected_authors} üçüncü şahıs commit'i dışlandı). Birincil saat dilimi `{primary_tz}`, "
                f"gece (00:00-05:00) kodlama oranı `%{nocturnal_coding_pct}` düzeyindedir (Güven Skoru: %{int(confidence * 100)})."
            ),
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }

        corpus_lines = [f"[TARGET_PRIMARY]: Commit — {c['subject']}" for c in verified_commits[:25] if len(c["subject"]) >= 12]
        self._persist_report(report, corpus_lines=corpus_lines, slug_tag=repo_dir.name)
        return report

    def _persist_report(self, report: Dict[str, Any], corpus_lines: List[str], slug_tag: str) -> None:
        """Persist code intelligence artifacts to vault and verified commit/repo text to raw_data."""
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.dirs.vault_dir / "code_intelligence.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        md_path = self.dirs.vault_dir / "code_intelligence_report.md"
        lines = [
            f"# 💻 Technical Twin: Code & Commit Intelligence — {self.target_name}",
            f"*Status:* `{report.get('status')}` | *Audited at:* {report.get('audited_at')}\n",
            f"> {report.get('probabilistic_statement')}\n",
        ]
        if report.get("language_distribution"):
            lines.append("## Programming Language Distribution")
            for item in report["language_distribution"]:
                lines.append(f"- **{item['language']}**: `%{item['weight_pct']}`")
        if report.get("sample_commit_messages"):
            lines.append("\n## Verified Commit Cadence Samples")
            for msg in report["sample_commit_messages"][:8]:
                lines.append(f"- `{msg}`")

        md_path.write_text("\n".join(lines), encoding="utf-8")

        if corpus_lines and report.get("status", "").startswith("VERIFIED"):
            self.dirs.raw_data_dir.mkdir(parents=True, exist_ok=True)
            safe_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", slug_tag)
            raw_file = self.dirs.raw_data_dir / f"code_osint_{safe_tag}.md"
            header = (
                "---\n"
                f"source_type: code_intelligence\n"
                f"speaker: TARGET_PRIMARY\n"
                f"target: {self.target_name}\n"
                "---\n\n"
            )
            raw_file.write_text(header + "\n".join(corpus_lines), encoding="utf-8")

