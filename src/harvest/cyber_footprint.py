"""
Cyber Footprint, Email Reconnaissance & Breach Intelligence Engine for who??.
Integrates passive account enumeration (Holehe principles), DNS/MX/SPF/DMARC routing analysis,
and strictly verified breach exposure auditing (Zero-Hallucination compliant — AGENTS.md Rule #3).
Corresponds to Layer 1 (Direct Evidence) & Layer 2 (Cyber Threat Intelligence).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from src.config import settings
from src.utils.entity_validator import is_strict_entity_match
from src.utils.logger import get_logger

logger = get_logger("who.harvest.cyber")


class CyberFootprintAuditor:
    """Analyzes email domains, DNS/MX/SPF/DMARC posture, passive platform registration, and verified breach exposure."""

    DISPOSABLE_DOMAINS = {
        "tempmail.com",
        "10minutemail.com",
        "guerrillamail.com",
        "throwawaymail.com",
        "mailinator.com",
        "yopmail.com",
        "sharklasers.com",
        "trashmail.com",
        "getnada.com",
        "temp-mail.org",
    }

    def __init__(self, target_name: str, target_dirs: Optional[Any] = None):
        self.target_name = target_name
        self.dirs = target_dirs or settings.get_target_dirs(target_name)

    def validate_email_syntax(self, email: str) -> Dict[str, Any]:
        """Validate email structure and isolate domain."""
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        cleaned = email.strip()
        is_valid = bool(re.match(pattern, cleaned))
        domain = cleaned.split("@")[-1].lower() if "@" in cleaned else ""
        return {
            "email": cleaned,
            "is_valid_format": is_valid,
            "domain": domain,
        }

    def _query_doh(self, name: str, record_type: str = "MX") -> List[str]:
        """Query Google DNS-over-HTTPS passively for MX or TXT records without requiring extra packages."""
        if not name:
            return []
        url = f"https://dns.google/resolve?name={urllib.parse.quote(name)}&type={urllib.parse.quote(record_type)}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/dns-json", "User-Agent": "who-OSINT/4.5"})
            with urllib.request.urlopen(req, timeout=4) as res:
                if res.getcode() == 200:
                    payload = json.loads(res.read().decode("utf-8"))
                    answers = payload.get("Answer", []) or []
                    return [str(a.get("data", "")).strip('" ') for a in answers if a.get("data")]
        except Exception:
            pass
        return []

    def check_domain_dns(self, domain: str) -> Dict[str, Any]:
        """Inspect DNS A, MX, SPF, and DMARC security posture passively."""
        if not domain:
            return {
                "domain": "",
                "resolves": False,
                "resolved_ip": None,
                "mx_records": [],
                "has_mx": False,
                "has_spf": False,
                "has_dmarc": False,
                "is_disposable": False,
            }

        is_disposable = domain.lower() in self.DISPOSABLE_DOMAINS

        resolves = False
        resolved_ip = None
        try:
            resolved_ip = socket.gethostbyname(domain)
            resolves = True
        except Exception:
            pass

        mx_records = self._query_doh(domain, "MX")
        txt_records = self._query_doh(domain, "TXT")
        dmarc_records = self._query_doh(f"_dmarc.{domain}", "TXT")

        has_mx = len(mx_records) > 0
        has_spf = any("v=spf1" in r.lower() for r in txt_records)
        has_dmarc = any("v=dmarc1" in r.lower() for r in dmarc_records)

        if has_mx and not resolves:
            resolves = True

        return {
            "domain": domain,
            "resolves": resolves,
            "resolved_ip": resolved_ip,
            "mx_records": mx_records[:5],
            "has_mx": has_mx,
            "has_spf": has_spf,
            "has_dmarc": has_dmarc,
            "is_disposable": is_disposable,
        }

    def probe_passive_services(self, email: str) -> List[Dict[str, Any]]:
        """
        Passive service registration checks (Holehe principle):
        Checks public Gravatar and GitHub commit/account email index without sending target notifications.
        Enforces AGENTS.md Rule #1 (Zero-Substitution) when comparing named profiles.
        """
        email_clean = email.strip().lower()
        if not email_clean or "@" not in email_clean:
            return []

        email_md5 = hashlib.md5(email_clean.encode("utf-8")).hexdigest()
        email_sha256 = hashlib.sha256(email_clean.encode("utf-8")).hexdigest()

        found_services: List[Dict[str, Any]] = []

        # 1. Gravatar check (SHA-256 and MD5)
        for g_hash in [email_sha256, email_md5]:
            gravatar_url = f"https://www.gravatar.com/avatar/{g_hash}?d=404"
            try:
                req = urllib.request.Request(gravatar_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=4) as res:
                    if res.getcode() == 200:
                        found_services.append({
                            "service": "Gravatar / WordPress",
                            "status": "REGISTERED",
                            "avatar_url": gravatar_url,
                            "category": "Identity Profile",
                        })
                        break
            except Exception:
                pass

        # 2. GitHub user email search (verified directly by exact email identifier)
        github_search_url = f"https://api.github.com/search/users?q={urllib.parse.quote(email_clean)}+in:email"
        try:
            req = urllib.request.Request(
                github_search_url,
                headers={
                    "User-Agent": "who-OSINT-Engine/4.5",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=4) as res:
                if res.getcode() == 200:
                    gh_data = json.loads(res.read().decode("utf-8"))
                    if gh_data.get("total_count", 0) > 0:
                        top_user = gh_data["items"][0]
                        login = top_user.get("login", "")
                        # Check full profile name if available so namesakes are never conflated
                        name_verified = True
                        try:
                            prof_req = urllib.request.Request(
                                f"https://api.github.com/users/{urllib.parse.quote(login)}",
                                headers={"User-Agent": "who-OSINT-Engine/4.5"},
                            )
                            with urllib.request.urlopen(prof_req, timeout=4) as pres:
                                prof = json.loads(pres.read().decode("utf-8"))
                                declared_name = (prof.get("name") or "").strip()
                                if declared_name and self.target_name:
                                    matched, _ = is_strict_entity_match(self.target_name, declared_name)
                                    name_verified = matched
                        except Exception:
                            pass

                        if name_verified:
                            found_services.append({
                                "service": "GitHub",
                                "status": "REGISTERED",
                                "username": login,
                                "profile_url": top_user.get("html_url"),
                                "category": "Code & Development",
                            })
        except Exception:
            pass

        return found_services

    def audit_breach_exposure(
        self,
        email: str,
        dns_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Audit breach exposure strictly via verified primary evidence (AGENTS.md Rule #3: Zero-Hallucination).
        - NEVER fabricates fake LinkedIn/Dropbox/Adobe breach records.
        - Queries official HaveIBeenPwned v3 API if HIBP_API_KEY is configured.
        - Otherwise returns an honest UNVERIFIED status and evaluates digital hygiene strictly
          from verifiable DNS/MX/SPF/DMARC/Burner signals.
        """
        email_clean = email.strip().lower()
        dns = dns_info or {}
        hibp_key = os.getenv("HIBP_API_KEY", "").strip()

        # Compute verifiable infrastructure hygiene baseline (0-100)
        infra_score = 50
        if dns.get("resolves"):
            infra_score += 15
        if dns.get("has_mx"):
            infra_score += 10
        if dns.get("has_spf"):
            infra_score += 13
        if dns.get("has_dmarc"):
            infra_score += 12
        if dns.get("is_disposable"):
            infra_score = max(10, infra_score - 60)

        if hibp_key and "@" in email_clean:
            hibp_url = (
                f"https://haveibeenpwned.com/api/v3/breachedaccount/"
                f"{urllib.parse.quote(email_clean)}?truncateResponse=false"
            )
            req = urllib.request.Request(
                hibp_url,
                headers={
                    "hibp-api-key": hibp_key,
                    "user-agent": "who-OSINT-Engine/4.5",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=6) as res:
                    if res.getcode() == 200:
                        raw_breaches = json.loads(res.read().decode("utf-8"))
                        verified_incidents = [
                            {
                                "source": f"{b.get('Title', b.get('Name', 'Verified Breach'))} ({b.get('BreachDate', 'Unknown Date')})",
                                "data_types": b.get("DataClasses", []),
                                "is_verified": bool(b.get("IsVerified", True)),
                            }
                            for b in raw_breaches
                        ]
                        count = len(verified_incidents)
                        risk_tier = "HIGH" if count >= 3 else ("MEDIUM" if count >= 1 else "LOW")
                        hygiene = max(15, min(100, infra_score - (count * 18)))
                        return {
                            "email": email_clean,
                            "verification_status": "VERIFIED_HIBP_PRIMARY",
                            "risk_tier": risk_tier,
                            "digital_hygiene_score_pct": hygiene,
                            "estimated_breaches_count": count,
                            "documented_breach_incidents": verified_incidents,
                            "probabilistic_statement": (
                                f"Doğrulanmış Siber İhlal Kanıtı: '{email_clean}' adresi resmi HIBP veri tabanında "
                                f"{count} doğrulanmış sızıntı kaydında tespit edilmiştir (Risk: {risk_tier}, Güven Skoru: %98)."
                            ),
                        }
            except urllib.error.HTTPError as http_err:
                if http_err.code == 404:
                    return {
                        "email": email_clean,
                        "verification_status": "VERIFIED_CLEAN_HIBP",
                        "risk_tier": "LOW",
                        "digital_hygiene_score_pct": min(100, infra_score),
                        "estimated_breaches_count": 0,
                        "documented_breach_incidents": [],
                        "probabilistic_statement": (
                            f"Doğrulanmış Siber İhlal Kanıtı: '{email_clean}' adresi resmi HIBP veri tabanında "
                            "hiçbir bilinen veri sızıntısında yer almamaktadır (LOW Risk, Güven Skoru: %98)."
                        ),
                    }
            except Exception as exc:
                logger.debug(f"HIBP API check failed ({exc}); returning unverified zero-hallucination report.")

        # Zero-Hallucination Mode when no HIBP_API_KEY is configured
        risk_tier = "HIGH" if dns.get("is_disposable") else "UNVERIFIED"
        return {
            "email": email_clean,
            "verification_status": "UNVERIFIED_NO_BREACH_API_KEY",
            "risk_tier": risk_tier,
            "digital_hygiene_score_pct": min(100, infra_score),
            "estimated_breaches_count": 0,
            "documented_breach_incidents": [],
            "probabilistic_statement": (
                f"Siber İhlal Analizi (Sıfır Halüsinasyon Modu): Doğrulanmış sızıntı API anahtarı (HIBP_API_KEY) "
                f"tanımlı olmadığından '{email_clean}' için hiçbir varsayımsal veya uydurma sızıntı kaydı üretilmemiştir. "
                f"Dijital hijyen skoru (%{min(100, infra_score)}) yalnızca doğrulanabilir DNS/MX/SPF/DMARC altyapı kanıtlarından hesaplanmıştır."
            ),
        }

    def run_full_audit(self, email: str) -> Dict[str, Any]:
        """Run complete end-to-end cyber footprint audit for a target's email."""
        logger.info(f"Executing cyber footprint reconnaissance for: {email}")

        val = self.validate_email_syntax(email)
        dns = self.check_domain_dns(val["domain"])
        services = self.probe_passive_services(email) if val["is_valid_format"] else []
        breach = self.audit_breach_exposure(email, dns_info=dns)

        report = {
            "target_name": self.target_name,
            "email": email.strip(),
            "validation": val,
            "dns_infrastructure": dns,
            "passive_services_detected": services,
            "breach_exposure": breach,
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def _persist_report(self, report: Dict[str, Any]) -> None:
        """Save cyber footprint report to vault."""
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        domain_slug = re.sub(r"[^a-zA-Z0-9._-]", "_", report["validation"]["domain"] or "invalid")
        out_json = self.dirs.vault_dir / f"cyber_footprint_{domain_slug}.json"

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        dns = report["dns_infrastructure"]
        b = report["breach_exposure"]
        md_file = self.dirs.vault_dir / "cyber_footprint_report.md"
        lines = [
            f"# 🛡️ Cyber Footprint & Breach Intelligence Report: {self.target_name}",
            f"*Target Identifier:* `{report['email']}`",
            f"*Audited at:* {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
            "---",
            "## 1. 🌐 DNS & Mail Security Posture",
            f"- **Domain:** `{dns.get('domain')}`",
            f"- **Resolves:** `{'Yes (Active)' if dns.get('resolves') else 'No'}`",
            f"- **Resolved Host IP:** `{dns.get('resolved_ip') or 'N/A'}`",
            f"- **MX Records:** `{', '.join(dns.get('mx_records', [])) or 'Not detected / Offline'}`",
            f"- **SPF / DMARC Policy:** `SPF={'Yes' if dns.get('has_spf') else 'No'}, DMARC={'Yes' if dns.get('has_dmarc') else 'No'}`",
            f"- **Disposable/Burner Flag:** `{'YES (Risk)' if dns.get('is_disposable') else 'No (Legitimate)'}`\n",
            "## 2. 🔍 Passive Account Discovery",
        ]

        if report["passive_services_detected"]:
            for s in report["passive_services_detected"]:
                lines.append(f"- ✓ **{s['service']}** ({s['category']}) - Status: `{s['status']}`")
        else:
            lines.append("- No public Gravatar or verified GitHub commit email identity found for this address.\n")

        lines.extend([
            "\n## 3. ⚠️ Breach Exposure & Digital Hygiene",
            f"- **Doğrulama Durumu:** `{b.get('verification_status', 'UNVERIFIED')}`",
            f"- **Risk Seviyesi:** `{b['risk_tier']}`",
            f"- **Altyapı & Dijital Hijyen Skoru:** `%{b['digital_hygiene_score_pct']}`",
            f"- **{b['probabilistic_statement']}**\n",
            "### Doğrulanmış Sızıntı Kayıtları:",
        ])

        incidents = b.get("documented_breach_incidents", [])
        if incidents:
            for inc in incidents:
                lines.append(f"- **{inc['source']}**: {', '.join(inc.get('data_types', []))}")
        else:
            lines.append("- Doğrulanmış birincil sızıntı kaydı bulunmamaktadır (Sıfır Halüsinasyon Protokolü).")

        md_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved Cyber Footprint reports to {out_json} and {md_file}")
