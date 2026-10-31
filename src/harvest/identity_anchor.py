"""
Cryptographic & Bidirectional Cross-Platform Identity Anchor for who??.
Enforces AGENTS.md Rule #1 (Strict Zero-Substitution Protocol) & Rule #3 (Epistemic Honesty):
- Verifies bidirectional <link rel="me"> / <a rel="me"> links across personal websites and profiles.
- Extracts cryptographic GPG Key IDs and SSH public key SHA-256 fingerprints from verified GitHub accounts.
- Validates ORCID scholarly identities against exact target names (blocking namesake expansions).
- Audits DNS TXT ownership records (_atproto, keybase, github, google-site-verification, openpgpkey).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

from src.config import settings
from src.utils.entity_validator import DISAMBIGUATION_NOTICE_TR, is_strict_entity_match
from src.utils.logger import get_logger

logger = get_logger("who.harvest.identity_anchor")


class CryptographicIdentityAnchor:
    """Constructs a cryptographically and bidirectionally verified identity proof graph."""

    def __init__(self, target_name: str, target_dirs: Optional[Any] = None):
        self.target_name = target_name
        self.dirs = target_dirs or settings.get_target_dirs(target_name)

    def _http_get_text(self, url: str, accept: str = "text/html,application/xhtml+xml") -> Optional[str]:
        """Fetch raw UTF-8 response body safely with timeout."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "who-OSINT-IdentityAnchor/4.5",
                    "Accept": accept,
                },
            )
            with urllib.request.urlopen(req, timeout=6) as res:
                if res.getcode() == 200:
                    return res.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug(f"HTTP fetch skipped for {url}: {exc}")
        return None

    def extract_rel_me_links(self, html: str, base_url: str) -> List[str]:
        """Extract all <link rel='me'> and <a rel='me'> targets from HTML."""
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        discovered: List[str] = []

        for tag in soup.find_all(["link", "a"]):
            rel_val = tag.get("rel")
            rels = [r.lower() for r in rel_val] if isinstance(rel_val, list) else [str(rel_val or "").lower()]
            if "me" in rels:
                href = (tag.get("href") or "").strip()
                if href and not href.startswith(("javascript:", "#")):
                    full_url = urllib.parse.urljoin(base_url, href)
                    if full_url not in discovered:
                        discovered.append(full_url)
        return discovered

    def verify_website_rel_me(self, website_url: str) -> Dict[str, Any]:
        """Verify outbound rel='me' declarations on target's primary website and test bidirectional backlinks."""
        normalized_home = website_url.strip()
        if normalized_home and not normalized_home.startswith(("http://", "https://")):
            normalized_home = f"https://{normalized_home}"

        home_domain = urllib.parse.urlparse(normalized_home).netloc.lower().lstrip("www.")
        html = self._http_get_text(normalized_home)
        if not html:
            return {
                "website_url": normalized_home,
                "reachable": False,
                "outbound_rel_me": [],
                "bidirectional_verified": [],
                "unidirectional_claims": [],
            }

        outbound = self.extract_rel_me_links(html, normalized_home)
        bidirectional: List[Dict[str, Any]] = []
        unidirectional: List[Dict[str, Any]] = []

        for ext_url in outbound[:12]:
            if ext_url.startswith("mailto:"):
                unidirectional.append({
                    "target_uri": ext_url,
                    "proof_tier": "DECLARED_MAILTO_REL_ME",
                    "confidence_score": 0.82,
                })
                continue

            ext_html = self._http_get_text(ext_url)
            backlinks_found = False
            if ext_html and home_domain:
                ext_rel_me = self.extract_rel_me_links(ext_html, ext_url)
                if any(home_domain in u.lower() for u in ext_rel_me) or (home_domain in ext_html.lower()):
                    backlinks_found = True

            if backlinks_found:
                bidirectional.append({
                    "source_url": normalized_home,
                    "verified_profile_url": ext_url,
                    "proof_tier": "BIDIRECTIONAL_REL_ME",
                    "confidence_score": 0.96,
                })
            else:
                unidirectional.append({
                    "source_url": normalized_home,
                    "claimed_profile_url": ext_url,
                    "proof_tier": "OUTBOUND_REL_ME_ONLY",
                    "confidence_score": 0.78,
                })

        return {
            "website_url": normalized_home,
            "reachable": True,
            "outbound_rel_me": outbound,
            "bidirectional_verified": bidirectional,
            "unidirectional_claims": unidirectional,
        }

    def verify_github_cryptographic_keys(self, github_username: str) -> Dict[str, Any]:
        """
        Verify GitHub identity under Strict Zero-Substitution and extract public GPG & SSH cryptographic fingerprints.
        """
        handle = github_username.strip().lstrip("@")
        if not handle:
            return {"status": "SKIPPED"}

        prof_raw = self._http_get_text(
            f"https://api.github.com/users/{urllib.parse.quote(handle)}",
            accept="application/vnd.github.v3+json",
        )
        if not prof_raw:
            return {"username": handle, "status": "UNREACHABLE_OR_NOT_FOUND"}

        profile = json.loads(prof_raw)
        declared_name = (profile.get("name") or "").strip()

        # STRICT ZERO-SUBSTITUTION GATE (AGENTS.md Rule #1)
        if declared_name:
            is_match, reason = is_strict_entity_match(self.target_name, declared_name)
            if not is_match:
                logger.warning(
                    f"[Zero-Substitution Gate] Rejected GitHub @{handle} declared as '{declared_name}': {reason}"
                )
                return {
                    "username": handle,
                    "declared_name": declared_name,
                    "status": "REJECTED_ZERO_SUBSTITUTION",
                    "rejection_reason": reason,
                    "disambiguation_notice": DISAMBIGUATION_NOTICE_TR,
                }

        # Fetch public GPG keys
        gpg_anchors: List[Dict[str, Any]] = []
        gpg_raw = self._http_get_text(
            f"https://api.github.com/users/{urllib.parse.quote(handle)}/gpg_keys",
            accept="application/vnd.github.v3+json",
        )
        if gpg_raw:
            try:
                for k in json.loads(gpg_raw):
                    gpg_anchors.append({
                        "key_id": k.get("key_id"),
                        "emails": [e.get("email") for e in k.get("emails", []) if e.get("email")],
                        "can_sign": k.get("can_sign", True),
                        "created_at": k.get("created_at"),
                    })
            except Exception:
                pass

        # Fetch public SSH signing/auth keys and compute SHA-256 fingerprint
        ssh_anchors: List[Dict[str, Any]] = []
        ssh_raw = self._http_get_text(
            f"https://api.github.com/users/{urllib.parse.quote(handle)}/keys",
            accept="application/vnd.github.v3+json",
        )
        if ssh_raw:
            try:
                for sk in json.loads(ssh_raw):
                    raw_key = (sk.get("key") or "").strip()
                    parts = raw_key.split()
                    fingerprint = ""
                    if len(parts) >= 2:
                        try:
                            der = base64.b64decode(parts[1])
                            digest = base64.b64encode(hashlib.sha256(der).digest()).decode("ascii").rstrip("=")
                            fingerprint = f"SHA256:{digest}"
                        except Exception:
                            fingerprint = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]
                    ssh_anchors.append({
                        "id": sk.get("id"),
                        "key_type": parts[0] if parts else "ssh",
                        "sha256_fingerprint": fingerprint,
                    })
            except Exception:
                pass

        has_crypto = bool(gpg_anchors or ssh_anchors)
        return {
            "username": handle,
            "declared_name": declared_name or self.target_name,
            "profile_url": profile.get("html_url", f"https://github.com/{handle}"),
            "blog_url": profile.get("blog") or "",
            "status": "VERIFIED_CRYPTOGRAPHIC" if has_crypto else "VERIFIED_PROFILE_MATCH",
            "proof_tier": "CRYPTOGRAPHIC_KEY_ANCHOR" if has_crypto else "STRICT_NAME_VERIFIED",
            "confidence_score": 0.98 if has_crypto else (0.90 if declared_name else 0.75),
            "gpg_keys": gpg_anchors,
            "ssh_keys": ssh_anchors,
        }

    def verify_orcid_identity(self, orcid_id: str) -> Dict[str, Any]:
        """Verify ORCID scholarly record against Strict Zero-Substitution name rules."""
        clean_orcid = orcid_id.strip().replace("https://orcid.org/", "").strip("/")
        if not re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", clean_orcid):
            return {"orcid": clean_orcid, "status": "INVALID_ORCID_FORMAT"}

        raw = self._http_get_text(
            f"https://pub.orcid.org/v3.0/{clean_orcid}/person",
            accept="application/orcid+json",
        )
        if not raw:
            return {"orcid": clean_orcid, "status": "UNREACHABLE"}

        data = json.loads(raw)
        name_obj = data.get("name") or {}
        given = ((name_obj.get("given-names") or {}).get("value") or "").strip()
        family = ((name_obj.get("family-name") or {}).get("value") or "").strip()
        full_name = f"{given} {family}".strip()

        if full_name:
            is_match, reason = is_strict_entity_match(self.target_name, full_name)
            if not is_match:
                return {
                    "orcid": clean_orcid,
                    "declared_name": full_name,
                    "status": "REJECTED_ZERO_SUBSTITUTION",
                    "rejection_reason": reason,
                    "disambiguation_notice": DISAMBIGUATION_NOTICE_TR,
                }

        urls = []
        for u in (data.get("researcher-urls") or {}).get("researcher-url", []) or []:
            val = ((u.get("url") or {}).get("value") or "").strip()
            if val:
                urls.append({"name": (u.get("url-name") or ""), "url": val})

        ext_ids = []
        for eid in (data.get("external-identifiers") or {}).get("external-identifier", []) or []:
            ext_ids.append({
                "type": eid.get("external-id-type"),
                "value": eid.get("external-id-value"),
            })

        return {
            "orcid": clean_orcid,
            "orcid_url": f"https://orcid.org/{clean_orcid}",
            "declared_name": full_name or self.target_name,
            "status": "VERIFIED_SCHOLARLY_ANCHOR",
            "proof_tier": "SCHOLARLY_AUTHORITY_ID",
            "confidence_score": 0.97,
            "researcher_urls": urls,
            "external_identifiers": ext_ids,
        }

    def inspect_domain_txt_proofs(self, domain: str) -> Dict[str, Any]:
        """Inspect DNS TXT records on a personal domain for decentralized & platform identity proofs."""
        clean_dom = domain.strip().lower().replace("https://", "").replace("http://", "").split("/")[0]
        if not clean_dom:
            return {"domain": "", "txt_proofs": []}

        proofs: List[Dict[str, str]] = []
        for host, label in [(clean_dom, "root"), (f"_atproto.{clean_dom}", "bluesky_atproto")]:
            raw = self._http_get_text(
                f"https://dns.google/resolve?name={urllib.parse.quote(host)}&type=TXT",
                accept="application/dns-json",
            )
            if not raw:
                continue
            try:
                answers = json.loads(raw).get("Answer", []) or []
                for ans in answers:
                    txt = str(ans.get("data", "")).strip('" ')
                    txt_low = txt.lower()
                    if any(
                        marker in txt_low
                        for marker in (
                            "did=did:",
                            "keybase-site-verification",
                            "github-verification",
                            "google-site-verification",
                            "openpgp",
                            "orcid",
                            "protonmail-verification",
                        )
                    ):
                        proofs.append({"record_host": host, "proof_type": label, "value": txt})
            except Exception:
                pass

        return {
            "domain": clean_dom,
            "txt_proofs": proofs,
            "proof_count": len(proofs),
        }

    def run_anchor_verification(
        self,
        website: Optional[str] = None,
        github: Optional[str] = None,
        orcid: Optional[str] = None,
        domain: Optional[str] = None,
        pgp_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end cryptographic and bidirectional identity anchoring."""
        logger.info(f"Running Cryptographic & Bidirectional Identity Anchoring for: {self.target_name}")

        verified_anchors: List[Dict[str, Any]] = []
        rejected_records: List[Dict[str, Any]] = []

        web_res = self.verify_website_rel_me(website) if website else None
        if web_res and web_res.get("reachable"):
            for bi in web_res["bidirectional_verified"]:
                verified_anchors.append({
                    "anchor_type": "BIDIRECTIONAL_REL_ME",
                    "identifier": bi["verified_profile_url"],
                    "confidence_score": bi["confidence_score"],
                    "evidence": f"{bi['source_url']} <-> {bi['verified_profile_url']}",
                })

        gh_res = self.verify_github_cryptographic_keys(github) if github else None
        if gh_res:
            if gh_res.get("status") == "REJECTED_ZERO_SUBSTITUTION":
                rejected_records.append(gh_res)
            elif gh_res.get("status", "").startswith("VERIFIED"):
                verified_anchors.append({
                    "anchor_type": gh_res["proof_tier"],
                    "identifier": gh_res["profile_url"],
                    "confidence_score": gh_res["confidence_score"],
                    "evidence": (
                        f"GPG keys: {len(gh_res.get('gpg_keys', []))}, "
                        f"SSH keys: {len(gh_res.get('ssh_keys', []))}"
                    ),
                })

        orcid_res = self.verify_orcid_identity(orcid) if orcid else None
        if orcid_res:
            if orcid_res.get("status") == "REJECTED_ZERO_SUBSTITUTION":
                rejected_records.append(orcid_res)
            elif orcid_res.get("status") == "VERIFIED_SCHOLARLY_ANCHOR":
                verified_anchors.append({
                    "anchor_type": "ORCID_SCHOLARLY_ID",
                    "identifier": orcid_res["orcid_url"],
                    "confidence_score": orcid_res["confidence_score"],
                    "evidence": f"Verified ORCID name match: {orcid_res['declared_name']}",
                })

        effective_domain = domain or (
            urllib.parse.urlparse(web_res["website_url"]).netloc if web_res and web_res.get("website_url") else ""
        )
        dns_res = self.inspect_domain_txt_proofs(effective_domain) if effective_domain else None
        if dns_res and dns_res.get("txt_proofs"):
            for p in dns_res["txt_proofs"]:
                verified_anchors.append({
                    "anchor_type": "DNS_TXT_DOMAIN_PROOF",
                    "identifier": p["record_host"],
                    "confidence_score": 0.94,
                    "evidence": p["value"][:120],
                })

        if pgp_fingerprint:
            clean_fp = re.sub(r"\s+", "", pgp_fingerprint).upper()
            if re.match(r"^[0-9A-F]{16,40}$", clean_fp):
                verified_anchors.append({
                    "anchor_type": "USER_DECLARED_OPENPGP_FINGERPRINT",
                    "identifier": f"openpgp:{clean_fp}",
                    "confidence_score": 0.99,
                    "evidence": f"Primary user-provided PGP fingerprint ({len(clean_fp) * 4}-bit hex ID)",
                })

        overall_conf = (
            round(max(a["confidence_score"] for a in verified_anchors), 2)
            if verified_anchors
            else 0.0
        )

        report = {
            "target_name": self.target_name,
            "verified_anchors_count": len(verified_anchors),
            "rejected_namesakes_count": len(rejected_records),
            "overall_identity_confidence": overall_conf,
            "verified_anchors": verified_anchors,
            "rejected_zero_substitution_records": rejected_records,
            "details": {
                "website_rel_me": web_res,
                "github_cryptographic": gh_res,
                "orcid_scholarly": orcid_res,
                "dns_txt_proofs": dns_res,
            },
            "disambiguation_notice": DISAMBIGUATION_NOTICE_TR if rejected_records else None,
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }

        self._persist_report(report)
        return report

    def _persist_report(self, report: Dict[str, Any]) -> None:
        """Persist identity anchor graph to vault (never writing rejected third-party data as target traits)."""
        self.dirs.vault_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.dirs.vault_dir / "identity_anchors.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        md_path = self.dirs.vault_dir / "identity_anchor_report.md"
        lines = [
            f"# 🔐 Cryptographic & Bidirectional Identity Anchor Report: {self.target_name}",
            f"*Audited at:* {report['audited_at']}",
            f"*Overall Primary Link Confidence:* `{int(report['overall_identity_confidence'] * 100)}%`\n",
            "## 1. Verified Primary Identity Anchors",
        ]
        if report["verified_anchors"]:
            for a in report["verified_anchors"]:
                lines.append(
                    f"- **[{a['anchor_type']}]** `{a['identifier']}` — "
                    f"Confidence: `{int(a['confidence_score'] * 100)}%` ({a['evidence']})"
                )
        else:
            lines.append("- Henüz doğrulanmış bir kriptografik veya çift yönlü kimlik çapası tespit edilmedi.")

        if report.get("disambiguation_notice"):
            lines.extend([
                "\n## 2. Sıfır İkame Kapısı Bildirimi (Zero-Substitution Gate)",
                f"> {report['disambiguation_notice']} (Engellenen kayıt sayısı: `{report['rejected_namesakes_count']}`)",
            ])

        md_path.write_text("\n".join(lines), encoding="utf-8")

