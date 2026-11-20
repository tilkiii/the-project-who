"""
Cross-platform username search and profile reconnaissance.
Extracts public biographies and account references across online services.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from who.config import settings
from who.utils.logger import get_logger
from who.utils.text_cleaner import clean_text

logger = get_logger("who.harvest.social")

# Platform probe definitions (probe URL, error type, category)
PROBE_CATALOG: Dict[str, Dict[str, Any]] = {
    "GitHub": {
        "url": "https://api.github.com/users/{username}",
        "web_url": "https://github.com/{username}",
        "check_type": "json_api",
        "category": "Code & Development",
        "icon": "🐙",
    },
    "Reddit": {
        "url": "https://www.reddit.com/user/{username}/about.json",
        "web_url": "https://www.reddit.com/user/{username}",
        "check_type": "json_api",
        "category": "Social & Discussions",
        "icon": "🤖",
    },
    "HackerNews": {
        "url": "https://hacker-news.firebaseio.com/v0/user/{username}.json",
        "web_url": "https://news.ycombinator.com/user?id={username}",
        "check_type": "json_api",
        "category": "Tech Community",
        "icon": "🔶",
    },
    "Dev.to": {
        "url": "https://dev.to/api/users/by_username?url={username}",
        "web_url": "https://dev.to/{username}",
        "check_type": "json_api",
        "category": "Developer Blog",
        "icon": "👩‍💻",
    },
    "GitLab": {
        "url": "https://gitlab.com/api/v4/users?username={username}",
        "web_url": "https://gitlab.com/{username}",
        "check_type": "json_api",
        "category": "Code & Development",
        "icon": "🦊",
    },
    "Medium": {
        "url": "https://medium.com/@{username}",
        "check_type": "status_code",
        "category": "Publishing & Articles",
        "icon": "✍️",
    },
    "Wikipedia": {
        "url": "https://en.wikipedia.org/wiki/User:{username}",
        "check_type": "status_code",
        "category": "Knowledge Base",
        "icon": "📚",
    },
    "Gravatar": {
        "url": "https://en.gravatar.com/{username}.json",
        "check_type": "json_api",
        "category": "Identity Profile",
        "icon": "👤",
    },
    "Steam": {
        "url": "https://steamcommunity.com/id/{username}",
        "check_type": "status_code",
        "category": "Gaming & Culture",
        "icon": "🎮",
    },
    "Spotify": {
        "url": "https://open.spotify.com/user/{username}",
        "check_type": "status_code",
        "category": "Audio & Media",
        "icon": "🎧",
        "user_agent": "Mozilla/5.0",
    },
    "Substack": {
        "url": "https://{username}.substack.com",
        "check_type": "status_code",
        "category": "Newsletters & Articles",
        "icon": "📰",
    },
    "Mastodon": {
        "url": "https://mastodon.social/@{username}",
        "check_type": "status_code",
        "category": "Decentralized Social",
        "icon": "🐘",
    },
    "Pinterest": {
        "url": "https://www.pinterest.com/{username}/",
        "check_type": "status_code",
        "category": "Visual Media",
        "icon": "📌",
    },
    "Kaggle": {
        "url": "https://www.kaggle.com/{username}",
        "check_type": "status_code",
        "category": "Data Science",
        "icon": "📊",
    },
    "Dribbble": {
        "url": "https://dribbble.com/{username}",
        "check_type": "status_code",
        "category": "Design & Portfolio",
        "icon": "🎨",
    },
    "Instagram": {
        "url": "https://www.instagram.com/{username}/",
        "check_type": "instagram",
        "category": "Social Media & Photos",
        "icon": "📸",
        "user_agent": "facebookexternalhit/1.1",
    },
    "YouTube": {
        "url": "https://www.youtube.com/@{username}",
        "check_type": "status_code",
        "category": "Video & Content",
        "icon": "🎥",
    },
    "Telegram": {
        "url": "https://t.me/{username}",
        "check_type": "telegram",
        "category": "Messaging & Community",
        "icon": "✈️",
    },
}


class SocialOSINTHunter:
    """Discovers and parses multi-platform digital footprints by username across public endpoints."""

    def __init__(self, target_name: str, target_dirs: Optional[Any] = None):
        self.target_name = target_name
        self.dirs = target_dirs or settings.get_target_dirs(target_name)
        self.discovered_profiles: List[Dict[str, Any]] = []

    def probe_single_platform(self, platform: str, spec: Dict[str, Any], username: str) -> Optional[Dict[str, Any]]:
        """Probe a specific platform for username existence and extract public metadata."""
        url = spec["url"].format(username=username)
        check_type = spec["check_type"]
        category = spec["category"]
        icon = spec.get("icon", "🌐")

        custom_ua = spec.get("user_agent")
        headers = {
            "User-Agent": custom_ua or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html, */*",
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as response:
                status = response.getcode()
                if status == 200:
                    raw_data = response.read().decode("utf-8", errors="replace")

                    # Handle API-based metadata extraction
                    meta: Dict[str, Any] = {}
                    bio_snippet = ""

                    if check_type == "json_api":
                        try:
                            parsed_json = json.loads(raw_data)
                            if parsed_json is None:
                                return None
                            if isinstance(parsed_json, list):
                                if len(parsed_json) == 0:
                                    return None
                            elif isinstance(parsed_json, dict):
                                if parsed_json.get("message") == "Not Found":
                                    return None
                                if platform == "GitHub":
                                    bio_snippet = parsed_json.get("bio") or ""
                                    meta = {
                                        "name": parsed_json.get("name"),
                                        "location": parsed_json.get("location"),
                                        "company": parsed_json.get("company"),
                                        "public_repos": parsed_json.get("public_repos"),
                                        "followers": parsed_json.get("followers"),
                                        "created_at": parsed_json.get("created_at"),
                                    }
                                elif platform == "Reddit":
                                    r_data = parsed_json.get("data", {})
                                    if not r_data:
                                        return None
                                    bio_snippet = r_data.get("subreddit", {}).get("public_description", "")
                                    meta = {
                                        "total_karma": r_data.get("total_karma"),
                                        "verified": r_data.get("verified"),
                                        "created_utc": r_data.get("created_utc"),
                                    }
                                elif platform == "Dev.to":
                                    bio_snippet = parsed_json.get("summary") or ""
                                    meta = {
                                        "name": parsed_json.get("name"),
                                        "website": parsed_json.get("website_url"),
                                    }
                        except Exception:
                            return None

                    # HTML preview extraction if no API snippet
                    if not bio_snippet:
                        # Extract title or description tag
                        title_m = re.search(r"<title>(.*?)</title>", raw_data, re.IGNORECASE)
                        if title_m:
                            bio_snippet = clean_text(title_m.group(1))[:180]
                            # Detect bot-challenge or captcha false positives
                            if any(bot_indicator in bio_snippet.lower() for bot_indicator in ["recaptcha", "cloudflare", "just a moment", "attention required"]):
                                return None

                    if check_type == "instagram":
                        title_lower = bio_snippet.lower()
                        if title_lower == "instagram" or "page not found" in title_lower or "sayfa bulunamadı" in title_lower:
                            return None
                        if f"@{username.lower()}" not in title_lower and f"&#064;{username.lower()}" not in title_lower:
                            return None

                    if check_type == "telegram":
                        if "tgme_page_extra" not in raw_data:
                            return None

                    if platform == "Steam":
                        if "the specified profile could not be found" in raw_data.lower() or "steam community :: error" in bio_snippet.lower():
                            return None

                    if platform == "Pinterest":
                        if not bio_snippet or f"({username.lower()})" not in bio_snippet.lower():
                            return None

                    display_url = spec.get("web_url", url).format(username=username)

                    return {
                        "platform": platform,
                        "url": display_url,
                        "category": category,
                        "icon": icon,
                        "exists": True,
                        "metadata": meta,
                        "bio_snippet": bio_snippet,
                    }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            logger.debug(f"Platform {platform} probe failed for {username}: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error probing {platform}: {e}")

        return None

    def hunt_username(self, username: str, max_workers: int = 8) -> List[Dict[str, Any]]:
        """Hunt a username across the full probe catalog concurrently."""
        clean_user = username.strip().lstrip("@")
        logger.info(f"Initiating multi-platform OSINT username hunt for: '{clean_user}'")

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_platform = {
                executor.submit(self.probe_single_platform, p, spec, clean_user): p
                for p, spec in PROBE_CATALOG.items()
            }

            for future in as_completed(future_to_platform):
                res = future.result()
                if res and res.get("exists"):
                    results.append(res)

        self.discovered_profiles = sorted(results, key=lambda x: x["platform"])
        logger.info(f"Discovered {len(self.discovered_profiles)} active profile(s) for '{clean_user}'.")

        # Ingest discovered bio snippets into target raw data
        if self.discovered_profiles:
            self._ingest_discovered_dossier(clean_user, self.discovered_profiles)

        return self.discovered_profiles

    def _ingest_discovered_dossier(self, username: str, profiles: List[Dict[str, Any]]) -> None:
        """Persist structured OSINT social dossier directly into target raw data directory."""
        self.dirs.raw_data_dir.mkdir(parents=True, exist_ok=True)
        filename = f"social_osint_dossier_{username}.md"
        out_file = self.dirs.raw_data_dir / filename

        lines = [
            f"# 🌐 Social Footprint OSINT Dossier: {self.target_name}",
            f"*Queried Username:* `@{username}`",
            f"*Harvested at:* {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
            "## 1. Discovered Active Accounts\n",
        ]

        for p in profiles:
            lines.append(f"- **{p['icon']} {p['platform']}** ({p['category']}): [{p['url']}]({p['url']})")
            if p.get("bio_snippet"):
                lines.append(f"  * Bio/Snippet: [TARGET_PRIMARY]: \"{p['bio_snippet']}\"")
            if p.get("metadata"):
                meta_str = ", ".join(f"{k}: {v}" for k, v in p["metadata"].items() if v is not None)
                if meta_str:
                    lines.append(f"  * Metadata: `{meta_str}`")

        out_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved Social OSINT dossier to: {out_file}")
