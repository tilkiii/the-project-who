"""
Web and RSS scraper for who??.
Leverages crawl4ai and feedparser to extract LLM-ready markdown from web articles and feeds.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from bs4 import BeautifulSoup

from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger("who.harvest.web")


class WebHarvester:
    """Harvests web articles, blogs, and RSS feeds into normalized markdown."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL using crawl4ai with httpx/bs4 fallback."""
        logger.info(f"Scraping web resource: {url}")
        content_markdown = ""
        title = ""

        # Attempt crawl4ai first
        try:
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                if result and result.markdown:
                    content_markdown = result.markdown
                    title = getattr(result, "title", "") or url
        except Exception as e:
            logger.debug(f"crawl4ai not available or encountered error ({e}), falling back to httpx+BeautifulSoup")

        # Fallback to httpx + BeautifulSoup
        if not content_markdown:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, "html.parser")
                        # Strip script and style
                        for s in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                            s.decompose()
                        title = soup.title.string.strip() if soup.title and soup.title.string else url
                        # Extract article or body paragraphs
                        paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]) if p.get_text().strip()]
                        content_markdown = f"# {title}\n\n" + "\n\n".join(paragraphs)
            except Exception as e:
                logger.error(f"Failed to scrape URL {url}: {e}")
                content_markdown = f"# Scrape Failure\n\nCould not fetch content from {url}."

        clean_md = clean_text(content_markdown)
        doc_record = {
            "source_url": url,
            "title": title or url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "content": clean_md,
            "source_type": "web_article",
            "speaker": "TARGET_PRIMARY",
        }

        if self.output_dir:
            self._save_record(doc_record, url)

        return doc_record

    def parse_rss_feed(self, feed_url: str, max_entries: int = 10) -> List[Dict[str, Any]]:
        """Parse an RSS or Atom feed using feedparser."""
        logger.info(f"Parsing RSS feed: {feed_url}")
        records = []
        try:
            import feedparser
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_entries]:
                title = entry.get("title", "Untitled")
                link = entry.get("link", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                # Clean html tags from summary
                soup = BeautifulSoup(summary, "html.parser")
                clean_body = soup.get_text(separator="\n").strip()
                record = {
                    "source_url": link or feed_url,
                    "title": title,
                    "published": entry.get("published", datetime.now(timezone.utc).isoformat()),
                    "content": f"# {title}\n\n{clean_body}",
                    "source_type": "rss_entry",
                    "speaker": "TARGET_PRIMARY",
                }
                records.append(record)
                if self.output_dir:
                    self._save_record(record, link or title)
        except Exception as e:
            logger.error(f"Failed to parse RSS feed {feed_url}: {e}")
        return records

    def harvest_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Synchronous wrapper to scrape a batch of URLs."""
        results = []
        for url in urls:
            if "rss" in url.lower() or "feed" in url.lower() or url.endswith(".xml"):
                results.extend(self.parse_rss_feed(url))
            else:
                try:
                    res = asyncio.run(self.scrape_url(url))
                    results.append(res)
                except Exception as e:
                    logger.error(f"Async scrape error for {url}: {e}")
        return results

    def _save_record(self, record: Dict[str, Any], identifier: str) -> None:
        """Persist harvested record to disk as markdown."""
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(identifier.encode()).hexdigest()[:10]
        safe_name = f"web_{h}.md"
        out_path = self.output_dir / safe_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"---\nsource_url: {record.get('source_url')}\ntitle: {record.get('title')}\nsource_type: {record.get('source_type')}\n---\n\n")
            f.write(record.get("content", ""))
