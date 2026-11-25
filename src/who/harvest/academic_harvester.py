"""
Academic harvester for who??.
Extracts scholarly publications and citations authored by the target.
Uses scholarly, Semantic Scholar API, and arXiv.
"""

from __future__ import annotations

import hashlib
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
import httpx

from who.utils.entity_validator import is_strict_entity_match
from who.utils.logger import get_logger
from who.utils.text_cleaner import clean_text

logger = get_logger("who.harvest.academic")


class AcademicHarvester:
    """Harvests target author publications, citations, and abstracts."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def search_author_publications(self, author_name: str, max_results: int = 15) -> List[Dict[str, Any]]:
        """Search publications for a specific author with strict Zero-Substitution identity verification."""
        logger.info(f"Harvesting academic footprint for author: {author_name}")
        publications: List[Dict[str, Any]] = []
        encoded_author = quote_plus(author_name)

        # 1. Attempt scholarly (Google Scholar)
        try:
            from scholarly import scholarly
            search_query = scholarly.search_author(author_name)
            matched_author = None
            for cand in islice(search_query, 20):
                cand_name = cand.get("name", "") if isinstance(cand, dict) else ""
                is_match, reason = is_strict_entity_match(author_name, cand_name)
                if is_match:
                    matched_author = cand
                    break
                logger.debug(f"Disambiguation rejected Google Scholar candidate '{cand_name}': {reason}")
            if matched_author:
                author_full = scholarly.fill(matched_author, sections=["publications"])
                pubs = author_full.get("publications", [])[:max_results]
                for p in pubs:
                    title = p.get("bib", {}).get("title", "Untitled Paper")
                    year = p.get("bib", {}).get("pub_year", "Unknown")
                    abstract = p.get("bib", {}).get("abstract", "")
                    citation_count = p.get("num_citations", 0)
                    record = {
                        "title": title,
                        "authors": [author_name],
                        "year": year,
                        "abstract": abstract,
                        "citation_count": citation_count,
                        "source": "Google Scholar via scholarly",
                        "content": f"# {title} ({year})\n**Author(s):** {author_name}\n**Citations:** {citation_count}\n\n## Abstract\n{abstract}",
                        "source_type": "academic_paper",
                        "speaker": "TARGET_PRIMARY",
                    }
                    publications.append(record)
                    self._save_record(record, title)
                if publications:
                    return publications
        except Exception as e:
            logger.debug(f"scholarly lookup skipped or failed ({e}). Proceeding to Semantic Scholar & arXiv APIs.")

        # 2. Query Semantic Scholar Open API
        try:
            with httpx.Client(timeout=10.0) as client:
                url = f"https://api.semanticscholar.org/graph/v1/author/search?query={encoded_author}&fields=name,papers.title,papers.year,papers.abstract,papers.citationCount"
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    authors_list = data.get("data", [])
                    matched_entry = None
                    for cand in authors_list:
                        cand_name = cand.get("name", "")
                        is_match, reason = is_strict_entity_match(author_name, cand_name)
                        if is_match:
                            matched_entry = cand
                            break
                        logger.debug(f"Disambiguation rejected Semantic Scholar candidate '{cand_name}': {reason}")
                    if matched_entry:
                        papers = matched_entry.get("papers", [])[:max_results]
                        for p in papers:
                            title = p.get("title", "Untitled Paper")
                            year = str(p.get("year", "Unknown"))
                            abstract = p.get("abstract", "") or "No abstract provided."
                            citations = p.get("citationCount", 0)
                            record = {
                                "title": title,
                                "authors": [author_name],
                                "year": year,
                                "abstract": abstract,
                                "citation_count": citations,
                                "source": "Semantic Scholar API",
                                "content": f"# {title} ({year})\n**Author:** {author_name}\n**Citations:** {citations}\n\n## Abstract\n{abstract}",
                                "source_type": "academic_paper",
                                "speaker": "TARGET_PRIMARY",
                            }
                            publications.append(record)
                            self._save_record(record, title)
                        if publications:
                            return publications
        except Exception as e:
            logger.debug(f"Semantic Scholar query exception: {e}")

        # 3. Query arXiv API
        try:
            import xml.etree.ElementTree as ET
            with httpx.Client(timeout=10.0) as client:
                arxiv_url = f"http://export.arxiv.org/api/query?search_query=au:{encoded_author}&start=0&max_results={max_results}"
                resp = client.get(arxiv_url)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall("atom:entry", ns):
                        entry_authors = [
                            clean_text(a_el.text)
                            for a_el in entry.findall("atom:author/atom:name", ns)
                            if a_el.text
                        ]
                        if not any(is_strict_entity_match(author_name, a_name)[0] for a_name in entry_authors):
                            logger.debug(f"Disambiguation rejected arXiv entry authors {entry_authors} for '{author_name}'")
                            continue
                        title_el = entry.find("atom:title", ns)
                        summary_el = entry.find("atom:summary", ns)
                        pub_el = entry.find("atom:published", ns)
                        title = clean_text(title_el.text) if title_el is not None and title_el.text else "Untitled"
                        abstract = clean_text(summary_el.text) if summary_el is not None and summary_el.text else ""
                        year = pub_el.text[:4] if pub_el is not None and pub_el.text else "Unknown"
                        record = {
                            "title": title,
                            "authors": entry_authors or [author_name],
                            "year": year,
                            "abstract": abstract,
                            "citation_count": 0,
                            "source": "arXiv API",
                            "content": f"# {title} ({year})\n**Author(s):** {', '.join(entry_authors) or author_name}\n\n## Abstract\n{abstract}",
                            "source_type": "academic_paper",
                            "speaker": "TARGET_PRIMARY",
                        }
                        publications.append(record)
                        self._save_record(record, title)
        except Exception as e:
            logger.debug(f"arXiv fallback exception: {e}")

        return publications

    def _save_record(self, record: Dict[str, Any], identifier: str) -> None:
        """Persist harvested academic record to disk as markdown."""
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(identifier.encode()).hexdigest()[:10]
        safe_name = f"academic_{h}.md"
        out_path = self.output_dir / safe_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"---\ntitle: {record.get('title')}\nyear: {record.get('year')}\nsource: {record.get('source')}\n---\n\n")
            f.write(record.get("content", ""))
