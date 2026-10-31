"""
Document ingestor for who?? using Docling.
Converts PDFs, DOCX, PPTX, and HTML into structured, normalized Markdown.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger("who.harvest.documents")


class DocumentIngestor:
    """Ingests multi-format office and scientific documents using Docling with fallbacks."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def ingest_file(self, file_path: Path | str) -> Dict[str, Any]:
        """Convert a single PDF, DOCX, PPTX, or text document into structured Markdown."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Target document not found: {path}")

        logger.info(f"Ingesting structured document: {path.name}")
        content_markdown = ""
        doc_type = path.suffix.lower()

        # 1. Attempt deep parsing via Docling
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(path))
            content_markdown = result.document.export_to_markdown()
            logger.info(f"Successfully parsed {path.name} via Docling.")
        except Exception as e:
            logger.debug(f"Docling parsing not active or failed ({e}). Attempting native format fallback.")

        # 2. Native fallbacks by file type
        if not content_markdown:
            if doc_type in (".txt", ".md", ".csv", ".json"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content_markdown = f.read()
            elif doc_type == ".docx":
                content_markdown = self._parse_docx_fallback(path)
            elif doc_type == ".pdf":
                content_markdown = self._parse_pdf_fallback(path)
            elif doc_type == ".pptx":
                content_markdown = self._parse_pptx_fallback(path)
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content_markdown = f.read()

        clean_md = clean_text(content_markdown)
        record = {
            "file_name": path.name,
            "source_path": str(path.absolute()),
            "doc_type": doc_type,
            "content": clean_md,
            "source_type": f"document_{doc_type.lstrip('.')}",
            "speaker": "TARGET_PRIMARY",
        }

        if self.output_dir:
            self._save_record(record, path.name)

        return record

    def ingest_directory(self, docs_dir: Path | str) -> List[Dict[str, Any]]:
        """Scan and ingest all supported document files within a directory."""
        directory = Path(docs_dir)
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Docs directory does not exist: {docs_dir}")
            return []

        supported_exts = {".pdf", ".docx", ".pptx", ".txt", ".md", ".html", ".htm"}
        results = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                try:
                    res = self.ingest_file(file_path)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Failed to ingest document {file_path}: {e}")
        return results

    def _parse_docx_fallback(self, path: Path) -> str:
        """Fallback DOCX parser using zipfile xml extraction."""
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(path) as z:
                xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for p in root.findall(".//w:p", ns):
                    texts = [node.text for node in p.findall(".//w:t", ns) if node.text]
                    if texts:
                        paragraphs.append("".join(texts))
                return f"# {path.stem}\n\n" + "\n\n".join(paragraphs)
        except Exception as e:
            logger.debug(f"DOCX fallback extraction error: {e}")
            return f"# Document {path.name}\n\n[Content extraction failed]"

    def _parse_pdf_fallback(self, path: Path) -> str:
        """Fallback PDF parser using pypdf/pymupdf or binary stream text extraction."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return f"# {path.stem}\n\n" + "\n\n---\n\n".join(pages)
        except Exception:
            # Simple binary text scrape fallback
            try:
                with open(path, "rb") as f:
                    raw = f.read().decode("latin1", errors="ignore")
                    import re
                    matches = re.findall(r"\(([\w\s.,;:!?-]{4,})\)", raw)
                    return f"# {path.stem}\n\n" + " ".join(matches)
            except Exception as e:
                logger.debug(f"PDF fallback extraction error: {e}")
                return f"# PDF Document {path.name}\n\n[Content extraction failed]"

    def _parse_pptx_fallback(self, path: Path) -> str:
        """Fallback PPTX parser using zipfile xml extraction."""
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(path) as z:
                slides_text = []
                slide_files = sorted([n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")])
                for sf in slide_files:
                    xml_content = z.read(sf)
                    root = ET.fromstring(xml_content)
                    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                    texts = [node.text for node in root.findall(".//a:t", ns) if node.text]
                    if texts:
                        slides_text.append("\n".join(texts))
                return f"# Presentation: {path.stem}\n\n" + "\n\n---\n\n".join(slides_text)
        except Exception as e:
            logger.debug(f"PPTX fallback extraction error: {e}")
            return f"# PPTX {path.name}\n\n[Content extraction failed]"

    def _save_record(self, record: Dict[str, Any], identifier: str) -> None:
        """Persist ingested document markdown to disk."""
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(identifier.encode()).hexdigest()[:10]
        safe_name = f"doc_{h}_{Path(identifier).stem}.md"
        out_path = self.output_dir / safe_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"---\nfile_name: {record.get('file_name')}\nsource_type: {record.get('source_type')}\n---\n\n")
            f.write(record.get("content", ""))
