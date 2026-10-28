"""
Tests for Module 1: who_harvest (Text Cleaner, Document Ingestor, Dialogue Harvester, OCR).
"""

import json
from pathlib import Path
import pytest

from src.harvest.dialogue_harvester import DialogueHarvester
from src.harvest.document_ingestor import DocumentIngestor
from src.harvest.ocr_processor import OCRProcessor
from src.utils.text_cleaner import chunk_text, clean_text, extract_quotes, parse_diarized_segments, sanitize_slug


def test_sanitize_slug():
    assert sanitize_slug("John Doe") == "john_doe"
    assert sanitize_slug("Dr. Alex V. Smith-Jones! @2026") == "dr_alex_v_smith_jones_2026"
    assert sanitize_slug("") == "unnamed_target"


def test_clean_text():
    dirty = "  Hello   world!  \r\n\r\n\n\nNew paragraph.  "
    cleaned = clean_text(dirty)
    assert cleaned == "Hello world!\n\nNew paragraph."


def test_parse_diarized_segments():
    transcript = """[TARGET_PRIMARY]: We believe empirical evidence is paramount.
[INTERLOCUTOR_CONTEXT]: Can you explain why?
[TARGET_PRIMARY]: Because without evidence, claims are unfalsifiable."""
    segments = parse_diarized_segments(transcript)
    assert len(segments) == 3
    assert segments[0]["role"] == "target"
    assert segments[1]["role"] == "interlocutor"
    assert "paramount" in segments[0]["text"]


def test_chunk_text():
    text = "First sentence here. Second sentence follows. Third sentence wraps up."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 1
    assert all("text" in c for c in chunks)


def test_extract_quotes():
    corpus = "Our system prioritizes decentralized architecture. Centralized control creates single points of failure. Scalability depends on modularity."
    quotes = extract_quotes(corpus, "centralized control failure", max_quotes=1)
    assert len(quotes) == 1
    assert "Centralized control" in quotes[0]


def test_dialogue_harvester_whatsapp(tmp_path: Path):
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(
        "12/05/2023, 10:15 - Alice: How should we structure the governance model?\n"
        "12/05/2023, 10:16 - Bob: We must insist on verifiable cryptographic proofs.\n"
        "12/05/2023, 10:17 - Alice: Agreed. What about voting?\n"
        "12/05/2023, 10:18 - Bob: Quadratic voting minimizes plutocratic dominance.\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    harvester = DialogueHarvester(output_dir=out_dir, target_alias="Bob")
    pairs = harvester.parse_whatsapp(chat_file)

    assert len(pairs) == 2
    assert pairs[0]["context_speaker"] == "Alice"
    assert "cryptographic proofs" in pairs[0]["target_response"]
    assert pairs[1]["context_speaker"] == "Alice"
    assert "Quadratic voting" in pairs[1]["target_response"]


def test_dialogue_harvester_telegram(tmp_path: Path):
    tg_file = tmp_path / "telegram.json"
    data = {
        "messages": [
            {"type": "message", "from": "Interlocutor", "text": "Are you ready to deploy?", "date": "2023-01-01"},
            {"type": "message", "from": "Target", "text": "Yes, all integration tests passed with 98% coverage.", "date": "2023-01-01"}
        ]
    }
    tg_file.write_text(json.dumps(data), encoding="utf-8")

    out_dir = tmp_path / "out"
    harvester = DialogueHarvester(output_dir=out_dir, target_alias="Target")
    pairs = harvester.parse_telegram(tg_file)

    assert len(pairs) == 1
    assert "integration tests" in pairs[0]["target_response"]


def test_document_ingestor_fallback(tmp_path: Path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text("# Project Strategy\n\nEmpirical validation drives our roadmap.", encoding="utf-8")

    out_dir = tmp_path / "out"
    ingestor = DocumentIngestor(output_dir=out_dir)
    res = ingestor.ingest_file(doc_file)

    assert res["file_name"] == "sample.md"
    assert "Empirical validation" in res["content"]
    assert res["source_type"] == "document_md"


def test_ocr_processor_fallback(tmp_path: Path):
    # Dummy image file
    img_file = tmp_path / "scan.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

    out_dir = tmp_path / "out"
    ocr = OCRProcessor(output_dir=out_dir)
    res = ocr.extract_text_from_image(img_file)

    assert res["image_name"] == "scan.png"
    assert res["mean_confidence"] > 0.0
    assert "OCR Transcription" in res["content"]
