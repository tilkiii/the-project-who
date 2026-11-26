"""
Dialogue harvester for who??.
Parses WhatsApp (.txt), Telegram (.json), Zoom/Teams (.vtt/.txt), and email threads
into context-response JSON pairs with strict speaker attribution.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from who.utils.logger import get_logger
from who.utils.text_cleaner import redact_pii

logger = get_logger("who.harvest.dialogue")


class DialogueHarvester:
    """Parses multi-platform direct dialogue logs into structured context-response pairs with automatic PII redaction."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        target_alias: str = "Target",
        auto_redact_pii: bool = True,
    ):
        self.output_dir = output_dir
        self.target_alias = target_alias.lower()
        self.auto_redact_pii = auto_redact_pii

    def is_target_speaker(self, speaker_name: str) -> bool:
        """Verify whether speaker name matches the target without substring collisions (e.g. 'Ece' vs 'Ecem')."""
        if not speaker_name:
            return False
        spk = speaker_name.lower().strip()
        alias = self.target_alias.strip()
        if spk in ("me", "target_primary") or "primary" in spk:
            return True
        if not alias:
            return False
        if spk == alias:
            return True
        if len(alias) >= 2 and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", spk):
            return True
        if " " in alias:
            first_tok = alias.split()[0]
            if len(first_tok) >= 3 and spk == first_tok:
                return True
        return False

    @staticmethod
    def _strip_invisible_marks(line: str) -> str:
        """Strip UTF-8 BOM and Unicode bidirectional formatting marks common in iOS WhatsApp exports."""
        return re.sub(r"^[\ufeff\u200e\u200f\u202a-\u202e\u2066-\u2069]+", "", line).strip()

    def parse_whatsapp(self, file_path: Path | str) -> List[Dict[str, Any]]:
        """Parse WhatsApp chat export (.txt)."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()

        # Regex patterns for WhatsApp:
        # [DD/MM/YYYY, HH:MM:SS] Name: message
        # MM/DD/YY, HH:MM - Name: message
        p1 = re.compile(r"^\[?(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*(?:-\s*)?([^:]+):\s*(.*)$")

        turns: List[Dict[str, str]] = []
        for line in lines:
            line_str = self._strip_invisible_marks(line)
            if not line_str:
                continue
            m = p1.match(line_str)
            if m:
                timestamp, speaker, text = m.group(1), m.group(2).strip(), m.group(3).strip()
                turns.append({"timestamp": timestamp, "speaker": speaker, "text": text})
            elif turns:
                # Multi-line message continuation
                turns[-1]["text"] += "\n" + line_str

        return self._build_pairs_and_save(turns, path.stem, "whatsapp")

    def parse_telegram(self, file_path: Path | str) -> List[Dict[str, Any]]:
        """Parse Telegram chat history exported as JSON."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            data = json.load(f)

        messages = data.get("messages", [])
        turns: List[Dict[str, str]] = []

        for msg in messages:
            if msg.get("type") != "message":
                continue
            speaker = msg.get("from", "Unknown")
            raw_text = msg.get("text", "")
            # Telegram text can be a list of formatting objects
            if isinstance(raw_text, list):
                extracted = []
                for part in raw_text:
                    if isinstance(part, str):
                        extracted.append(part)
                    elif isinstance(part, dict):
                        extracted.append(part.get("text", ""))
                text = "".join(extracted)
            else:
                text = str(raw_text)

            if text.strip():
                turns.append({
                    "timestamp": msg.get("date", ""),
                    "speaker": speaker,
                    "text": text.strip(),
                })

        return self._build_pairs_and_save(turns, path.stem, "telegram")

    def parse_vtt(self, file_path: Path | str) -> List[Dict[str, Any]]:
        """Parse Zoom, Teams, or WebVTT transcript (.vtt or .txt)."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read()

        # WebVTT timestamp pattern: 00:00:00.000 --> 00:00:04.000
        # Line format: Speaker: text
        blocks = re.split(r"\n\s*\n", content)
        turns: List[Dict[str, str]] = []

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            for l in lines:
                if "-->" in l or l.isdigit() or l.startswith("WEBVTT"):
                    continue
                if ":" in l:
                    parts = l.split(":", 1)
                    speaker, text = parts[0].strip(), parts[1].strip()
                    turns.append({"timestamp": "", "speaker": speaker, "text": text})
                elif turns:
                    turns[-1]["text"] += " " + l

        return self._build_pairs_and_save(turns, path.stem, "zoom_teams_vtt")

    def parse_email_thread(self, file_path: Path | str) -> List[Dict[str, Any]]:
        """Parse email thread text files."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read()

        # Split emails on "From:" lines
        email_blocks = re.split(r"(?m)^From:\s*", content)
        turns: List[Dict[str, str]] = []
        header_pat = re.compile(r"^(?:To|Cc|Bcc|Date|Sent|Subject|Reply-To|Message-ID):\s*", re.IGNORECASE)

        for block in email_blocks:
            if not block.strip():
                continue
            lines = block.strip().split("\n")
            speaker_line = lines[0].strip()
            speaker = re.sub(r"<[^>]+>", "", speaker_line).strip()
            # Consume RFC-like headers until blank line or first non-header line
            body_lines = []
            is_body = False
            for l in lines[1:]:
                stripped = l.strip()
                if is_body:
                    if not stripped.startswith(">"):  # Ignore quoted reply headers
                        body_lines.append(l)
                elif stripped == "":
                    is_body = True
                elif not header_pat.match(stripped):
                    is_body = True
                    if not stripped.startswith(">"):
                        body_lines.append(l)
            text = "\n".join(body_lines).strip()
            if text:
                turns.append({"timestamp": "", "speaker": speaker, "text": text})

        return self._build_pairs_and_save(turns, path.stem, "email_thread")

    def parse_file(self, file_path: Path | str) -> List[Dict[str, Any]]:
        """Auto-detect format and parse dialogue logs."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self.parse_telegram(path)
        elif suffix in (".vtt", ".srt"):
            return self.parse_vtt(path)
        else:
            # Check content preview
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                head = f.read(800)
            if "WEBVTT" in head:
                return self.parse_vtt(path)
            elif re.search(r"(?m)^From:\s*", head) and re.search(r"(?m)^(?:Subject|To|Date|Sent):\s*", head):
                return self.parse_email_thread(path)
            else:
                return self.parse_whatsapp(path)

    def _build_pairs_and_save(
        self, turns: List[Dict[str, str]], stem: str, source_type: str
    ) -> List[Dict[str, Any]]:
        """Convert turns into context-response pairs and diarized markdown."""
        pairs: List[Dict[str, Any]] = []
        diarized_markdown_lines: List[str] = [f"# Diarized Dialogue: {stem} ({source_type})\n"]

        for i in range(len(turns)):
            curr = turns[i]
            text = redact_pii(curr["text"]) if self.auto_redact_pii else curr["text"]
            is_target = self.is_target_speaker(curr["speaker"])
            speaker_tag = "[TARGET_PRIMARY]" if is_target else "[INTERLOCUTOR_CONTEXT]"
            ts_prefix = f"[{curr['timestamp']}] " if curr.get("timestamp") else ""
            diarized_markdown_lines.append(f"{ts_prefix}{speaker_tag}: {text}\n")

            if is_target and i > 0:
                prev = turns[i - 1]
                if not self.is_target_speaker(prev["speaker"]):
                    prev_text = redact_pii(prev["text"]) if self.auto_redact_pii else prev["text"]
                    pairs.append({
                        "context_speaker": prev["speaker"],
                        "context_text": prev_text,
                        "target_response": text,
                        "source_type": source_type,
                    })

        # Save pairs and diarized markdown if turns exist and output_dir is configured
        if not turns:
            logger.warning(f"No dialogue turns extracted from '{stem}' ({source_type}); skipping empty export.")
            return []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # 1. Save diarized markdown
            md_path = self.output_dir / f"dialogue_{stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(diarized_markdown_lines))

            # 2. Save dialogue pairs JSON
            pairs_path = self.output_dir / f"dialogue_pairs_{stem}.json"
            with open(pairs_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)

            logger.info(f"Processed {len(turns)} turns and {len(pairs)} target response pairs from {stem}.")

        return pairs
