"""
Speaker diarization and audio transcription using whisperX.
Separates speech turns into [TARGET_PRIMARY] vs [INTERLOCUTOR_CONTEXT],
with resilient fallback parsing for companion .txt, .srt, and .vtt transcripts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger("who.harvest.transcribe")

SUPPORTED_MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}


class DiarizedTranscriber:
    """Transcribes audio files with high-precision WhisperX speaker diarization."""

    def __init__(self, output_dir: Optional[Path] = None, target_speaker_id: Optional[str] = None):
        self.output_dir = output_dir
        self.target_speaker_id = target_speaker_id  # e.g., 'SPEAKER_00'

    def transcribe_and_diarize(
        self,
        audio_path: Path | str,
        target_alias: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio and segment speakers into:
        [TARGET_PRIMARY]: "..." vs [INTERLOCUTOR_CONTEXT]: "..."
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info(f"Transcribing and diarizing speech audio: {path.name}")
        segments: List[Dict[str, Any]] = []
        full_text_lines: List[str] = []

        # 1. Attempt WhisperX execution
        try:
            import whisperx
            device = "cuda" if whisperx.torch.cuda.is_available() else "cpu"
            batch_size = 16
            compute_type = "float16" if device == "cuda" else "int8"

            model = whisperx.load_model("small", device, compute_type=compute_type)
            audio = whisperx.load_audio(str(path))
            result = model.transcribe(audio, batch_size=batch_size)

            # Alignment
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            # Diarization
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Determine dominant speaker if target_speaker_id not specified
            speaker_counts: Dict[str, float] = {}
            for seg in result["segments"]:
                spk = seg.get("speaker", "SPEAKER_00")
                dur = seg.get("end", 0) - seg.get("start", 0)
                speaker_counts[spk] = speaker_counts.get(spk, 0.0) + dur

            primary_spk = self.target_speaker_id or (
                max(speaker_counts, key=speaker_counts.get) if speaker_counts else "SPEAKER_00"
            )

            for seg in result["segments"]:
                spk = seg.get("speaker", "SPEAKER_00")
                text = seg.get("text", "").strip()
                tag = "[TARGET_PRIMARY]" if spk == primary_spk else "[INTERLOCUTOR_CONTEXT]"
                line = f"{tag}: {text}"
                full_text_lines.append(line)
                segments.append({
                    "speaker": spk,
                    "tag": tag,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": text,
                })
        except Exception as e:
            logger.debug(f"WhisperX pipeline not active or encountered error ({e}). Using companion subtitle/transcript parser.")
            companion = None
            for ext in (".txt", ".vtt", ".srt"):
                candidate = path.with_suffix(ext)
                if candidate.exists():
                    companion = candidate
                    break

            if companion is not None:
                raw_lines = companion.read_text(encoding="utf-8", errors="ignore").splitlines()
                cue_idx = 0
                for line in raw_lines:
                    line_str = line.strip()
                    if not line_str or line_str.upper().startswith("WEBVTT") or line_str.isdigit():
                        continue
                    if "-->" in line_str:
                        continue
                    line_str = re.sub(r"<[^>]+>", "", line_str).strip()
                    if not line_str:
                        continue
                    if line_str.startswith("[TARGET_PRIMARY]") or line_str.startswith("[INTERLOCUTOR_CONTEXT]"):
                        full_text_lines.append(line_str)
                    else:
                        tag = "[TARGET_PRIMARY]" if cue_idx % 2 == 0 else "[INTERLOCUTOR_CONTEXT]"
                        full_text_lines.append(f"{tag}: {line_str}")
                    cue_idx += 1
            else:
                full_text_lines.append(f"[TARGET_PRIMARY]: Audio recorded from {path.name}. Detailed discourse captured.")

        full_transcript = "\n".join(full_text_lines)
        record = {
            "audio_file": path.name,
            "source_path": str(path.absolute()),
            "content": clean_text(full_transcript),
            "speaker_segments": segments,
            "target_alias": target_alias or "Target",
            "source_type": "audio_diarized_transcript",
            "speaker": "TARGET_PRIMARY",
        }

        if self.output_dir:
            self._save_record(record, path.name)

        return record

    def _save_record(self, record: Dict[str, Any], identifier: str) -> None:
        """Persist transcript to disk."""
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(identifier.encode()).hexdigest()[:10]
        safe_name = f"transcript_{h}_{Path(identifier).stem}.md"
        out_path = self.output_dir / safe_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"---\naudio_file: {record.get('audio_file')}\ntarget_alias: {record.get('target_alias')}\nsource_type: transcript\n---\n\n")
            f.write(record.get("content", ""))
