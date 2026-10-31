"""
Visual harvester for who??.
Extracts frames and key visual slides from downloaded media assets for OCR analysis.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from who.utils.logger import get_logger

logger = get_logger("who.harvest.visual")


class VisualHarvester:
    """Extracts keyframes from video files using ffmpeg or OpenCV."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def extract_keyframes(
        self,
        video_path: Path | str,
        interval_seconds: int = 30,
        max_frames: int = 20,
    ) -> List[Path]:
        """Extract periodic keyframes from a video for OCR and visual indexing."""
        v_path = Path(video_path)
        if not v_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return []

        out_frames_dir = (self.output_dir or v_path.parent) / f"{v_path.stem}_frames"
        out_frames_dir.mkdir(parents=True, exist_ok=True)
        extracted: List[Path] = []

        # 1. Try ffmpeg
        try:
            out_pattern = str(out_frames_dir / "frame_%04d.jpg")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(v_path),
                "-vf", f"fps=1/{interval_seconds}",
                "-vframes", str(max_frames),
                out_pattern,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            extracted = sorted(list(out_frames_dir.glob("frame_*.jpg")))
            if extracted:
                logger.info(f"Extracted {len(extracted)} keyframes using ffmpeg.")
                return extracted
        except Exception as e:
            logger.debug(f"ffmpeg extraction not available: {e}")

        # 2. Try OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(str(v_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_interval = int(fps * interval_seconds)
            count = 0
            saved = 0

            while cap.isOpened() and saved < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % frame_interval == 0:
                    frame_file = out_frames_dir / f"frame_{saved:04d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                    extracted.append(frame_file)
                    saved += 1
                count += 1
            cap.release()
            logger.info(f"Extracted {len(extracted)} keyframes using OpenCV.")
        except Exception as e:
            logger.debug(f"OpenCV frame extraction unavailable: {e}")

        return extracted
