"""
Visual harvester for who??.
Extracts keyframes from video/animated media assets, deduplicates static slides via perceptual dHash,
and routes unique visual scenes into VisionAnalyzer for Multimodal OSINT synthesis.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageSequence

from src.harvest.visual_intelligence import ImageForensics, VisionAnalyzer
from src.utils.logger import get_logger
from src.utils.text_cleaner import sanitize_slug

logger = get_logger("who.harvest.visual")


class VisualHarvester:
    """Extracts keyframes from video files and deduplicates near-identical presentation slides."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def extract_keyframes(
        self,
        video_path: Path | str,
        interval_seconds: int = 30,
        max_frames: int = 20,
    ) -> List[Path]:
        """Extract periodic keyframes from a video (ffmpeg / OpenCV) or multi-frame visual asset."""
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
                "-i",
                str(v_path),
                "-vf",
                f"fps=1/{max(1, interval_seconds)}",
                "-vframes",
                str(max_frames),
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
            frame_interval = max(1, int(fps * max(1, interval_seconds)))
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
            if extracted:
                logger.info(f"Extracted {len(extracted)} keyframes using OpenCV.")
                return extracted
        except Exception as e:
            logger.debug(f"OpenCV frame extraction unavailable: {e}")

        # 3. Fallback: Pillow multi-frame container (e.g., animated GIF / WebP / multi-page TIFF slide deck)
        try:
            with Image.open(v_path) as img:
                for idx, frame in enumerate(ImageSequence.Iterator(img)):
                    if idx >= max_frames:
                        break
                    frame_file = out_frames_dir / f"frame_{idx:04d}.jpg"
                    frame.convert("RGB").save(frame_file, format="JPEG", quality=90)
                    extracted.append(frame_file)
            if extracted:
                logger.info(f"Extracted {len(extracted)} frames via Pillow ImageSequence.")
        except Exception as e:
            logger.debug(f"Pillow multi-frame fallback note: {e}")

        return extracted

    @staticmethod
    def deduplicate_frames(
        frames: List[Path],
        hamming_threshold: int = 6,
    ) -> List[Path]:
        """
        Filter out near-identical consecutive frames/slides using 64-bit perceptual dHash.
        Retains a frame only when its dHash differs from previously retained frames by >= hamming_threshold.
        """
        unique_frames: List[Path] = []
        retained_hashes: List[str] = []

        for frame_path in frames:
            if not frame_path.exists():
                continue
            try:
                with Image.open(frame_path) as img:
                    f_hash = ImageForensics.compute_dhash(img)
            except Exception:
                continue

            is_duplicate = any(
                ImageForensics.hamming_distance(f_hash, prev_h) < hamming_threshold
                for prev_h in retained_hashes
            )
            if not is_duplicate:
                unique_frames.append(frame_path)
                retained_hashes.append(f_hash)

        return unique_frames

    def analyze_video_visuals(
        self,
        video_path: Path | str,
        vision_analyzer: VisionAnalyzer,
        interval_seconds: int = 30,
        max_frames: int = 12,
        hamming_threshold: int = 6,
    ) -> Dict[str, Any]:
        """
        End-to-end Video Visual OSINT Pipeline:
        1. Extract keyframes from video / slide stream.
        2. Deduplicate static scenes using perceptual dHash.
        3. Analyze each unique scene via VisionAnalyzer (enforcing Zero-Substitution).
        4. Persist a consolidated Markdown slide dossier into raw_data/.
        """
        v_path = Path(video_path)
        raw_frames = self.extract_keyframes(
            v_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )
        unique_frames = self.deduplicate_frames(raw_frames, hamming_threshold=hamming_threshold)

        verified_scenes: List[Dict[str, Any]] = []
        rejected_scenes = 0

        for idx, frame in enumerate(unique_frames, 1):
            hint = f"Keyframe #{idx} extracted from video '{v_path.name}' (approx. {(idx - 1) * interval_seconds}s mark)"
            res = vision_analyzer.analyze_image(
                frame,
                context_hint=hint,
                ingest_to_corpus=False,
            )
            if res.get("rejected"):
                rejected_scenes += 1
            else:
                verified_scenes.append(res)

        dossier_path: Optional[Path] = None
        if verified_scenes and vision_analyzer.raw_data_dir:
            vision_analyzer.raw_data_dir.mkdir(parents=True, exist_ok=True)
            safe_stem = sanitize_slug(v_path.stem) or "video"
            dossier_path = vision_analyzer.raw_data_dir / f"video_visual_dossier_{safe_stem}.md"

            sections = [
                f"# Video Visual & Slide Dossier: {v_path.name}",
                f"**Target:** {vision_analyzer.target_name}",
                f"**Total Extracted Frames:** {len(raw_frames)} | **Unique Scenes Analyzed:** {len(verified_scenes)}",
                "\n---\n",
            ]
            for s_idx, scene in enumerate(verified_scenes, 1):
                sections.append(
                    f"## Scene {s_idx}: `{scene['file_name']}` (dHash: `{scene['forensics']['dhash']}`)\n"
                    f"{scene['vision_synthesis']}\n"
                )
            dossier_path.write_text("\n".join(sections), encoding="utf-8")

        return {
            "video_name": v_path.name,
            "total_frames_extracted": len(raw_frames),
            "unique_frames_retained": len(unique_frames),
            "verified_scenes_count": len(verified_scenes),
            "rejected_scenes_count": rejected_scenes,
            "scenes": verified_scenes,
            "consolidated_dossier_path": str(dossier_path) if dossier_path else None,
        }
