"""
Media harvester for who?? using yt-dlp.
Downloads audio and video files from YouTube, podcasts, Spotify, and Vimeo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("who.harvest.media")


class MediaHarvester:
    """Downloads audio streams and metadata from video/podcast URLs using yt-dlp."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("profiles/temp/raw_data/media")

    def download_audio(self, url: str) -> Dict[str, Any]:
        """Download highest-quality audio stream from URL and extract metadata."""
        logger.info(f"Downloading media stream for: {url}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_template = str(self.output_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown Media")
                video_id = info.get("id", "media")
                duration = info.get("duration", 0)
                description = info.get("description", "")
                channel = info.get("uploader", "Unknown Channel")
                expected_audio = self.output_dir / f"{video_id}.mp3"

                meta_record = {
                    "source_url": url,
                    "title": title,
                    "channel": channel,
                    "duration_seconds": duration,
                    "description": description,
                    "audio_file_path": str(expected_audio) if expected_audio.exists() else str(self.output_dir / f"{video_id}.webm"),
                    "source_type": "media_audio",
                }

                # Save metadata json
                meta_path = self.output_dir / f"{video_id}_meta.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_record, f, indent=2)

                return meta_record
        except Exception as e:
            logger.error(f"yt-dlp download failed for {url}: {e}")
            # Generate placeholder descriptor if download failed or yt-dlp missing
            placeholder = {
                "source_url": url,
                "title": f"Media stream from {url}",
                "channel": "Unknown",
                "duration_seconds": 0,
                "description": f"Failed extraction: {e}",
                "audio_file_path": "",
                "source_type": "media_audio",
            }
            return placeholder

    def download_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Download multiple audio sources."""
        results = []
        for u in urls:
            results.append(self.download_audio(u))
        return results
