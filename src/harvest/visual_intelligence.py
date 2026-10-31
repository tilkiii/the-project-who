"""
Multimodal Visual Intelligence & Forensic OSINT Engine for who??.

Capabilities:
1. ImageForensics:
   - Cryptographic SHA-256 hashing, 64-bit Perceptual Difference Hash (dHash), 24-bin RGB histogram.
   - EXIF / Sub-IFD / PNG-Info / GPS coordinate extraction & ISO-8601 timestamp normalization
     (bridging photo capture times directly into CognitiveChronomics circadian drift analysis).
   - Editing Software & AI-Generator metadata signature auditing + JPEG Error Level Analysis (ELA).
2. AvatarIdentityMatcher:
   - Cross-platform avatar perceptual comparison (dHash Hamming similarity + RGB cosine similarity)
     with calibrated probabilistic confidence scoring (AGENTS.md Rule #3).
3. VisionAnalyzer:
   - Multimodal Vision-LLM slide, diagram, scene, and environmental cue extraction across
     Google Gemini Vision, OpenAI/OpenRouter Vision, Anthropic Claude Vision, and Ollama Vision,
     with resilient offline OCR + forensic fallback.
   - Enforces Strict Zero-Substitution Disambiguation Gate (AGENTS.md Rule #1) on EXIF author tags
     and extracted visual text — zero third-party namesake files are ever written to disk.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import ExifTags, Image, ImageChops

from src.config import settings
from src.harvest.ocr_processor import OCRProcessor
from src.utils.entity_validator import (
    DISAMBIGUATION_NOTICE_TR,
    detect_unauthorized_name_expansion,
)
from src.utils.logger import get_logger
from src.utils.text_cleaner import sanitize_slug

logger = get_logger("who.harvest.visual_intelligence")

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}

_EDITING_SOFTWARE_SIGNATURES = (
    "photoshop",
    "lightroom",
    "gimp",
    "canva",
    "affinity",
    "snapseed",
    "picsart",
    "pixelmator",
    "paint.net",
    "illustrator",
)

_AI_GENERATOR_SIGNATURES = (
    "midjourney",
    "dall-e",
    "dalle",
    "stable diffusion",
    "automatic1111",
    "comfyui",
    "adobe firefly",
    "firefly",
    "c2pa",
    "flux.1",
    "imagen",
    "leonardo.ai",
)


def _rational_to_float(val: Any) -> float:
    """Safely convert Pillow IFDRational, (num, den) tuple, or numeric value to float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, tuple) and len(val) == 2:
        den = float(val[1])
        return float(val[0]) / den if den != 0.0 else 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


def _decode_exif_value(val: Any) -> str:
    """Convert raw EXIF tag value (bytes, UTF-16LE XP tags, strings) into clean UTF-8 text."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        # Windows XP* tags (XPAuthor, XPComment, XPTitle) are UTF-16LE null-terminated
        if b"\x00" in val and len(val) >= 4:
            try:
                decoded = val.decode("utf-16le", errors="ignore").rstrip("\x00").strip()
                if decoded and all(ord(c) >= 32 or c in "\n\r\t" for c in decoded):
                    return decoded
            except Exception:
                pass
        if val.startswith(b"ASCII\x00\x00\x00"):
            val = val[8:]
        elif val.startswith(b"UNICODE\x00"):
            try:
                return val[8:].decode("utf-16", errors="ignore").rstrip("\x00").strip()
            except Exception:
                pass
        try:
            return val.decode("utf-8", errors="strict").rstrip("\x00").strip()
        except UnicodeDecodeError:
            return val.decode("cp1254", errors="ignore").rstrip("\x00").strip()
    text_val = str(val).rstrip("\x00").strip()
    # Pillow decodes UTF-8 TIFF ASCII tags via latin-1; recover original UTF-8 text losslessly if valid
    if any("\x80" <= ch <= "\xff" for ch in text_val):
        try:
            text_val = text_val.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text_val


def _normalize_exif_datetime(raw_dt: str) -> Optional[str]:
    """Normalize EXIF 'YYYY:MM:DD HH:MM:SS' or ISO strings into ISO-8601 'YYYY-MM-DDTHH:MM:SS'."""
    if not raw_dt:
        return None
    cleaned = raw_dt.strip()
    m = re.match(r"^(\d{4})[:\-](\d{2})[:\-](\d{2})[T\s](\d{2}:\d{2}:\d{2})", cleaned)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}"
    return None


def _dms_to_decimal(dms: Any, ref: Any) -> Optional[float]:
    """Convert EXIF GPS (degrees, minutes, seconds) and hemisphere ref into signed decimal degrees."""
    if not dms or not isinstance(dms, (tuple, list)) or len(dms) < 3:
        return None
    deg = _rational_to_float(dms[0])
    minutes = _rational_to_float(dms[1])
    seconds = _rational_to_float(dms[2])
    decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
    ref_str = _decode_exif_value(ref).upper()
    if ref_str.startswith(("S", "W")):
        decimal = -decimal
    return round(decimal, 6)


class ImageForensics:
    """Extracts cryptographic, perceptual, EXIF/GPS, and ELA tampering forensics from images."""

    @staticmethod
    def compute_dhash(img: Image.Image) -> str:
        """Compute 64-bit horizontal difference perceptual hash (16-char hex)."""
        gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        raw = gray.tobytes()
        bits = 0
        for row in range(8):
            offset = row * 9
            for col in range(8):
                bits = (bits << 1) | (1 if raw[offset + col] > raw[offset + col + 1] else 0)
        return f"{bits:016x}"

    @staticmethod
    def hamming_distance(hash_a: str, hash_b: str) -> int:
        """Bitwise Hamming distance (0..64) between two 64-bit hex dHashes."""
        try:
            return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()
        except ValueError:
            return 64

    @staticmethod
    def compute_color_histogram(img: Image.Image) -> List[float]:
        """Compute L2-normalized 24-bin RGB color histogram (8 bins per channel)."""
        raw_hist = img.convert("RGB").histogram()  # 768 ints (256 R + 256 G + 256 B)
        bins: List[float] = []
        for b_idx in range(24):
            start = b_idx * 32
            bins.append(float(sum(raw_hist[start : start + 32])))
        norm = math.sqrt(sum(v * v for v in bins))
        if norm <= 0.0:
            return [0.0] * 24
        return [round(v / norm, 6) for v in bins]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity in [0.0, 1.0] between two normalized vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def classify_aspect_layout(width: int, height: int) -> str:
        """Classify visual asset geometry for OSINT context inference."""
        if width <= 0 or height <= 0:
            return "unknown_geometry"
        ratio = width / float(height)
        if 0.92 <= ratio <= 1.08:
            return "square_avatar_or_social"
        if 1.65 <= ratio <= 1.88:
            return "presentation_slide_16_9"
        if ratio > 1.88:
            return "panoramic_or_banner"
        if ratio <= 0.65:
            return "mobile_vertical_screenshot"
        if 1.25 <= ratio < 1.65:
            return "standard_photograph_or_4_3_slide"
        return "document_or_portrait_capture"

    @staticmethod
    def compute_ela_metrics(img: Image.Image, quality: int = 90) -> Dict[str, Any]:
        """
        Perform lightweight in-memory JPEG Error Level Analysis (ELA) to estimate
        localized recompression variance. Always framed probabilistically (AGENTS.md Rule #3).
        """
        rgb = img.convert("RGB")
        # Cap evaluation resolution for deterministic speed
        if max(rgb.size) > 1024:
            rgb = rgb.copy()
            rgb.thumbnail((1024, 1024), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as resaved:
            diff = ImageChops.difference(rgb, resaved.convert("RGB"))
            gray_diff = diff.convert("L")
            diff_bytes = gray_diff.tobytes()

        if not diff_bytes:
            return {
                "mean_error": 0.0,
                "max_error": 0,
                "regional_block_std": 0.0,
                "ela_anomaly_score": 0.0,
                "assessment": "LOW_RECOMPRESSION_VARIANCE",
            }

        mean_err = sum(diff_bytes) / float(len(diff_bytes))
        max_err = max(diff_bytes)

        # Regional 8x8 grid variance to detect localized splice/overlay contrast
        grid_bytes = gray_diff.resize((8, 8), Image.Resampling.BOX).tobytes()
        grid_mean = sum(grid_bytes) / float(len(grid_bytes))
        grid_var = sum((b - grid_mean) ** 2 for b in grid_bytes) / float(len(grid_bytes))
        block_std = math.sqrt(grid_var)

        # Normalized heuristic anomaly index in [0.0, 1.0]
        raw_score = min(1.0, (block_std / 18.0) * 0.7 + (max_err / 255.0) * 0.3)
        if raw_score >= 0.55:
            tier = "ELEVATED_LOCALIZED_RECOMPRESSION_VARIANCE"
        elif raw_score >= 0.28:
            tier = "MODERATE_COMPRESSION_VARIANCE"
        else:
            tier = "LOW_RECOMPRESSION_VARIANCE"

        return {
            "mean_error": round(mean_err, 3),
            "max_error": int(max_err),
            "regional_block_std": round(block_std, 3),
            "ela_anomaly_score": round(raw_score, 3),
            "assessment": tier,
        }

    @classmethod
    def extract_exif_and_metadata(cls, img: Image.Image) -> Dict[str, Any]:
        """Extract IFD0, Exif sub-IFD, GPS coordinates, and PNG/XMP metadata tags."""
        tags: Dict[str, str] = {}
        gps_raw: Dict[str, Any] = {}

        # 1. Inspect PNG / XMP / format info dictionary
        for k, v in (img.info or {}).items():
            if k in ("exif", "icc_profile", "photoshop"):
                continue
            if isinstance(v, (str, int, float)):
                val_str = str(v).strip()
                if val_str and len(val_str) <= 2000:
                    tags[str(k)] = val_str

        # 2. Inspect Pillow Exif container (IFD0 + Exif Sub-IFD 0x8769 + GPS Sub-IFD 0x8825)
        try:
            exif = img.getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "GPSInfo" and isinstance(val, dict):
                        for gk, gv in val.items():
                            gps_raw[ExifTags.GPSTAGS.get(gk, str(gk))] = gv
                    elif tag_name not in ("MakerNote", "PrintImageMatching"):
                        decoded = _decode_exif_value(val)
                        if decoded:
                            tags[tag_name] = decoded

                # Exif Sub-IFD (DateTimeOriginal = 36867, LensModel = 42036, UserComment = 37510)
                try:
                    exif_ifd = exif.get_ifd(0x8769)
                    for tag_id, val in (exif_ifd or {}).items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name not in ("MakerNote",):
                            decoded = _decode_exif_value(val)
                            if decoded:
                                tags[tag_name] = decoded
                except Exception:
                    pass

                # GPS Sub-IFD (0x8825)
                try:
                    gps_ifd = exif.get_ifd(0x8825)
                    for tag_id, val in (gps_ifd or {}).items():
                        gps_name = ExifTags.GPSTAGS.get(tag_id, str(tag_id))
                        gps_raw[gps_name] = val
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"EXIF read note: {e}")

        # 3. Resolve normalized ISO timestamp
        raw_ts = (
            tags.get("DateTimeOriginal")
            or tags.get("DateTimeDigitized")
            or tags.get("DateTime")
            or tags.get("Creation Time")
            or ""
        )
        capture_ts_iso = _normalize_exif_datetime(raw_ts)

        # 4. Resolve camera / capture device
        make = tags.get("Make", "").strip()
        model = tags.get("Model", "").strip()
        if make and model:
            camera_device = model if model.lower().startswith(make.lower()) else f"{make} {model}"
        else:
            camera_device = make or model or ""

        # 5. Resolve declared author metadata
        author_meta = (
            tags.get("Artist")
            or tags.get("Author")
            or tags.get("XPAuthor")
            or tags.get("Creator")
            or ""
        ).strip()

        # 6. Resolve GPS decimal coordinates
        lat = _dms_to_decimal(gps_raw.get("GPSLatitude"), gps_raw.get("GPSLatitudeRef", "N"))
        lon = _dms_to_decimal(gps_raw.get("GPSLongitude"), gps_raw.get("GPSLongitudeRef", "E"))
        gps_info: Dict[str, Any] = {"has_gps": lat is not None and lon is not None}
        if lat is not None and lon is not None:
            gps_info["latitude"] = lat
            gps_info["longitude"] = lon
            gps_info["maps_query"] = f"{lat},{lon}"

        # 7. Software & AI Generator signature audit
        searchable_blob = " | ".join(
            [
                tags.get("Software", ""),
                tags.get("ProcessingSoftware", ""),
                tags.get("UserComment", ""),
                tags.get("ImageDescription", ""),
                tags.get("parameters", ""),
                tags.get("prompt", ""),
                tags.get("Comment", ""),
            ]
        ).lower()

        detected_editors = [sig for sig in _EDITING_SOFTWARE_SIGNATURES if sig in searchable_blob]
        detected_ai = [sig for sig in _AI_GENERATOR_SIGNATURES if sig in searchable_blob]

        return {
            "raw_tags": tags,
            "capture_timestamp_raw": raw_ts,
            "capture_timestamp_iso": capture_ts_iso,
            "camera_device": camera_device,
            "author_metadata": author_meta,
            "description_metadata": (
                tags.get("ImageDescription")
                or tags.get("UserComment")
                or tags.get("Description")
                or tags.get("Comment")
                or ""
            ).strip(),
            "software_tag": tags.get("Software") or tags.get("ProcessingSoftware") or "",
            "detected_editing_tools": detected_editors,
            "detected_ai_signatures": detected_ai,
            "gps": gps_info,
        }

    @classmethod
    def inspect_image(
        cls,
        image_path: Path | str,
        target_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full forensic, perceptual, EXIF/GPS, ELA, and Zero-Substitution audit on an image."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Visual asset not found: {path}")

        raw_bytes = path.read_bytes()
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = (img.format or path.suffix.lstrip(".") or "UNKNOWN").upper()
            width, height = img.size
            mode = img.mode
            dhash = cls.compute_dhash(img)
            histogram = cls.compute_color_histogram(img)
            meta = cls.extract_exif_and_metadata(img)
            ela = cls.compute_ela_metrics(img)

        layout_class = cls.classify_aspect_layout(width, height)

        # Strict Zero-Substitution Check (AGENTS.md Rule #1) on EXIF Author & Description
        is_contaminated = False
        contamination_reason = "CLEAN"
        if target_name and target_name.strip():
            is_contaminated, contamination_reason = detect_unauthorized_name_expansion(
                target_name=target_name,
                text=meta["description_metadata"],
                declared_author=meta["author_metadata"] or None,
            )

        # Calibrated probabilistic authenticity assessment (AGENTS.md Rule #3)
        penalty = 0.0
        flags: List[str] = []
        if meta["detected_ai_signatures"]:
            penalty += 0.55
            flags.append(f"AI generator metadata marker detected ({', '.join(meta['detected_ai_signatures'])})")
        if meta["detected_editing_tools"]:
            penalty += 0.18
            flags.append(f"Post-processing software tag present ({', '.join(meta['detected_editing_tools'])})")
        if ela["ela_anomaly_score"] >= 0.55:
            penalty += 0.22
            flags.append(f"Elevated regional ELA variance ({ela['ela_anomaly_score']})")
        if not meta["camera_device"] and not meta["capture_timestamp_iso"]:
            penalty += 0.08
            flags.append("Stripped or absent camera EXIF provenance (typical of social/web redistribution)")

        authenticity_conf = round(max(12.0, min(96.0, (1.0 - penalty) * 92.0)), 1)

        return {
            "file_name": path.name,
            "source_path": str(path.resolve()),
            "file_size_bytes": len(raw_bytes),
            "sha256": sha256_hex,
            "format": fmt,
            "mode": mode,
            "width": width,
            "height": height,
            "layout_class": layout_class,
            "dhash": dhash,
            "rgb_histogram": histogram,
            "capture_timestamp_iso": meta["capture_timestamp_iso"],
            "camera_device": meta["camera_device"],
            "author_metadata": meta["author_metadata"],
            "description_metadata": meta["description_metadata"],
            "software_tag": meta["software_tag"],
            "detected_editing_tools": meta["detected_editing_tools"],
            "detected_ai_signatures": meta["detected_ai_signatures"],
            "gps": meta["gps"],
            "exif_tags": meta["raw_tags"],
            "ela": ela,
            "forensic_flags": flags,
            "authenticity_confidence_pct": authenticity_conf,
            "disambiguation_gate": {
                "rejected": is_contaminated,
                "reason": contamination_reason,
                "notice": DISAMBIGUATION_NOTICE_TR if is_contaminated else "",
            },
        }


class AvatarIdentityMatcher:
    """
    Perceptual Cross-Platform Avatar Matcher.
    Combines 64-bit dHash Hamming similarity and 24-bin RGB histogram cosine similarity
    to evaluate whether two profile avatars plausibly originate from the same visual photograph.
    """

    @staticmethod
    def _resolve_to_local_path(source: Path | str, temp_files: List[Path]) -> Path:
        src_str = str(source).strip()
        if src_str.startswith(("http://", "https://")):
            resp = httpx.get(src_str, timeout=15.0, follow_redirects=True)
            resp.raise_for_status()
            suffix = Path(src_str.split("?")[0]).suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(resp.content)
            tmp.close()
            tmp_path = Path(tmp.name)
            temp_files.append(tmp_path)
            return tmp_path
        return Path(src_str)

    @classmethod
    def compare_avatars(
        cls,
        image_a: Path | str,
        image_b: Path | str,
        target_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare two avatar images and return calibrated perceptual correspondence metrics."""
        temp_files: List[Path] = []
        try:
            path_a = cls._resolve_to_local_path(image_a, temp_files)
            path_b = cls._resolve_to_local_path(image_b, temp_files)

            info_a = ImageForensics.inspect_image(path_a, target_name=target_name)
            info_b = ImageForensics.inspect_image(path_b, target_name=target_name)

            if info_a["disambiguation_gate"]["rejected"] or info_b["disambiguation_gate"]["rejected"]:
                reason = (
                    info_a["disambiguation_gate"]["reason"]
                    if info_a["disambiguation_gate"]["rejected"]
                    else info_b["disambiguation_gate"]["reason"]
                )
                return {
                    "image_a": info_a["file_name"],
                    "image_b": info_b["file_name"],
                    "dhash_a": info_a["dhash"],
                    "dhash_b": info_b["dhash"],
                    "hamming_distance": 64,
                    "dhash_similarity": 0.0,
                    "histogram_cosine": 0.0,
                    "visual_similarity_score": 0.0,
                    "similarity_pct": 0.0,
                    "match_tier": "REJECTED_BY_DISAMBIGUATION_GATE",
                    "confidence_score_pct": 95.0,
                    "disambiguation_rejected": True,
                    "disambiguation_reason": reason,
                    "probabilistic_statement": DISAMBIGUATION_NOTICE_TR,
                }

            # Exact SHA-256 byte match shortcut
            if info_a["sha256"] == info_b["sha256"]:
                hamming = 0
                dhash_sim = 1.0
                hist_sim = 1.0
                combined = 1.0
            else:
                hamming = ImageForensics.hamming_distance(info_a["dhash"], info_b["dhash"])
                dhash_sim = max(0.0, 1.0 - (hamming / 64.0))
                hist_sim = ImageForensics.cosine_similarity(
                    info_a["rgb_histogram"], info_b["rgb_histogram"]
                )
                combined = round(0.68 * dhash_sim + 0.32 * hist_sim, 4)

            sim_pct = round(combined * 100.0, 1)

            if combined >= 0.85 or hamming <= 7:
                tier = "HIGH_VISUAL_CORRESPONDENCE"
                conf_pct = round(min(96.0, 78.0 + combined * 18.0), 1)
                statement = (
                    f"İki profil görseli arasında yüksek algısal örtüşme tespit edildi "
                    f"(Benzerlik: %{sim_pct}, dHash Hamming Mesafesi: {hamming}/64). "
                    f"Görsellerin aynı kaynak fotoğrafın yeniden boyutlandırılmış veya sıkıştırılmış "
                    f"versiyonları olma olasılığı yüksektir (Confidence Score: {conf_pct}%)."
                )
            elif combined >= 0.68 or hamming <= 16:
                tier = "MODERATE_VISUAL_SIMILARITY"
                conf_pct = round(58.0 + combined * 20.0, 1)
                statement = (
                    f"Görseller arasında orta düzeyde yapısal/renksel benzerlik gözlemlendi "
                    f"(Benzerlik: %{sim_pct}, dHash Hamming Mesafesi: {hamming}/64). "
                    f"Kırpma (crop) veya filtre uygulanmış olabilir; birincil @handle/isim doğrulaması "
                    f"ile desteklenmelidir (Confidence Score: {conf_pct}%)."
                )
            else:
                tier = "DISTINCT_VISUAL_ASSETS"
                conf_pct = round(min(94.0, 70.0 + (1.0 - combined) * 24.0), 1)
                statement = (
                    f"İki görsel algısal olarak farklı kaynaklara işaret etmektedir "
                    f"(Benzerlik: %{sim_pct}, dHash Hamming Mesafesi: {hamming}/64). "
                    f"Sıfır İkame Protokolü gereği yalnızca görsel varsayımla hesap birleştirmesi "
                    f"yapılamaz (Confidence Score: {conf_pct}%)."
                )

            return {
                "image_a": info_a["file_name"],
                "image_b": info_b["file_name"],
                "sha256_identical": info_a["sha256"] == info_b["sha256"],
                "dhash_a": info_a["dhash"],
                "dhash_b": info_b["dhash"],
                "hamming_distance": hamming,
                "dhash_similarity": round(dhash_sim, 4),
                "histogram_cosine": round(hist_sim, 4),
                "visual_similarity_score": round(combined, 4),
                "similarity_pct": sim_pct,
                "match_tier": tier,
                "confidence_score_pct": conf_pct,
                "disambiguation_rejected": False,
                "probabilistic_statement": statement,
            }
        finally:
            for tf in temp_files:
                try:
                    tf.unlink(missing_ok=True)
                except Exception:
                    pass


class VisionAnalyzer:
    """
    Multimodal Vision-LLM & Forensic OSINT Analyzer.
    Extracts slide arguments, diagrams, environmental/institutional cues, and OCR text
    while strictly enforcing the Zero-Substitution Disambiguation Gate (AGENTS.md Rule #1).
    """

    def __init__(
        self,
        target_name: str,
        raw_data_dir: Optional[Path] = None,
        vault_dir: Optional[Path] = None,
    ):
        self.target_name = target_name
        self.raw_data_dir = raw_data_dir
        self.vault_dir = vault_dir
        self.ocr_processor = OCRProcessor(output_dir=None)
        self.rejected_namesakes_count: int = 0

    @staticmethod
    def _encode_image_for_vision_api(image_path: Path, max_edge: int = 1280) -> Tuple[str, str]:
        """Downscale large images if needed and return (mime_type, base64_string)."""
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            if max(rgb.size) > max_edge:
                rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=88)
            return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_vision_prompt(
        self,
        forensics: Dict[str, Any],
        ocr_text: str,
        context_hint: Optional[str] = None,
    ) -> str:
        hint_line = f"Additional Context: {context_hint}\n" if context_hint else ""
        ocr_line = f"Pre-extracted OCR / Metadata Text:\n{ocr_text}\n" if ocr_text else ""
        return (
            f"Analyze this visual OSINT asset associated with target '{self.target_name}'.\n"
            f"Geometry: {forensics['width']}x{forensics['height']} ({forensics['layout_class']}).\n"
            f"{hint_line}{ocr_line}\n"
            "Provide an epistemically calibrated analysis with:\n"
            "1. **Visible Text & Slide/Document Claims**: Transcribe key titles, bullet points, or code/diagrams verbatim.\n"
            "2. **Contextual & Institutional Cues**: Note observable conference banners, academic/corporate affiliations, software tools, or work environment cues strictly visible in the frame.\n"
            "3. **Cognitive & Thematic Signals**: Summarize what intellectual priorities or technical focus areas this visual artifact evidences (include a calibrated Confidence Score %).\n"
            "Do NOT speculate beyond visible primary evidence."
        )

    def _call_multimodal_vision_llm(
        self,
        image_path: Path,
        forensics: Dict[str, Any],
        ocr_text: str,
        context_hint: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Attempt multimodal Vision-LLM analysis via configured provider REST API.
        Returns (synthesis_markdown, vision_engine_used).
        """
        system_prompt = (
            "You are an epistemic visual OSINT analyst for the who?? Cognitive Digital Twin framework. "
            "Ground every claim strictly in observable visual or textual evidence and include calibrated confidence percentages."
        )
        user_prompt = self._build_vision_prompt(forensics, ocr_text, context_hint)
        mime_type, b64_img = self._encode_image_for_vision_api(image_path)

        override = (settings.provider_override or "").lower().strip()

        # 1. Google Gemini Vision (gemini-2.5-flash)
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.gemini_api_key
        if gemini_key and override in ("", "gemini", "google"):
            try:
                model_name = os.getenv("VISION_LLM_MODEL", "gemini-2.5-flash")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                payload = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [
                        {
                            "parts": [
                                {"text": user_prompt},
                                {"inline_data": {"mime_type": mime_type, "data": b64_img}},
                            ]
                        }
                    ],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
                }
                resp = httpx.post(url, json=payload, timeout=35.0)
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_out = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
                        if text_out:
                            return text_out, f"gemini-vision ({model_name})"
            except Exception as e:
                logger.debug(f"Gemini Vision API fallback note: {e}")

        # 2. OpenAI / OpenRouter Vision
        openai_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
        if (openai_key and override in ("", "openai")) or (openrouter_key and override in ("", "openrouter")):
            use_router = bool(openrouter_key and (override == "openrouter" or not openai_key))
            api_key = openrouter_key if use_router else openai_key
            base_url = "https://openrouter.ai/api/v1" if use_router else "https://api.openai.com/v1"
            model_name = os.getenv("VISION_LLM_MODEL", "openai/gpt-4o-mini" if use_router else "gpt-4o-mini")
            try:
                resp = httpx.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model_name,
                        "temperature": 0.2,
                        "max_tokens": 1200,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                                ],
                            },
                        ],
                    },
                    timeout=35.0,
                )
                if resp.status_code == 200:
                    choices = resp.json().get("choices", [])
                    if choices:
                        text_out = (choices[0].get("message", {}).get("content") or "").strip()
                        if text_out:
                            return text_out, f"openai-vision ({model_name})"
            except Exception as e:
                logger.debug(f"OpenAI/OpenRouter Vision API fallback note: {e}")

        # 3. Anthropic Claude Vision
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or settings.anthropic_api_key
        if anthropic_key and override in ("", "anthropic", "claude"):
            try:
                model_name = os.getenv("VISION_LLM_MODEL", "claude-3-5-sonnet-latest")
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "max_tokens": 1200,
                        "system": system_prompt,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": mime_type, "data": b64_img},
                                    },
                                    {"type": "text", "text": user_prompt},
                                ],
                            }
                        ],
                    },
                    timeout=35.0,
                )
                if resp.status_code == 200:
                    blocks = resp.json().get("content", [])
                    text_out = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                    if text_out:
                        return text_out, f"anthropic-vision ({model_name})"
            except Exception as e:
                logger.debug(f"Anthropic Vision API fallback note: {e}")

        # 4. Local Ollama Vision (e.g., llava / qwen2.5-vl)
        if override == "ollama":
            try:
                model_name = os.getenv("VISION_LLM_MODEL", "llava")
                resp = httpx.post(
                    f"{settings.ollama_api_base.rstrip('/')}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "images": [b64_img],
                        "stream": False,
                    },
                    timeout=40.0,
                )
                if resp.status_code == 200:
                    text_out = (resp.json().get("response") or "").strip()
                    if text_out:
                        return text_out, f"ollama-vision ({model_name})"
            except Exception as e:
                logger.debug(f"Ollama Vision API fallback note: {e}")

        # 5. Deterministic Offline Forensic + OCR Synthesis Fallback
        return self._offline_visual_synthesis(forensics, ocr_text, context_hint), "offline-forensic-ocr"

    def _offline_visual_synthesis(
        self,
        forensics: Dict[str, Any],
        ocr_text: str,
        context_hint: Optional[str] = None,
    ) -> str:
        """Synthesize deterministic, evidence-grounded visual findings when offline."""
        layout_labels = {
            "presentation_slide_16_9": "16:9 Sunum Slaydı / Konferans Ekran Görüntüsü",
            "square_avatar_or_social": "1:1 Profil Avatarı / Sosyal Medya Görseli",
            "mobile_vertical_screenshot": "Dikey Mobil Ekran Görüntüsü",
            "standard_photograph_or_4_3_slide": "4:3 Fotoğraf / Akademik Slayt Karesi",
            "panoramic_or_banner": "Panoramik Kapak / Banner Görseli",
            "document_or_portrait_capture": "Belge Taraması / Portre Kaydı",
        }
        human_layout = layout_labels.get(forensics["layout_class"], forensics["layout_class"])

        lines = [
            f"### Görsel Yapı & Sahne Sınıflandırması",
            f"- **Varlık Tipi:** {human_layout} (`{forensics['width']}x{forensics['height']}`, `{forensics['format']}`)",
            f"- **Algısal Parmak İzi (dHash):** `{forensics['dhash']}`",
            f"- **Adli Bütünlük Güveni:** %{forensics['authenticity_confidence_pct']} (`{forensics['ela']['assessment']}`)",
        ]
        if forensics.get("camera_device"):
            lines.append(f"- **Kayıt Cihazı (EXIF):** {forensics['camera_device']}")
        if forensics.get("capture_timestamp_iso"):
            lines.append(f"- **Çekim Zaman Damgası (ISO-8601):** `{forensics['capture_timestamp_iso']}`")
        if forensics.get("gps", {}).get("has_gps"):
            gps = forensics["gps"]
            lines.append(f"- **Coğrafi Koordinat (EXIF GPS):** `{gps['latitude']}, {gps['longitude']}`")
        if context_hint:
            lines.append(f"- **Analitik Bağlam Notu:** {context_hint}")

        if ocr_text:
            lines.append("\n### Çıkarılan Birincil Görsel Metin / Slayt İçeriği")
            lines.append(ocr_text)
            conf = 84
        else:
            conf = 72

        lines.append(
            f"\n### Epistemik Değerlendirme\n"
            f"Görsel meta-verileri ve yapısal öznitelikler `{self.target_name}` dijital ikiz veri tabanına "
            f"birincil kanıt olarak kaydedilmiştir (Confidence Score: {conf}%)."
        )
        return "\n".join(lines)

    def _extract_clean_ocr_text(self, image_path: Path, forensics: Dict[str, Any]) -> str:
        """Combine real OCR output (ignoring synthetic placeholders) with embedded description tags."""
        segments: List[str] = []
        desc = forensics.get("description_metadata", "").strip()
        if desc:
            segments.append(desc)

        try:
            ocr_res = self.ocr_processor.extract_text_from_image(image_path)
            raw_content = ocr_res.get("content", "")
            # Strip markdown title header added by OCRProcessor
            body = re.sub(r"^#\s*OCR Transcription:[^\n]*\n+", "", raw_content).strip()
            placeholder = f"[Visual Text extracted from {image_path.name}]"
            if body and placeholder not in body:
                if body not in segments:
                    segments.append(body)
        except Exception as e:
            logger.debug(f"OCR extraction skipped on {image_path.name}: {e}")

        return "\n\n".join(segments).strip()

    def analyze_image(
        self,
        image_path: Path | str,
        context_hint: Optional[str] = None,
        ingest_to_corpus: bool = True,
    ) -> Dict[str, Any]:
        """
        Inspect an image for EXIF/GPS/ELA forensics and Vision-LLM content, enforce Zero-Substitution,
        and optionally persist verified findings into raw_data/ and vault/visual_forensics_report.json.
        """
        path = Path(image_path)
        forensics = ImageForensics.inspect_image(path, target_name=self.target_name)

        # Gate 1: EXIF Author / Description Zero-Substitution Enforcement (AGENTS.md Rule #1)
        if forensics["disambiguation_gate"]["rejected"]:
            self.rejected_namesakes_count += 1
            logger.warning(
                f"Disambiguation Gate blocked visual asset '{path.name}': "
                f"{forensics['disambiguation_gate']['reason']}"
            )
            return {
                "file_name": path.name,
                "rejected": True,
                "rejection_reason": forensics["disambiguation_gate"]["reason"],
                "notice": DISAMBIGUATION_NOTICE_TR,
                "forensics": forensics,
                "vision_synthesis": "",
                "vision_engine": "blocked_by_disambiguation_gate",
                "ingested_markdown_path": None,
            }

        ocr_text = self._extract_clean_ocr_text(path, forensics)
        vision_md, engine_used = self._call_multimodal_vision_llm(
            image_path=path,
            forensics=forensics,
            ocr_text=ocr_text,
            context_hint=context_hint,
        )

        # Gate 2: Post-OCR / Post-Vision Text Zero-Substitution Enforcement (AGENTS.md Rule #1)
        combined_textual_evidence = f"{ocr_text}\n{vision_md}".strip()
        is_contaminated, reason = detect_unauthorized_name_expansion(
            target_name=self.target_name,
            text=combined_textual_evidence,
            declared_author=forensics.get("author_metadata") or None,
        )
        if is_contaminated:
            self.rejected_namesakes_count += 1
            forensics["disambiguation_gate"] = {
                "rejected": True,
                "reason": reason,
                "notice": DISAMBIGUATION_NOTICE_TR,
            }
            logger.warning(
                f"Disambiguation Gate blocked visual text in '{path.name}': {reason}"
            )
            return {
                "file_name": path.name,
                "rejected": True,
                "rejection_reason": reason,
                "notice": DISAMBIGUATION_NOTICE_TR,
                "forensics": forensics,
                "vision_synthesis": "",
                "vision_engine": "blocked_by_disambiguation_gate",
                "ingested_markdown_path": None,
            }

        ingested_md_path: Optional[Path] = None
        if ingest_to_corpus and self.raw_data_dir:
            ingested_md_path = self._persist_corpus_markdown(
                image_path=path,
                forensics=forensics,
                vision_md=vision_md,
                engine_used=engine_used,
            )

        if ingest_to_corpus and self.vault_dir:
            self._upsert_vault_forensics_report(
                forensics=forensics,
                vision_md=vision_md,
                engine_used=engine_used,
                ingested_md_path=ingested_md_path,
            )

        conf_score = round(
            min(95.0, forensics["authenticity_confidence_pct"] * 0.5 + (44.0 if "vision" in engine_used else 36.0)),
            1,
        )

        return {
            "file_name": path.name,
            "rejected": False,
            "rejection_reason": "CLEAN",
            "forensics": forensics,
            "ocr_text": ocr_text,
            "vision_synthesis": vision_md,
            "vision_engine": engine_used,
            "confidence_score_pct": conf_score,
            "ingested_markdown_path": str(ingested_md_path) if ingested_md_path else None,
        }

    def analyze_inputs(
        self,
        paths: List[Path | str],
        context_hint: Optional[str] = None,
        ingest_to_corpus: bool = True,
    ) -> List[Dict[str, Any]]:
        """Analyze a list of image files or directories containing visual assets."""
        resolved_files: List[Path] = []
        for p in paths:
            candidate = Path(p)
            if candidate.is_dir():
                for child in sorted(candidate.rglob("*")):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                        resolved_files.append(child)
            elif candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                resolved_files.append(candidate)

        results: List[Dict[str, Any]] = []
        for img_file in resolved_files:
            try:
                results.append(
                    self.analyze_image(
                        img_file,
                        context_hint=context_hint,
                        ingest_to_corpus=ingest_to_corpus,
                    )
                )
            except Exception as e:
                logger.error(f"Visual intelligence error on {img_file}: {e}")
        return results

    def _persist_corpus_markdown(
        self,
        image_path: Path,
        forensics: Dict[str, Any],
        vision_md: str,
        engine_used: str,
    ) -> Path:
        """Write verified visual intelligence document into raw_data/ with Chronomics-ready timestamp header."""
        assert self.raw_data_dir is not None
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        short_hash = forensics["sha256"][:10]
        safe_stem = sanitize_slug(image_path.stem) or "image"
        out_path = self.raw_data_dir / f"vision_{short_hash}_{safe_stem}.md"

        header_lines = [
            f"# Visual OSINT Dossier: {image_path.name}",
            f"**Target:** {self.target_name}",
            f"**Source Type:** visual_osint ({forensics['layout_class']})",
            f"**Vision Engine:** {engine_used}",
            f"**Perceptual dHash:** `{forensics['dhash']}`",
            f"**Forensic Authenticity Confidence:** {forensics['authenticity_confidence_pct']}%",
        ]
        # Bridge EXIF DateTimeOriginal directly into CognitiveChronomics (**Date:** ISO-8601)
        if forensics.get("capture_timestamp_iso"):
            header_lines.append(f"**Date:** {forensics['capture_timestamp_iso']}")
        if forensics.get("camera_device"):
            header_lines.append(f"**Device:** {forensics['camera_device']}")
        if forensics.get("gps", {}).get("has_gps"):
            gps = forensics["gps"]
            header_lines.append(f"**GPS:** {gps['latitude']}, {gps['longitude']}")

        doc_text = (
            "\n".join(header_lines)
            + "\n\n---\n\n"
            + f"[TARGET_PRIMARY]: Visual Evidence Record ({image_path.name})\n\n"
            + vision_md.strip()
            + "\n"
        )
        out_path.write_text(doc_text, encoding="utf-8")
        return out_path

    def _upsert_vault_forensics_report(
        self,
        forensics: Dict[str, Any],
        vision_md: str,
        engine_used: str,
        ingested_md_path: Optional[Path],
    ) -> Path:
        """Persist structured visual forensics ledger into vault/visual_forensics_report.json."""
        assert self.vault_dir is not None
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.vault_dir / "visual_forensics_report.json"

        existing_items: List[Dict[str, Any]] = []
        if report_file.exists():
            try:
                loaded = json.loads(report_file.read_text(encoding="utf-8"))
                existing_items = loaded.get("assets", [])
            except Exception:
                existing_items = []

        # Strip raw 24-float histogram from vault summary to keep JSON compact and human-readable
        compact_entry = {
            "file_name": forensics["file_name"],
            "sha256": forensics["sha256"],
            "dhash": forensics["dhash"],
            "format": forensics["format"],
            "width": forensics["width"],
            "height": forensics["height"],
            "layout_class": forensics["layout_class"],
            "capture_timestamp_iso": forensics["capture_timestamp_iso"],
            "camera_device": forensics["camera_device"],
            "software_tag": forensics["software_tag"],
            "detected_editing_tools": forensics["detected_editing_tools"],
            "detected_ai_signatures": forensics["detected_ai_signatures"],
            "gps": forensics["gps"],
            "ela": forensics["ela"],
            "forensic_flags": forensics["forensic_flags"],
            "authenticity_confidence_pct": forensics["authenticity_confidence_pct"],
            "vision_engine": engine_used,
            "vision_summary": vision_md[:1800],
            "ingested_markdown_path": str(ingested_md_path) if ingested_md_path else None,
        }

        updated: List[Dict[str, Any]] = [
            item for item in existing_items if item.get("sha256") != forensics["sha256"]
        ]
        updated.append(compact_entry)

        payload = {
            "target_name": self.target_name,
            "total_verified_visual_assets": len(updated),
            "rejected_namesakes_count": self.rejected_namesakes_count,
            "assets": updated,
        }
        report_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_file
