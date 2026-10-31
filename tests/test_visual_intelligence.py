"""
Unit & Integration Tests for the Multimodal Visual Intelligence & Forensic OSINT Package:
- ImageForensics (EXIF, Sub-IFD, GPS DMS-to-decimal, ELA tampering variance, 64-bit dHash, Chronomics bridge)
- Zero-Substitution Disambiguation Gate (AGENTS.md Rule #1) on EXIF Author & Visual Text
- AvatarIdentityMatcher (Perceptual dHash Hamming + 24-bin RGB histogram cosine)
- VisualHarvester (Multi-frame extraction & dHash scene deduplication)
- `who vision` CLI command & Web Dashboard 2.0 Visual OSINT Lab REST endpoints
"""

from __future__ import annotations

import base64
from http.server import ThreadingHTTPServer
import io
import threading
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from src.cli import app
from src.config import settings
from src.engine.dossier_exporter import DossierExporter
from src.harvest.visual_harvester import VisualHarvester
from src.harvest.visual_intelligence import (
    AvatarIdentityMatcher,
    ImageForensics,
    VisionAnalyzer,
)
from src.learn.chronomics import CognitiveChronomics
from src.ui.dashboard import DashboardRequestHandler
from src.utils.entity_validator import DISAMBIGUATION_NOTICE_TR

runner = CliRunner()


def _create_gradient_slide(
    path: Path,
    size: tuple[int, int] = (640, 360),
    exif_bytes: bytes | None = None,
    invert: bool = False,
) -> Path:
    """Create a deterministic structured image with horizontal/vertical luminance gradients."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    w, h = size
    for x in range(0, w, 16):
        val = int((x / max(1, w)) * 240)
        if invert:
            val = 255 - val
        color = (val, (val * 2) % 255, 180 if not invert else 40)
        draw.rectangle([x, 0, min(w - 1, x + 15), h - 1], fill=color)
    # Add asymmetric geometric anchor so dHash has non-trivial horizontal gradients
    box_color = (20, 220, 90) if not invert else (240, 30, 200)
    draw.rectangle([int(w * 0.15), int(h * 0.2), int(w * 0.55), int(h * 0.75)], fill=box_color)

    save_kwargs = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 92
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    img.save(path, **save_kwargs)
    return path


def test_image_forensics_exif_gps_ela_and_chronomics_bridge(tmp_path: Path) -> None:
    """Verify EXIF timestamp, camera, software audit, GPS coordinates, ELA, and Chronomics nocturnal bridge."""
    exif = Image.Exif()
    exif[271] = "Canon"  # Make
    exif[272] = "Canon EOS R5"  # Model
    exif[305] = "Adobe Photoshop 2026"  # Software
    exif[315] = "Mucteba Özdemir".encode("utf-8")  # Artist (Exif 2.31 UTF-8 tag bytes)
    exif[270] = "Keynote slide: Kesinlikle ampirik dogrulama sarttir."  # ImageDescription
    exif[36867] = "2026:10:30 02:15:00"  # DateTimeOriginal (Nocturnal 02:15!)
    # GPS Sub-IFD (0x8825 = 34853): 41 deg 0 min 30 sec N, 28 deg 58 min 42 sec E
    exif[34853] = {
        1: "N",
        2: (41.0, 0.0, 30.0),
        3: "E",
        4: (28.0, 58.0, 42.0),
    }

    slide_path = _create_gradient_slide(
        tmp_path / "keynote_slide.jpg",
        size=(640, 360),
        exif_bytes=exif.tobytes(),
    )

    info = ImageForensics.inspect_image(slide_path, target_name="Mucteba Özdemir")
    assert info["layout_class"] == "presentation_slide_16_9"
    assert info["capture_timestamp_iso"] == "2026-10-30T02:15:00"
    assert info["camera_device"] == "Canon EOS R5"
    assert "photoshop" in info["detected_editing_tools"]
    assert info["gps"]["has_gps"] is True
    assert abs(info["gps"]["latitude"] - 41.008333) < 0.001
    assert abs(info["gps"]["longitude"] - 28.978333) < 0.001
    assert len(info["dhash"]) == 16
    assert 0.0 <= info["ela"]["ela_anomaly_score"] <= 1.0
    assert info["disambiguation_gate"]["rejected"] is False

    # Ingest via VisionAnalyzer and verify CognitiveChronomics parses the EXIF nocturnal timestamp
    raw_dir = tmp_path / "raw_data"
    vault_dir = tmp_path / "vault"
    analyzer = VisionAnalyzer("Mucteba Özdemir", raw_data_dir=raw_dir, vault_dir=vault_dir)
    res = analyzer.analyze_image(slide_path, ingest_to_corpus=True)
    assert res["rejected"] is False
    assert res["ingested_markdown_path"] is not None

    md_content = Path(res["ingested_markdown_path"]).read_text(encoding="utf-8")
    assert "**Date:** 2026-10-30T02:15:00" in md_content

    segments = CognitiveChronomics().extract_timestamp_segments([{"content": md_content}])
    assert len(segments) >= 1
    assert segments[0]["window"] == "NIGHT"


def test_visual_zero_substitution_gate_blocks_prefixed_namesake(tmp_path: Path) -> None:
    """Ensure prefixed/expanded namesakes in EXIF Artist or visual text never touch raw_data/ or vault/."""
    # Case 1: EXIF Artist expansion ("Muhammet Mücteba Özdemir" vs target "Mucteba Özdemir")
    exif_a = Image.Exif()
    exif_a[315] = "Muhammet Mücteba Özdemir".encode("utf-8")
    img_a = _create_gradient_slide(tmp_path / "namesake_exif.jpg", exif_bytes=exif_a.tobytes())

    # Case 2: Clean EXIF author, but visual slide byline contains prefixed namesake
    exif_b = Image.Exif()
    exif_b[270] = "Author: Muhammet Mücteba Özdemir\nDistributed Consensus Lecture".encode("utf-8")
    img_b = _create_gradient_slide(tmp_path / "namesake_byline.jpg", exif_bytes=exif_b.tobytes())

    raw_dir = tmp_path / "target_raw"
    vault_dir = tmp_path / "target_vault"
    analyzer = VisionAnalyzer("Mucteba Özdemir", raw_data_dir=raw_dir, vault_dir=vault_dir)

    res_a = analyzer.analyze_image(img_a, ingest_to_corpus=True)
    res_b = analyzer.analyze_image(img_b, ingest_to_corpus=True)

    assert res_a["rejected"] is True
    assert "UNAUTHORIZED_NAME_EXPANSION" in res_a["rejection_reason"]
    assert res_b["rejected"] is True
    assert "UNAUTHORIZED_NAME_EXPANSION" in res_b["rejection_reason"]
    assert res_a["notice"] == DISAMBIGUATION_NOTICE_TR
    assert analyzer.rejected_namesakes_count == 2

    # Zero files may be written to target's raw_data/ or vault/
    assert not raw_dir.exists() or list(raw_dir.glob("*")) == []
    assert not vault_dir.exists() or list(vault_dir.glob("*")) == []


def test_avatar_identity_matcher_same_vs_distinct_images(tmp_path: Path) -> None:
    """Verify perceptual dHash + RGB histogram matcher distinguishes resized clones from distinct avatars."""
    avatar_orig = _create_gradient_slide(tmp_path / "github_avatar.png", size=(256, 256), invert=False)

    # Create resized & JPEG-compressed version of the same avatar
    avatar_resized = tmp_path / "x_avatar_small.jpg"
    with Image.open(avatar_orig) as im:
        im.resize((96, 96), Image.Resampling.BILINEAR).save(avatar_resized, format="JPEG", quality=80)

    # Create a completely distinct visual pattern
    avatar_other = Image.new("RGB", (256, 256), color=(15, 15, 25))
    draw = ImageDraw.Draw(avatar_other)
    for y in range(0, 256, 32):
        for x in range(0, 256, 32):
            if ((x // 32) + (y // 32)) % 2 == 0:
                draw.rectangle([x, y, x + 31, y + 31], fill=(245, 20, 40))
    avatar_distinct = tmp_path / "stranger_avatar.png"
    avatar_other.save(avatar_distinct)

    same_cmp = AvatarIdentityMatcher.compare_avatars(
        avatar_orig, avatar_resized, target_name="Mucteba Özdemir"
    )
    assert same_cmp["match_tier"] == "HIGH_VISUAL_CORRESPONDENCE"
    assert same_cmp["visual_similarity_score"] >= 0.85
    assert same_cmp["hamming_distance"] <= 6
    assert "Confidence Score:" in same_cmp["probabilistic_statement"]

    diff_cmp = AvatarIdentityMatcher.compare_avatars(
        avatar_orig, avatar_distinct, target_name="Mucteba Özdemir"
    )
    assert diff_cmp["match_tier"] == "DISTINCT_VISUAL_ASSETS"
    assert diff_cmp["visual_similarity_score"] < 0.65


def test_video_keyframe_dhash_deduplication_pipeline(tmp_path: Path) -> None:
    """Verify multi-frame extraction deduplicates consecutive static slides via perceptual dHash."""
    slide_1_path = _create_gradient_slide(tmp_path / "s1.png", size=(320, 180), invert=False)
    slide_2 = Image.new("RGB", (320, 180), color=(240, 240, 250))
    d2 = ImageDraw.Draw(slide_2)
    d2.ellipse([20, 20, 300, 160], fill=(180, 15, 45))

    with Image.open(slide_1_path) as s1:
        f1 = s1.copy()
        f1_cursor = s1.copy()
        f1_cursor.putpixel((5, 5), (255, 255, 255))  # Minor cursor/noise shift on same slide
        s2_cursor = slide_2.copy()
        s2_cursor.putpixel((5, 5), (0, 0, 0))
        # 4 frames: [Slide 1, Slide 1 w/ cursor jitter, Slide 2, Slide 2 w/ cursor jitter]
        deck_gif = tmp_path / "lecture_deck.gif"
        f1.save(
            deck_gif,
            save_all=True,
            append_images=[f1_cursor, slide_2.copy(), s2_cursor],
            duration=200,
            loop=0,
        )

    raw_dir = tmp_path / "raw_video_corpus"
    vault_dir = tmp_path / "vault_video"
    analyzer = VisionAnalyzer("Mucteba Özdemir", raw_data_dir=raw_dir, vault_dir=vault_dir)
    harvester = VisualHarvester(output_dir=tmp_path / "frames_out")

    v_res = harvester.analyze_video_visuals(
        video_path=deck_gif,
        vision_analyzer=analyzer,
        interval_seconds=15,
        max_frames=10,
        hamming_threshold=6,
    )

    assert v_res["total_frames_extracted"] == 4
    assert v_res["unique_frames_retained"] == 2
    assert v_res["verified_scenes_count"] == 2
    assert v_res["consolidated_dossier_path"] is not None
    assert Path(v_res["consolidated_dossier_path"]).exists()


def test_who_vision_cli_and_dashboard_visual_lab_endpoints(tmp_path: Path, monkeypatch) -> None:
    """Verify `who vision` CLI, Dossier visual ledger integration, and Dashboard `/api/vision/*` endpoints."""
    isolated_profiles = tmp_path / "profiles"
    isolated_profiles.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "profiles_dir", isolated_profiles)

    img_clean = _create_gradient_slide(tmp_path / "clean_slide.jpg", size=(640, 360))
    img_clone = _create_gradient_slide(tmp_path / "clean_slide_copy.png", size=(320, 180))

    # 1. Test `who vision` CLI
    cli_res = runner.invoke(
        app,
        [
            "vision",
            "--target",
            "Mucteba Özdemir",
            "--images",
            str(img_clean),
            "--compare-avatars",
            f"{img_clean},{img_clone}",
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert "0/64" in cli_res.output and "Confidence Score:" in cli_res.output

    # Verify standalone Dossier Exporter includes the Visual OSINT & EXIF Forensic Ledger
    target_dirs = settings.get_target_dirs("Mucteba Özdemir")
    dossier_html = DossierExporter("Mucteba Özdemir", target_dirs).render_html()
    assert "Visual OSINT &amp; EXIF Forensic Ledger" in dossier_html
    assert "clean_slide.jpg" in dossier_html

    # 2. Test Web Dashboard 2.0 Visual OSINT Lab HTTP API
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        ui_resp = httpx.get(f"{base_url}/", timeout=10.0)
        assert ui_resp.status_code == 200
        assert "Visual OSINT Lab" in ui_resp.text

        # Valid base64 image upload
        b64_clean = base64.b64encode(img_clone.read_bytes()).decode("ascii")
        an_resp = httpx.post(
            f"{base_url}/api/vision/analyze",
            json={
                "target": "Mucteba Özdemir",
                "filename": "dashboard_upload.png",
                "image_base64": f"data:image/png;base64,{b64_clean}",
                "ingest_to_corpus": True,
            },
            timeout=15.0,
        )
        assert an_resp.status_code == 200
        an_json = an_resp.json()
        assert an_json["rejected"] is False
        assert len(an_json["forensics"]["dhash"]) == 16

        # Contaminated namesake base64 upload -> must be blocked without polluting vault ledger
        bad_exif = Image.Exif()
        bad_exif[315] = "Muhammet Mücteba Özdemir".encode("utf-8")
        bad_buf = io.BytesIO()
        Image.new("RGB", (200, 200), (80, 90, 120)).save(bad_buf, format="JPEG", exif=bad_exif.tobytes())
        b64_bad = base64.b64encode(bad_buf.getvalue()).decode("ascii")

        bad_resp = httpx.post(
            f"{base_url}/api/vision/analyze",
            json={
                "target": "Mucteba Özdemir",
                "filename": "contaminated_namesake.jpg",
                "image_base64": b64_bad,
                "ingest_to_corpus": True,
            },
            timeout=15.0,
        )
        assert bad_resp.status_code == 200
        assert bad_resp.json()["rejected"] is True

        # Check GET /api/target/{slug}/vision ledger contains ONLY the 2 verified assets
        ledger_resp = httpx.get(f"{base_url}/api/target/mucteba_ozdemir/vision", timeout=10.0)
        assert ledger_resp.status_code == 200
        ledger_assets = ledger_resp.json()["assets"]
        filenames = [a["file_name"] for a in ledger_assets]
        assert "clean_slide.jpg" in filenames
        assert "dashboard_upload.png" in filenames
        assert "contaminated_namesake.jpg" not in filenames

        # Check POST /api/vision/compare
        cmp_resp = httpx.post(
            f"{base_url}/api/vision/compare",
            json={
                "target": "Mucteba Özdemir",
                "filename_a": "clean_slide.jpg",
                "image_a_base64": base64.b64encode(img_clean.read_bytes()).decode("ascii"),
                "filename_b": "clean_slide_copy.png",
                "image_b_base64": b64_clean,
            },
            timeout=15.0,
        )
        assert cmp_resp.status_code == 200
        assert cmp_resp.json()["match_tier"] == "HIGH_VISUAL_CORRESPONDENCE"
    finally:
        server.shutdown()
        server.server_close()
