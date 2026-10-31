"""
Harvest module for who??.
Handles multimodal data acquisition across web, documents, OCR, media, and dialogue logs.
"""

from src.harvest.web_harvester import WebHarvester
from src.harvest.academic_harvester import AcademicHarvester
from src.harvest.document_ingestor import DocumentIngestor
from src.harvest.ocr_processor import OCRProcessor
from src.harvest.media_harvester import MediaHarvester
from src.harvest.diarized_transcriber import DiarizedTranscriber
from src.harvest.visual_harvester import VisualHarvester
from src.harvest.dialogue_harvester import DialogueHarvester
from src.harvest.live_monitor import LiveMonitor
from src.harvest.social_osint import SocialOSINTHunter

__all__ = [
    "WebHarvester",
    "AcademicHarvester",
    "DocumentIngestor",
    "OCRProcessor",
    "MediaHarvester",
    "DiarizedTranscriber",
    "VisualHarvester",
    "DialogueHarvester",
    "LiveMonitor",
    "SocialOSINTHunter",
]
