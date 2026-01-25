"""
Services package for CasperPrep backend.
"""

from .audio import analyze_audio, extract_audio, transcribe, detect_filler_words
from .feedback import generate_text_feedback, generate_video_feedback, generate_scenario_feedback
from .r2 import r2_service
from .analyzer import analyze_attempt

try:
    from .video import analyze_eye_contact
except ImportError:
    analyze_eye_contact = None

__all__ = [
    "analyze_eye_contact",
    "analyze_audio",
    "extract_audio",
    "transcribe",
    "detect_filler_words",
    "generate_text_feedback",
    "generate_video_feedback",
    "generate_scenario_feedback",
    "r2_service",
    "analyze_attempt",
]
