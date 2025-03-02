"""
PyHearingAI - Audio transcription with speaker diarization.

This package provides tools for transcribing audio files with speaker diarization.
"""

__version__ = "0.1.0"

# Import the core functionality
from pyhearingai.application.transcribe import transcribe

# Import public API
from pyhearingai.core.models import DiarizationSegment, Segment, TranscriptionResult

# Import adapters to ensure they are registered
from pyhearingai.infrastructure import adapters

# Import registry functions for public API
from pyhearingai.infrastructure.registry import (
    get_diarizer,
    get_output_formatter,
    get_transcriber,
    list_diarizers,
    list_output_formatters,
    list_transcribers,
)

__all__ = [
    "transcribe",
    "TranscriptionResult",
    "Segment",
    "DiarizationSegment",
    "list_output_formatters",
    "get_output_formatter",
    "list_transcribers",
    "get_transcriber",
    "list_diarizers",
    "get_diarizer",
]
