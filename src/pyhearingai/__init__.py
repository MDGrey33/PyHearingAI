"""
PyHearingAI - Audio transcription with speaker diarization.

This package provides tools for transcribing audio files with speaker diarization.
"""

__version__ = "0.1.0"

# Import initialization module to ensure all components are registered
from pyhearingai import initialization
from pyhearingai.application.resource_manager import cleanup_resources
from pyhearingai.application.session import pipeline_session

# Import the core functionality
from pyhearingai.application.transcribe import transcribe, transcribe_chunked
from pyhearingai.config import set_memory_limit

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
    "transcribe_chunked",
    "pipeline_session",
    "cleanup_resources",
    "set_memory_limit",
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
