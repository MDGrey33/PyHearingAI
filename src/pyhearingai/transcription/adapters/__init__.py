"""
Adapters for transcription services.

This package contains adapters for various transcription services,
providing a consistent interface for the TranscriptionService.
"""

from pyhearingai.transcription.adapters.whisper import WhisperAdapter

__all__ = ["WhisperAdapter"]
