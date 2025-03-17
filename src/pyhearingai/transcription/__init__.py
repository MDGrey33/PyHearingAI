"""
Transcription service module for PyHearingAI.

This module provides services for transcribing audio chunks and speaker segments,
with support for idempotent processing and resumability.
"""

from pyhearingai.transcription.service import TranscriptionService

__all__ = ["TranscriptionService"]
