"""
Transcribers package initialization.

This module imports all transcriber implementations to ensure they are registered
with the registry when the package is imported.
"""

# Import all transcribers so they are registered
from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber

__all__ = ["WhisperOpenAITranscriber"]






