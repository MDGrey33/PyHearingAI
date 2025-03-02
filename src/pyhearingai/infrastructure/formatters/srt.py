"""
SRT formatter implementation.

This module provides a concrete implementation of the OutputFormatter port
for SRT subtitle format.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from pyhearingai.application.outputs import to_srt
from pyhearingai.core.models import TranscriptionResult
from pyhearingai.core.ports import OutputFormatter
from pyhearingai.infrastructure.registry import register_output_formatter

logger = logging.getLogger(__name__)


@register_output_formatter("srt")
class SRTFormatter(OutputFormatter):
    """Output formatter implementation for SRT subtitle format."""

    @property
    def format_name(self) -> str:
        """Get the name of the format."""
        return "srt"

    def format(self, result: TranscriptionResult, **kwargs) -> str:
        """
        Format a transcription result as SRT subtitles.

        Args:
            result: The transcription result to format
            **kwargs: Additional formatting options

        Returns:
            SRT subtitle representation of the transcription result
        """
        return to_srt(result)

    def save(self, result: TranscriptionResult, path: Path, **kwargs) -> Path:
        """
        Save a transcription result to an SRT subtitle file.

        Args:
            result: The transcription result to save
            path: Path to save the file to
            **kwargs: Additional saving options

        Returns:
            Path to the saved file
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Format the transcription as SRT
        srt_content = self.format(result, **kwargs)

        # Write to file
        path.write_text(srt_content, encoding="utf-8")

        logger.debug(f"Saved transcription to SRT file: {path}")
        return path
