"""
JSON formatter implementation.

This module provides a concrete implementation of the OutputFormatter port
for JSON output format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from pyhearingai.application.outputs import to_json
from pyhearingai.core.models import TranscriptionResult
from pyhearingai.core.ports import OutputFormatter
from pyhearingai.infrastructure.registry import register_output_formatter

logger = logging.getLogger(__name__)


@register_output_formatter("json")
class JSONFormatter(OutputFormatter):
    """Output formatter implementation for JSON format."""

    @property
    def format_name(self) -> str:
        """Get the name of the format."""
        return "json"

    def format(self, result: TranscriptionResult, **kwargs) -> str:
        """
        Format a transcription result as JSON.

        Args:
            result: The transcription result to format
            **kwargs: Additional formatting options

        Returns:
            JSON representation of the transcription result
        """
        return to_json(result)

    def save(self, result: TranscriptionResult, path: Path, **kwargs) -> Path:
        """
        Save a transcription result to a JSON file.

        Args:
            result: The transcription result to save
            path: Path to save the file to
            **kwargs: Additional saving options

        Returns:
            Path to the saved file
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Format the transcription as JSON
        json_content = self.format(result, **kwargs)

        # Write to file
        path.write_text(json_content, encoding="utf-8")

        logger.debug(f"Saved transcription to JSON file: {path}")
        return path
