"""
WebVTT formatter implementation.

This module provides a concrete implementation of the OutputFormatter port
for WebVTT subtitle format.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from pyhearingai.application.outputs import to_vtt
from pyhearingai.core.models import TranscriptionResult
from pyhearingai.core.ports import OutputFormatter
from pyhearingai.infrastructure.registry import register_output_formatter


logger = logging.getLogger(__name__)


@register_output_formatter("vtt")
class VTTFormatter(OutputFormatter):
    """Output formatter implementation for WebVTT subtitle format."""
    
    @property
    def format_name(self) -> str:
        """Get the name of the format."""
        return "vtt"
    
    def format(self, result: TranscriptionResult, **kwargs) -> str:
        """
        Format a transcription result as WebVTT subtitles.
        
        Args:
            result: The transcription result to format
            **kwargs: Additional formatting options
        
        Returns:
            WebVTT subtitle representation of the transcription result
        """
        return to_vtt(result)
    
    def save(self, result: TranscriptionResult, path: Path, **kwargs) -> Path:
        """
        Save a transcription result to a WebVTT subtitle file.
        
        Args:
            result: The transcription result to save
            path: Path to save the file to
            **kwargs: Additional saving options
        
        Returns:
            Path to the saved file
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format the transcription as WebVTT
        vtt_content = self.format(result, **kwargs)
        
        # Write to file
        path.write_text(vtt_content, encoding="utf-8")
        
        logger.debug(f"Saved transcription to WebVTT file: {path}")
        return path 