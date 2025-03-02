"""
Audio converter implementation using FFmpeg.

This module provides a concrete implementation of the AudioConverter port
using FFmpeg for audio conversion.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import ffmpeg

from pyhearingai.core.ports import AudioConverter
from pyhearingai.infrastructure.registry import register_converter

logger = logging.getLogger(__name__)


@register_converter
class FFmpegAudioConverter(AudioConverter):
    """Audio converter implementation using FFmpeg."""

    def convert(self, audio_path: Path, target_format: str = "wav", **kwargs) -> Path:
        """
        Convert an audio file to the specified format using FFmpeg.

        Args:
            audio_path: Path to the audio file to convert
            target_format: Target format to convert to (e.g., 'wav')
            **kwargs: Additional conversion options
                - sample_rate: Sample rate in Hz (default: 16000)
                - channels: Number of audio channels (default: 1)
                - output_dir: Directory to save the converted file (default: temporary directory)
                - codec: Audio codec to use (default: 'pcm_s16le' for wav)

        Returns:
            Path to the converted audio file
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If the file is already in the target format, return it
        if audio_path.suffix.lower() == f".{target_format.lower()}" and not kwargs:
            return audio_path

        # Extract conversion options
        sample_rate = kwargs.get("sample_rate", 16000)
        channels = kwargs.get("channels", 1)
        output_dir = kwargs.get("output_dir")
        codec = kwargs.get("codec")

        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = Path(output_dir) / f"{audio_path.stem}.{target_format}"
        else:
            # Create a temporary directory if no output_dir is specified
            temp_dir = tempfile.mkdtemp()
            output_path = Path(temp_dir) / f"{audio_path.stem}.{target_format}"

        try:
            # Build FFmpeg input
            input_stream = ffmpeg.input(str(audio_path))

            # Build output options
            output_args = {"ar": sample_rate, "ac": channels}

            # Add codec if specified
            if codec:
                output_args["codec:a"] = codec
            elif target_format == "wav" and "codec" not in kwargs:
                # Default codec for WAV
                output_args["codec:a"] = "pcm_s16le"

            # Run the conversion
            logger.debug(f"Converting {audio_path} to {output_path} (format: {target_format})")

            (
                input_stream.output(str(output_path), **output_args).run(
                    quiet=True, overwrite_output=True
                )
            )

            logger.debug(f"Conversion complete: {output_path}")
            return output_path

        except ffmpeg.Error as e:
            error_message = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            raise
