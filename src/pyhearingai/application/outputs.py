"""
Output formatters for transcription results.

This module provides functions to save transcription results in various formats.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from pyhearingai.core.models import Segment, TranscriptionResult
from pyhearingai.infrastructure.registry import get_output_formatter


def save_transcript(
    result: TranscriptionResult, path: Union[str, Path], format: Optional[str] = None
) -> Path:
    """
    Save a transcription result to a file in the specified format.

    Args:
        result: The transcription result to save
        path: Path to save the file to
        format: Format to save as (if None, inferred from path extension)

    Returns:
        Path to the saved file
    """
    # Convert path to Path object
    if isinstance(path, str):
        path = Path(path)

    # Infer format from file extension if not specified
    if format is None:
        format = path.suffix.lstrip(".")
        if not format:
            format = "txt"  # Default to text if no extension

    # Get the appropriate formatter
    formatter = get_output_formatter(format)

    # Save the transcript
    return formatter.save(result, path)


def to_text(result: TranscriptionResult) -> str:
    """
    Convert a transcription result to plain text format.

    Args:
        result: The transcription result to format

    Returns:
        Formatted plain text representation
    """
    lines = []

    for segment in result.segments:
        if segment.speaker_id:
            lines.append(f"**{segment.speaker_id}:** {segment.text}")
        else:
            lines.append(segment.text)

    return "\n\n".join(lines)


def to_json(result: TranscriptionResult) -> str:
    """
    Convert a transcription result to JSON format.

    Args:
        result: The transcription result to format

    Returns:
        JSON string representation
    """
    data = {
        "metadata": result.metadata,
        "segments": [
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "speaker_id": segment.speaker_id,
            }
            for segment in result.segments
        ],
    }

    return json.dumps(data, indent=2)


def to_srt(result: TranscriptionResult) -> str:
    """
    Convert a transcription result to SRT subtitle format.

    Args:
        result: The transcription result to format

    Returns:
        SRT subtitle string
    """
    lines = []

    for i, segment in enumerate(result.segments, 1):
        start_time = _format_srt_time(segment.start)
        end_time = _format_srt_time(segment.end)

        lines.append(str(i))
        lines.append(f"{start_time} --> {end_time}")

        if segment.speaker_id:
            lines.append(f"{segment.speaker_id}: {segment.text}")
        else:
            lines.append(segment.text)

        lines.append("")  # Empty line between entries

    return "\n".join(lines)


def to_vtt(result: TranscriptionResult) -> str:
    """
    Convert a transcription result to WebVTT format.

    Args:
        result: The transcription result to format

    Returns:
        WebVTT subtitle string
    """
    lines = ["WEBVTT", ""]

    for i, segment in enumerate(result.segments, 1):
        start_time = _format_vtt_time(segment.start)
        end_time = _format_vtt_time(segment.end)

        lines.append(f"{start_time} --> {end_time}")

        if segment.speaker_id:
            lines.append(f"{segment.speaker_id}: {segment.text}")
        else:
            lines.append(segment.text)

        lines.append("")  # Empty line between entries

    return "\n".join(lines)


def to_markdown(result: TranscriptionResult) -> str:
    """
    Convert a transcription result to Markdown format.

    Args:
        result: The transcription result to format

    Returns:
        Markdown formatted string
    """
    lines = ["# Transcript", ""]

    current_speaker = None
    for segment in result.segments:
        # Start a new speaker section if the speaker changes
        if segment.speaker_id != current_speaker:
            current_speaker = segment.speaker_id
            if current_speaker:
                lines.append(f"## {current_speaker}")
                lines.append("")

        # Add timestamp and text
        time_str = f"{segment.start_timecode} - {segment.end_timecode}"
        lines.append(f"*{time_str}*")
        lines.append("")
        lines.append(segment.text)
        lines.append("")

    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    """Format time in SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


def _format_vtt_time(seconds: float) -> str:
    """Format time in WebVTT format: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
