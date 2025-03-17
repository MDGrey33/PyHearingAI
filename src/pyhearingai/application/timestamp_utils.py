"""
Timestamp utility functions for PyHearingAI.

This module provides utility functions for working with audio timestamps,
converting between different formats, and reconciling overlapping segments.
"""

from datetime import timedelta
from typing import List, Tuple, Union

from pyhearingai.core.idempotent import SpeakerSegment
from pyhearingai.core.models import Segment


def format_timestamp(seconds: float, include_ms: bool = True) -> str:
    """
    Format a timestamp in seconds to a human-readable string.

    Args:
        seconds: Time in seconds
        include_ms: Whether to include milliseconds

    Returns:
        Formatted timestamp string (HH:MM:SS.mmm or HH:MM:SS)
    """
    td = timedelta(seconds=seconds)

    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if include_ms:
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_timestamp(timestamp: str) -> float:
    """
    Parse a timestamp string to seconds.

    Args:
        timestamp: Timestamp string (HH:MM:SS.mmm or HH:MM:SS)

    Returns:
        Time in seconds
    """
    # Split by colon
    parts = timestamp.strip().split(":")

    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    # Parse hours, minutes, seconds
    try:
        hours = int(parts[0])
        minutes = int(parts[1])

        # Handle seconds with optional milliseconds
        if "." in parts[2]:
            seconds_parts = parts[2].split(".")
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1].ljust(3, "0")[:3])
        else:
            seconds = int(parts[2])
            milliseconds = 0

        # Convert to seconds
        total_seconds = (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)
        return total_seconds
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}") from e


def adjust_timestamp(timestamp: float, offset: float) -> float:
    """
    Adjust a timestamp by adding an offset.

    Args:
        timestamp: Original timestamp in seconds
        offset: Offset to add in seconds

    Returns:
        Adjusted timestamp in seconds
    """
    return max(0.0, timestamp + offset)


def relative_to_absolute_time(relative_time: float, chunk_start_time: float) -> float:
    """
    Convert a time relative to a chunk to absolute time.

    Args:
        relative_time: Time in seconds relative to chunk start
        chunk_start_time: Start time of the chunk in absolute time

    Returns:
        Absolute time in seconds
    """
    return chunk_start_time + relative_time


def absolute_to_relative_time(absolute_time: float, chunk_start_time: float) -> float:
    """
    Convert an absolute time to time relative to a chunk.

    Args:
        absolute_time: Absolute time in seconds
        chunk_start_time: Start time of the chunk in absolute time

    Returns:
        Relative time in seconds
    """
    return max(0.0, absolute_time - chunk_start_time)


def adjust_segment_timestamps(
    segments: List[Union[Segment, SpeakerSegment]], offset: float
) -> List[Union[Segment, SpeakerSegment]]:
    """
    Adjust the timestamps of a list of segments by an offset.

    Args:
        segments: List of segments to adjust
        offset: Offset to add to timestamps

    Returns:
        List of segments with adjusted timestamps
    """
    adjusted_segments = []

    for segment in segments:
        # Create a copy of the segment with adjusted timestamps
        # This depends on the segment type
        if isinstance(segment, SpeakerSegment):
            adjusted_segment = SpeakerSegment(
                id=segment.id,
                job_id=segment.job_id,
                chunk_id=segment.chunk_id,
                speaker_id=segment.speaker_id,
                start_time=adjust_timestamp(segment.start_time, offset),
                end_time=adjust_timestamp(segment.end_time, offset),
                confidence=segment.confidence,
                metadata=segment.metadata.copy() if segment.metadata else {},
            )
        elif isinstance(segment, Segment):
            adjusted_segment = Segment(
                text=segment.text,
                start=adjust_timestamp(segment.start, offset),
                end=adjust_timestamp(segment.end, offset),
                speaker_id=segment.speaker_id,
            )
        else:
            raise TypeError(f"Unsupported segment type: {type(segment)}")

        adjusted_segments.append(adjusted_segment)

    return adjusted_segments


def find_overlapping_segments(
    segments: List[Union[Segment, SpeakerSegment]], window_start: float, window_end: float
) -> List[Union[Segment, SpeakerSegment]]:
    """
    Find segments that overlap with a specified time window.

    Args:
        segments: List of segments to search
        window_start: Start of the time window
        window_end: End of the time window

    Returns:
        List of segments that overlap with the window
    """
    overlapping = []

    for segment in segments:
        # Check if segment overlaps with window
        if isinstance(segment, SpeakerSegment):
            segment_start = segment.start_time
            segment_end = segment.end_time
        elif isinstance(segment, Segment):
            segment_start = segment.start
            segment_end = segment.end
        else:
            raise TypeError(f"Unsupported segment type: {type(segment)}")

        # Check for overlap
        if segment_end > window_start and segment_start < window_end:
            overlapping.append(segment)

    return overlapping


def merge_overlapping_segments(
    segments: List[Union[Segment, SpeakerSegment]], tolerance: float = 0.1
) -> List[Union[Segment, SpeakerSegment]]:
    """
    Merge segments that overlap or are very close together.

    Args:
        segments: List of segments to merge
        tolerance: Maximum gap between segments to consider for merging

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort segments by start time
    segments_sorted = sorted(
        segments, key=lambda s: s.start_time if isinstance(s, SpeakerSegment) else s.start
    )

    merged = [segments_sorted[0]]

    for segment in segments_sorted[1:]:
        last = merged[-1]

        # Extract start and end times based on segment type
        if isinstance(segment, SpeakerSegment) and isinstance(last, SpeakerSegment):
            # Only merge segments with the same speaker
            if segment.speaker_id != last.speaker_id:
                merged.append(segment)
                continue

            segment_start = segment.start_time
            segment_end = segment.end_time
            last_end = last.end_time

            # Check if segments should be merged
            if segment_start <= last_end + tolerance:
                # Merge segments
                last.end_time = max(last_end, segment_end)
            else:
                merged.append(segment)

        elif isinstance(segment, Segment) and isinstance(last, Segment):
            # Only merge segments with the same speaker
            if segment.speaker_id != last.speaker_id:
                merged.append(segment)
                continue

            segment_start = segment.start
            segment_end = segment.end
            last_end = last.end

            # Check if segments should be merged
            if segment_start <= last_end + tolerance:
                # Merge segments by extending the last segment and appending text
                last.end = max(last_end, segment_end)
                last.text += " " + segment.text
            else:
                merged.append(segment)

        else:
            # Different types, can't merge
            merged.append(segment)

    return merged
