"""
Test helper functions and factories for PyHearingAI tests.

This module provides reusable components for creating test objects and mocks,
which helps keep individual test files smaller and more focused.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

from pyhearingai.core.models import Segment


# Domain model factories
def create_segment(
    start: float = 0.0,
    end: float = 1.0,
    text: str = "Test segment",
    speaker_id: str = "SPEAKER_0",
) -> "Segment":
    """
    Create a Segment instance for testing.

    Args:
        start: Segment start time in seconds
        end: Segment end time in seconds
        text: Segment text content
        speaker_id: ID of the speaker for this segment

    Returns:
        A Segment instance with the provided properties
    """
    segment = Segment(start=start, end=end, text=text)
    if speaker_id:
        segment.speaker_id = speaker_id
    return segment


def create_segment_list(count: int = 3, speaker_count: int = 2) -> List["Segment"]:
    """
    Create a list of test segments.

    Args:
        count: Number of segments to create
        speaker_count: Number of unique speakers to distribute

    Returns:
        List of Segment objects with alternating speakers
    """
    segments = []

    for i in range(count):
        speaker_id = f"SPEAKER_{i % speaker_count:02d}"
        start = i * 2.0
        end = start + 1.5
        text = f"This is test segment {i+1} from {speaker_id}."

        segments.append(create_segment(start=start, end=end, text=text, speaker_id=speaker_id))

    return segments


# Mock factories
def mock_openai_client(segment_data: Optional[List[Dict]] = None) -> Tuple[MagicMock, MagicMock]:
    """
    Create a pre-configured mock OpenAI client.

    Args:
        segment_data: Optional list of segment data to include in the response

    Returns:
        Tuple of (mock_client, mock_transcription)
    """
    if segment_data is None:
        segment_data = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test."},
            {"id": 1, "start": 2.5, "end": 4.5, "text": "Testing the transcriber."},
        ]

    mock_client = MagicMock()
    mock_audio = MagicMock()
    mock_client.audio = mock_audio

    mock_transcription = MagicMock()
    mock_audio.transcriptions = mock_transcription

    mock_response = MagicMock()
    mock_response.segments = segment_data
    mock_transcription.create.return_value = mock_response

    return mock_client, mock_transcription


def mock_pyannote_pipeline(speaker_segments: Optional[List[Tuple]] = None) -> MagicMock:
    """
    Create a pre-configured mock Pyannote Pipeline.

    Args:
        speaker_segments: Optional list of speaker segments as (region, track, label) tuples

    Returns:
        Configured mock Pipeline
    """
    if speaker_segments is None:
        # Default: two speakers with three segments
        speaker_segments = [
            (MagicMock(start=0.5, end=2.0), 0, "speaker"),  # SPEAKER_00
            (MagicMock(start=2.5, end=3.5), 1, "speaker"),  # SPEAKER_01
            (MagicMock(start=4.0, end=6.0), 0, "speaker"),  # SPEAKER_00
        ]

    mock_pipeline = MagicMock()
    mock_instance = MagicMock()
    mock_pipeline.from_pretrained.return_value = mock_instance

    # Configure the return value
    mock_instance.return_value = MagicMock()
    mock_instance.return_value.itertracks.return_value = speaker_segments

    return mock_pipeline


# File helpers
def create_temp_audio_file(suffix: str = ".wav") -> Tuple[Path, tempfile._TemporaryFileWrapper]:
    """
    Create a temporary audio file for testing.

    Args:
        suffix: File extension to use

    Returns:
        Tuple of (file_path, file_handle) - keep the handle to prevent deletion
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    return Path(temp_file.name), temp_file


def create_temp_dir() -> Path:
    """
    Create a temporary directory for test outputs.

    Returns:
        Path to the temporary directory
    """
    return Path(tempfile.mkdtemp())


# Assertion helpers
def assert_segments_equal(
    actual: "Segment", expected: "Segment", check_speaker: bool = True
) -> None:
    """
    Assert that two segments have equal properties.

    Args:
        actual: The actual segment to check
        expected: The expected segment to compare against
        check_speaker: Whether to check speaker_id equality
    """
    assert (
        actual.start == expected.start
    ), f"Start time mismatch: {actual.start} != {expected.start}"
    assert actual.end == expected.end, f"End time mismatch: {actual.end} != {expected.end}"
    assert actual.text == expected.text, f"Text mismatch: {actual.text} != {expected.text}"

    if check_speaker:
        actual_speaker = getattr(actual, "speaker_id", None)
        expected_speaker = getattr(expected, "speaker_id", None)
        assert (
            actual_speaker == expected_speaker
        ), f"Speaker ID mismatch: {actual_speaker} != {expected_speaker}"


def assert_segment_lists_equal(
    actual: List["Segment"], expected: List["Segment"], check_speaker: bool = True
) -> None:
    """
    Assert that two lists of segments have equal properties.

    Args:
        actual: The actual segment list to check
        expected: The expected segment list to compare against
        check_speaker: Whether to check speaker_id equality
    """
    assert len(actual) == len(expected), f"Segment count mismatch: {len(actual)} != {len(expected)}"

    for i, (act, exp) in enumerate(zip(actual, expected)):
        try:
            assert_segments_equal(act, exp, check_speaker)
        except AssertionError as e:
            raise AssertionError(f"Segment {i} mismatch: {e}")


# Patch helpers
def patch_openai():
    """
    Create a patch for the OpenAI client.

    Returns:
        A context manager for patching openai.OpenAI
    """
    return patch("openai.OpenAI")


def patch_pyannote_pipeline():
    """
    Create a patch for the Pyannote Pipeline.

    Returns:
        A context manager for patching pyannote.audio.Pipeline
    """
    return patch("pyhearingai.infrastructure.diarizers.pyannote.Pipeline")
