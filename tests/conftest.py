import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import our test helpers
from tests.helpers import (
    assert_segment_lists_equal,
    assert_segments_equal,
    create_segment,
    create_segment_list,
    create_temp_audio_file,
    create_temp_dir,
    mock_openai_client,
    mock_pyannote_pipeline,
)


# Path fixtures
@pytest.fixture(scope="session")
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def example_audio_path(fixtures_dir):
    """Return the path to the example audio file."""
    return fixtures_dir / "example_audio.m4a"


@pytest.fixture(scope="session")
def reference_transcript_path(fixtures_dir):
    """Return the path to the reference transcript file."""
    return fixtures_dir / "labeled_transcript.txt"


# Environment fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_audio_file():
    """Create a temporary WAV file for testing."""
    path, file_handle = create_temp_audio_file(suffix=".wav")

    # Write some dummy content to make it a valid file
    file_handle.write(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    file_handle.flush()

    yield path

    # Clean up
    file_handle.close()
    if path.exists():
        path.unlink()


# Domain model fixtures
@pytest.fixture
def transcript_segments():
    """Return a list of transcript segments for testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is a test."),
        create_segment(start=2.5, end=4.5, text="Testing the transcriber."),
        create_segment(start=5.0, end=7.0, text="Final test segment."),
    ]


@pytest.fixture
def diarization_segments():
    """Return a list of diarization segments for testing."""
    return [
        create_segment(start=0.0, end=2.0, text="", speaker_id="SPEAKER_00"),
        create_segment(start=2.5, end=4.5, text="", speaker_id="SPEAKER_01"),
        create_segment(start=5.0, end=7.0, text="", speaker_id="SPEAKER_00"),
    ]


@pytest.fixture
def labeled_segments():
    """Return a list of segments with speaker labels for testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is a test.", speaker_id="SPEAKER_00"),
        create_segment(
            start=2.5, end=4.5, text="Testing the transcriber.", speaker_id="SPEAKER_01"
        ),
        create_segment(start=5.0, end=7.0, text="Final test segment.", speaker_id="SPEAKER_00"),
    ]


# Formatter test fixtures
@pytest.fixture
def formatter_segments_with_speakers():
    """Return a list of segments with speaker IDs for formatter testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is segment one.", speaker_id="Speaker 1"),
        create_segment(start=2.5, end=4.5, text="This is segment two.", speaker_id="Speaker 2"),
        create_segment(start=5.0, end=7.0, text="This is segment three.", speaker_id="Speaker 1"),
    ]


@pytest.fixture
def formatter_segments_without_speakers():
    """Return a list of segments without speaker IDs for formatter testing."""
    return [
        create_segment(start=0.0, end=2.0, text="This is segment one.", speaker_id=None),
        create_segment(start=2.5, end=4.5, text="This is segment two.", speaker_id=None),
    ]


@pytest.fixture
def formatter_single_segment():
    """Return a single segment for simple formatter tests."""
    return [
        create_segment(start=0.0, end=2.0, text="This is a test segment.", speaker_id="Speaker 1"),
    ]


@pytest.fixture
def test_transcription_result(request):
    """Create a TranscriptionResult with specified segments.

    Usage:
        test_transcription_result(formatter_segments_with_speakers)
        test_transcription_result(formatter_segments_without_speakers)
        test_transcription_result(formatter_single_segment)
    """
    from pyhearingai.core.models import TranscriptionResult

    # Get the segments from the provided fixture
    segments = request.getfixturevalue(request.param)

    # Create a transcription result with optional metadata
    result = TranscriptionResult(
        segments=segments, audio_path=Path("test_audio.wav"), metadata={"test": True}
    )
    return result


# Mock fixtures
@pytest.fixture
def mock_openai():
    """Return a mock OpenAI client and transcription."""
    return mock_openai_client()


@pytest.fixture
def mock_pyannote():
    """Return a mock Pyannote Pipeline."""
    return mock_pyannote_pipeline()


# Utility functions
def clean_text(text):
    """Clean text by removing speaker labels, punctuation, extra spaces, and line breaks."""
    import re

    # Remove speaker labels like "**Speaker 1:**"
    text = re.sub(r"\*\*Speaker \d+:\*\*", "", text)
    # Remove any non-alphanumeric characters (except spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_similarity(text1, text2):
    """Calculate similarity ratio between two texts."""
    import difflib

    # Clean both texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    # Split into words for word-level comparison
    words1 = set(text1.split())
    words2 = set(text2.split())

    # Calculate word overlap
    common_words = words1.intersection(words2)
    total_words = words1.union(words2)

    word_similarity = len(common_words) / max(len(total_words), 1)

    # Use SequenceMatcher for character-level similarity
    char_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

    # Combine both metrics (weighted average)
    combined_similarity = (word_similarity * 0.7) + (char_similarity * 0.3)

    return combined_similarity, word_similarity, char_similarity


# Make these utility functions available to tests
@pytest.fixture
def text_utils():
    """Return utility functions for text processing and comparison."""
    return {"clean_text": clean_text, "calculate_similarity": calculate_similarity}


# Component fixtures
@pytest.fixture
def audio_converter():
    """Return an initialized FFmpegAudioConverter."""
    from pyhearingai.infrastructure.audio_converter import FFmpegAudioConverter

    return FFmpegAudioConverter()


@pytest.fixture
def transcriber():
    """Return an initialized WhisperOpenAITranscriber."""
    from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber

    return WhisperOpenAITranscriber()


@pytest.fixture
def diarizer():
    """Return an initialized PyannoteDiarizer."""
    from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer

    return PyannoteDiarizer()


@pytest.fixture
def speaker_assigner():
    """Return an initialized DefaultSpeakerAssigner."""
    from pyhearingai.infrastructure.speaker_assignment import DefaultSpeakerAssigner

    return DefaultSpeakerAssigner()


# Assertion helpers
@pytest.fixture
def assert_segments():
    """Return the assert_segments_equal function."""
    return assert_segments_equal


@pytest.fixture
def assert_segment_lists():
    """Return the assert_segment_lists_equal function."""
    return assert_segment_lists_equal
