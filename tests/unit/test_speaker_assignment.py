"""
Unit tests for the DefaultSpeakerAssigner class.

Tests the default speaker assignment including:
- Basic functionality
- Edge cases
- Different option configurations
"""

import pytest

from pyhearingai.core.models import DiarizationSegment, Segment
from pyhearingai.infrastructure.speaker_assignment import DefaultSpeakerAssigner


@pytest.fixture
def transcript_segments():
    """
    Sample transcript segments for testing.

    Returns:
        List[Segment]: A list of transcript segments
    """
    return [
        Segment(text="Hello, how are you today?", start=0.0, end=2.0),
        Segment(text="I'm doing well, thank you. How about you?", start=2.5, end=5.0),
        Segment(text="Pretty good, thanks for asking.", start=5.5, end=7.0),
    ]


@pytest.fixture
def diarization_segments():
    """
    Sample diarization segments for testing.

    Returns:
        List[DiarizationSegment]: A list of diarization segments with speaker IDs
    """
    return [
        DiarizationSegment(start=0.0, end=2.0, speaker_id="SPEAKER_01"),
        DiarizationSegment(start=2.5, end=5.0, speaker_id="SPEAKER_02"),
        DiarizationSegment(start=5.5, end=7.0, speaker_id="SPEAKER_01"),
    ]


def test_basic_speaker_assignment(transcript_segments, diarization_segments):
    """
    Test basic speaker assignment functionality.

    Given: Transcript segments and diarization segments that align perfectly
    When: Calling assign_speakers
    Then: The segments have the correct speaker IDs assigned
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()

    # Act
    result = assigner.assign_speakers(transcript_segments, diarization_segments)

    # Assert
    assert len(result) == 3
    assert result[0].speaker_id == "Speaker 0"  # Normalized from SPEAKER_01
    assert result[1].speaker_id == "Speaker 1"  # Normalized from SPEAKER_02
    assert result[2].speaker_id == "Speaker 0"  # Normalized from SPEAKER_01

    # Verify text is preserved
    assert result[0].text == "Hello, how are you today?"
    assert result[1].text == "I'm doing well, thank you. How about you?"
    assert result[2].text == "Pretty good, thanks for asking."

    # Verify timing is preserved
    assert result[0].start == 0.0
    assert result[0].end == 2.0


def test_empty_transcript_segments():
    """
    Test behavior with empty transcript segments.

    Given: Empty transcript segments
    When: Calling assign_speakers
    Then: An empty list is returned
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()
    diarization_segments = [
        DiarizationSegment(start=0.0, end=2.0, speaker_id="SPEAKER_01"),
    ]

    # Act
    result = assigner.assign_speakers([], diarization_segments)

    # Assert
    assert len(result) == 0
    assert isinstance(result, list)


def test_empty_diarization_segments(transcript_segments):
    """
    Test behavior with empty diarization segments.

    Given: Empty diarization segments
    When: Calling assign_speakers
    Then: Original transcript segments are returned without speaker IDs
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()

    # Act
    result = assigner.assign_speakers(transcript_segments, [])

    # Assert
    assert len(result) == 3
    assert result[0].speaker_id is None
    assert result[1].speaker_id is None
    assert result[2].speaker_id is None

    # Verify text is preserved
    assert result[0].text == "Hello, how are you today?"


def test_partial_overlap():
    """
    Test behavior with partial overlaps between transcript and diarization.

    Given: Transcript and diarization segments with partial overlaps
    When: Calling assign_speakers
    Then: Speaker IDs are assigned based on maximum overlap
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()

    transcript_segments = [
        Segment(text="This spans two speakers.", start=1.0, end=3.0),
    ]

    diarization_segments = [
        DiarizationSegment(start=0.0, end=2.0, speaker_id="SPEAKER_01"),
        DiarizationSegment(start=2.0, end=4.0, speaker_id="SPEAKER_02"),
    ]

    # Act
    result = assigner.assign_speakers(transcript_segments, diarization_segments)

    # Assert
    assert len(result) == 1
    # Since both speakers overlap equally (1.0 seconds each), it should pick the first one
    assert result[0].speaker_id == "Speaker 0"


def test_min_overlap_threshold():
    """
    Test the min_overlap option.

    Given: Transcript and diarization segments with small overlaps
    When: Calling assign_speakers with a high min_overlap
    Then: No speaker is assigned when overlap is below threshold
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()

    transcript_segments = [
        Segment(text="This has minimal overlap.", start=0.0, end=2.0),
    ]

    diarization_segments = [
        DiarizationSegment(start=1.5, end=2.5, speaker_id="SPEAKER_01"),
    ]

    # The overlap is 0.5/2.0 = 25%

    # Act - Set min_overlap to 0.2 (20%)
    result_with_low_threshold = assigner.assign_speakers(
        transcript_segments, diarization_segments, min_overlap=0.2
    )

    # Set min_overlap to 0.3 (30%)
    result_with_high_threshold = assigner.assign_speakers(
        transcript_segments, diarization_segments, min_overlap=0.3
    )

    # Assert
    # With low threshold (20%), speaker should be assigned (overlap is 25%)
    assert result_with_low_threshold[0].speaker_id == "Speaker 0"

    # With high threshold (30%), no speaker should be assigned (overlap is 25%)
    assert result_with_high_threshold[0].speaker_id is None


def test_custom_speaker_prefix():
    """
    Test custom speaker prefix option.

    Given: Standard transcript and diarization segments
    When: Calling assign_speakers with a custom speaker_prefix
    Then: Speaker IDs use the custom prefix
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()

    transcript_segments = [
        Segment(text="Hello", start=0.0, end=1.0),
    ]

    diarization_segments = [
        DiarizationSegment(start=0.0, end=1.0, speaker_id="SPEAKER_01"),
    ]

    # Act
    result = assigner.assign_speakers(
        transcript_segments, diarization_segments, speaker_prefix="Person_"
    )

    # Assert
    assert result[0].speaker_id == "Person_0"


def test_disable_speaker_normalization():
    """
    Test disabling speaker normalization.

    Given: Standard transcript and diarization segments
    When: Calling assign_speakers with normalize_speakers=False
    Then: Original speaker IDs are preserved
    """
    # Arrange
    assigner = DefaultSpeakerAssigner()

    transcript_segments = [
        Segment(text="Hello", start=0.0, end=1.0),
    ]

    diarization_segments = [
        DiarizationSegment(start=0.0, end=1.0, speaker_id="SPEAKER_01"),
    ]

    # Act
    result = assigner.assign_speakers(
        transcript_segments, diarization_segments, normalize_speakers=False
    )

    # Assert
    assert result[0].speaker_id == "SPEAKER_01"
