"""
Tests for timestamp utility functions.

This module tests utilities for working with audio timestamps,
converting between formats, and reconciling overlapping segments.
"""

import pytest

from pyhearingai.application.timestamp_utils import (
    absolute_to_relative_time,
    adjust_segment_timestamps,
    adjust_timestamp,
    find_overlapping_segments,
    format_timestamp,
    merge_overlapping_segments,
    parse_timestamp,
    relative_to_absolute_time,
)
from pyhearingai.core.idempotent import SpeakerSegment
from pyhearingai.core.models import Segment


class TestTimestampFormatting:
    """Tests for timestamp formatting and parsing functions."""

    def test_format_timestamp(self):
        """Test formatting timestamps from seconds to string."""
        # Test basic formatting
        assert format_timestamp(0) == "00:00:00.000"
        assert format_timestamp(1.5) == "00:00:01.500"
        assert format_timestamp(61.25) == "00:01:01.250"
        assert format_timestamp(3661.75) == "01:01:01.750"

        # Test without milliseconds
        assert format_timestamp(0, include_ms=False) == "00:00:00"
        assert format_timestamp(1.5, include_ms=False) == "00:00:01"
        assert format_timestamp(61.25, include_ms=False) == "00:01:01"
        assert format_timestamp(3661.75, include_ms=False) == "01:01:01"

    def test_parse_timestamp(self):
        """Test parsing timestamps from string to seconds."""
        # Test basic parsing
        assert parse_timestamp("00:00:00") == 0.0
        assert parse_timestamp("00:00:01.500") == 1.5
        assert parse_timestamp("00:01:01.250") == 61.25
        assert parse_timestamp("01:01:01.750") == 3661.75

        # Test with leading/trailing whitespace
        assert parse_timestamp(" 00:00:01.500 ") == 1.5

        # Test invalid format
        with pytest.raises(ValueError):
            parse_timestamp("00:00")

        with pytest.raises(ValueError):
            parse_timestamp("invalid")


class TestTimestampConversion:
    """Tests for timestamp conversion and adjustment functions."""

    def test_adjust_timestamp(self):
        """Test adjusting timestamps by an offset."""
        # Test positive offset
        assert adjust_timestamp(10.0, 2.5) == 12.5

        # Test negative offset (but not below zero)
        assert adjust_timestamp(10.0, -2.5) == 7.5
        assert adjust_timestamp(1.0, -2.5) == 0.0  # Clamps to zero

    def test_relative_to_absolute_time(self):
        """Test converting chunk-relative time to absolute time."""
        # Basic conversion
        assert relative_to_absolute_time(1.5, 10.0) == 11.5

        # Zero relative time
        assert relative_to_absolute_time(0.0, 10.0) == 10.0

    def test_absolute_to_relative_time(self):
        """Test converting absolute time to chunk-relative time."""
        # Basic conversion
        assert absolute_to_relative_time(11.5, 10.0) == 1.5

        # Absolute time before chunk start (should clamp to zero)
        assert absolute_to_relative_time(5.0, 10.0) == 0.0


class TestSegmentOperations:
    """Tests for operations on segment objects."""

    @pytest.fixture
    def speaker_segments(self):
        """Create a list of speaker segments for testing."""
        return [
            SpeakerSegment(
                job_id="test-job",
                chunk_id="test-chunk",
                speaker_id="speaker1",
                start_time=0.0,
                end_time=2.5,
            ),
            SpeakerSegment(
                job_id="test-job",
                chunk_id="test-chunk",
                speaker_id="speaker2",
                start_time=2.5,
                end_time=5.0,
            ),
        ]

    @pytest.fixture
    def text_segments(self):
        """Create a list of text segments for testing."""
        return [
            Segment(
                text="Hello",
                start=0.0,
                end=2.5,
                speaker_id="speaker1",
            ),
            Segment(
                text="world",
                start=2.5,
                end=5.0,
                speaker_id="speaker2",
            ),
        ]

    def test_adjust_segment_timestamps(self, speaker_segments, text_segments):
        """Test adjusting timestamps of segments."""
        # Test SpeakerSegment adjustment
        adjusted = adjust_segment_timestamps(speaker_segments, 10.0)

        assert len(adjusted) == 2
        assert adjusted[0].start_time == 10.0
        assert adjusted[0].end_time == 12.5
        assert adjusted[1].start_time == 12.5
        assert adjusted[1].end_time == 15.0

        # Test Segment adjustment
        adjusted = adjust_segment_timestamps(text_segments, 10.0)

        assert len(adjusted) == 2
        assert adjusted[0].start == 10.0
        assert adjusted[0].end == 12.5
        assert adjusted[1].start == 12.5
        assert adjusted[1].end == 15.0

        # Test mixed types
        with pytest.raises(TypeError):
            adjust_segment_timestamps([{"not": "a segment"}], 10.0)

    def test_find_overlapping_segments(self, speaker_segments, text_segments):
        """Test finding segments that overlap with a time window."""
        # Test SpeakerSegment overlap
        overlapping = find_overlapping_segments(speaker_segments, 1.0, 3.0)

        assert len(overlapping) == 2
        assert overlapping[0].start_time == 0.0
        assert overlapping[1].start_time == 2.5

        # No overlap
        overlapping = find_overlapping_segments(speaker_segments, 10.0, 15.0)
        assert len(overlapping) == 0

        # Test Segment overlap
        overlapping = find_overlapping_segments(text_segments, 1.0, 3.0)

        assert len(overlapping) == 2
        assert overlapping[0].start == 0.0
        assert overlapping[1].start == 2.5

        # Test mixed types
        with pytest.raises(TypeError):
            find_overlapping_segments([{"not": "a segment"}], 1.0, 3.0)

    def test_merge_overlapping_segments(self):
        """Test merging overlapping segments."""
        # Test merging SpeakerSegments
        segments = [
            SpeakerSegment(
                job_id="test-job",
                chunk_id="test-chunk",
                speaker_id="speaker1",
                start_time=0.0,
                end_time=2.5,
            ),
            SpeakerSegment(
                job_id="test-job",
                chunk_id="test-chunk",
                speaker_id="speaker1",  # Same speaker
                start_time=2.3,  # Overlaps with previous
                end_time=5.0,
            ),
            SpeakerSegment(
                job_id="test-job",
                chunk_id="test-chunk",
                speaker_id="speaker2",  # Different speaker
                start_time=4.8,  # Overlaps but different speaker, shouldn't merge
                end_time=7.0,
            ),
        ]

        merged = merge_overlapping_segments(segments)

        assert len(merged) == 2
        assert merged[0].speaker_id == "speaker1"
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 5.0
        assert merged[1].speaker_id == "speaker2"

        # Test merging Segments
        segments = [
            Segment(
                text="Hello",
                start=0.0,
                end=2.5,
                speaker_id="speaker1",
            ),
            Segment(
                text="world",
                start=2.3,  # Overlaps with previous
                end=5.0,
                speaker_id="speaker1",  # Same speaker
            ),
            Segment(
                text="!",
                start=4.8,  # Overlaps but different speaker, shouldn't merge
                end=7.0,
                speaker_id="speaker2",  # Different speaker
            ),
        ]

        merged = merge_overlapping_segments(segments)

        assert len(merged) == 2
        assert merged[0].speaker_id == "speaker1"
        assert merged[0].start == 0.0
        assert merged[0].end == 5.0
        assert merged[0].text == "Hello world"  # Texts merged
        assert merged[1].speaker_id == "speaker2"
        assert merged[1].text == "!"

        # Test empty list
        merged = merge_overlapping_segments([])
        assert merged == []
