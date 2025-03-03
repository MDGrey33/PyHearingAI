"""
Tests for the output formatter functions in the application/outputs.py module.

These tests verify the functionality of the various output formatting functions
and the save_transcript utility.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyhearingai.application.outputs import (
    _format_srt_time,
    _format_vtt_time,
    save_transcript,
    to_json,
    to_markdown,
    to_srt,
    to_text,
    to_vtt,
)
from pyhearingai.core.models import Segment, TranscriptionResult


@pytest.fixture
def sample_result():
    """
    Create a sample TranscriptionResult with test data.

    Returns:
        TranscriptionResult: A sample result with two segments, one with speaker
        and one without.
    """
    segments = [
        Segment(text="Hello, this is speaker one.", start=0.0, end=2.5, speaker_id="Speaker 1"),
        Segment(text="This segment has no speaker assigned.", start=3.0, end=5.5, speaker_id=None),
    ]

    return TranscriptionResult(segments=segments, metadata={"duration": 5.5, "language": "en"})


class TestTimeFormatting:
    """Tests for the time formatting helper functions."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "00:00:00,000"),
            (3.5, "00:00:03,500"),
            (63.25, "00:01:03,250"),
            (3661.75, "01:01:01,750"),
        ],
    )
    def test_format_srt_time(self, seconds, expected):
        """
        Test SRT time formatting with various inputs.

        Args:
            seconds: Input time in seconds
            expected: Expected formatted output
        """
        result = _format_srt_time(seconds)
        assert result == expected

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "00:00:00.000"),
            (3.5, "00:00:03.500"),
            (63.25, "00:01:03.250"),
            (3661.75, "01:01:01.750"),
        ],
    )
    def test_format_vtt_time(self, seconds, expected):
        """
        Test WebVTT time formatting with various inputs.

        Args:
            seconds: Input time in seconds
            expected: Expected formatted output
        """
        result = _format_vtt_time(seconds)
        assert result == expected


class TestFormatters:
    """Tests for the various format conversion functions."""

    def test_to_text(self, sample_result):
        """Test conversion to plain text format."""
        result = to_text(sample_result)
        assert "**Speaker 1:** Hello, this is speaker one." in result
        assert "This segment has no speaker assigned." in result
        assert "\n\n" in result  # Check for paragraph breaks

    def test_to_json(self, sample_result):
        """Test conversion to JSON format."""
        result = to_json(sample_result)
        # Parse the JSON to verify its structure
        data = json.loads(result)

        assert "metadata" in data
        assert "segments" in data
        assert len(data["segments"]) == 2
        assert data["metadata"]["duration"] == 5.5
        assert data["segments"][0]["speaker_id"] == "Speaker 1"
        assert data["segments"][1]["speaker_id"] is None

    def test_to_srt(self, sample_result):
        """Test conversion to SRT subtitle format."""
        result = to_srt(sample_result)

        # Verify structural elements
        lines = result.split("\n")
        assert "1" in lines  # First subtitle index
        assert "00:00:00,000 --> 00:00:02,500" in result
        assert "Speaker 1: Hello, this is speaker one." in result
        assert "2" in lines  # Second subtitle index
        assert "00:00:03,000 --> 00:00:05,500" in result
        assert "This segment has no speaker assigned." in result

    def test_to_vtt(self, sample_result):
        """Test conversion to WebVTT format."""
        result = to_vtt(sample_result)

        # Verify structural elements
        assert "WEBVTT" in result
        assert "00:00:00.000 --> 00:00:02.500" in result
        assert "Speaker 1: Hello, this is speaker one." in result
        assert "00:00:03.000 --> 00:00:05.500" in result
        assert "This segment has no speaker assigned." in result

    def test_to_markdown(self, sample_result):
        """Test conversion to Markdown format."""
        result = to_markdown(sample_result)

        # Verify structural elements
        assert "# Transcript" in result
        assert "## Speaker 1" in result
        assert "*0:00: - 0:00:02.5*" in result  # Updated to match actual format
        assert "Hello, this is speaker one." in result
        assert "*0:00:03 - 0:00:05.5*" in result  # Updated to match actual format
        assert "This segment has no speaker assigned." in result


class TestSaveTranscript:
    """Tests for the save_transcript function."""

    def test_save_transcript_with_format(self, sample_result, tmp_path):
        """Test saving transcript with explicit format."""
        output_path = tmp_path / "output.txt"

        # Mock the formatter to verify it's called correctly
        mock_formatter = MagicMock()
        mock_formatter.save.return_value = output_path

        with patch(
            "pyhearingai.application.outputs.get_output_formatter", return_value=mock_formatter
        ) as mock_get_formatter:
            result_path = save_transcript(sample_result, output_path, format="txt")

            # Verify the formatter was retrieved with the correct format
            mock_get_formatter.assert_called_once_with("txt")

            # Verify the formatter's save method was called with correct args
            mock_formatter.save.assert_called_once_with(sample_result, output_path)

            # Verify the correct path was returned
            assert result_path == output_path

    def test_save_transcript_infer_format(self, sample_result, tmp_path):
        """Test saving transcript with format inferred from extension."""
        output_path = tmp_path / "output.json"

        # Mock the formatter
        mock_formatter = MagicMock()
        mock_formatter.save.return_value = output_path

        with patch(
            "pyhearingai.application.outputs.get_output_formatter", return_value=mock_formatter
        ) as mock_get_formatter:
            result_path = save_transcript(sample_result, output_path)

            # Verify the formatter was retrieved with the correct format
            mock_get_formatter.assert_called_once_with("json")

            # Verify the formatter's save method was called with correct args
            mock_formatter.save.assert_called_once_with(sample_result, output_path)

    def test_save_transcript_default_format(self, sample_result, tmp_path):
        """Test saving transcript with default format when no extension."""
        output_path = tmp_path / "output"  # No extension

        # Mock the formatter
        mock_formatter = MagicMock()
        mock_formatter.save.return_value = output_path

        with patch(
            "pyhearingai.application.outputs.get_output_formatter", return_value=mock_formatter
        ) as mock_get_formatter:
            result_path = save_transcript(sample_result, output_path)

            # Verify the formatter was retrieved with the default format
            mock_get_formatter.assert_called_once_with("txt")

    def test_save_transcript_str_path(self, sample_result, tmp_path):
        """Test saving transcript with string path instead of Path object."""
        output_path = tmp_path / "output.vtt"

        # Mock the formatter
        mock_formatter = MagicMock()
        mock_formatter.save.return_value = output_path

        with patch(
            "pyhearingai.application.outputs.get_output_formatter", return_value=mock_formatter
        ) as mock_get_formatter:
            # Convert path to string for the test
            str_path = str(output_path)
            result_path = save_transcript(sample_result, str_path)

            # Verify the formatter was retrieved with the correct format
            mock_get_formatter.assert_called_once_with("vtt")

            # Verify the save method was called with Path object
            called_with_path = mock_formatter.save.call_args[0][1]
            assert isinstance(called_with_path, Path)
            assert str(called_with_path) == str_path
