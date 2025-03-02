import tempfile
from pathlib import Path

import pytest

from pyhearingai.core.models import Segment, TranscriptionResult
from pyhearingai.infrastructure.formatters.vtt import VTTFormatter


@pytest.mark.parametrize(
    "test_transcription_result", ["formatter_segments_with_speakers"], indirect=True
)
def test_vtt_formatter_format(test_transcription_result):
    """Test that the VTT formatter correctly formats a transcription result."""
    # Format the result using the VTT formatter
    formatter = VTTFormatter()
    vtt_output = formatter.format(test_transcription_result)

    # Split the output into lines for testing
    lines = vtt_output.strip().split("\n")

    # Verify structure
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""

    # Verify first segment
    assert lines[2] == "00:00:00.000 --> 00:00:02.000"
    assert lines[3] == "Speaker 1: This is segment one."
    assert lines[4] == ""

    # Verify second segment
    assert lines[5] == "00:00:02.500 --> 00:00:04.500"
    assert lines[6] == "Speaker 2: This is segment two."
    assert lines[7] == ""

    # Verify third segment
    assert lines[8] == "00:00:05.000 --> 00:00:07.000"
    assert lines[9] == "Speaker 1: This is segment three."


@pytest.mark.parametrize(
    "test_transcription_result", ["formatter_segments_without_speakers"], indirect=True
)
def test_vtt_formatter_format_no_speaker(test_transcription_result):
    """Test that the VTT formatter correctly formats segments without speaker IDs."""
    # Format the result using the VTT formatter
    formatter = VTTFormatter()
    vtt_output = formatter.format(test_transcription_result)

    # Split the output into lines
    lines = vtt_output.strip().split("\n")

    # Verify structure
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""

    # Verify first segment without speaker ID
    assert lines[2] == "00:00:00.000 --> 00:00:02.000"
    assert lines[3] == "This is segment one."  # No speaker ID prefix
    assert lines[4] == ""

    # Verify second segment without speaker ID
    assert lines[5] == "00:00:02.500 --> 00:00:04.500"
    assert lines[6] == "This is segment two."  # No speaker ID prefix


@pytest.mark.parametrize("test_transcription_result", ["formatter_single_segment"], indirect=True)
def test_vtt_formatter_save(test_transcription_result, tmp_path):
    """Test that the VTT formatter correctly saves a transcription result to a file."""
    # Create a formatter
    formatter = VTTFormatter()

    # Create a path to save to
    output_path = tmp_path / "output.vtt"

    # Save the result to the file
    saved_path = formatter.save(test_transcription_result, output_path)

    # Verify that the file was saved to the specified path
    assert saved_path == output_path
    assert output_path.exists()

    # Read the saved file
    saved_content = output_path.read_text(encoding="utf-8")

    # Verify the content
    assert "WEBVTT" in saved_content
    assert "00:00:00.000 --> 00:00:02.000" in saved_content
    assert "Speaker 1: This is a test segment." in saved_content


def test_vtt_formatter_format_name():
    """Test that the VTT formatter returns the correct format name."""
    formatter = VTTFormatter()
    assert formatter.format_name == "vtt"
