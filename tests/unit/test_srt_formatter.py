import tempfile
from pathlib import Path

import pytest

from pyhearingai.core.models import Segment, TranscriptionResult
from pyhearingai.infrastructure.formatters.srt import SRTFormatter


def test_srt_formatter_format():
    """Test that the SRT formatter correctly formats a transcription result."""
    # Create test segments
    segments = [
        Segment(text="This is segment one.", start=0.0, end=2.0, speaker_id="Speaker 1"),
        Segment(text="This is segment two.", start=2.5, end=4.5, speaker_id="Speaker 2"),
        Segment(text="This is segment three.", start=5.0, end=7.0, speaker_id="Speaker 1"),
    ]

    # Create a transcription result
    result = TranscriptionResult(segments=segments)

    # Format the result using the SRT formatter
    formatter = SRTFormatter()
    srt_output = formatter.format(result)

    # Split the output into lines
    lines = srt_output.strip().split("\n")

    # Verify structure - each segment has 4 lines except the last one has 3 (no trailing blank line)
    assert len(lines) == 11  # 2 segments * 4 lines + last segment * 3 lines

    # Verify first segment
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,000"
    assert lines[2] == "Speaker 1: This is segment one."
    assert lines[3] == ""

    # Verify second segment
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,500 --> 00:00:04,500"
    assert lines[6] == "Speaker 2: This is segment two."
    assert lines[7] == ""

    # Verify third segment
    assert lines[8] == "3"
    assert lines[9] == "00:00:05,000 --> 00:00:07,000"
    assert lines[10] == "Speaker 1: This is segment three."


def test_srt_formatter_format_no_speaker():
    """Test that the SRT formatter correctly formats segments without speaker IDs."""
    # Create test segments without speaker IDs
    segments = [
        Segment(text="This is segment one.", start=0.0, end=2.0),
        Segment(text="This is segment two.", start=2.5, end=4.5),
    ]

    # Create a transcription result
    result = TranscriptionResult(segments=segments)

    # Format the result using the SRT formatter
    formatter = SRTFormatter()
    srt_output = formatter.format(result)

    # Split the output into lines
    lines = srt_output.strip().split("\n")

    # Verify structure - each segment has 4 lines except the last one has 3 (no trailing blank line)
    assert len(lines) == 7  # 1 segment * 4 lines + last segment * 3 lines

    # Verify first segment
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,000"
    assert lines[2] == "This is segment one."  # No speaker ID prefix
    assert lines[3] == ""

    # Verify second segment
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,500 --> 00:00:04,500"
    assert lines[6] == "This is segment two."  # No speaker ID prefix


def test_srt_formatter_save():
    """Test that the SRT formatter correctly saves a transcription result to a file."""
    # Create test segment
    segments = [
        Segment(text="This is a test segment.", start=0.0, end=2.0, speaker_id="Speaker 1"),
    ]

    # Create a transcription result
    result = TranscriptionResult(segments=segments)

    # Create a formatter
    formatter = SRTFormatter()

    # Create a temporary file to save to
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.srt"

        # Save the result to the temporary file
        saved_path = formatter.save(result, output_path)

        # Verify that the file was saved to the specified path
        assert saved_path == output_path
        assert output_path.exists()

        # Read the saved file
        saved_content = output_path.read_text(encoding="utf-8")

        # Verify the content - the file has a trailing newline
        expected_content = "1\n00:00:00,000 --> 00:00:02,000\nSpeaker 1: This is a test segment.\n"
        assert saved_content == expected_content


def test_srt_formatter_format_name():
    """Test that the SRT formatter returns the correct format name."""
    formatter = SRTFormatter()
    assert formatter.format_name == "srt"
