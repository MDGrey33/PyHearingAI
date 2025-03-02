import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pyhearingai.core.models import Segment, TranscriptionResult
from pyhearingai.infrastructure.formatters.json import JSONFormatter


def test_json_formatter_format():
    """Test that the JSON formatter correctly formats a transcription result."""
    # Create test segments
    segments = [
        Segment(text="This is segment one.", start=0.0, end=2.0, speaker_id="speaker_1"),
        Segment(text="This is segment two.", start=2.5, end=4.5, speaker_id="speaker_2"),
        Segment(text="This is segment three.", start=5.0, end=7.0, speaker_id="speaker_1"),
    ]

    # Create a transcription result with metadata
    metadata = {"audio_file": "test.wav", "duration": 7.0, "language": "en"}
    result = TranscriptionResult(segments=segments, metadata=metadata)

    # Format the result using the JSON formatter
    formatter = JSONFormatter()
    json_output = formatter.format(result)

    # Parse the JSON output
    parsed = json.loads(json_output)

    # Verify structure
    assert "metadata" in parsed
    assert "segments" in parsed

    # Verify metadata
    assert parsed["metadata"] == metadata

    # Verify segments
    assert len(parsed["segments"]) == 3

    # Verify first segment
    assert parsed["segments"][0]["text"] == "This is segment one."
    assert parsed["segments"][0]["start"] == 0.0
    assert parsed["segments"][0]["end"] == 2.0
    assert parsed["segments"][0]["speaker_id"] == "speaker_1"

    # Verify second segment
    assert parsed["segments"][1]["text"] == "This is segment two."
    assert parsed["segments"][1]["start"] == 2.5
    assert parsed["segments"][1]["end"] == 4.5
    assert parsed["segments"][1]["speaker_id"] == "speaker_2"

    # Verify third segment
    assert parsed["segments"][2]["text"] == "This is segment three."
    assert parsed["segments"][2]["start"] == 5.0
    assert parsed["segments"][2]["end"] == 7.0
    assert parsed["segments"][2]["speaker_id"] == "speaker_1"


def test_json_formatter_save():
    """Test that the JSON formatter correctly saves a transcription result to a file."""
    # Create test segments
    segments = [
        Segment(text="This is a test segment.", start=0.0, end=2.0, speaker_id="speaker_1"),
    ]

    # Create a transcription result
    result = TranscriptionResult(segments=segments, metadata={"test": True})

    # Create a formatter
    formatter = JSONFormatter()

    # Create a temporary file to save to
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.json"

        # Save the result to the temporary file
        saved_path = formatter.save(result, output_path)

        # Verify that the file was saved to the specified path
        assert saved_path == output_path
        assert output_path.exists()

        # Read the saved file
        saved_content = output_path.read_text(encoding="utf-8")

        # Parse the saved content
        parsed = json.loads(saved_content)

        # Verify the content
        assert "metadata" in parsed
        assert "segments" in parsed
        assert parsed["metadata"] == {"test": True}
        assert len(parsed["segments"]) == 1
        assert parsed["segments"][0]["text"] == "This is a test segment."
        assert parsed["segments"][0]["speaker_id"] == "speaker_1"


def test_json_formatter_format_name():
    """Test that the JSON formatter returns the correct format name."""
    formatter = JSONFormatter()
    assert formatter.format_name == "json"
