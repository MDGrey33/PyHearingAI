import tempfile
from pathlib import Path

import pytest

from pyhearingai.core.models import Segment, TranscriptionResult
from pyhearingai.infrastructure.formatters.markdown import MarkdownFormatter


@pytest.mark.parametrize(
    "test_transcription_result", ["formatter_segments_with_speakers"], indirect=True
)
def test_markdown_formatter_format(test_transcription_result):
    """Test that the Markdown formatter correctly formats a transcription result."""
    # Format the result using the Markdown formatter
    formatter = MarkdownFormatter()
    md_output = formatter.format(test_transcription_result)

    # Split the output into lines for testing
    lines = md_output.strip().split("\n")

    # Verify structure
    assert lines[0] == "# Transcript"
    assert lines[1] == ""

    # Verify first speaker section
    assert lines[2] == "## Speaker 1"
    assert lines[3] == ""
    # Time format will vary based on implementation, but we can check for asterisks
    assert lines[4].startswith("*") and lines[4].endswith("*")
    assert lines[5] == ""
    assert lines[6] == "This is segment one."

    # Verify second speaker section
    assert "## Speaker 2" in md_output
    assert "This is segment two." in md_output

    # Verify that Speaker 1 appears again after Speaker 2
    assert md_output.count("## Speaker 1") == 2  # Should appear twice since the speaker changes


@pytest.mark.parametrize(
    "test_transcription_result", ["formatter_segments_without_speakers"], indirect=True
)
def test_markdown_formatter_format_no_speaker(test_transcription_result):
    """Test that the Markdown formatter correctly formats segments without speaker IDs."""
    # Format the result using the Markdown formatter
    formatter = MarkdownFormatter()
    md_output = formatter.format(test_transcription_result)

    # Split the output into lines
    lines = md_output.strip().split("\n")

    # Verify structure - should still have transcript header
    assert lines[0] == "# Transcript"
    assert lines[1] == ""

    # Verify the content doesn't have speaker headers
    assert "## " not in md_output

    # Verify segments are included
    assert "This is segment one." in md_output
    assert "This is segment two." in md_output


@pytest.mark.parametrize("test_transcription_result", ["formatter_single_segment"], indirect=True)
def test_markdown_formatter_save(test_transcription_result, tmp_path):
    """Test that the Markdown formatter correctly saves a transcription result to a file."""
    # Create a formatter
    formatter = MarkdownFormatter()

    # Create a path to save to
    output_path = tmp_path / "output.md"

    # Save the result to the file
    saved_path = formatter.save(test_transcription_result, output_path)

    # Verify that the file was saved to the specified path
    assert saved_path == output_path
    assert output_path.exists()

    # Read the saved file
    saved_content = output_path.read_text(encoding="utf-8")

    # Verify the content includes expected markdown elements
    assert "# Transcript" in saved_content
    assert "## Speaker 1" in saved_content
    assert "This is a test segment." in saved_content


def test_markdown_formatter_format_name():
    """Test that the Markdown formatter returns the correct format name."""
    formatter = MarkdownFormatter()
    assert formatter.format_name == "md"
